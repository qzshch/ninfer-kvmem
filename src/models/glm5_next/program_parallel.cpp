#include "models/glm5_next/math.h"
#include "models/glm5_next/program.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/indexed_linear.h"
#include "ninfer/ops/resident_transform.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::models::glm5_next {
namespace {
constexpr int H = 4096;
// Categories 0..44 are the fixed GLM layers, with attention/FFN components.
enum GraphRegion { EmbeddingGraph = 45, FinalizeGraph, HeadGraph, ArgmaxGraph, ReductionGraph };
}

void Program::tensor_launch(int category, int component, const std::function<void()>& body) {
    if (!options_.graph || frame_->kind != runtime::TokenBatchKind::Decode) { body(); return; }
    const int d = component % 3;
    const std::array<int, 4> key{frame_->batch, frame_->tier, category, component};
    auto [it, fresh] = frame_->tensor_graphs.try_emplace(key);
    if (fresh) {
        DecodeGraphDefinition definition;
        definition.capture(frame_->devices[d]->stream, body);
        it->second.instantiate(definition);
        it->second.upload(frame_->devices[d]->stream);
        ++stats_.graph_captures;
    }
    it->second.launch(frame_->devices[d]->stream);
    ++stats_.graph_replays;
}

void Program::tensor_phase(int layer, int d, bool ffn) {
    auto& work = *frame_->state[d].workspace;
    work.reset();
    const auto stream = frame_->devices[d]->stream;
    const auto& p = model_.parameters(d).layers[layer];
    auto* residual = static_cast<float*>(frame_->state[d].streams.p);
    auto* coeff = static_cast<float*>(frame_->state[d].coefficients.p);
    auto* collapsed = floats(work, H * frame_->batch);
    Tensor source(collapsed, DType::FP32, {H, frame_->batch});
    Tensor x(frame_->state[d].normalized.p, options_.bf16_activations ? DType::BF16 : DType::FP32, {H, frame_->batch});
    auto* partial = static_cast<float*>(frame_->state[d].partial.p);
    if (layer || ffn)
        ops::hyper_connection_expand(residual, static_cast<float*>(frame_->state[d].hidden.p),
            coeff, residual, H, frame_->batch, stream);
    hyper_prepare(residual, ffn ? p.ffn_hc : p.attention_hc, coeff, collapsed, d, frame_->batch);
    const auto& norm = ffn ? p.ffn_norm : p.attention_norm;
    if (options_.bf16_activations) {
        ops::resident_norm(source, &norm, nullptr, nullptr, x, rms_norm_epsilon, false, stream);
        if (ffn) feed_forward(x, p, partial, frame_->batch);
        else if (p.linear) kda(x, p.kda, layer_state(d, layer), partial, d, frame_->batch);
        else mla(x, p.mla, layer_state(d, layer), partial, d, frame_->batch);
    } else {
        auto* input = static_cast<float*>(x.data);
        ops::resident_norm(collapsed, &norm, nullptr, nullptr, input, H, frame_->batch, rms_norm_epsilon, false, true, stream);
        if (ffn) feed_forward(input, p, partial, frame_->batch);
        else if (p.linear) kda(input, p.kda, layer_state(d, layer), partial, d, frame_->batch);
        else mla(input, p.mla, layer_state(d, layer), partial, d, frame_->batch);
    }
    stats_.workspace_peak[d] = std::max(stats_.workspace_peak[d], work.peak_used());
}

void Program::tensor_reduce(bool routed) {
    const auto count = static_cast<std::size_t>(H) * frame_->batch;
    const auto bytes = count * sizeof(float) * (routed ? 2 : 1);
    // Producers recorded readiness on their already-bound device immediately
    // after enqueueing the phase. Only the root gather needs a context switch.
    frame_->devices[0]->bind_to_current_thread();
    auto* received = static_cast<float*>(frame_->state[0].received.p);
    for (int d = 1; d < 3; ++d) {
        frame_->state[d].ready->wait(frame_->devices[0]->stream);
        CUDA_CHECK(hipMemcpyPeerAsync(reinterpret_cast<std::byte*>(received) + (d - 1) * bytes,
            frame_->devices[0]->device, frame_->state[d].partial.p, frame_->devices[d]->device, bytes, frame_->devices[0]->stream));
        stats_.peer_bytes += bytes;
    }
    tensor_launch(ReductionGraph, routed ? 3 : 0, [&] {
        ops::resident_tp_sum(static_cast<const float*>(frame_->state[0].partial.p), received,
            received + count * (routed ? 2 : 1), static_cast<float*>(frame_->state[0].hidden.p),
            count, routed, frame_->devices[0]->stream);
    });
    frame_->state[0].ready->record(frame_->devices[0]->stream);
    for (int d = 1; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        frame_->state[0].ready->wait(frame_->devices[d]->stream);
        CUDA_CHECK(hipMemcpyPeerAsync(frame_->state[d].hidden.p, frame_->devices[d]->device,
            frame_->state[0].hidden.p, frame_->devices[0]->device, count * sizeof(float), frame_->devices[d]->stream));
        stats_.peer_bytes += count * sizeof(float);
    }
    // A rank cannot overwrite its partials before root has copied them: its
    // next producer is ordered after this root-result wait on the same stream.
}

void Program::tensor_finalize(int d) {
    const auto stream = frame_->devices[d]->stream;
    auto* residual = static_cast<float*>(frame_->state[d].streams.p);
    ops::hyper_connection_expand(residual, static_cast<float*>(frame_->state[d].hidden.p),
        static_cast<float*>(frame_->state[d].coefficients.p), residual, H, frame_->batch, stream);
    if (d || !frame_->outputs) return;
    auto& work = *frame_->state[0].workspace;
    work.reset();
    const float* selected = residual;
    if (frame_->outputs != frame_->batch) {
        auto* compact = floats(work, 4 * H * frame_->outputs);
        ops::resident_gather_rows(selected, metadata(0, OutputRows), compact, 4 * H, frame_->outputs, stream);
        selected = compact;
    }
    ops::mean_streams(selected, static_cast<float*>(frame_->state[0].hidden.p), H, frame_->outputs, stream);
    Tensor source(frame_->state[0].hidden.p, DType::FP32, {H, frame_->outputs});
    Tensor normalized(frame_->state[0].normalized.p, options_.bf16_activations ? DType::BF16 : DType::FP32, {H, frame_->outputs});
    if (options_.bf16_activations)
        ops::resident_norm(source, &parameters_.output_norm, nullptr, nullptr, normalized, rms_norm_epsilon, false, stream);
    else ops::resident_norm(static_cast<const float*>(source.data), &parameters_.output_norm,
        nullptr, nullptr, static_cast<float*>(normalized.data), H, frame_->outputs, rms_norm_epsilon, false, true, stream);
}

void Program::tensor_execute() {
    const bool graph = options_.graph && frame_->kind == runtime::TokenBatchKind::Decode;
    if (graph) {
        const std::array<int, 2> shape{frame_->batch, frame_->tier};
        if (!frame_->tensor_graph_families.contains(shape) && frame_->tensor_graph_families.size() == 2) {
            const auto victim = std::min_element(frame_->tensor_graph_families.begin(), frame_->tensor_graph_families.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            for (auto it = frame_->tensor_graphs.begin(); it != frame_->tensor_graphs.end();) {
                if (it->first[0] == victim->first[0] && it->first[1] == victim->first[1]) {
                    frame_->devices[it->first[3] % 3]->bind_to_current_thread();
                    it = frame_->tensor_graphs.erase(it);
                } else ++it;
            }
            frame_->tensor_graph_families.erase(victim);
        }
        frame_->tensor_graph_families[shape] = ++frame_->tensor_graph_clock;
    }
    frame_->devices[0]->bind_to_current_thread();
    auto& work = *frame_->state[0].workspace;
    tensor_launch(EmbeddingGraph, 0, [&] {
        work.reset();
        auto* embedding = floats(work, H * frame_->batch);
        ops::embedding_fp32(parameters_.embedding, metadata(0, TokenIds), embedding, frame_->batch, frame_->devices[0]->stream);
        ops::replicate_streams(embedding, static_cast<float*>(frame_->state[0].streams.p), H, frame_->batch, frame_->devices[0]->stream);
    });
    frame_->state[0].ready->record(frame_->devices[0]->stream);
    for (int d = 1; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        frame_->state[0].ready->wait(frame_->devices[d]->stream);
        const auto bytes = static_cast<std::size_t>(4 * H) * frame_->batch * sizeof(float);
        CUDA_CHECK(hipMemcpyPeerAsync(frame_->state[d].streams.p, frame_->devices[d]->device,
            frame_->state[0].streams.p, frame_->devices[0]->device, bytes, frame_->devices[d]->stream));
        stats_.peer_bytes += bytes;
    }
    for (int layer = 0; layer < 45; ++layer)
        for (int ffn = 0; ffn < 2; ++ffn) {
            for (int d = 0; d < 3; ++d) {
                frame_->devices[d]->bind_to_current_thread();
                tensor_launch(layer, ffn * 3 + d, [&] { tensor_phase(layer, d, ffn); });
                frame_->state[d].ready->record(frame_->devices[d]->stream);
            }
            tensor_reduce(ffn && !parameters_.layers[layer].dense);
        }
    for (int d = 1; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        tensor_launch(FinalizeGraph, d, [&] { tensor_finalize(d); });
        frame_->state[d].ready->record(frame_->devices[d]->stream);
    }
    frame_->devices[0]->bind_to_current_thread();
    // Last-layer peers still read root's result until their expand completes.
    // The next head may overwrite that same hidden buffer only after this join.
    for (int d = 1; d < 3; ++d) frame_->state[d].ready->wait(frame_->devices[0]->stream);
    tensor_launch(FinalizeGraph, 0, [&] { tensor_finalize(0); });
    if (frame_->outputs) {
        frame_->state[0].ready->record(frame_->devices[0]->stream);
        for (int d = 1; d < 3; ++d) {
            frame_->devices[d]->bind_to_current_thread();
            frame_->state[0].ready->wait(frame_->devices[d]->stream);
            const auto bytes = static_cast<std::size_t>(H) * frame_->outputs * (options_.bf16_activations ? 2 : 4);
            CUDA_CHECK(hipMemcpyPeerAsync(frame_->state[d].normalized.p, frame_->devices[d]->device,
                frame_->state[0].normalized.p, frame_->devices[0]->device, bytes, frame_->devices[d]->stream));
            stats_.peer_bytes += bytes;
        }
        for (int d = 0; d < 3; ++d) {
            frame_->devices[d]->bind_to_current_thread();
            auto& local_work = *frame_->state[d].workspace;
            tensor_launch(HeadGraph, d, [&] {
                local_work.reset();
                const auto& p = model_.parameters(d);
                auto* logits = floats(local_work, static_cast<std::size_t>(p.output_head.n) * frame_->outputs);
                Tensor input(frame_->state[d].normalized.p, options_.bf16_activations ? DType::BF16 : DType::FP32, {H, frame_->outputs});
                Tensor scores(logits, DType::FP32, {p.output_head.n, frame_->outputs});
                projection(input, p.output_head, scores, d);
                ops::resident_argmax_pairs(logits, static_cast<ops::ArgmaxPair*>(frame_->state[d].partial.p),
                    p.output_head.n, frame_->outputs, p.output_begin, frame_->devices[d]->stream);
            });
            frame_->state[d].ready->record(frame_->devices[d]->stream);
            stats_.workspace_peak[d] = std::max(stats_.workspace_peak[d], local_work.peak_used());
        }
        frame_->devices[0]->bind_to_current_thread();
        const auto bytes = frame_->outputs * sizeof(ops::ArgmaxPair);
        auto* pairs = static_cast<ops::ArgmaxPair*>(frame_->state[0].received.p);
        CUDA_CHECK(cudaMemcpyAsync(pairs, frame_->state[0].partial.p, bytes, cudaMemcpyDeviceToDevice, frame_->devices[0]->stream));
        for (int d = 1; d < 3; ++d) {
            frame_->state[d].ready->wait(frame_->devices[0]->stream);
            CUDA_CHECK(hipMemcpyPeerAsync(pairs + d * frame_->outputs, frame_->devices[0]->device,
                frame_->state[d].partial.p, frame_->devices[d]->device, bytes, frame_->devices[0]->stream));
            stats_.peer_bytes += bytes;
        }
        tensor_launch(ArgmaxGraph, 0, [&] {
            ops::resident_argmax_merge(pairs, metadata(0, PredictedIds), frame_->outputs, frame_->devices[0]->stream);
        });
    }
}
} // namespace ninfer::models::glm5_next
