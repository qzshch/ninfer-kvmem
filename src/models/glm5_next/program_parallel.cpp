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
    if (!options_.graph || kind_ != runtime::TokenBatchKind::Decode) { body(); return; }
    const int d = component % 3;
    const std::array<int, 4> key{batch_, tier_, category, component};
    auto [it, fresh] = tensor_graphs_.try_emplace(key);
    if (fresh) {
        DecodeGraphDefinition definition;
        definition.capture(devices_[d]->stream, body);
        it->second.instantiate(definition);
        it->second.upload(devices_[d]->stream);
        ++stats_.graph_captures;
    }
    it->second.launch(devices_[d]->stream);
    ++stats_.graph_replays;
}

void Program::tensor_phase(int layer, int d, bool ffn) {
    auto& work = *state_[d].workspace;
    work.reset();
    const auto stream = devices_[d]->stream;
    const auto& p = model_.parameters(d).layers[layer];
    auto* residual = static_cast<float*>(state_[d].streams.p);
    auto* coeff = static_cast<float*>(state_[d].coefficients.p);
    auto* x = static_cast<float*>(state_[d].normalized.p);
    auto* partial = static_cast<float*>(state_[d].partial.p);
    if (layer || ffn)
        ops::hyper_connection_expand(residual, static_cast<float*>(state_[d].hidden.p),
            coeff, residual, H, batch_, stream);
    hyper_prepare(residual, ffn ? p.ffn_hc : p.attention_hc, coeff, x, d, batch_);
    const auto& norm = ffn ? p.ffn_norm : p.attention_norm;
    ops::resident_norm(x, &norm, nullptr, nullptr, x, H, batch_, 1e-5f, false, true, stream);
    if (ffn) feed_forward(x, p, partial, batch_);
    else if (p.linear) kda(x, p.kda, layer_state(d, layer), partial, d, batch_);
    else mla(x, p.mla, layer_state(d, layer), partial, d, batch_);
    stats_.workspace_peak[d] = std::max(stats_.workspace_peak[d], work.peak_used());
}

void Program::tensor_reduce(bool routed) {
    const auto count = static_cast<std::size_t>(H) * batch_;
    const auto bytes = count * sizeof(float) * (routed ? 2 : 1);
    for (int d = 0; d < 3; ++d) {
        devices_[d]->bind_to_current_thread();
        state_[d].ready->record(devices_[d]->stream);
    }
    devices_[0]->bind_to_current_thread();
    auto* received = static_cast<float*>(state_[0].received.p);
    for (int d = 1; d < 3; ++d) {
        state_[d].ready->wait(devices_[0]->stream);
        CUDA_CHECK(hipMemcpyPeerAsync(reinterpret_cast<std::byte*>(received) + (d - 1) * bytes,
            devices_[0]->device, state_[d].partial.p, devices_[d]->device, bytes, devices_[0]->stream));
        stats_.peer_bytes += bytes;
    }
    tensor_launch(ReductionGraph, routed ? 3 : 0, [&] {
        ops::resident_tp_sum(static_cast<const float*>(state_[0].partial.p), received,
            received + count * (routed ? 2 : 1), static_cast<float*>(state_[0].hidden.p),
            count, routed, devices_[0]->stream);
    });
    state_[0].ready->record(devices_[0]->stream);
    for (int d = 1; d < 3; ++d) {
        devices_[d]->bind_to_current_thread();
        state_[0].ready->wait(devices_[d]->stream);
        CUDA_CHECK(hipMemcpyPeerAsync(state_[d].hidden.p, devices_[d]->device,
            state_[0].hidden.p, devices_[0]->device, count * sizeof(float), devices_[d]->stream));
        stats_.peer_bytes += count * sizeof(float);
    }
    // A rank cannot overwrite its partials before root has copied them: its
    // next producer is ordered after this root-result wait on the same stream.
}

void Program::tensor_finalize(int d) {
    const auto stream = devices_[d]->stream;
    auto* residual = static_cast<float*>(state_[d].streams.p);
    ops::hyper_connection_expand(residual, static_cast<float*>(state_[d].hidden.p),
        static_cast<float*>(state_[d].coefficients.p), residual, H, batch_, stream);
    if (d || !outputs_) return;
    auto& work = *state_[0].workspace;
    work.reset();
    const float* selected = residual;
    if (outputs_ != batch_) {
        auto* compact = floats(work, 4 * H * outputs_);
        ops::resident_gather_rows(selected, metadata(0, OutputRows), compact, 4 * H, outputs_, stream);
        selected = compact;
    }
    ops::mean_streams(selected, static_cast<float*>(state_[0].hidden.p), H, outputs_, stream);
    ops::resident_norm(static_cast<float*>(state_[0].hidden.p), &parameters_.output_norm,
        nullptr, nullptr, static_cast<float*>(state_[0].hidden.p), H, outputs_, 1e-5f, false, true, stream);
}

std::vector<std::int32_t> Program::tensor_execute() {
    const bool graph = options_.graph && kind_ == runtime::TokenBatchKind::Decode;
    if (graph) {
        const std::array<int, 2> shape{batch_, tier_};
        if (!tensor_graph_families_.contains(shape) && tensor_graph_families_.size() == 2) {
            const auto victim = std::min_element(tensor_graph_families_.begin(), tensor_graph_families_.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            for (auto it = tensor_graphs_.begin(); it != tensor_graphs_.end();) {
                if (it->first[0] == victim->first[0] && it->first[1] == victim->first[1]) {
                    devices_[it->first[3] % 3]->bind_to_current_thread();
                    it = tensor_graphs_.erase(it);
                } else ++it;
            }
            tensor_graph_families_.erase(victim);
        }
        tensor_graph_families_[shape] = ++tensor_graph_clock_;
    }
    devices_[0]->bind_to_current_thread();
    auto& work = *state_[0].workspace;
    tensor_launch(EmbeddingGraph, 0, [&] {
        work.reset();
        auto* embedding = floats(work, H * batch_);
        ops::embedding_fp32(parameters_.embedding, metadata(0, TokenIds), embedding, batch_, devices_[0]->stream);
        ops::replicate_streams(embedding, static_cast<float*>(state_[0].streams.p), H, batch_, devices_[0]->stream);
    });
    state_[0].ready->record(devices_[0]->stream);
    for (int d = 1; d < 3; ++d) {
        devices_[d]->bind_to_current_thread();
        state_[0].ready->wait(devices_[d]->stream);
        const auto bytes = static_cast<std::size_t>(4 * H) * batch_ * sizeof(float);
        CUDA_CHECK(hipMemcpyPeerAsync(state_[d].streams.p, devices_[d]->device,
            state_[0].streams.p, devices_[0]->device, bytes, devices_[d]->stream));
        stats_.peer_bytes += bytes;
    }
    for (int layer = 0; layer < 45; ++layer)
        for (int ffn = 0; ffn < 2; ++ffn) {
            for (int d = 0; d < 3; ++d) {
                devices_[d]->bind_to_current_thread();
                tensor_launch(layer, ffn * 3 + d, [&] { tensor_phase(layer, d, ffn); });
            }
            tensor_reduce(ffn && !parameters_.layers[layer].dense);
        }
    std::vector<std::int32_t> result(outputs_);
    for (int d = 1; d < 3; ++d) {
        devices_[d]->bind_to_current_thread();
        tensor_launch(FinalizeGraph, d, [&] { tensor_finalize(d); });
        state_[d].ready->record(devices_[d]->stream);
    }
    devices_[0]->bind_to_current_thread();
    // Last-layer peers still read root's result until their expand completes.
    // The next head may overwrite that same hidden buffer only after this join.
    for (int d = 1; d < 3; ++d) state_[d].ready->wait(devices_[0]->stream);
    tensor_launch(FinalizeGraph, 0, [&] { tensor_finalize(0); });
    if (outputs_) {
        state_[0].ready->record(devices_[0]->stream);
        for (int d = 1; d < 3; ++d) {
            devices_[d]->bind_to_current_thread();
            state_[0].ready->wait(devices_[d]->stream);
            const auto bytes = static_cast<std::size_t>(H) * outputs_ * sizeof(float);
            CUDA_CHECK(hipMemcpyPeerAsync(state_[d].hidden.p, devices_[d]->device,
                state_[0].hidden.p, devices_[0]->device, bytes, devices_[d]->stream));
            stats_.peer_bytes += bytes;
        }
        for (int d = 0; d < 3; ++d) {
            devices_[d]->bind_to_current_thread();
            auto& local_work = *state_[d].workspace;
            tensor_launch(HeadGraph, d, [&] {
                local_work.reset();
                const auto& p = model_.parameters(d);
                auto* logits = floats(local_work, static_cast<std::size_t>(p.output_head.n) * outputs_);
                projection(static_cast<const float*>(state_[d].hidden.p), p.output_head, logits, outputs_, d, false);
                ops::resident_argmax_pairs(logits, static_cast<ops::ArgmaxPair*>(state_[d].partial.p),
                    p.output_head.n, outputs_, p.output_begin, devices_[d]->stream);
            });
            state_[d].ready->record(devices_[d]->stream);
            stats_.workspace_peak[d] = std::max(stats_.workspace_peak[d], local_work.peak_used());
        }
        devices_[0]->bind_to_current_thread();
        const auto bytes = outputs_ * sizeof(ops::ArgmaxPair);
        auto* pairs = static_cast<ops::ArgmaxPair*>(state_[0].received.p);
        CUDA_CHECK(cudaMemcpyAsync(pairs, state_[0].partial.p, bytes, cudaMemcpyDeviceToDevice, devices_[0]->stream));
        for (int d = 1; d < 3; ++d) {
            state_[d].ready->wait(devices_[0]->stream);
            CUDA_CHECK(hipMemcpyPeerAsync(pairs + d * outputs_, devices_[0]->device,
                state_[d].partial.p, devices_[d]->device, bytes, devices_[0]->stream));
            stats_.peer_bytes += bytes;
        }
        tensor_launch(ArgmaxGraph, 0, [&] {
            ops::resident_argmax_merge(pairs, metadata(0, PredictedIds), outputs_, devices_[0]->stream);
        });
        devices_[0]->synchronize();
        CUDA_CHECK(cudaMemcpy(result.data(), metadata(0, PredictedIds), outputs_ * sizeof(int), cudaMemcpyDeviceToHost));
    }
    // Complete every rank before the owning frame publishes sequence frontiers
    // or releases metadata. No per-projection or per-layer host waits.
    for (int d = 0; d < 3; ++d) { devices_[d]->bind_to_current_thread(); devices_[d]->synchronize(); }
    for (auto id : result) if (id < 0) throw std::runtime_error("GLM TP nonfinite logits");
    return result;
}
} // namespace ninfer::models::glm5_next
