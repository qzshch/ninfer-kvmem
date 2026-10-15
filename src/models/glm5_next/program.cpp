#include "models/glm5_next/program.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/indexed_linear.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/kimi_delta_attention.h"
#include "ninfer/ops/resident_transform.h"
#include "ninfer/ops/sparse_latent_attention.h"
#include "ninfer/ops/dflash_block.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::glm5_next {
namespace {
constexpr int H = 4096, Features = 20480, FeatureSlots = 2056;

const float* fp(const Tensor& x) { return static_cast<const float*>(x.data); }
} // namespace

Program::Program(const Model& model, std::array<DeviceContext*, 3> devices, ProgramOptions options)
    : model_(model), parameters_(model.parameters()), options_(options),
      ledger_(options.max_lanes, options.decode_frames),
      tensor_parallel_(model.parallelism() == Parallelism::TensorExpert) {
    if (tensor_parallel_ && (options.draft || options.mtp))
        throw std::invalid_argument("GLM tensor/expert Program requires ordinary execution");
    if (options.max_lanes < 1 || options.max_lanes > 8 ||
        (options.draft && (options.max_lanes != 1 || options.graph)) || options.max_context < 1 ||
        options.max_context > 32768 ||
        (options.kvmem_window_pages &&
         (options.kvmem_window_pages < 4 || options.kvmem_window_pages % 2)))
        throw std::invalid_argument(
            "GLM resident Program requires context 1..32768 and even KVMem window >=4 pages");
    if (options.mtp && (options.draft || options.max_lanes != 1))
        throw std::invalid_argument("native MTP requires one compact target lane");
    if (!options.prefill_chunk) throw std::invalid_argument("GLM prefill chunk must be positive");
    // The input extent is bounded by this Program's fixed resident workspace.
    // Speculative profiles retain their one-input execution and resource quote.
    const int constructor_rows = (options.draft || options.mtp)
                    ? options.max_lanes
                    : std::max(options.max_lanes, static_cast<int>(std::min(options.prefill_chunk, 1024u)));
    if (options.decode_frames > 1 && (options.draft || options.mtp))
        throw std::invalid_argument("GLM asynchronous frames require ordinary execution");
    const int primary_rows = constructor_rows;
    for (unsigned f = 0; f < options.decode_frames; ++f) {
        frames_[f] = std::make_unique<Frame>();
        auto& frame = *frames_[f];
        frame.capacity = f ? options.max_lanes : primary_rows;
        frame.host_metadata.resize(5 * frame.capacity + 2 * options.max_lanes + 1);
        frame.upload = std::make_unique<PinnedHostBuffer>(frame.host_metadata.size() * sizeof(int), true);
        frame.result = std::make_unique<PinnedHostBuffer>(frame.capacity * sizeof(int), true);
        for (int d = 0; d < 3; ++d) {
            if (f) frame.owned_devices[d] = std::make_unique<DeviceContext>(devices[d]->device);
            frame.devices[d] = f ? frame.owned_devices[d].get() : devices[d];
        }
        frame.completed = std::make_unique<CudaCompletionEvent>(*frame.devices[0]);
    }
    frame_ = frames_[0].get();
    std::array<std::size_t, 3> counts{};
    for (const auto& l : parameters_.layers)
        if (l.linear) {
            if (tensor_parallel_) for (auto& count : counts) ++count;
            else ++counts[l.device];
        }
    if (tensor_parallel_) for (int d = 0; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        const auto context = static_cast<std::size_t>(options.max_context);
        const auto channels = static_cast<std::size_t>(model_.parameters(d).layers[0].kda.query.n);
        std::size_t need = counts[d] * channels * (128 + 9) * sizeof(float) * options.max_lanes;
        const auto mla = (context * (512 + 128 + 128) * 2 +
            ((context + 3) / 4) * 128 * 2 + ((context + 127) / 128) * 128 * 4) * options.max_lanes;
        for (const auto& p : model_.parameters(d).layers) if (!p.linear) need += mla;
        for (unsigned f = 0; f < options.decode_frames; ++f) {
            const auto& frame = *frames_[f];
            need += static_cast<std::size_t>(8 * H + (d == 0 ? 4 * H : 0) + 24) * 4 * frame.capacity;
            need += frame.host_metadata.size() * sizeof(int) + (f ? (256ULL << 20) : (2ULL << 30));
        }
        std::size_t available = 0, total = 0;
        CUDA_CHECK(hipMemGetInfo(&available, &total));
        if (need > available || available - need < (512ULL << 20))
            throw std::runtime_error("GLM TP/EP resident state/workspace and reserve exceed free device memory; reduce resident context/lanes");
    }
    for (int d = 0; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        for (int p = 0; p < 3; ++p)
            if (p != d) {
                int supported = 0;
                CUDA_CHECK(hipDeviceCanAccessPeer(&supported, d, p));
                if (!supported) throw std::runtime_error("GLM requires direct device peer access");
                auto e = hipDeviceEnablePeerAccess(p, 0);
                if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
            }
        const auto channels = tensor_parallel_ ? model_.parameters(d).layers[0].kda.query.n : 8192;
        const auto bytes = counts[d] * static_cast<std::size_t>(channels) * (128 + 9) * 4;
        for (int slot = 0; slot < (options.draft ? 9 : 1); ++slot)
            recurrent_[d][slot] = DeviceBuffer(bytes * options.max_lanes);
        stats_.persistent_bytes[d] = bytes * (options.draft ? 9 : options.max_lanes);
        for (unsigned f = 0; f < options.decode_frames; ++f) {
            frame_ = frames_[f].get();
            const auto rows = frame_->capacity;
            auto& local = frame_->state[d];
            local.streams = DeviceBuffer(4ULL * H * 4 * rows);
            local.metadata = DeviceBuffer(frame_->host_metadata.size() * sizeof(int));
            local.hidden = DeviceBuffer(H * 4ULL * rows);
            local.workspace = std::make_unique<WorkspaceArena>(f ? (256ULL << 20) : (2ULL << 30));
            local.ready = std::make_unique<CudaCompletionEvent>(*frame_->devices[d]);
            local.copied = std::make_unique<CudaCompletionEvent>(*frame_->devices[d]);
            local.done = std::make_unique<CudaCompletionEvent>(*frame_->devices[d]);
            if (tensor_parallel_) {
                local.partial = DeviceBuffer(2ULL * H * 4 * rows);
                if (!d) local.received = DeviceBuffer(4ULL * H * 4 * rows);
                local.coefficients = DeviceBuffer(24ULL * 4 * rows);
                local.normalized = DeviceBuffer(H * (options.bf16_activations ? 2ULL : 4ULL) * rows);
            }
            stats_.persistent_bytes[d] += local.streams.bytes + local.metadata.bytes + local.hidden.bytes +
                local.partial.bytes + local.received.bytes + local.coefficients.bytes + local.normalized.bytes;
        }
        frame_ = frames_[0].get();
    }
    std::array<std::size_t, 3> cursor{};
    for (int d = 0; d < 3; ++d)
    for (int i = 0; i < 45; ++i) {
        const auto& p = model_.parameters(d).layers[i];
        if (!tensor_parallel_ && p.device != d) continue;
        auto& l = layer_state(d, i);
        frame_->devices[d]->bind_to_current_thread();
        if (p.linear) {
            const auto channels = static_cast<std::size_t>(p.kda.query.n);
            const auto state_floats = channels * 128;
            // Layer-major pools: a compact execution row selects a physical
            // state directly, without gathering the complete recurrent image.
            auto* base  = static_cast<float*>(recurrent_[d][0].p) + cursor[d];
            l.recurrent = base;
            for (int j = 0; j < 3; ++j)
                l.conv[j] = base + state_floats * options.max_lanes +
                            j * channels * 3 * options.max_lanes;
            cursor[d] += (state_floats + 9 * channels) * options.max_lanes;
        } else {
            const auto context = options_.max_context * options_.max_lanes;
            l.latent           = DeviceBuffer(static_cast<std::size_t>(context) * 512 * 2);
            l.keys             = DeviceBuffer(static_cast<std::size_t>(context) * 128 * 2);
            l.gates            = DeviceBuffer(l.keys.bytes);
            l.pools = DeviceBuffer(static_cast<std::size_t>((options_.max_context + 3) / 4) *
                                   options_.max_lanes * 128 * 2);
            l.means = DeviceBuffer(static_cast<std::size_t>((options_.max_context + 127) / 128) *
                                   options_.max_lanes * 128 * 4);
            if (options_.draft)
                l.retrieval = std::make_unique<kvmem::RetrievalIndex>(128, 1, 1, 128);
            stats_.persistent_bytes[d] +=
                l.latent.bytes + l.keys.bytes + l.gates.bytes + l.pools.bytes + l.means.bytes;
        }
    }
    frame_->devices[2]->bind_to_current_thread();
    if (options.draft)
        features_ = DeviceBuffer(static_cast<std::size_t>(FeatureSlots) * Features * 2);
    stats_.persistent_bytes[2] += features_.bytes;
    if (options.draft)
        for (auto& c : draft_cache_) {
            c.key   = DeviceBuffer(static_cast<std::size_t>(FeatureSlots) * 1024 * 2);
            c.value = DeviceBuffer(c.key.bytes);
            stats_.persistent_bytes[2] += c.key.bytes + c.value.bytes;
        }
    if (options.mtp) {
        auto& l  = mtp_state_;
        l.latent = DeviceBuffer(static_cast<std::size_t>(options.max_context) * 512 * 2);
        l.keys   = DeviceBuffer(static_cast<std::size_t>(options.max_context) * 128 * 2);
        l.gates  = DeviceBuffer(l.keys.bytes);
        l.pools  = DeviceBuffer(static_cast<std::size_t>((options.max_context + 3) / 4) * 128 * 2);
        l.means =
            DeviceBuffer(static_cast<std::size_t>((options.max_context + 127) / 128) * 128 * 4);
        stats_.persistent_bytes[2] +=
            l.latent.bytes + l.keys.bytes + l.gates.bytes + l.pools.bytes + l.means.bytes;
    }
    reset();
}

Program::~Program() {
    drain();
    for (unsigned f = 0; f < options_.decode_frames; ++f) {
        auto& frame = *frames_[f];
        for (int d = 0; d < 3; ++d) {
            frame.devices[d]->bind_to_current_thread_noexcept();
            for (auto& [key, family] : frame.graphs) {
                family[d].reset();
                if (!d) family[3].reset();
            }
            for (auto& [key, graph] : frame.tensor_graphs) if (key[3] % 3 == d) graph.reset();
            frame.state[d] = {};
            if (!d) frame.completed.reset();
        }
    }
    frame_ = frames_[0].get();
    for (int d = 0; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread_noexcept();
        for (int i = 0; i < 45; ++i)
            if (tensor_parallel_ || parameters_.layers[i].device == d) layer_state(d, i) = {};
        for (auto& s : recurrent_[d]) s = {};
        for (auto& sequence : sequence_ready_) sequence[d].reset();
        if (d == 2) { mtp_state_ = {}; features_ = {}; for (auto& c : draft_cache_) c = {}; }
    }
}

void Program::drain() noexcept {
    for (auto& frame : frames_) if (frame)
        for (auto* device : frame->devices) if (device) {
            device->bind_to_current_thread_noexcept();
            try {
                device->synchronize();
                CUDA_CHECK(cudaStreamSynchronize(device->transfer_stream));
            } catch (...) {}
        }
}

std::size_t Program::workspace_capacity_bytes() const noexcept {
    return 3 * ((2ULL << 30) + (options_.decode_frames - 1) * (256ULL << 20));
}

float* Program::floats(WorkspaceArena& w, std::size_t n) {
    return static_cast<float*>(w.alloc_bytes(n * 4).data);
}

int* Program::metadata(int device, MetadataField field) const noexcept {
    return static_cast<int*>(frame_->state[device].metadata.p) + static_cast<int>(field) * frame_->capacity;
}

void Program::projection(const float* x, const Weight& w, float* y, int t, int d, bool round) {
    Tensor a(const_cast<float*>(x), DType::FP32, {w.k, t}), b(y, DType::FP32, {w.n, t});
    ops::linear_fp32(a, w, b, frame_->devices[d]->stream, round);
}

void Program::mlp(const float* x, const MlpParameters& p, float* out, int intermediate, int t,
                  int d, float limit) {
    auto& w    = *frame_->state[d].workspace;
    auto scope = w.scope();
    auto* g    = floats(w, intermediate * t);
    auto* u    = floats(w, intermediate * t);
    projection(x, p.gate, g, t, d);
    projection(x, p.up, u, t, d);
    ops::resident_swiglu(g, u, g, static_cast<std::size_t>(intermediate) * t, limit,
                         frame_->devices[d]->stream);
    if (tensor_parallel_)
        ops::linear_partial_fp32(g, p.down, out, p.column_begin, intermediate, t, frame_->devices[d]->stream);
    else projection(g, p.down, out, t, d);
}

void Program::hyper_prepare(const float* residual, const HyperParameters& p, float* coeff,
                            float* out, int d, int t) {
    auto& w      = *frame_->state[d].workspace;
    auto scope   = w.scope();
    auto* norm   = floats(w, 4 * H * t);
    auto* mix    = floats(w, 24 * t);
    const auto s = frame_->devices[d]->stream;
    ops::resident_norm(residual, nullptr, nullptr, nullptr, norm, 4 * H, t, 1e-5f, false, false, s);
    projection(norm, p.projection, mix, t, d, false);
    ops::hyper_connection_coefficients(mix, fp(p.base), fp(p.scale), coeff, t, 20, 1e-6f, s);
    ops::hyper_connection_collapse(residual, coeff, out, H, t, s);
}

void Program::kda(const float* x, const KdaParameters& p, LayerState& state, float* out, int d,
                  int t) {
    const int channels = p.query.n, heads = channels / 128;
    auto& w      = *frame_->state[d].workspace;
    auto scope   = w.scope();
    const auto s = frame_->devices[d]->stream;
    float *q = floats(w, channels * t), *k = floats(w, channels * t), *v = floats(w, channels * t),
          *g = floats(w, channels * t), *gate = floats(w, channels * t), *small = floats(w, 128 * t),
          *beta = floats(w, heads * t), *result = floats(w, channels * t);
    const auto* slots = metadata(d, SequenceOffsets) + options_.max_lanes + 1;
    const auto* offsets = metadata(d, SequenceOffsets);
    projection(x, p.query, q, t, d);
    projection(x, p.key, k, t, d);
    projection(x, p.value, v, t, d);
    if (options_.draft) {
        ops::resident_causal_conv(q, p.query_conv, state.conv[0], q, channels, 1, s);
        ops::resident_causal_conv(k, p.key_conv, state.conv[1], k, channels, 1, s);
        ops::resident_causal_conv(v, p.value_conv, state.conv[2], v, channels, 1, s);
    } else {
        ops::resident_causal_conv_packed(q, p.query_conv, state.conv[0], slots, offsets,
                                         q, channels, frame_->sequences, s);
        ops::resident_causal_conv_packed(k, p.key_conv, state.conv[1], slots, offsets,
                                         k, channels, frame_->sequences, s);
        ops::resident_causal_conv_packed(v, p.value_conv, state.conv[2], slots, offsets,
                                         v, channels, frame_->sequences, s);
    }
    projection(x, p.forget_in, small, t, d);
    projection(small, p.forget_out, g, t, d);
    projection(x, p.output_gate_in, small, t, d);
    projection(small, p.output_gate_out, gate, t, d);
    projection(x, p.input_gate, beta, t, d);
    {
        auto temp = w.scope();
        auto qb = w.alloc(DType::BF16, {128, heads, t}),
             kb = w.alloc(DType::BF16, {128, heads, t}),
             vb = w.alloc(DType::BF16, {128, heads, t}),
             gb = w.alloc(DType::BF16, {128, heads, t}),
             bb = w.alloc(DType::BF16, {heads, t}),
             yb = w.alloc(DType::BF16, {128, heads, t});
        ops::pack_bf16(q, static_cast<std::uint16_t*>(qb.data), channels * t, s);
        ops::pack_bf16(k, static_cast<std::uint16_t*>(kb.data), channels * t, s);
        ops::pack_bf16(v, static_cast<std::uint16_t*>(vb.data), channels * t, s);
        ops::pack_bf16(g, static_cast<std::uint16_t*>(gb.data), channels * t, s);
        ops::pack_bf16(beta, static_cast<std::uint16_t*>(bb.data), heads * t, s);
        Tensor image(state.recurrent, DType::FP32, {128, 128, heads, options_.max_lanes});
        auto bias = p.decay_bias.reshape({128, heads});
        if (options_.draft) {
            ops::kimi_delta_attention(
                qb, kb, vb, gb, bb, p.negative_decay_rate, bias, -5.f,
                1.f / std::sqrt(128.f), w, image, yb, frame_->devices[d]->execution_view(),
                ops::KdaDecayParameter::NegativeRate);
        } else {
            Tensor ids(const_cast<int*>(slots), DType::I32, {frame_->sequences});
            Tensor ranges(const_cast<int*>(offsets), DType::I32, {frame_->sequences + 1});
            ops::kimi_delta_attention_packed_update(
                qb, kb, vb, gb, bb, p.negative_decay_rate, bias, -5.f,
                1.f / std::sqrt(128.f), w, image, ids, ranges, yb, s,
                ops::KdaDecayParameter::NegativeRate);
        }
        ops::unpack_bf16(static_cast<const std::uint16_t*>(yb.data), result, channels * t, s);
    }
    ops::resident_norm(result, &p.output_norm, nullptr, gate, result, 128, heads * t, 1e-5f, false,
                       true, s);
    if (tensor_parallel_)
        ops::linear_partial_fp32(result, p.output, out, p.head_begin * 128, channels, t, s);
    else projection(result, p.output, out, t, d);
}

void Program::publish_retrieval_mean(LayerState& layer, int block, int d) {
    auto& w       = *frame_->state[d].workspace;
    auto scope    = w.scope();
    auto* feature = floats(w, 128);
    ops::resident_mean_bf16(static_cast<const std::uint16_t*>(layer.keys.p) + block * 128 * 128,
                            feature, 128, 128, frame_->devices[d]->stream);
    frame_->devices[d]->synchronize();
    std::array<float, 128> mean{};
    CUDA_CHECK(cudaMemcpy(mean.data(), feature, sizeof(mean), cudaMemcpyDeviceToHost));
    layer.retrieval->write_block_mean(block, 0, mean);
}

void Program::mla(const float* x, const MlaParameters& p, LayerState& state, float* out, int d,
                  int t) {
    const int heads = p.query_heads.n / 256;
    auto& w      = *frame_->state[d].workspace;
    auto scope   = w.scope();
    const auto s = frame_->devices[d]->stream;
    auto *low = floats(w, (1536) * t), *query = floats(w, ((heads * 256)) * t),
         *latent = floats(w, (512) * t), *absorbed = floats(w, (heads * 512) * t),
         *attended = floats(w, (heads * 512) * t), *expanded = floats(w, ((heads * 256)) * t),
         *iq = floats(w, (4096) * t), *ik = floats(w, (128) * t), *ig = floats(w, (128) * t),
         *ih = floats(w, (32) * t), *scores = floats(w, (8192) * t),
         *prob = floats(w, (heads * 2051) * t);
    auto* ids  = static_cast<std::int32_t*>(w.alloc_bytes(2051 * 4 * t).data);
    projection(x, p.query_lowrank, low, t, d);
    ops::resident_norm(low, &p.query_norm, nullptr, nullptr, low, 1536, t, 1e-5f, false, true, s);
    projection(low, p.query_heads, query, t, d);
    projection(x, p.kv_latent, latent, t, d);
    ops::resident_norm(latent, &p.kv_norm, nullptr, nullptr, latent, 512, t, 1e-5f, false, true, s);
    if (options_.draft)
        ops::pack_bf16(latent, static_cast<std::uint16_t*>(state.latent.p) + position_ * 512, 512,
                       s);
    projection(low, p.index.query, iq, t, d);
    projection(x, p.index.key, ik, t, d);
    ops::resident_norm(ik, &p.index.key_norm, &p.index.key_norm_bias, nullptr, ik, 128, t, 1e-6f,
                       true, true, s);
    projection(x, p.index.pool_gate, ig, t, d);
    projection(x, p.index.head_weights, ih, t, d, false);
    const auto* slots     = metadata(d, StateSlots);
    const auto* positions = metadata(d, Positions);
    if (options_.draft) {
        ops::index_pool_append(ik, ig, fp(p.index.position_bias),
                               static_cast<std::uint16_t*>(state.keys.p),
                               static_cast<std::uint16_t*>(state.gates.p),
                               static_cast<std::uint16_t*>(state.pools.p), position_, s);
        const std::uint8_t* allowed = nullptr;
        if (options_.kvmem_window_pages) {
            auto full = state.retrieval->append(1);
            for (const auto& block : full) publish_retrieval_mean(state, block.block_id, d);
            auto* feature = floats(w, (128) * t);
            ops::resident_sum_rows(iq, feature, 128, 32, s);
            frame_->devices[d]->synchronize();
            std::array<float, 128> q{};
            CUDA_CHECK(cudaMemcpy(q.data(), feature, sizeof(q), cudaMemcpyDeviceToHost));
            std::vector<float> rank(state.retrieval->block_count(),
                                    std::numeric_limits<float>::quiet_NaN());
            const std::array<std::uint32_t, 1> count{1};
            for (unsigned b = 0; b < rank.size(); ++b)
                if (state.retrieval->block(b).full)
                    (void)state.retrieval->score(b, q, count, rank[b]);
            kvmem::BlockSelectionConfig config{128, options_.kvmem_window_pages / 2, 1, 1, 0};
            auto selection = kvmem::select_blocks(*state.retrieval, rank, config);
            std::vector<std::uint8_t> pages((options_.max_context + 63) / 64);
            for (auto b : selection.selected)
                for (int j = 0; j < 2; ++j)
                    if (b * 2 + j < pages.size()) pages[b * 2 + j] = 1;
            auto mask = w.alloc_bytes(pages.size());
            CUDA_CHECK(cudaMemcpy(mask.data, pages.data(), pages.size(), cudaMemcpyHostToDevice));
            allowed = static_cast<const std::uint8_t*>(mask.data);
            ++stats_.kvmem_selections;
            if (selection.selected.size() < state.retrieval->block_count())
                ++stats_.kvmem_restricted_steps;
        }
        ops::index_pool_scores(iq, ih, static_cast<const std::uint16_t*>(state.pools.p), allowed,
                               scores, (position_ + 1) / 4, s);
        ops::index_pool_select(scores, ids, position_ + 1, s);
    } else {
        ops::resident_index_append(
            latent, ik, ig, fp(p.index.position_bias), static_cast<std::uint16_t*>(state.latent.p),
            static_cast<std::uint16_t*>(state.keys.p), static_cast<std::uint16_t*>(state.gates.p),
            static_cast<std::uint16_t*>(state.pools.p), slots, positions, options_.max_context, t,
            s);
        auto* allowed =
            static_cast<std::uint8_t*>(w.alloc_bytes(t * ((options_.max_context + 63) / 64)).data);
        if (options_.kvmem_window_pages)
            ops::resident_retrieval(iq, static_cast<const std::uint16_t*>(state.keys.p),
                                    static_cast<float*>(state.means.p), allowed, slots, positions,
                                    options_.max_context, t, options_.kvmem_window_pages / 2, s);
        ops::resident_pool_select(iq, ih, static_cast<const std::uint16_t*>(state.pools.p),
                                  options_.kvmem_window_pages ? allowed : nullptr, scores, ids,
                                  slots, positions, options_.max_context, frame_->tier, t, s);
    }
    ops::indexed_linear_fp32(query, p.key_absorb, nullptr, absorbed, 512, heads, t, false, s);
    if (options_.draft) {
        ops::sparse_latent_attention(absorbed, static_cast<const std::uint16_t*>(state.latent.p),
                                     ids, prob, attended, heads, 512, t, 2051, 1.f / 16, s);
    } else
        ops::resident_sparse_attention(absorbed, static_cast<const std::uint16_t*>(state.latent.p),
                                       ids, prob, attended, slots, options_.max_context, t, s, heads);
    ops::indexed_linear_fp32(attended, p.value_expand, nullptr, expanded, 256, heads, t, false, s,
                             true);
    if (tensor_parallel_)
        ops::linear_partial_fp32(expanded, p.output, out, p.head_begin * 256, heads * 256, t, s);
    else projection(expanded, p.output, out, t, d);
}

void Program::feed_forward(const float* x, const LayerParameters& p, float* out, int t) {
    const int d  = p.device;
    auto& w      = *frame_->state[d].workspace;
    auto scope   = w.scope();
    const auto s = frame_->devices[d]->stream;
    if (p.dense) {
        mlp(x, p.ffn, out, p.ffn.gate.n, t, d, 10.f);
        return;
    }
    auto *logits = floats(w, (288) * t), *weights = floats(w, (8) * t),
         *g = floats(w, (8 * 2048) * t), *u = floats(w, (8 * 2048) * t),
         *experts = floats(w, (8 * H) * t), *shared = floats(w, (H)*t);
    auto* ids = static_cast<std::int32_t*>(w.alloc_bytes(8 * 4 * t).data);
    projection(x, p.router, logits, t, d, false);
    ops::resident_router(logits, fp(p.router_bias), ids, weights, 288, 8, t, 2.5f, s);
    const auto indexed = [&](const float* input, const Weight& bank, float* result, int rows, bool same) {
        if (t >= 16) ops::indexed_linear_grouped_fp32(input, bank, ids, result, rows, 8, t,
            same, tensor_parallel_ ? p.expert_begin : 0, w, s, true);
        else if (tensor_parallel_) ops::indexed_linear_owned_fp32(input, bank, ids, result,
            rows, 8, t, same, p.expert_begin, s, true);
        else ops::indexed_linear_fp32(input, bank, ids, result, rows, 8, t, same, s, true);
    };
    if (tensor_parallel_) {
        indexed(x, p.experts.gate, g, 2048, true);
        indexed(x, p.experts.up, u, 2048, true);
        Tensor gate(g, DType::FP32, {2048, 8 * t}), up(u, DType::FP32, {2048, 8 * t});
        ops::resident_swiglu_owned(gate, up, ids, p.expert_begin, p.experts.gate.n / 2048, gate, 10.f, s);
        indexed(g, p.experts.down, experts, H, false);
        mlp(x, p.ffn, out, p.ffn.gate.n, t, d, 10.f);
        ops::resident_moe_partial(experts, weights, out + H * t, H, 8, t, s);
    } else {
    indexed(x, p.experts.gate, g, 2048, true);
    indexed(x, p.experts.up, u, 2048, true);
    ops::resident_swiglu(g, u, g, 8 * 2048 * t, 10.f, s);
    indexed(g, p.experts.down, experts, H, false);
    mlp(x, p.ffn, shared, 2048, t, d, 10.f);
    ops::resident_moe_sum(experts, weights, shared, out, H, 8, t, s);
    }
}

void Program::peer(void* dst, int d, const void* src, int source, std::size_t bytes) {
    if (d == source) {
        frame_->devices[d]->bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, frame_->devices[d]->stream));
        return;
    }
    frame_->devices[source]->bind_to_current_thread();
    frame_->state[source].ready->record(frame_->devices[source]->stream);
    frame_->devices[d]->bind_to_current_thread();
    // Destination consumers from the preceding use must finish before this
    // transfer overwrites their buffer. Both dependencies stay on device.
    frame_->state[d].ready->record(frame_->devices[d]->stream);
    frame_->state[source].ready->wait(frame_->devices[d]->transfer_stream);
    frame_->state[d].ready->wait(frame_->devices[d]->transfer_stream);
    CUDA_CHECK(hipMemcpyPeerAsync(dst, frame_->devices[d]->device, src, frame_->devices[source]->device, bytes,
                                  frame_->devices[d]->transfer_stream));
    frame_->state[d].copied->record(frame_->devices[d]->transfer_stream);
    frame_->state[d].copied->wait(frame_->devices[d]->stream);
    frame_->devices[source]->bind_to_current_thread();
    // Protect source-buffer lifetime before its next producer can reuse it.
    frame_->state[d].copied->wait(frame_->devices[source]->stream);
    frame_->devices[d]->bind_to_current_thread();
    stats_.peer_bytes += bytes;
}

void Program::capture(const float* streams, int layer, int d) {
    constexpr int taps[5] = {5, 14, 24, 33, 42};
    const auto found      = std::find(std::begin(taps), std::end(taps), layer);
    if (found == std::end(taps)) return;
    const int tap = found - std::begin(taps);
    auto& w       = *frame_->state[d].workspace;
    auto scope    = w.scope();
    auto* mean    = floats(w, H);
    ops::mean_streams(streams, mean, H, 1, frame_->devices[d]->stream);
    auto* destination = static_cast<std::uint16_t*>(features_.p) +
                        (static_cast<std::size_t>(position_ % FeatureSlots) * 5 + tap) * H;
    if (d == 2)
        ops::pack_bf16(mean, destination, H, frame_->devices[d]->stream);
    else {
        auto& target   = *frame_->state[2].workspace;
        auto ts        = target.scope();
        auto* received = floats(target, H);
        peer(received, 2, mean, d, H * 4);
        ops::pack_bf16(received, destination, H, frame_->devices[2]->stream);
        frame_->devices[2]->synchronize();
        frame_->devices[d]->bind_to_current_thread();
    }
}

std::int32_t Program::target(std::int32_t token, int snapshot) {
    if (!options_.draft) {
        if (snapshot) throw std::invalid_argument("ordinary compact Program has no draft snapshot");
        const int lane = 0;
        return target_batch({&token, 1}, {&lane, 1})[0];
    }
    if (token < 0 || token >= 154880 || position_ >= options_.max_context || snapshot < 0 ||
        snapshot > 8)
        throw std::invalid_argument("GLM target token/frontier/snapshot");
    for (auto& d : frame_->state) d.workspace->reset();
    frame_->devices[0]->bind_to_current_thread();
    auto& work = *frame_->state[0].workspace;
    auto* id   = static_cast<std::int32_t*>(work.alloc_bytes(4).data);
    CUDA_CHECK(cudaMemcpy(id, &token, 4, cudaMemcpyHostToDevice));
    auto* embedding = floats(work, H);
    ops::embedding_fp32(parameters_.embedding, id, embedding, 1, frame_->devices[0]->stream);
    ops::replicate_streams(embedding, static_cast<float*>(frame_->state[0].streams.p), H, 1,
                           frame_->devices[0]->stream);
    work.reset();
    for (int i = 0; i < 45; ++i) {
        const auto& p = parameters_.layers[i];
        const int d   = p.device;
        if (i == 16 || i == 31)
            peer(frame_->state[d].streams.p, d, frame_->state[d - 1].streams.p, d - 1, 4 * H * 4);
        frame_->devices[d]->bind_to_current_thread();
        auto& w     = *frame_->state[d].workspace;
        auto scope  = w.scope();
        auto *coeff = floats(w, 24), *x = floats(w, H), *y = floats(w, H);
        auto* residual = static_cast<float*>(frame_->state[d].streams.p);
        const auto s   = frame_->devices[d]->stream;
        hyper_prepare(residual, p.attention_hc, coeff, x, d);
        ops::resident_norm(x, &p.attention_norm, nullptr, nullptr, x, H, 1, 1e-5f, false, true, s);
        if (p.linear)
            kda(x, p.kda, layers_[i], y, d);
        else
            mla(x, p.mla, layers_[i], y, d);
        ops::hyper_connection_expand(residual, y, coeff, residual, H, 1, s);
        hyper_prepare(residual, p.ffn_hc, coeff, x, d);
        ops::resident_norm(x, &p.ffn_norm, nullptr, nullptr, x, H, 1, 1e-5f, false, true, s);
        feed_forward(x, p, y);
        ops::hyper_connection_expand(residual, y, coeff, residual, H, 1, s);
        capture(residual, i, d);
        stats_.workspace_peak[d] = std::max(stats_.workspace_peak[d], w.peak_used());
    }
    frame_->devices[2]->bind_to_current_thread();
    auto& last   = *frame_->state[2].workspace;
    auto* hidden = floats(last, H);
    ops::mean_streams(static_cast<float*>(frame_->state[2].streams.p), hidden, H, 1, frame_->devices[2]->stream);
    auto& first = *frame_->state[0].workspace;
    auto* final = floats(first, H);
    peer(final, 0, hidden, 2, H * 4);
    ops::resident_norm(final, &parameters_.output_norm, nullptr, nullptr, final, H, 1, 1e-5f, false,
                       true, frame_->devices[0]->stream);
    auto* logits = floats(first, 154880);
    projection(final, parameters_.output_head, logits, 1, 0, false);
    auto* result = static_cast<std::int32_t*>(first.alloc_bytes(4).data);
    ops::resident_argmax(logits, result, 154880, 1, frame_->devices[0]->stream);
    frame_->devices[0]->synchronize();
    std::int32_t next;
    CUDA_CHECK(cudaMemcpy(&next, result, 4, cudaMemcpyDeviceToHost));
    if (next < 0) throw std::runtime_error("GLM target produced nonfinite logits");
    if (snapshot)
        for (int d = 0; d < 3; ++d) {
            frame_->devices[d]->bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(recurrent_[d][snapshot].p, recurrent_[d][0].p,
                                       recurrent_[d][0].bytes, cudaMemcpyDeviceToDevice,
                                       frame_->devices[d]->stream));
            frame_->devices[d]->synchronize();
        }
    ++position_;
    ++stats_.target_steps;
    ++stats_.predicted_rows;
    return next;
}

// Native checkpoint NextN: concatenate independently normalized next-token
// embedding and target post-output-norm hidden, fuse, one plain MLA/MoE block,
// then its own output norm and the resident target head. The MTP cache advances
// only on actual/greedily accepted input tokens; no rejected proposal owns state.
std::int32_t Program::mtp_forward(std::int32_t token, bool proposal) {
    if (!options_.mtp || token < 0 || token >= 154880 || position_ != mtp_frontier_ + 1)
        throw std::invalid_argument("native MTP token/teacher frontier");
    const auto& p = parameters_.mtp;
    for (auto& state : frame_->state) state.workspace->reset();
    frame_->devices[0]->bind_to_current_thread();
    auto& first = *frame_->state[0].workspace;
    auto* id    = static_cast<int*>(first.alloc_bytes(4).data);
    CUDA_CHECK(cudaMemcpy(id, &token, 4, cudaMemcpyHostToDevice));
    auto* embedded = floats(first, H);
    ops::embedding_fp32(parameters_.embedding, id, embedded, 1, frame_->devices[0]->stream);
    auto& work   = *frame_->state[2].workspace;
    auto* joined = floats(work, 2 * H);
    peer(joined, 2, embedded, 0, H * 4);
    peer(joined + H, 2, frame_->state[0].hidden.p, 0, H * 4);
    const auto stream = frame_->devices[2]->stream;
    ops::resident_norm(joined, &p.embedding_norm, nullptr, nullptr, joined, H, 1, 1e-5f, false,
                       true, stream);
    ops::resident_norm(joined + H, &p.hidden_norm, nullptr, nullptr, joined + H, H, 1, 1e-5f, false,
                       true, stream);
    auto* residual = floats(work, H);
    auto* x        = floats(work, H);
    auto* y        = floats(work, H);
    projection(joined, p.fusion, residual, 1, 2);
    ops::resident_norm(residual, &p.layer.attention_norm, nullptr, nullptr, x, H, 1, 1e-5f, false,
                       true, stream);
    mla(x, p.layer.mla, mtp_state_, y, 2, 1);
    ops::resident_add(residual, y, residual, H, stream);
    ops::resident_norm(residual, &p.layer.ffn_norm, nullptr, nullptr, x, H, 1, 1e-5f, false, true,
                       stream);
    feed_forward(x, p.layer, y, 1);
    ops::resident_add(residual, y, residual, H, stream);
    ++mtp_frontier_;
    if (options_.kvmem_window_pages) {
        ++stats_.kvmem_selections;
        if ((mtp_frontier_ + 127) / 128 > static_cast<int>(options_.kvmem_window_pages / 2))
            ++stats_.kvmem_restricted_steps;
    }
    if (!proposal) {
        frame_->devices[2]->synchronize();
        return -1;
    }
    ops::resident_norm(residual, &p.output_norm, nullptr, nullptr, residual, H, 1, 1e-5f, false,
                       true, stream);
    auto* input = floats(first, H);
    peer(input, 0, residual, 2, H * 4);
    auto* logits = floats(first, 154880);
    projection(input, parameters_.output_head, logits, 1, 0, false);
    auto* output = static_cast<int*>(first.alloc_bytes(4).data);
    ops::resident_argmax(logits, output, 154880, 1, frame_->devices[0]->stream);
    frame_->devices[0]->synchronize();
    int result;
    CUDA_CHECK(cudaMemcpy(&result, output, 4, cudaMemcpyDeviceToHost));
    if (result < 0) throw std::runtime_error("native MTP nonfinite logits");
    ++stats_.draft_calls;
    ++stats_.drafted_tokens;
    return result;
}

void Program::reset_lane(int lane) {
    if (lane < 0 || lane >= options_.max_lanes || options_.draft)
        throw std::invalid_argument("compact Program lane reset");
    ledger_.reset_lane(lane);
    frame_ = frames_[0].get();
    for (int d = 0; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        for (int i = 0; i < 45; ++i) {
            const auto& p = model_.parameters(d).layers[i];
            if (!p.linear || (!tensor_parallel_ && p.device != d)) continue;
            const auto channels = static_cast<std::size_t>(p.kda.query.n);
            const auto state_floats = channels * 128;
            auto& layer = layer_state(d, i);
            CUDA_CHECK(cudaMemsetAsync(layer.recurrent + lane * state_floats, 0,
                                       state_floats * sizeof(float), frame_->devices[d]->stream));
            for (auto* conv : layer.conv)
                CUDA_CHECK(cudaMemsetAsync(conv + lane * channels * 3, 0,
                                           channels * 3 * sizeof(float), frame_->devices[d]->stream));
        }
    }
    for (int d = 0; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        if (!sequence_ready_[lane][d]) sequence_ready_[lane][d] =
            std::make_unique<CudaCompletionEvent>(*frame_->devices[d]);
        sequence_ready_[lane][d]->record(frame_->devices[d]->stream);
    }
    // KV, pools and derived means are overwritten before every visible read.
}

void Program::batch_stage(int d) {
    auto& work = *frame_->state[d].workspace;
    work.reset();
    const auto stream = frame_->devices[d]->stream;
    auto* residual    = static_cast<float*>(frame_->state[d].streams.p);
    if (d == 0) {
        auto scope      = work.scope();
        auto* embedding = floats(work, H * frame_->batch);
        ops::embedding_fp32(parameters_.embedding, metadata(0, TokenIds),
                            embedding, frame_->batch, stream);
        ops::replicate_streams(embedding, residual, H, frame_->batch, stream);
    }
    for (int i = 0; i < 45; ++i) {
        const auto& p = parameters_.layers[i];
        if (p.device != d) continue;
        auto scope  = work.scope();
        auto* coeff = floats(work, 24 * frame_->batch);
        auto* x     = floats(work, H * frame_->batch);
        auto* y     = floats(work, H * frame_->batch);
        Tensor collapsed(x, DType::FP32, {H, frame_->batch});
        auto normalized = work.alloc(DType::BF16, {H, frame_->batch});
        hyper_prepare(residual, p.attention_hc, coeff, x, d, frame_->batch);
        if (options_.bf16_activations) {
            ops::resident_norm(collapsed, &p.attention_norm, nullptr, nullptr, normalized, 1e-5f, false, stream);
            if (p.linear)
                kda(normalized, p.kda, layers_[i], y, d, frame_->batch);
            else
                mla(normalized, p.mla, layers_[i], y, d, frame_->batch);
        } else {
            ops::resident_norm(x, &p.attention_norm, nullptr, nullptr, x, H, frame_->batch, 1e-5f, false, true, stream);
            if (p.linear) kda(x, p.kda, layers_[i], y, d, frame_->batch);
            else mla(x, p.mla, layers_[i], y, d, frame_->batch);
        }
        ops::hyper_connection_expand(residual, y, coeff, residual, H, frame_->batch, stream);
        hyper_prepare(residual, p.ffn_hc, coeff, x, d, frame_->batch);
        if (options_.bf16_activations) {
            ops::resident_norm(collapsed, &p.ffn_norm, nullptr, nullptr, normalized, 1e-5f, false, stream);
            feed_forward(normalized, p, y, frame_->batch);
        } else {
            ops::resident_norm(x, &p.ffn_norm, nullptr, nullptr, x, H, frame_->batch, 1e-5f, false, true, stream);
            feed_forward(x, p, y, frame_->batch);
        }
        ops::hyper_connection_expand(residual, y, coeff, residual, H, frame_->batch, stream);
    }
    if (d == 2 && frame_->outputs) {
        auto scope = work.scope();
        const float* selected = residual;
        if (frame_->outputs != frame_->batch) {
            auto* compact = floats(work, 4 * H * frame_->outputs);
            ops::resident_gather_rows(residual, metadata(d, OutputRows), compact,
                                      4 * H, frame_->outputs, stream);
            selected = compact;
        }
        ops::mean_streams(selected, static_cast<float*>(frame_->state[2].hidden.p), H, frame_->outputs, stream);
    }
    stats_.workspace_peak[d] = std::max(stats_.workspace_peak[d], work.peak_used());
}

void Program::batch_head() {
    auto& work = *frame_->state[0].workspace;
    work.reset();
    const auto stream = frame_->devices[0]->stream;
    auto* final       = static_cast<float*>(frame_->state[0].hidden.p);
    Tensor final_input(final, DType::FP32, {H, frame_->outputs});
    auto normalized = work.alloc(DType::BF16, {H, frame_->outputs});
    if (options_.bf16_activations)
        ops::resident_norm(final_input, &parameters_.output_norm, nullptr, nullptr, normalized, 1e-5f, false, stream);
    else {
        ops::resident_norm(final, &parameters_.output_norm, nullptr, nullptr, final, H, frame_->outputs, 1e-5f, false, true, stream);
        normalized = final_input;
    }
    auto* logits = floats(work, 154880 * frame_->outputs);
    Tensor scores(logits, DType::FP32, {154880, frame_->outputs});
    projection(normalized, parameters_.output_head, scores, 0);
    ops::resident_argmax(logits, metadata(0, PredictedIds), 154880, frame_->outputs,
                         stream);
    stats_.workspace_peak[0] = std::max(stats_.workspace_peak[0], work.peak_used());
}

std::vector<std::int32_t> Program::target_batch(std::span<const int> tokens,
                                                std::span<const int> lanes) {
    if (tokens.empty() || tokens.size() > static_cast<unsigned>(options_.max_lanes) ||
        tokens.size() != lanes.size())
        throw std::invalid_argument("compact Program batch geometry");
    std::array<runtime::TokenSpan, 8> spans{};
    std::array<std::int32_t, 8> outputs{};
    for (unsigned row = 0; row < tokens.size(); ++row) {
        const int slot = lanes[row];
        if (slot < 0 || slot >= options_.max_lanes)
            throw std::invalid_argument("compact Program lane");
        spans[row] = {static_cast<unsigned>(slot), static_cast<unsigned>(ledger_.frontier(slot)), row, 1};
        outputs[row] = row;
    }
    return execute({runtime::TokenBatchKind::Decode, tokens,
                    {spans.data(), tokens.size()}, {outputs.data(), tokens.size()}});
}

std::vector<std::int32_t> Program::execute(const runtime::TokenBatch& input) {
    return finish(submit(input));
}

FrameTicket Program::submit(const runtime::TokenBatch& input) {
    if (options_.draft) throw std::invalid_argument("async Program excludes draft execution");
    input.validate(options_.max_lanes, options_.max_context, input_capacity());
    for (auto token : input.tokens)
        if (token < 0 || token >= 154880)
            throw std::invalid_argument("packed Program token outside vocabulary");
    // FrameLedger reserves one writer per physical sequence, with owning spans.
    const auto ticket = ledger_.reserve(input);
    frame_ = frames_[ticket.slot].get();
    try {
    frame_->spans.assign(input.spans.begin(), input.spans.end());
    frame_->kind = input.kind;
    frame_->batch = input.tokens.size();
    frame_->sequences = input.spans.size();
    frame_->outputs = input.output_rows.size();
    frame_->tier = 128;
    auto* offsets = frame_->host_metadata.data() + 5 * frame_->capacity;
    auto* sequence_slots = offsets + options_.max_lanes + 1;
    for (int r = 0; r < frame_->sequences; ++r) {
        const auto& span = input.spans[r];
        offsets[r] = span.begin;
        sequence_slots[r] = span.lane;
        for (unsigned t = 0; t < span.count; ++t) {
            const int row = span.begin + t;
            frame_->host_metadata[row] = input.tokens[row];
            frame_->host_metadata[frame_->capacity + row] = span.lane;
            frame_->host_metadata[2 * frame_->capacity + row] = span.input_frontier + t;
        }
        while (frame_->tier < static_cast<int>(span.input_frontier + span.count)) frame_->tier *= 2;
    }
    offsets[frame_->sequences] = frame_->batch;
    std::copy(input.output_rows.begin(), input.output_rows.end(),
              frame_->host_metadata.begin() + 3 * frame_->capacity);
    std::memcpy(frame_->upload->data(), frame_->host_metadata.data(), frame_->upload->size());
    for (int d = 0; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        for (const auto& span : input.spans)
            if (sequence_ready_[span.lane][d]) sequence_ready_[span.lane][d]->wait(frame_->devices[d]->stream);
        CUDA_CHECK(cudaMemcpyAsync(frame_->state[d].metadata.p, frame_->upload->data(),
                                   frame_->host_metadata.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, frame_->devices[d]->stream));
        // Consumers use this same stream. The owning host frame remains stable
        // until the complete stage chain finishes; no per-upload host wait.
    }
    if (tensor_parallel_) tensor_execute();
    else {
    // Temporal prefill uses the same stage composition eagerly. Decode Graphs
    // retain exact request geometry; no temporal span is relabeled as a lane.
    const bool graph = options_.graph && frame_->kind == runtime::TokenBatchKind::Decode;
    const int key = frame_->batch * 65536 + frame_->tier;
    if (graph && !frame_->graphs.contains(key)) {
        auto& family = frame_->graphs[key];
        for (int g = 0; g < 4; ++g) {
            const int d = g == 3 ? 0 : g;
            frame_->devices[d]->bind_to_current_thread();
            DecodeGraphDefinition definition;
            definition.capture(frame_->devices[d]->stream, [&] {
                if (g == 3)
                    batch_head();
                else
                    batch_stage(d);
            });
            family[g].instantiate(definition);
            family[g].upload(frame_->devices[d]->stream);
            ++stats_.graph_captures;
        }
    }
    for (int d = 0; d < 3; ++d) {
        if (d)
            peer(frame_->state[d].streams.p, d, frame_->state[d - 1].streams.p, d - 1,
                 static_cast<std::size_t>(4 * H * 4) * frame_->batch);
        frame_->devices[d]->bind_to_current_thread();
        if (graph) {
            frame_->graphs.at(key)[d].launch(frame_->devices[d]->stream);
            ++stats_.graph_replays;
        } else
            batch_stage(d);
    }
    if (frame_->outputs) {
        peer(frame_->state[0].hidden.p, 0, frame_->state[2].hidden.p, 2,
              static_cast<std::size_t>(H * 4) * frame_->outputs);
        frame_->devices[0]->bind_to_current_thread();
        if (graph) {
            frame_->graphs.at(key)[3].launch(frame_->devices[0]->stream);
            ++stats_.graph_replays;
        } else
            batch_head();
    }
    }
    // Join every rank before D2H/publication; no host wait during enqueue.
    for (int d = 0; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        frame_->state[d].done->record(frame_->devices[d]->stream);
    }
    frame_->devices[0]->bind_to_current_thread();
    for (int d = 1; d < 3; ++d) frame_->state[d].done->wait(frame_->devices[0]->stream);
    if (frame_->outputs)
        CUDA_CHECK(cudaMemcpyAsync(frame_->result->data(), metadata(0, PredictedIds),
            frame_->outputs * sizeof(int), cudaMemcpyDeviceToHost, frame_->devices[0]->stream));
    frame_->completed->record(frame_->devices[0]->stream);
    return ticket;
    } catch (...) {
        // A partial enqueue has already mutated model state. Fail closed; no
        // rollback claim, slot recycling, or second admission after this point.
        ledger_.fail();
        drain();
        throw;
    }
}

bool Program::ready(FrameTicket ticket) const {
    ledger_.validate(ticket);
    frames_[ticket.slot]->devices[0]->bind_to_current_thread();
    return frames_[ticket.slot]->completed->ready();
}

std::vector<std::int32_t> Program::finish(FrameTicket ticket) {
    ledger_.validate(ticket);
    frame_ = frames_[ticket.slot].get();
    frame_->devices[0]->bind_to_current_thread();
    frame_->completed->synchronize();
    const auto* ids = static_cast<const int*>(frame_->result->data());
    std::vector<int> result(ids, ids + frame_->outputs);
    for (auto token : result) if (token < 0) {
        ledger_.fail();
        drain();
        throw std::runtime_error("packed Program nonfinite logits");
    }
    for (const auto& span : frame_->spans) {

        for (unsigned t = 0; t < span.count; ++t) {
            if (options_.kvmem_window_pages) {
                stats_.kvmem_selections += 11;
                if ((span.input_frontier + t + 128) / 128 > options_.kvmem_window_pages / 2)
                    stats_.kvmem_restricted_steps += 11;
            }
        }
    }
    for (int d = 0; d < 3; ++d) {
        std::size_t peak = 0;
        for (const auto& frame : frames_) if (frame) peak += frame->state[d].workspace->peak_used();
        stats_.workspace_peak[d] = std::max(stats_.workspace_peak[d], peak);
    }
    ledger_.complete(ticket);
    position_ = ledger_.frontier(0);
    stats_.target_steps += frame_->batch;
    ++stats_.batch_rounds;
    stats_.projection_columns += frame_->batch;
    stats_.predicted_rows += frame_->outputs;
    return result;
}

std::vector<std::int32_t> Program::propose(std::int32_t anchor, int t) {
    if (anchor < 0 || anchor >= 154880 || t < 1 || t > 8 || position_ < 1 ||
        position_ + t > options_.max_context)
        throw std::invalid_argument("GLM DFlash proposal anchor/width/context");
    std::vector<std::int32_t> block(t, 154856);
    block[0] = anchor;
    if (t == 1) return block;
    for (auto& d : frame_->state) d.workspace->reset();
    frame_->devices[0]->bind_to_current_thread();
    auto& first  = *frame_->state[0].workspace;
    auto* tokens = static_cast<std::int32_t*>(first.alloc_bytes(t * 4).data);
    CUDA_CHECK(cudaMemcpy(tokens, block.data(), t * 4, cudaMemcpyHostToDevice));
    auto* embedded = floats(first, t * H);
    ops::embedding_fp32(parameters_.embedding, tokens, embedded, t, frame_->devices[0]->stream);
    auto& w      = *frame_->state[2].workspace;
    auto* hidden = floats(w, t * H);
    peer(hidden, 2, embedded, 0, static_cast<std::size_t>(t) * H * 4);
    const auto s              = frame_->devices[2]->stream;
    const auto& p             = parameters_.draft;
    const int context_begin   = std::max(draft_frontier_, std::max(0, position_ - 2047)),
              context         = position_ - context_begin;
    float* normalized_context = nullptr;
    if (context) {
        auto* feature      = floats(w, static_cast<std::size_t>(context) * Features);
        normalized_context = floats(w, static_cast<std::size_t>(context) * H);
        ops::dflash_features(static_cast<const std::uint16_t*>(features_.p), feature, FeatureSlots,
                             context_begin, context, s);
        projection(feature, p.feature_projection, normalized_context, context, 2);
        ops::resident_norm(normalized_context, &p.context_norm, nullptr, nullptr,
                           normalized_context, H, context, 1e-5f, false, true, s);
    }
    for (int layer = 0; layer < 5; ++layer) {
        const auto& l    = p.layers[layer];
        auto layer_scope = w.scope();
        if (context) {
            auto scope = w.scope();
            auto *k = floats(w, context * 1024), *v = floats(w, context * 1024);
            projection(normalized_context, l.key, k, context, 2);
            projection(normalized_context, l.value, v, context, 2);
            ops::resident_norm(k, &l.key_norm, nullptr, nullptr, k, 128, context * 8, 1e-5f, false,
                               true, s);
            ops::dflash_rope(k, 8, context, context_begin, s);
            ops::dflash_cache_append(k, v, static_cast<std::uint16_t*>(draft_cache_[layer].key.p),
                                     static_cast<std::uint16_t*>(draft_cache_[layer].value.p),
                                     FeatureSlots, context_begin, context, s);
        }
        auto *normalized = floats(w, t * H), *prepared = floats(w, t * H),
             *dynamic = floats(w, t * 1024), *output = floats(w, t * H),
             *finished = floats(w, t * H);
        ops::resident_norm(hidden, &l.attention_norm, nullptr, nullptr, normalized, H, t, 1e-5f,
                           false, true, s);
        projection(normalized, l.attention_conv.kernel_projection, dynamic, t, 2);
        ops::dflash_dynamic_conv(
            normalized, dynamic,
            static_cast<const std::uint16_t*>(l.attention_conv.base_kernel.data), prepared, t, 0,
            s);
        {
            auto scope = w.scope();
            auto *q = floats(w, t * 4096), *k = floats(w, t * 1024), *v = floats(w, t * 1024),
                 *attended = floats(w, t * 4096),
                 *prob     = floats(w, static_cast<std::size_t>(t) * 32 * (2048 + t));
            projection(prepared, l.query, q, t, 2);
            projection(prepared, l.key, k, t, 2);
            projection(prepared, l.value, v, t, 2);
            ops::resident_norm(q, &l.query_norm, nullptr, nullptr, q, 128, t * 32, 1e-5f, false,
                               true, s);
            ops::resident_norm(k, &l.key_norm, nullptr, nullptr, k, 128, t * 8, 1e-5f, false, true,
                               s);
            ops::dflash_rope(q, 32, t, position_, s);
            ops::dflash_rope(k, 8, t, position_, s);
            ops::dflash_attention(q, static_cast<const std::uint16_t*>(draft_cache_[layer].key.p),
                                  static_cast<const std::uint16_t*>(draft_cache_[layer].value.p), k,
                                  v, prob, attended, FeatureSlots, position_, t, s);
            projection(attended, l.output, output, t, 2);
        }
        ops::dflash_dynamic_conv(
            output, dynamic, static_cast<const std::uint16_t*>(l.attention_conv.base_kernel.data),
            finished, t, 1, s);
        ops::resident_add(hidden, finished, hidden, static_cast<std::size_t>(t) * H, s);
        ops::resident_norm(hidden, &l.ffn_norm, nullptr, nullptr, normalized, H, t, 1e-5f, false,
                           true, s);
        projection(normalized, l.mlp_conv.kernel_projection, dynamic, t, 2);
        ops::dflash_dynamic_conv(normalized, dynamic,
                                 static_cast<const std::uint16_t*>(l.mlp_conv.base_kernel.data),
                                 prepared, t, 0, s);
        mlp(prepared, l.ffn, output, 12288, t, 2, 0.f);
        ops::dflash_dynamic_conv(output, dynamic,
                                 static_cast<const std::uint16_t*>(l.mlp_conv.base_kernel.data),
                                 finished, t, 1, s);
        ops::resident_add(hidden, finished, hidden, static_cast<std::size_t>(t) * H, s);
    }
    ops::resident_norm(hidden, &p.final_norm, nullptr, nullptr, hidden, H, t, 1e-5f, false, true,
                       s);
    auto* rank = floats(w, (t - 1) * 256);
    projection(hidden + H, p.selector_projection, rank, t - 1, 2);
    auto* head_input = floats(first, (t - 1) * H);
    peer(head_input, 0, hidden + H, 2, static_cast<std::size_t>(t - 1) * H * 4);
    auto* logits = floats(first, static_cast<std::size_t>(t - 1) * 154880);
    projection(head_input, parameters_.output_head, logits, t - 1, 0, false);
    auto* candidate0 = static_cast<std::int32_t*>(first.alloc_bytes((t - 1) * 16 * 4).data);
    auto* unary0     = floats(first, (t - 1) * 16);
    ops::dflash_top16(logits, candidate0, unary0, 154880, t - 1, frame_->devices[0]->stream);
    auto* candidates = static_cast<std::int32_t*>(w.alloc_bytes((t - 1) * 16 * 4).data);
    auto* unary      = floats(w, (t - 1) * 16);
    peer(candidates, 2, candidate0, 0, (t - 1) * 16 * 4);
    peer(unary, 2, unary0, 0, (t - 1) * 16 * 4);
    auto* selected = static_cast<std::int32_t*>(w.alloc_bytes((t - 1) * 4).data);
    ops::dflash_select(rank, static_cast<const std::uint16_t*>(p.predecessor.qdata),
                       static_cast<const std::uint16_t*>(p.successor.qdata), candidates, unary,
                       selected, anchor, t - 1, s);
    frame_->devices[2]->synchronize();
    CUDA_CHECK(cudaMemcpy(block.data() + 1, selected, (t - 1) * 4, cudaMemcpyDeviceToHost));
    draft_frontier_ = position_;
    ++stats_.draft_calls;
    stats_.drafted_tokens += t - 1;
    stats_.draft_context_tokens += context;
    stats_.workspace_peak[2] = std::max(stats_.workspace_peak[2], w.peak_used());
    stats_.workspace_peak[0] = std::max(stats_.workspace_peak[0], first.peak_used());
    return block;
}

void Program::reset() {
    mtp_frontier_ = 0;
    ledger_.reset();
    frame_ = frames_[0].get();
    position_       = 0;
    draft_frontier_ = 0;
    for (int d = 0; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        CUDA_CHECK(cudaMemsetAsync(recurrent_[d][0].p, 0, recurrent_[d][0].bytes, frame_->devices[d]->stream));
        if (!options_.draft) for (int lane = 0; lane < options_.max_lanes; ++lane) {
            if (!sequence_ready_[lane][d]) sequence_ready_[lane][d] =
                std::make_unique<CudaCompletionEvent>(*frame_->devices[d]);
            sequence_ready_[lane][d]->record(frame_->devices[d]->stream);
        }
        for (auto& frame : frames_) if (frame) frame->state[d].workspace->reset();
    }
    for (auto& l : layers_)
        if (l.retrieval) l.retrieval = std::make_unique<kvmem::RetrievalIndex>(128, 1, 1, 128);
}

void Program::restore(int frontier, int slot) {
    if (frontier < 1 || frontier > position_ || slot < 1 || slot > 8)
        throw std::invalid_argument("GLM restore frontier/snapshot");
    for (int d = 0; d < 3; ++d) {
        frame_->devices[d]->bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(recurrent_[d][0].p, recurrent_[d][slot].p,
                                   recurrent_[d][0].bytes, cudaMemcpyDeviceToDevice,
                                   frame_->devices[d]->stream));
        frame_->devices[d]->synchronize();
    }
    for (auto& l : layers_)
        if (l.retrieval) l.retrieval->truncate_to(frontier);
    position_ = frontier;
}
} // namespace ninfer::models::glm5_next
