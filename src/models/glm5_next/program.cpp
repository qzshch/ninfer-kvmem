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
#include <stdexcept>

namespace ninfer::models::glm5_next {
namespace {
constexpr int H = 4096, Features = 20480, FeatureSlots = 2056;

const float* fp(const Tensor& x) { return static_cast<const float*>(x.data); }
} // namespace

Program::Program(const Model& model, std::array<DeviceContext*, 3> devices, ProgramOptions options)
    : model_(model), parameters_(model.parameters()), devices_(devices), options_(options),
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
    max_rows_ = (options.draft || options.mtp)
                    ? options.max_lanes
                    : std::max(options.max_lanes, static_cast<int>(std::min(options.prefill_chunk, 1024u)));
    host_metadata_.resize(5 * max_rows_ + 2 * options.max_lanes + 1);
    std::array<std::size_t, 3> counts{};
    for (const auto& l : parameters_.layers)
        if (l.linear) {
            if (tensor_parallel_) for (auto& count : counts) ++count;
            else ++counts[l.device];
        }
    if (tensor_parallel_) for (int d = 0; d < 3; ++d) {
        devices_[d]->bind_to_current_thread();
        const auto context = static_cast<std::size_t>(options.max_context);
        const auto channels = static_cast<std::size_t>(model_.parameters(d).layers[0].kda.query.n);
        std::size_t need = counts[d] * channels * (128 + 9) * sizeof(float) * options.max_lanes;
        const auto mla = (context * (512 + 128 + 128) * 2 +
            ((context + 3) / 4) * 128 * 2 + ((context + 127) / 128) * 128 * 4) * options.max_lanes;
        for (const auto& p : model_.parameters(d).layers) if (!p.linear) need += mla;
        need += static_cast<std::size_t>(8 * H + (d == 0 ? 4 * H : 0) + 24) * 4 * max_rows_;
        need += host_metadata_.size() * sizeof(int) + (2ULL << 30);
        std::size_t available = 0, total = 0;
        CUDA_CHECK(hipMemGetInfo(&available, &total));
        if (need > available || available - need < (512ULL << 20))
            throw std::runtime_error("GLM TP/EP resident state/workspace and reserve exceed free device memory; reduce resident context/lanes");
    }
    for (int d = 0; d < 3; ++d) {
        devices_[d]->bind_to_current_thread();
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
            state_[d].recurrent[slot] = DeviceBuffer(bytes * options.max_lanes);
        state_[d].streams          = DeviceBuffer(4 * H * 4 * max_rows_);
        state_[d].metadata         = DeviceBuffer(host_metadata_.size() * sizeof(int));
        state_[d].hidden           = DeviceBuffer(H * 4 * max_rows_);
        state_[d].workspace        = std::make_unique<WorkspaceArena>(2ULL * 1024 * 1024 * 1024);
        if (tensor_parallel_) {
            state_[d].partial = DeviceBuffer(2ULL * H * 4 * max_rows_);
            if (!d) state_[d].received = DeviceBuffer(4ULL * H * 4 * max_rows_);
            state_[d].coefficients = DeviceBuffer(24ULL * 4 * max_rows_);
            state_[d].normalized = DeviceBuffer(H * 4ULL * max_rows_);
            state_[d].ready = std::make_unique<CudaCompletionEvent>(*devices_[d]);
        }
        stats_.persistent_bytes[d] = bytes * (options.draft ? 9 : options.max_lanes) +
                                     state_[d].streams.bytes + state_[d].metadata.bytes +
                                     state_[d].hidden.bytes + state_[d].partial.bytes + state_[d].received.bytes +
                                     state_[d].coefficients.bytes + state_[d].normalized.bytes;
    }
    std::array<std::size_t, 3> cursor{};
    for (int d = 0; d < 3; ++d)
    for (int i = 0; i < 45; ++i) {
        const auto& p = model_.parameters(d).layers[i];
        if (!tensor_parallel_ && p.device != d) continue;
        auto& l = layer_state(d, i);
        devices_[d]->bind_to_current_thread();
        if (p.linear) {
            const auto channels = static_cast<std::size_t>(p.kda.query.n);
            const auto state_floats = channels * 128;
            // Layer-major pools: a compact execution row selects a physical
            // state directly, without gathering the complete recurrent image.
            auto* base  = static_cast<float*>(state_[d].recurrent[0].p) + cursor[d];
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
    devices_[2]->bind_to_current_thread();
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
    for (int d = 0; d < 3; ++d) {
        devices_[d]->bind_to_current_thread_noexcept();
        try {
            devices_[d]->synchronize();
        } catch (...) {}
        for (int i = 0; i < 45; ++i)
            if (tensor_parallel_ || parameters_.layers[i].device == d) {
                auto& l = layer_state(d, i);
                l.latent = {}; l.keys = {}; l.gates = {}; l.pools = {}; l.means = {};
            }
        for (auto& [key, family] : graphs_) {
            family[d].reset();
            if (d == 0) family[3].reset();
        }
        for (auto& [key, graph] : tensor_graphs_) if (key[3] % 3 == d) graph.reset();
        state_[d].ready.reset();
        state_[d].partial = {}; state_[d].received = {};
        state_[d].coefficients = {}; state_[d].normalized = {};
        state_[d].workspace.reset();
        state_[d].streams  = {};
        state_[d].metadata = {};
        state_[d].hidden   = {};
        for (auto& s : state_[d].recurrent) s = {};
        if (d == 2) {
            mtp_state_ = {};
            features_  = {};
            for (auto& c : draft_cache_) {
                c.key   = {};
                c.value = {};
            }
        }
    }
}

float* Program::floats(WorkspaceArena& w, std::size_t n) {
    return static_cast<float*>(w.alloc_bytes(n * 4).data);
}

int* Program::metadata(int device, MetadataField field) const noexcept {
    return static_cast<int*>(state_[device].metadata.p) + static_cast<int>(field) * max_rows_;
}

void Program::projection(const float* x, const Weight& w, float* y, int t, int d, bool round) {
    Tensor a(const_cast<float*>(x), DType::FP32, {w.k, t}), b(y, DType::FP32, {w.n, t});
    ops::linear_fp32(a, w, b, devices_[d]->stream, round);
}

void Program::mlp(const float* x, const MlpParameters& p, float* out, int intermediate, int t,
                  int d, float limit) {
    auto& w    = *state_[d].workspace;
    auto scope = w.scope();
    auto* g    = floats(w, intermediate * t);
    auto* u    = floats(w, intermediate * t);
    projection(x, p.gate, g, t, d);
    projection(x, p.up, u, t, d);
    ops::resident_swiglu(g, u, g, static_cast<std::size_t>(intermediate) * t, limit,
                         devices_[d]->stream);
    if (tensor_parallel_)
        ops::linear_partial_fp32(g, p.down, out, p.column_begin, intermediate, t, devices_[d]->stream);
    else projection(g, p.down, out, t, d);
}

void Program::hyper_prepare(const float* residual, const HyperParameters& p, float* coeff,
                            float* out, int d, int t) {
    auto& w      = *state_[d].workspace;
    auto scope   = w.scope();
    auto* norm   = floats(w, 4 * H * t);
    auto* mix    = floats(w, 24 * t);
    const auto s = devices_[d]->stream;
    ops::resident_norm(residual, nullptr, nullptr, nullptr, norm, 4 * H, t, 1e-5f, false, false, s);
    projection(norm, p.projection, mix, t, d, false);
    ops::hyper_connection_coefficients(mix, fp(p.base), fp(p.scale), coeff, t, 20, 1e-6f, s);
    ops::hyper_connection_collapse(residual, coeff, out, H, t, s);
}

void Program::kda(const float* x, const KdaParameters& p, LayerState& state, float* out, int d,
                  int t) {
    const int channels = p.query.n, heads = channels / 128;
    auto& w      = *state_[d].workspace;
    auto scope   = w.scope();
    const auto s = devices_[d]->stream;
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
                                         q, channels, sequences_, s);
        ops::resident_causal_conv_packed(k, p.key_conv, state.conv[1], slots, offsets,
                                         k, channels, sequences_, s);
        ops::resident_causal_conv_packed(v, p.value_conv, state.conv[2], slots, offsets,
                                         v, channels, sequences_, s);
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
                1.f / std::sqrt(128.f), w, image, yb, devices_[d]->execution_view(),
                ops::KdaDecayParameter::NegativeRate);
        } else {
            Tensor ids(const_cast<int*>(slots), DType::I32, {sequences_});
            Tensor ranges(const_cast<int*>(offsets), DType::I32, {sequences_ + 1});
            ops::kimi_delta_attention_packed_update(
                qb, kb, vb, gb, bb, p.negative_decay_rate, bias, -5.f,
                1.f / std::sqrt(128.f), image, ids, ranges, yb, s,
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
    auto& w       = *state_[d].workspace;
    auto scope    = w.scope();
    auto* feature = floats(w, 128);
    ops::resident_mean_bf16(static_cast<const std::uint16_t*>(layer.keys.p) + block * 128 * 128,
                            feature, 128, 128, devices_[d]->stream);
    devices_[d]->synchronize();
    std::array<float, 128> mean{};
    CUDA_CHECK(cudaMemcpy(mean.data(), feature, sizeof(mean), cudaMemcpyDeviceToHost));
    layer.retrieval->write_block_mean(block, 0, mean);
}

void Program::mla(const float* x, const MlaParameters& p, LayerState& state, float* out, int d,
                  int t) {
    const int heads = p.query_heads.n / 256;
    auto& w      = *state_[d].workspace;
    auto scope   = w.scope();
    const auto s = devices_[d]->stream;
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
            devices_[d]->synchronize();
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
                                  slots, positions, options_.max_context, tier_, t, s);
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
    auto& w      = *state_[d].workspace;
    auto scope   = w.scope();
    const auto s = devices_[d]->stream;
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
        ops::resident_swiglu(g, u, g, 8 * 2048 * t, 10.f, s);
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
    devices_[source]->bind_to_current_thread();
    devices_[source]->synchronize();
    devices_[d]->bind_to_current_thread();
    CUDA_CHECK(hipMemcpyPeerAsync(dst, devices_[d]->device, src, devices_[source]->device, bytes,
                                  devices_[d]->stream));
    devices_[d]->synchronize();
    stats_.peer_bytes += bytes;
}

void Program::capture(const float* streams, int layer, int d) {
    constexpr int taps[5] = {5, 14, 24, 33, 42};
    const auto found      = std::find(std::begin(taps), std::end(taps), layer);
    if (found == std::end(taps)) return;
    const int tap = found - std::begin(taps);
    auto& w       = *state_[d].workspace;
    auto scope    = w.scope();
    auto* mean    = floats(w, H);
    ops::mean_streams(streams, mean, H, 1, devices_[d]->stream);
    auto* destination = static_cast<std::uint16_t*>(features_.p) +
                        (static_cast<std::size_t>(position_ % FeatureSlots) * 5 + tap) * H;
    if (d == 2)
        ops::pack_bf16(mean, destination, H, devices_[d]->stream);
    else {
        auto& target   = *state_[2].workspace;
        auto ts        = target.scope();
        auto* received = floats(target, H);
        peer(received, 2, mean, d, H * 4);
        ops::pack_bf16(received, destination, H, devices_[2]->stream);
        devices_[2]->synchronize();
        devices_[d]->bind_to_current_thread();
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
    for (auto& d : state_) d.workspace->reset();
    devices_[0]->bind_to_current_thread();
    auto& work = *state_[0].workspace;
    auto* id   = static_cast<std::int32_t*>(work.alloc_bytes(4).data);
    CUDA_CHECK(cudaMemcpy(id, &token, 4, cudaMemcpyHostToDevice));
    auto* embedding = floats(work, H);
    ops::embedding_fp32(parameters_.embedding, id, embedding, 1, devices_[0]->stream);
    ops::replicate_streams(embedding, static_cast<float*>(state_[0].streams.p), H, 1,
                           devices_[0]->stream);
    work.reset();
    for (int i = 0; i < 45; ++i) {
        const auto& p = parameters_.layers[i];
        const int d   = p.device;
        if (i == 16 || i == 31)
            peer(state_[d].streams.p, d, state_[d - 1].streams.p, d - 1, 4 * H * 4);
        devices_[d]->bind_to_current_thread();
        auto& w     = *state_[d].workspace;
        auto scope  = w.scope();
        auto *coeff = floats(w, 24), *x = floats(w, H), *y = floats(w, H);
        auto* residual = static_cast<float*>(state_[d].streams.p);
        const auto s   = devices_[d]->stream;
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
    devices_[2]->bind_to_current_thread();
    auto& last   = *state_[2].workspace;
    auto* hidden = floats(last, H);
    ops::mean_streams(static_cast<float*>(state_[2].streams.p), hidden, H, 1, devices_[2]->stream);
    auto& first = *state_[0].workspace;
    auto* final = floats(first, H);
    peer(final, 0, hidden, 2, H * 4);
    ops::resident_norm(final, &parameters_.output_norm, nullptr, nullptr, final, H, 1, 1e-5f, false,
                       true, devices_[0]->stream);
    auto* logits = floats(first, 154880);
    projection(final, parameters_.output_head, logits, 1, 0, false);
    auto* result = static_cast<std::int32_t*>(first.alloc_bytes(4).data);
    ops::resident_argmax(logits, result, 154880, 1, devices_[0]->stream);
    devices_[0]->synchronize();
    std::int32_t next;
    CUDA_CHECK(cudaMemcpy(&next, result, 4, cudaMemcpyDeviceToHost));
    if (next < 0) throw std::runtime_error("GLM target produced nonfinite logits");
    if (snapshot)
        for (int d = 0; d < 3; ++d) {
            devices_[d]->bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(state_[d].recurrent[snapshot].p, state_[d].recurrent[0].p,
                                       state_[d].recurrent[0].bytes, cudaMemcpyDeviceToDevice,
                                       devices_[d]->stream));
            devices_[d]->synchronize();
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
    for (auto& state : state_) state.workspace->reset();
    devices_[0]->bind_to_current_thread();
    auto& first = *state_[0].workspace;
    auto* id    = static_cast<int*>(first.alloc_bytes(4).data);
    CUDA_CHECK(cudaMemcpy(id, &token, 4, cudaMemcpyHostToDevice));
    auto* embedded = floats(first, H);
    ops::embedding_fp32(parameters_.embedding, id, embedded, 1, devices_[0]->stream);
    auto& work   = *state_[2].workspace;
    auto* joined = floats(work, 2 * H);
    peer(joined, 2, embedded, 0, H * 4);
    peer(joined + H, 2, state_[0].hidden.p, 0, H * 4);
    const auto stream = devices_[2]->stream;
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
        devices_[2]->synchronize();
        return -1;
    }
    ops::resident_norm(residual, &p.output_norm, nullptr, nullptr, residual, H, 1, 1e-5f, false,
                       true, stream);
    auto* input = floats(first, H);
    peer(input, 0, residual, 2, H * 4);
    auto* logits = floats(first, 154880);
    projection(input, parameters_.output_head, logits, 1, 0, false);
    auto* output = static_cast<int*>(first.alloc_bytes(4).data);
    ops::resident_argmax(logits, output, 154880, 1, devices_[0]->stream);
    devices_[0]->synchronize();
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
    frontiers_[lane] = 0;
    for (int d = 0; d < 3; ++d) {
        devices_[d]->bind_to_current_thread();
        for (int i = 0; i < 45; ++i) {
            const auto& p = model_.parameters(d).layers[i];
            if (!p.linear || (!tensor_parallel_ && p.device != d)) continue;
            const auto channels = static_cast<std::size_t>(p.kda.query.n);
            const auto state_floats = channels * 128;
            auto& layer = layer_state(d, i);
            CUDA_CHECK(cudaMemsetAsync(layer.recurrent + lane * state_floats, 0,
                                       state_floats * sizeof(float), devices_[d]->stream));
            for (auto* conv : layer.conv)
                CUDA_CHECK(cudaMemsetAsync(conv + lane * channels * 3, 0,
                                           channels * 3 * sizeof(float), devices_[d]->stream));
        }
    }
    // KV, pools and derived means are overwritten before every visible read.
}

void Program::batch_stage(int d) {
    auto& work = *state_[d].workspace;
    work.reset();
    const auto stream = devices_[d]->stream;
    auto* residual    = static_cast<float*>(state_[d].streams.p);
    if (d == 0) {
        auto scope      = work.scope();
        auto* embedding = floats(work, H * batch_);
        ops::embedding_fp32(parameters_.embedding, metadata(0, TokenIds),
                            embedding, batch_, stream);
        ops::replicate_streams(embedding, residual, H, batch_, stream);
    }
    for (int i = 0; i < 45; ++i) {
        const auto& p = parameters_.layers[i];
        if (p.device != d) continue;
        auto scope  = work.scope();
        auto* coeff = floats(work, 24 * batch_);
        auto* x     = floats(work, H * batch_);
        auto* y     = floats(work, H * batch_);
        hyper_prepare(residual, p.attention_hc, coeff, x, d, batch_);
        ops::resident_norm(x, &p.attention_norm, nullptr, nullptr, x, H, batch_, 1e-5f, false, true,
                           stream);
        if (p.linear)
            kda(x, p.kda, layers_[i], y, d, batch_);
        else
            mla(x, p.mla, layers_[i], y, d, batch_);
        ops::hyper_connection_expand(residual, y, coeff, residual, H, batch_, stream);
        hyper_prepare(residual, p.ffn_hc, coeff, x, d, batch_);
        ops::resident_norm(x, &p.ffn_norm, nullptr, nullptr, x, H, batch_, 1e-5f, false, true,
                           stream);
        feed_forward(x, p, y, batch_);
        ops::hyper_connection_expand(residual, y, coeff, residual, H, batch_, stream);
    }
    if (d == 2 && outputs_) {
        auto scope = work.scope();
        const float* selected = residual;
        if (outputs_ != batch_) {
            auto* compact = floats(work, 4 * H * outputs_);
            ops::resident_gather_rows(residual, metadata(d, OutputRows), compact,
                                      4 * H, outputs_, stream);
            selected = compact;
        }
        ops::mean_streams(selected, static_cast<float*>(state_[2].hidden.p), H, outputs_, stream);
    }
    stats_.workspace_peak[d] = std::max(stats_.workspace_peak[d], work.peak_used());
}

void Program::batch_head() {
    auto& work = *state_[0].workspace;
    work.reset();
    const auto stream = devices_[0]->stream;
    auto* final       = static_cast<float*>(state_[0].hidden.p);
    ops::resident_norm(final, &parameters_.output_norm, nullptr, nullptr, final, H, outputs_, 1e-5f,
                       false, true, stream);
    auto* logits = floats(work, 154880 * outputs_);
    projection(final, parameters_.output_head, logits, outputs_, 0, false);
    ops::resident_argmax(logits, metadata(0, PredictedIds), 154880, outputs_,
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
        spans[row] = {static_cast<unsigned>(slot), static_cast<unsigned>(frontiers_[slot]), row, 1};
        outputs[row] = row;
    }
    return execute({runtime::TokenBatchKind::Decode, tokens,
                    {spans.data(), tokens.size()}, {outputs.data(), tokens.size()}});
}

std::vector<std::int32_t> Program::execute(const runtime::TokenBatch& input) {
    if (options_.draft) throw std::invalid_argument("packed Program excludes draft execution");
    input.validate(options_.max_lanes, options_.max_context, max_rows_);
    for (const auto& span : input.spans)
        if (span.input_frontier != static_cast<unsigned>(frontiers_[span.lane]))
            throw std::invalid_argument("packed Program stale input frontier");
    for (auto token : input.tokens)
        if (token < 0 || token >= 154880)
            throw std::invalid_argument("packed Program token outside vocabulary");
    kind_ = input.kind;
    batch_ = input.tokens.size();
    sequences_ = input.spans.size();
    outputs_ = input.output_rows.size();
    tier_ = 128;
    auto* offsets = host_metadata_.data() + 5 * max_rows_;
    auto* sequence_slots = offsets + options_.max_lanes + 1;
    for (int r = 0; r < sequences_; ++r) {
        const auto& span = input.spans[r];
        offsets[r] = span.begin;
        sequence_slots[r] = span.lane;
        for (unsigned t = 0; t < span.count; ++t) {
            const int row = span.begin + t;
            host_metadata_[row] = input.tokens[row];
            host_metadata_[max_rows_ + row] = span.lane;
            host_metadata_[2 * max_rows_ + row] = span.input_frontier + t;
        }
        while (tier_ < static_cast<int>(span.input_frontier + span.count)) tier_ *= 2;
    }
    offsets[sequences_] = batch_;
    std::copy(input.output_rows.begin(), input.output_rows.end(),
              host_metadata_.begin() + 3 * max_rows_);
    for (int d = 0; d < 3; ++d) {
        devices_[d]->bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(state_[d].metadata.p, host_metadata_.data(),
                                   host_metadata_.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, devices_[d]->stream));
        // Consumers use this same stream. The owning host frame remains stable
        // until the complete stage chain finishes; no per-upload host wait.
    }
    std::vector<int> result;
    if (tensor_parallel_) result = tensor_execute();
    else {
    // Temporal prefill uses the same stage composition eagerly. Decode Graphs
    // retain exact request geometry; no temporal span is relabeled as a lane.
    const bool graph = options_.graph && kind_ == runtime::TokenBatchKind::Decode;
    const int key = batch_ * 65536 + tier_;
    if (graph && !graphs_.contains(key)) {
        auto& family = graphs_[key];
        for (int g = 0; g < 4; ++g) {
            const int d = g == 3 ? 0 : g;
            devices_[d]->bind_to_current_thread();
            DecodeGraphDefinition definition;
            definition.capture(devices_[d]->stream, [&] {
                if (g == 3)
                    batch_head();
                else
                    batch_stage(d);
            });
            family[g].instantiate(definition);
            family[g].upload(devices_[d]->stream);
            devices_[d]->synchronize();
            ++stats_.graph_captures;
        }
    }
    for (int d = 0; d < 3; ++d) {
        if (d)
            peer(state_[d].streams.p, d, state_[d - 1].streams.p, d - 1,
                 static_cast<std::size_t>(4 * H * 4) * batch_);
        devices_[d]->bind_to_current_thread();
        if (graph) {
            graphs_.at(key)[d].launch(devices_[d]->stream);
            ++stats_.graph_replays;
        } else
            batch_stage(d);
    }
    result.resize(outputs_);
    if (outputs_) {
        peer(state_[0].hidden.p, 0, state_[2].hidden.p, 2,
              static_cast<std::size_t>(H * 4) * outputs_);
        devices_[0]->bind_to_current_thread();
        if (graph) {
            graphs_.at(key)[3].launch(devices_[0]->stream);
            ++stats_.graph_replays;
        } else
            batch_head();
        devices_[0]->synchronize();
        CUDA_CHECK(cudaMemcpy(result.data(), metadata(0, PredictedIds),
                              outputs_ * sizeof(int), cudaMemcpyDeviceToHost));
        for (auto token : result)
            if (token < 0) throw std::runtime_error("packed Program nonfinite logits");
    } else {
        // No head means no final device-to-host result copy. Still complete the
        // last stage before publishing any input frontier or reusing scratch.
        devices_[2]->synchronize();
    }
    }
    for (const auto& span : input.spans) {
        for (unsigned t = 0; t < span.count; ++t) {
            if (options_.kvmem_window_pages) {
                stats_.kvmem_selections += 11;
                if ((span.input_frontier + t + 128) / 128 > options_.kvmem_window_pages / 2)
                    stats_.kvmem_restricted_steps += 11;
            }
        }
        frontiers_[span.lane] += span.count;
    }
    position_ = frontiers_[0];
    stats_.target_steps += batch_;
    ++stats_.batch_rounds;
    stats_.projection_columns += batch_;
    stats_.predicted_rows += outputs_;
    return result;
}

std::vector<std::int32_t> Program::propose(std::int32_t anchor, int t) {
    if (anchor < 0 || anchor >= 154880 || t < 1 || t > 8 || position_ < 1 ||
        position_ + t > options_.max_context)
        throw std::invalid_argument("GLM DFlash proposal anchor/width/context");
    std::vector<std::int32_t> block(t, 154856);
    block[0] = anchor;
    if (t == 1) return block;
    for (auto& d : state_) d.workspace->reset();
    devices_[0]->bind_to_current_thread();
    auto& first  = *state_[0].workspace;
    auto* tokens = static_cast<std::int32_t*>(first.alloc_bytes(t * 4).data);
    CUDA_CHECK(cudaMemcpy(tokens, block.data(), t * 4, cudaMemcpyHostToDevice));
    auto* embedded = floats(first, t * H);
    ops::embedding_fp32(parameters_.embedding, tokens, embedded, t, devices_[0]->stream);
    auto& w      = *state_[2].workspace;
    auto* hidden = floats(w, t * H);
    peer(hidden, 2, embedded, 0, static_cast<std::size_t>(t) * H * 4);
    const auto s              = devices_[2]->stream;
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
    ops::dflash_top16(logits, candidate0, unary0, 154880, t - 1, devices_[0]->stream);
    auto* candidates = static_cast<std::int32_t*>(w.alloc_bytes((t - 1) * 16 * 4).data);
    auto* unary      = floats(w, (t - 1) * 16);
    peer(candidates, 2, candidate0, 0, (t - 1) * 16 * 4);
    peer(unary, 2, unary0, 0, (t - 1) * 16 * 4);
    auto* selected = static_cast<std::int32_t*>(w.alloc_bytes((t - 1) * 4).data);
    ops::dflash_select(rank, static_cast<const std::uint16_t*>(p.predecessor.qdata),
                       static_cast<const std::uint16_t*>(p.successor.qdata), candidates, unary,
                       selected, anchor, t - 1, s);
    devices_[2]->synchronize();
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
    frontiers_.fill(0);
    position_       = 0;
    draft_frontier_ = 0;
    for (int d = 0; d < 3; ++d) {
        devices_[d]->bind_to_current_thread();
        state_[d].recurrent[0].fill(0);
        state_[d].workspace->reset();
    }
    for (auto& l : layers_)
        if (l.retrieval) l.retrieval = std::make_unique<kvmem::RetrievalIndex>(128, 1, 1, 128);
}

void Program::restore(int frontier, int slot) {
    if (frontier < 1 || frontier > position_ || slot < 1 || slot > 8)
        throw std::invalid_argument("GLM restore frontier/snapshot");
    for (int d = 0; d < 3; ++d) {
        devices_[d]->bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(state_[d].recurrent[0].p, state_[d].recurrent[slot].p,
                                   state_[d].recurrent[0].bytes, cudaMemcpyDeviceToDevice,
                                   devices_[d]->stream));
        devices_[d]->synchronize();
    }
    for (auto& l : layers_)
        if (l.retrieval) l.retrieval->truncate_to(frontier);
    position_ = frontier;
}
} // namespace ninfer::models::glm5_next
