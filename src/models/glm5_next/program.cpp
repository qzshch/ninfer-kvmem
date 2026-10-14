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
constexpr std::size_t StateFloats = 64 * 128 * 128, ConvFloats = 3 * 8192 * 3;

const float* fp(const Tensor& x) { return static_cast<const float*>(x.data); }
} // namespace

Program::Program(const Model& model, std::array<DeviceContext*, 3> devices, ProgramOptions options)
    : model_(model), parameters_(model.parameters()), devices_(devices), options_(options) {
    if (options.max_context < 1 || options.max_context > 32768 ||
        (options.kvmem_window_pages &&
         (options.kvmem_window_pages < 4 || options.kvmem_window_pages % 2)))
        throw std::invalid_argument(
            "GLM resident Program requires context 1..32768 and even KVMem window >=4 pages");
    std::array<std::size_t, 3> counts{};
    for (const auto& l : parameters_.layers)
        if (l.linear) ++counts[l.device];
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
        const auto bytes = counts[d] * (StateFloats + ConvFloats) * 4;
        for (auto& state : state_[d].recurrent) state = DeviceBuffer(bytes);
        state_[d].streams          = DeviceBuffer(4 * H * 4);
        state_[d].workspace        = std::make_unique<WorkspaceArena>(2ULL * 1024 * 1024 * 1024);
        stats_.persistent_bytes[d] = bytes * 9 + 4 * H * 4;
    }
    std::array<std::size_t, 3> cursor{};
    for (int i = 0; i < 45; ++i) {
        const auto& p = parameters_.layers[i];
        auto& l       = layers_[i];
        const int d   = p.device;
        devices_[d]->bind_to_current_thread();
        if (p.linear) {
            auto* base  = static_cast<float*>(state_[d].recurrent[0].p) + cursor[d];
            l.recurrent = base;
            for (int j = 0; j < 3; ++j) l.conv[j] = base + StateFloats + j * 8192 * 3;
            cursor[d] += StateFloats + ConvFloats;
        } else {
            const auto context = options_.max_context;
            l.latent           = DeviceBuffer(static_cast<std::size_t>(context) * 512 * 2);
            l.keys             = DeviceBuffer(static_cast<std::size_t>(context) * 128 * 2);
            l.gates            = DeviceBuffer(l.keys.bytes);
            l.pools     = DeviceBuffer(static_cast<std::size_t>((context + 3) / 4) * 128 * 2);
            l.retrieval = std::make_unique<kvmem::RetrievalIndex>(128, 1, 1, 128);
            stats_.persistent_bytes[d] +=
                l.latent.bytes + l.keys.bytes + l.gates.bytes + l.pools.bytes;
        }
    }
    devices_[2]->bind_to_current_thread();
    features_ = DeviceBuffer(static_cast<std::size_t>(FeatureSlots) * Features * 2);
    stats_.persistent_bytes[2] += features_.bytes;
    for (auto& c : draft_cache_) {
        c.key   = DeviceBuffer(static_cast<std::size_t>(FeatureSlots) * 1024 * 2);
        c.value = DeviceBuffer(c.key.bytes);
        stats_.persistent_bytes[2] += c.key.bytes + c.value.bytes;
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
            if (parameters_.layers[i].device == d) {
                layers_[i].latent = {};
                layers_[i].keys   = {};
                layers_[i].gates  = {};
                layers_[i].pools  = {};
            }
        state_[d].workspace.reset();
        state_[d].streams = {};
        for (auto& s : state_[d].recurrent) s = {};
        if (d == 2) {
            features_ = {};
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

void Program::projection(const float* x, const Weight& w, float* y, int t, int d, bool round) {
    Tensor a(const_cast<float*>(x), DType::FP32, {w.k, t}), b(y, DType::FP32, {w.n, t});
    ops::linear_fp32(a, w, b, devices_[d]->stream);
    if (round) ops::round_bf16(y, static_cast<std::size_t>(w.n) * t, devices_[d]->stream);
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
    projection(g, p.down, out, t, d);
}

void Program::hyper_prepare(const float* residual, const HyperParameters& p, float* coeff,
                            float* out, int d) {
    auto& w      = *state_[d].workspace;
    auto scope   = w.scope();
    auto* norm   = floats(w, 4 * H);
    auto* mix    = floats(w, 24);
    const auto s = devices_[d]->stream;
    ops::resident_norm(residual, nullptr, nullptr, nullptr, norm, 4 * H, 1, 1e-5f, false, false, s);
    projection(norm, p.projection, mix, 1, d, false);
    ops::hyper_connection_coefficients(mix, fp(p.base), fp(p.scale), coeff, 1, 20, 1e-6f, s);
    ops::hyper_connection_collapse(residual, coeff, out, H, 1, s);
}

void Program::kda(const float* x, const KdaParameters& p, LayerState& state, float* out, int d) {
    auto& w      = *state_[d].workspace;
    auto scope   = w.scope();
    const auto s = devices_[d]->stream;
    float *q = floats(w, 8192), *k = floats(w, 8192), *v = floats(w, 8192), *g = floats(w, 8192),
          *gate = floats(w, 8192), *small = floats(w, 128), *beta = floats(w, 64),
          *result = floats(w, 8192);
    projection(x, p.query, q, 1, d);
    projection(x, p.key, k, 1, d);
    projection(x, p.value, v, 1, d);
    ops::resident_causal_conv(q, p.query_conv, state.conv[0], q, 8192, 1, s);
    ops::resident_causal_conv(k, p.key_conv, state.conv[1], k, 8192, 1, s);
    ops::resident_causal_conv(v, p.value_conv, state.conv[2], v, 8192, 1, s);
    projection(x, p.forget_in, small, 1, d);
    projection(small, p.forget_out, g, 1, d);
    projection(x, p.output_gate_in, small, 1, d);
    projection(small, p.output_gate_out, gate, 1, d);
    projection(x, p.input_gate, beta, 1, d);
    auto qb = w.alloc(DType::BF16, {128, 64, 1}), kb = w.alloc(DType::BF16, {128, 64, 1}),
         vb = w.alloc(DType::BF16, {128, 64, 1}), gb = w.alloc(DType::BF16, {128, 64, 1}),
         bb = w.alloc(DType::BF16, {64, 1}), yb = w.alloc(DType::BF16, {128, 64, 1});
    ops::pack_bf16(q, static_cast<std::uint16_t*>(qb.data), 8192, s);
    ops::pack_bf16(k, static_cast<std::uint16_t*>(kb.data), 8192, s);
    ops::pack_bf16(v, static_cast<std::uint16_t*>(vb.data), 8192, s);
    ops::pack_bf16(g, static_cast<std::uint16_t*>(gb.data), 8192, s);
    ops::pack_bf16(beta, static_cast<std::uint16_t*>(bb.data), 64, s);
    Tensor recurrent(state.recurrent, DType::FP32, {128, 128, 64}),
        bias = p.decay_bias.reshape({128, 64});
    ops::kimi_delta_attention(qb, kb, vb, gb, bb, p.negative_decay_rate, bias, -5.f,
                              1.f / std::sqrt(128.f), w, recurrent, yb,
                              devices_[d]->execution_view(), ops::KdaDecayParameter::NegativeRate);
    ops::unpack_bf16(static_cast<std::uint16_t*>(yb.data), result, 8192, s);
    ops::resident_norm(result, &p.output_norm, nullptr, gate, result, 128, 64, 1e-5f, false, true,
                       s);
    projection(result, p.output, out, 1, d);
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

void Program::mla(const float* x, const MlaParameters& p, LayerState& state, float* out, int d) {
    auto& w      = *state_[d].workspace;
    auto scope   = w.scope();
    const auto s = devices_[d]->stream;
    auto *low = floats(w, 1536), *query = floats(w, 16384), *latent = floats(w, 512),
         *absorbed = floats(w, 64 * 512), *attended = floats(w, 64 * 512),
         *expanded = floats(w, 16384), *iq = floats(w, 4096), *ik = floats(w, 128),
         *ig = floats(w, 128), *ih = floats(w, 32), *scores = floats(w, 8192),
         *prob = floats(w, 64 * 2051);
    auto* ids  = static_cast<std::int32_t*>(w.alloc_bytes(2051 * 4).data);
    projection(x, p.query_lowrank, low, 1, d);
    ops::resident_norm(low, &p.query_norm, nullptr, nullptr, low, 1536, 1, 1e-5f, false, true, s);
    projection(low, p.query_heads, query, 1, d);
    projection(x, p.kv_latent, latent, 1, d);
    ops::resident_norm(latent, &p.kv_norm, nullptr, nullptr, latent, 512, 1, 1e-5f, false, true, s);
    ops::pack_bf16(latent, static_cast<std::uint16_t*>(state.latent.p) + position_ * 512, 512, s);
    projection(low, p.index.query, iq, 1, d);
    projection(x, p.index.key, ik, 1, d);
    ops::resident_norm(ik, &p.index.key_norm, &p.index.key_norm_bias, nullptr, ik, 128, 1, 1e-6f,
                       true, true, s);
    projection(x, p.index.pool_gate, ig, 1, d);
    projection(x, p.index.head_weights, ih, 1, d, false);
    ops::index_pool_append(ik, ig, fp(p.index.position_bias),
                           static_cast<std::uint16_t*>(state.keys.p),
                           static_cast<std::uint16_t*>(state.gates.p),
                           static_cast<std::uint16_t*>(state.pools.p), position_, s);
    const std::uint8_t* allowed = nullptr;
    if (options_.kvmem_window_pages) {
        auto full = state.retrieval->append(1);
        for (const auto& block : full) publish_retrieval_mean(state, block.block_id, d);
        auto* feature = floats(w, 128);
        ops::resident_sum_rows(iq, feature, 128, 32, s);
        devices_[d]->synchronize();
        std::array<float, 128> q{};
        CUDA_CHECK(cudaMemcpy(q.data(), feature, sizeof(q), cudaMemcpyDeviceToHost));
        std::vector<float> rank(state.retrieval->block_count(),
                                std::numeric_limits<float>::quiet_NaN());
        const std::array<std::uint32_t, 1> count{1};
        for (unsigned b = 0; b < rank.size(); ++b)
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
    ops::indexed_linear_fp32(query, p.key_absorb, nullptr, absorbed, 512, 64, 1, false, s);
    ops::sparse_latent_attention(absorbed, static_cast<const std::uint16_t*>(state.latent.p), ids,
                                 prob, attended, 64, 512, 1, 2051, 1.f / 16, s);
    ops::indexed_linear_fp32(attended, p.value_expand, nullptr, expanded, 256, 64, 1, false, s);
    ops::round_bf16(expanded, 16384, s);
    projection(expanded, p.output, out, 1, d);
}

void Program::feed_forward(const float* x, const LayerParameters& p, float* out) {
    const int d  = p.device;
    auto& w      = *state_[d].workspace;
    auto scope   = w.scope();
    const auto s = devices_[d]->stream;
    if (p.dense) {
        mlp(x, p.ffn, out, 12288, 1, d, 10.f);
        return;
    }
    auto *logits = floats(w, 288), *weights = floats(w, 8), *g = floats(w, 8 * 2048),
         *u = floats(w, 8 * 2048), *experts = floats(w, 8 * H), *shared = floats(w, H);
    auto* ids = static_cast<std::int32_t*>(w.alloc_bytes(8 * 4).data);
    projection(x, p.router, logits, 1, d, false);
    ops::resident_router(logits, fp(p.router_bias), ids, weights, 288, 8, 1, 2.5f, s);
    ops::indexed_linear_fp32(x, p.experts.gate, ids, g, 2048, 8, 1, true, s);
    ops::indexed_linear_fp32(x, p.experts.up, ids, u, 2048, 8, 1, true, s);
    ops::round_bf16(g, 8 * 2048, s);
    ops::round_bf16(u, 8 * 2048, s);
    ops::resident_swiglu(g, u, g, 8 * 2048, 10.f, s);
    ops::indexed_linear_fp32(g, p.experts.down, ids, experts, H, 8, 1, false, s);
    ops::round_bf16(experts, 8 * H, s);
    mlp(x, p.ffn, shared, 2048, 1, d, 10.f);
    ops::resident_moe_sum(experts, weights, shared, out, H, 8, 1, s);
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
    return next;
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
