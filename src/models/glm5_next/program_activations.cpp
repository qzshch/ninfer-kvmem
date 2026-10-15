#include "models/glm5_next/program.h"
#include "ninfer/ops/indexed_linear.h"
#include "ninfer/ops/kimi_delta_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/resident_transform.h"
#include "ninfer/ops/sparse_latent_attention.h"
#include <cmath>

namespace ninfer::models::glm5_next {
namespace { constexpr int H = 4096; }

void Program::projection(const Tensor& x, const Weight& weights, Tensor& out, int d) {
    ops::linear_mixed(x, weights, out, frame_->devices[d]->stream);
}

void Program::output_projection(const Tensor& input, const Weight& weights, float* out,
                                int begin, int d, int tokens) {
    auto& work = *frame_->state[d].workspace;
    auto scope = work.scope();
    const auto stream = frame_->devices[d]->stream;
    auto x = input.reshape({static_cast<int>(input.numel() / tokens), tokens});
    if (tensor_parallel_) {
        Tensor y(out, DType::FP32, {weights.n, tokens});
        ops::linear_partial(x, weights, y, begin, stream);
    } else {
        auto y = work.alloc(DType::BF16, {weights.n, tokens});
        projection(x, weights, y, d);
        ops::unpack_bf16(static_cast<const std::uint16_t*>(y.data), out, y.numel(), stream);
    }
}

void Program::mlp(const Tensor& x, const MlpParameters& p, float* out, int tokens, int d) {
    auto& work = *frame_->state[d].workspace;
    auto scope = work.scope();
    auto g = work.alloc(DType::BF16, {p.gate.n, tokens});
    auto u = work.alloc(DType::BF16, {p.up.n, tokens});
    projection(x, p.gate, g, d);
    projection(x, p.up, u, d);
    ops::resident_swiglu(g, u, g, 10.f, frame_->devices[d]->stream);
    output_projection(g, p.down, out, p.column_begin, d, tokens);
}

void Program::feed_forward(const Tensor& x, const LayerParameters& p, float* out, int tokens) {
    const int d = p.device;
    if (p.dense) { mlp(x, p.ffn, out, tokens, d); return; }
    auto& work = *frame_->state[d].workspace;
    auto scope = work.scope();
    const auto stream = frame_->devices[d]->stream;
    auto logits = work.alloc(DType::FP32, {288, tokens});
    auto* weights = floats(work, 8ULL * tokens);
    auto* ids = static_cast<int*>(work.alloc_bytes(8ULL * tokens * sizeof(int)).data);
    auto g = work.alloc(DType::BF16, {2048, 8 * tokens});
    auto u = work.alloc(DType::BF16, {2048, 8 * tokens});
    auto experts = work.alloc(DType::BF16, {H, 8 * tokens});
    projection(x, p.router, logits, d);
    ops::resident_router(static_cast<const float*>(logits.data), static_cast<const float*>(p.router_bias.data),
                          ids, weights, 288, 8, tokens, 2.5f, stream);
    const auto indexed = [&](const Tensor& input, const Weight& bank, Tensor& result, int rows, bool shared) {
        const int first = tensor_parallel_ ? p.expert_begin : 0;
        if (tokens >= 16)
            ops::indexed_linear_grouped(input, bank, ids, result, rows, 8, tokens,
                                        shared, first, work, stream);
        else ops::indexed_linear_mixed(input, bank, ids, result, rows, 8, tokens, shared, first, stream);
    };
    indexed(x, p.experts.gate, g, 2048, true);
    indexed(x, p.experts.up, u, 2048, true);
    if (tensor_parallel_)
        ops::resident_swiglu_owned(g, u, ids, p.expert_begin, p.experts.gate.n / 2048, g, 10.f, stream);
    else ops::resident_swiglu(g, u, g, 10.f, stream);
    indexed(g, p.experts.down, experts, H, false);
    if (tensor_parallel_) {
        mlp(x, p.ffn, out, tokens, d);
        ops::resident_moe_partial(experts, weights, out + H * tokens, H, 8, tokens, stream);
    } else {
        auto* shared = floats(work, H * tokens);
        mlp(x, p.ffn, shared, tokens, d);
        ops::resident_moe_sum(experts, weights, shared, out, H, 8, tokens, stream);
    }
}

void Program::kda(const Tensor& x, const KdaParameters& p, LayerState& state,
                   float* out, int d, int tokens) {
    const int channels = p.query.n, heads = channels / 128;
    auto& work = *frame_->state[d].workspace;
    auto scope = work.scope();
    const auto stream = frame_->devices[d]->stream;
    auto q = work.alloc(DType::BF16, {channels, tokens});
    auto k = work.alloc(DType::BF16, {channels, tokens});
    auto v = work.alloc(DType::BF16, {channels, tokens});
    auto g = work.alloc(DType::BF16, {channels, tokens});
    auto gate = work.alloc(DType::BF16, {channels, tokens});
    auto small = work.alloc(DType::BF16, {128, tokens});
    auto beta = work.alloc(DType::BF16, {heads, tokens});
    auto result = work.alloc(DType::BF16, {128, heads, tokens});
    const auto* slots = metadata(d, SequenceOffsets) + options_.max_lanes + 1;
    const auto* offsets = metadata(d, SequenceOffsets);
    projection(x, p.query, q, d); projection(x, p.key, k, d); projection(x, p.value, v, d);
    ops::resident_causal_conv_packed(q, p.query_conv, state.conv[0], slots, offsets, q, frame_->sequences, stream);
    ops::resident_causal_conv_packed(k, p.key_conv, state.conv[1], slots, offsets, k, frame_->sequences, stream);
    ops::resident_causal_conv_packed(v, p.value_conv, state.conv[2], slots, offsets, v, frame_->sequences, stream);
    projection(x, p.forget_in, small, d); projection(small, p.forget_out, g, d);
    projection(x, p.output_gate_in, small, d); projection(small, p.output_gate_out, gate, d);
    projection(x, p.input_gate, beta, d);
    Tensor image(state.recurrent, DType::FP32, {128, 128, heads, options_.max_lanes});
    Tensor ids(const_cast<int*>(slots), DType::I32, {frame_->sequences});
    Tensor ranges(const_cast<int*>(offsets), DType::I32, {frame_->sequences + 1});
    ops::kimi_delta_attention_packed_update(q.reshape({128, heads, tokens}), k.reshape({128, heads, tokens}),
        v.reshape({128, heads, tokens}), g.reshape({128, heads, tokens}), beta, p.negative_decay_rate,
        p.decay_bias.reshape({128, heads}), -5.f, 1.f / std::sqrt(128.f), work, image, ids, ranges,
        result, stream, ops::KdaDecayParameter::NegativeRate);
    auto gates = gate.reshape({128, heads, tokens});
    ops::resident_norm(result, &p.output_norm, nullptr, &gates, result, 1e-5f, false, stream);
    output_projection(result, p.output, out, p.head_begin * 128, d, tokens);
}

void Program::mla(const Tensor& x, const MlaParameters& p, LayerState& state,
                   float* out, int d, int tokens) {
    const int heads = p.query_heads.n / 256;
    auto& work = *frame_->state[d].workspace;
    auto scope = work.scope();
    const auto stream = frame_->devices[d]->stream;
    auto low = work.alloc(DType::BF16, {1536, tokens});
    auto query = work.alloc(DType::BF16, {heads * 256, tokens});
    auto latent = work.alloc(DType::BF16, {512, tokens});
    auto iq = work.alloc(DType::BF16, {4096, tokens});
    auto ik = work.alloc(DType::BF16, {128, tokens});
    auto ig = work.alloc(DType::BF16, {128, tokens});
    auto ih = work.alloc(DType::FP32, {32, tokens});
    auto absorbed = work.alloc(DType::FP32, {512, heads * tokens});
    auto attended = work.alloc(DType::FP32, {512, heads * tokens});
    auto expanded = work.alloc(DType::BF16, {256, heads * tokens});
    auto* scores = floats(work, 8192ULL * tokens);
    auto* prob = floats(work, 2051ULL * heads * tokens);
    auto* ids = static_cast<int*>(work.alloc_bytes(2051ULL * tokens * sizeof(int)).data);
    projection(x, p.query_lowrank, low, d);
    ops::resident_norm(low, &p.query_norm, nullptr, nullptr, low, 1e-6f, false, stream);
    projection(low, p.query_heads, query, d);
    projection(x, p.kv_latent, latent, d);
    ops::resident_norm(latent, &p.kv_norm, nullptr, nullptr, latent, 1e-6f, false, stream);
    projection(low, p.index.query, iq, d);
    projection(x, p.index.key, ik, d);
    ops::resident_norm(ik, &p.index.key_norm, &p.index.key_norm_bias, nullptr, ik, 1e-6f, true, stream);
    projection(x, p.index.pool_gate, ig, d);
    projection(x, p.index.head_weights, ih, d);
    const auto* slots = metadata(d, StateSlots);
    const auto* positions = metadata(d, Positions);
    const auto bf = [](const Tensor& t) { return static_cast<const std::uint16_t*>(t.data); };
    ops::resident_index_append(bf(latent), bf(ik), bf(ig), static_cast<const float*>(p.index.position_bias.data),
        static_cast<std::uint16_t*>(state.latent.p), static_cast<std::uint16_t*>(state.keys.p),
        static_cast<std::uint16_t*>(state.gates.p), static_cast<std::uint16_t*>(state.pools.p),
        slots, positions, options_.max_context, tokens, stream);
    auto* allowed = static_cast<std::uint8_t*>(work.alloc_bytes(tokens * ((options_.max_context + 63) / 64)).data);
    if (options_.kvmem_window_pages)
        ops::resident_retrieval(bf(iq), static_cast<const std::uint16_t*>(state.keys.p),
            static_cast<float*>(state.means.p), allowed, slots, positions, options_.max_context,
            tokens, options_.kvmem_window_pages / 2, stream);
    ops::resident_pool_select(bf(iq), static_cast<const float*>(ih.data), static_cast<const std::uint16_t*>(state.pools.p),
        options_.kvmem_window_pages ? allowed : nullptr, scores, ids, slots, positions,
        options_.max_context, frame_->tier, tokens, stream);
    auto query_heads = query.reshape({256, heads * tokens});
    ops::indexed_linear_mixed(query_heads, p.key_absorb, nullptr, absorbed, 512, heads, tokens, false, 0, stream);
    ops::resident_sparse_attention(static_cast<const float*>(absorbed.data), static_cast<const std::uint16_t*>(state.latent.p),
        ids, prob, static_cast<float*>(attended.data), slots, options_.max_context, tokens, stream, heads);
    ops::indexed_linear_mixed(attended, p.value_expand, nullptr, expanded, 256, heads, tokens, false, 0, stream);
    output_projection(expanded, p.output, out, p.head_begin * 256, d, tokens);
}
} // namespace ninfer::models::glm5_next
