#include "models/glm5_next/program.h"
#include "models/glm5_next/math.h"
#include "ninfer/ops/indexed_linear.h"
#include "ninfer/ops/kimi_delta_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/resident_transform.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::glm5_next {
namespace { constexpr int H = 4096, Vocabulary = 154880; }

// The MTP cache owns actual next-token inputs paired with the corresponding
// target post-output-norm teacher. Scratch and row metadata belong to this one
// Program frame; no mutable state is shared with another request.
std::int32_t Program::mtp_apply(std::span<const int> tokens, const float* teachers, bool proposal) {
    const int rows = tokens.size();
    if (rows < 1 || rows > frame_->capacity || mtp_frontier_ + rows > options_.max_context ||
        (proposal && rows != 1) || !teachers)
        throw std::invalid_argument("native MTP temporal extent/teacher");
    for (int token : tokens) if (token < 0 || token >= Vocabulary)
        throw std::invalid_argument("native MTP token outside vocabulary");
    try {
        for (auto& state : frame_->state) state.workspace->reset();
        std::fill(frame_->host_metadata.begin(), frame_->host_metadata.end(), 0);
        frame_->tier = 128;
        while (frame_->tier < mtp_frontier_ + rows) frame_->tier *= 2;
        for (int r = 0; r < rows; ++r) {
            frame_->host_metadata[r] = tokens[r];
            frame_->host_metadata[2 * frame_->capacity + r] = mtp_frontier_ + r;
        }
        std::memcpy(frame_->upload->data(), frame_->host_metadata.data(), frame_->upload->size());
        for (int d : {0, 2}) {
            frame_->devices[d]->bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(frame_->state[d].metadata.p, frame_->upload->data(),
                frame_->upload->size(), cudaMemcpyHostToDevice, frame_->devices[d]->stream));
        }
        const auto& p = parameters_.mtp;
        frame_->devices[0]->bind_to_current_thread();
        auto& first = *frame_->state[0].workspace;
        auto* embedded = floats(first, H * rows);
        ops::embedding_fp32(parameters_.embedding, metadata(0, TokenIds), embedded, rows, frame_->devices[0]->stream);
        auto& work = *frame_->state[2].workspace;
        auto* embed_input = floats(work, H * rows);
        auto* teacher_input = floats(work, H * rows);
        peer(embed_input, 2, embedded, 0, H * sizeof(float) * rows);
        peer(teacher_input, 2, teachers, 0, H * sizeof(float) * rows);
        const auto stream = frame_->devices[2]->stream;
        ops::resident_norm(embed_input, &p.embedding_norm, nullptr, nullptr, embed_input,
            H, rows, rms_norm_epsilon, false, true, stream);
        ops::resident_norm(teacher_input, &p.hidden_norm, nullptr, nullptr, teacher_input,
            H, rows, rms_norm_epsilon, false, true, stream);
        auto* joined = floats(work, 2 * H * rows);
        ops::resident_concat_rows(embed_input, teacher_input, joined, H, rows, stream);
        auto* residual = floats(work, H * rows);
        auto* x = floats(work, H * rows);
        auto* y = floats(work, H * rows);
        projection(joined, p.fusion, residual, rows, 2);
        ops::resident_norm(residual, &p.layer.attention_norm, nullptr, nullptr, x,
            H, rows, rms_norm_epsilon, false, true, stream);
        mla(x, p.layer.mla, mtp_state_, y, 2, rows);
        ops::resident_add(residual, y, residual, H * rows, stream);
        ops::resident_norm(residual, &p.layer.ffn_norm, nullptr, nullptr, x,
            H, rows, rms_norm_epsilon, false, true, stream);
        feed_forward(x, p.layer, y, rows);
        ops::resident_add(residual, y, residual, H * rows, stream);
        mtp_frontier_ += rows;
        if (!proposal) {
            frame_->devices[2]->synchronize();
            return -1;
        }
        ops::resident_norm(residual, &p.output_norm, nullptr, nullptr, residual,
            H, 1, rms_norm_epsilon, false, true, stream);
        auto* input = floats(first, H);
        peer(input, 0, residual, 2, H * sizeof(float));
        auto* logits = floats(first, Vocabulary);
        projection(input, parameters_.output_head, logits, 1, 0, false);
        ops::resident_argmax(logits, metadata(0, PredictedIds), Vocabulary, 1, frame_->devices[0]->stream);
        CUDA_CHECK(cudaMemcpyAsync(frame_->result->data(), metadata(0, PredictedIds), sizeof(int),
            cudaMemcpyDeviceToHost, frame_->devices[0]->stream));
        frame_->devices[0]->synchronize();
        const int result = *static_cast<const int*>(frame_->result->data());
        if (result < 0) throw std::runtime_error("native MTP nonfinite logits");
        ++stats_.draft_calls;
        ++stats_.drafted_tokens;
        return result;
    } catch (...) {
        ledger_.fail(); drain(); throw;
    }
}

std::int32_t Program::mtp_forward(std::int32_t token, bool proposal) {
    if (!options_.mtp || ledger_.busy(0) || mtp_proposal_pending_ || position_ != mtp_frontier_ + 1 ||
        (proposal && position_ + 2 > options_.max_context))
        throw std::invalid_argument("native MTP token/teacher frontier");
    frame_ = frames_[0].get();
    const int result = mtp_apply({&token, 1}, static_cast<const float*>(frame_->state[0].hidden.p), proposal);
    if (proposal) { mtp_proposal_pending_ = true; mtp_anchor_ = token; }
    return result;
}

std::int32_t Program::mtp_prefill_chunk(std::span<const int> tokens,
                                       std::span<const int> next_tokens, bool final) {
    if (!options_.mtp || tokens.empty() || next_tokens.size() != tokens.size() - (final ? 1 : 0) ||
        mtp_frontier_ != position_ || ledger_.busy(0) || mtp_proposal_pending_)
        throw std::invalid_argument("native MTP prefill input/teacher alignment");
    for (unsigned r = 0; r < next_tokens.size(); ++r)
        if (next_tokens[r] < 0 || next_tokens[r] >= Vocabulary ||
            (r + 1 < tokens.size() && next_tokens[r] != tokens[r + 1]))
            throw std::invalid_argument("native MTP prefill next-token alignment");
    const runtime::TokenSpan span{0, static_cast<unsigned>(position_), 0, static_cast<unsigned>(tokens.size())};
    const int last = tokens.size() - 1;
    const auto result = execute({runtime::TokenBatchKind::Prefill, tokens, {&span, 1},
                                final ? std::span<const int>(&last, 1) : std::span<const int>()});
    auto* teachers = static_cast<float*>(frame_->state[0].hidden.p);
    if (!next_tokens.empty()) mtp_apply(next_tokens, teachers, false);
    if (final && last) {
        frame_->devices[0]->bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(teachers, teachers + H * last, H * sizeof(float),
            cudaMemcpyDeviceToDevice, frame_->devices[0]->stream));
        frame_->devices[0]->synchronize();
    }
    return final ? result.at(0) : -1;
}

MtpVerification Program::verify_mtp(int anchor, int proposal) {
    const std::array<int, 2> inputs{anchor, proposal}, rows{0, 1};
    const runtime::TokenSpan span{0, static_cast<unsigned>(position_), 0, 2};
    const auto ticket = submit({runtime::TokenBatchKind::VerifyK1, inputs, {&span, 1}, rows});
    try {
        frame_->devices[0]->bind_to_current_thread();
        frame_->completed->synchronize();
        const auto* predictions = static_cast<const int*>(frame_->result->data());
        if (predictions[0] < 0 || predictions[1] < 0)
            throw std::runtime_error("native K1 target nonfinite logits");
        frame_->valid_prefix = predictions[0] == proposal ? 2 : 1;
        return {ticket, {predictions[0], predictions[1]}, static_cast<unsigned>(frame_->valid_prefix)};
    } catch (...) {
        ledger_.fail(); drain(); throw;
    }
}

void Program::commit_mtp(FrameTicket ticket, unsigned count) {
    ledger_.validate(ticket);
    frame_ = frames_[ticket.slot].get();
    if (!options_.mtp || frame_->kind != runtime::TokenBatchKind::VerifyK1 || count > 2)
        throw std::invalid_argument("native K1 commit exceeds verified input prefix");
    // Preserve the proposal before MTP catch-up reuses row metadata. No target
    // replay, full-state snapshot or CPU state transfer is needed.
    const int proposal = frame_->host_metadata[1];
    frame_->devices[0]->bind_to_current_thread();
    frame_->completed->synchronize();
    const auto* predictions = static_cast<const int*>(frame_->result->data());
    if (predictions[0] < 0 || predictions[1] < 0) {
        ledger_.fail(); drain();
        throw std::runtime_error("native K1 target nonfinite logits");
    }
    frame_->valid_prefix = predictions[0] == proposal ? 2 : 1;
    if (count > static_cast<unsigned>(frame_->valid_prefix))
        throw std::invalid_argument("native K1 commit exceeds verified input prefix");
    try {
        *static_cast<int*>(frame_->upload->data()) = count;
        for (int d = 0; d < 3; ++d) {
            frame_->devices[d]->bind_to_current_thread();
            const auto stream = frame_->devices[d]->stream;
            auto& local = frame_->state[d];
            CUDA_CHECK(cudaMemcpyAsync(local.commit_prefix.p, frame_->upload->data(), sizeof(int),
                cudaMemcpyHostToDevice, stream));
            Tensor prefix(local.commit_prefix.p, DType::I32, {1});
            for (int i = 0; i < 45; ++i) {
                const auto& p = parameters_.layers[i];
                if (!p.linear || p.device != d) continue;
                auto& state = layers_[i];
                const int channels = p.kda.query.n, heads = channels / 128;
                Tensor image(state.recurrent, DType::FP32, {128, 128, heads});
                Tensor factors(state.k1_factors.p, DType::FP32, {128, 3, heads, 2});
                ops::kimi_delta_attention_commit_k1(image, factors, prefix, stream);
                for (int j = 0; j < 3; ++j)
                    ops::resident_causal_conv_commit_k1(state.conv[j],
                        static_cast<const float*>(state.k1_conv[j].p), channels,
                        static_cast<const int*>(prefix.data), stream);
            }
            local.done->record(stream);
        }
        // Keep the writer reservation until every rank's prefix is durable.
        // The same pinned upload buffer is reused by mtp_apply only after this join.
        for (int d = 0; d < 3; ++d) {
            frame_->devices[d]->bind_to_current_thread();
            frame_->state[d].done->synchronize();
        }
        auto* teachers = static_cast<float*>(frame_->state[0].hidden.p);
        if (count == 2) mtp_apply({&proposal, 1}, teachers, false);
        frame_->devices[0]->bind_to_current_thread();
        if (count != 1) {
            const void* retained = count ? teachers + H : mtp_teacher_backup_.p;
            CUDA_CHECK(cudaMemcpyAsync(teachers, retained, H * sizeof(float),
                cudaMemcpyDeviceToDevice, frame_->devices[0]->stream));
            frame_->devices[0]->synchronize();
        }
        if (!count) --mtp_frontier_;
        // Appended MLA/index suffixes remain outside the logical frontier.
        // Each later replacement overwrites latent/key/gate and newly completed
        // pool entries before its per-row causal selector can observe them.
        ledger_.complete_prefix(ticket, count);
        mtp_proposal_pending_ = false;
        mtp_anchor_ = -1;
        position_ = ledger_.frontier(0);
        stats_.target_steps += count;
        ++stats_.batch_rounds;
        stats_.projection_columns += 2;
        stats_.predicted_rows += 2;
        for (int d = 0; d < 3; ++d)
            stats_.workspace_peak[d] = std::max(stats_.workspace_peak[d], frame_->state[d].workspace->peak_used());
    } catch (...) {
        ledger_.fail(); drain(); throw;
    }
}
} // namespace ninfer::models::glm5_next
