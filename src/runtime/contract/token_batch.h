#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ninfer::runtime {

enum class TokenBatchKind : std::uint8_t { Decode, Prefill };

// A contiguous span of known input tokens for one logical resident lane. Lane
// identity is independent of the compact token row and Program's physical state.
struct TokenSpan {
    std::uint32_t lane;
    std::uint32_t input_frontier;
    std::uint32_t begin;
    std::uint32_t count;
};

// Borrowed for one Program call. Stateful consumers follow spans in time order;
// stateless projections may process all input rows together. Each output row
// requests the next-token prediction after the last input of its span. Spans
// without an output row only advance input state and do not execute the head.
struct TokenBatch {
    TokenBatchKind kind;
    std::span<const std::int32_t> tokens;
    std::span<const TokenSpan> spans;
    std::span<const std::int32_t> output_rows;

    void validate(std::uint32_t lanes, std::uint32_t context,
                  std::uint32_t row_capacity) const {
        if (lanes < 1 || lanes > 8 || spans.empty() || spans.size() > lanes ||
            tokens.empty() || tokens.size() > row_capacity ||
            (kind != TokenBatchKind::Decode && kind != TokenBatchKind::Prefill))
            throw std::invalid_argument("token batch capacity/kind");
        std::array<bool, 8> used{};
        std::size_t end = 0;
        for (const auto& span : spans) {
            if (span.lane >= lanes || used[span.lane] || !span.count || span.begin != end ||
                span.count > tokens.size() - end || span.input_frontier > context ||
                span.count > context - span.input_frontier ||
                (kind == TokenBatchKind::Decode && span.count != 1))
                throw std::invalid_argument("token batch span/frontier");
            used[span.lane] = true;
            end += span.count;
        }
        if (end != tokens.size()) throw std::invalid_argument("token batch uncovered input");
        std::int32_t previous = -1;
        for (auto row : output_rows) {
            bool last = false;
            for (const auto& span : spans)
                last |= row >= 0 && static_cast<std::uint32_t>(row) == span.begin + span.count - 1;
            if (row <= previous || !last)
                throw std::invalid_argument("token batch output row");
            previous = row;
        }
        if (kind == TokenBatchKind::Decode && output_rows.size() != spans.size())
            throw std::invalid_argument("decode requires one output per span");
    }
};

} // namespace ninfer::runtime
