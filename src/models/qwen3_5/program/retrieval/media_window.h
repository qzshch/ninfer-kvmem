#pragma once

#include "ninfer/types.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/retrieval/block_retrieval.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct MediaPageGroup {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
};

// Include the predecessor consumed by the MTP embedding bridge. Items sharing a
// physical page form one indivisible group, including intervening video timestamps.
inline std::vector<MediaPageGroup> media_page_groups(std::span<const VisionItem> items) {
    std::vector<MediaPageGroup> groups;
    for (const auto& item : items) {
        if (item.token_spans.empty()) {
            throw std::invalid_argument("media item has no token span");
        }
        const auto first = item.token_spans.front().begin;
        const auto& last = item.token_spans.back();
        const MediaPageGroup group{static_cast<std::uint32_t>((first == 0 ? 0 : first - 1) / 64),
                                   static_cast<std::uint32_t>((last.begin + last.count + 63) / 64)};
        if (!groups.empty() && group.begin < groups.back().end) {
            groups.back().end = std::max(groups.back().end, group.end);
        } else {
            groups.push_back(group);
        }
    }
    return groups;
}

inline void validate_context_window(std::span<const MediaPageGroup> groups, std::uint32_t budget,
                                    std::span<const TokenSpan> instructions = {}) {
    if (groups.empty() && instructions.empty()) { return; }
    std::uint32_t extent = 2;
    for (const auto& span : instructions) {
        extent = std::max(extent, static_cast<std::uint32_t>((span.begin + span.count + 63U) / 64U));
    }
    for (const auto& group : groups) { extent = std::max(extent, group.end); }
    std::vector<std::uint8_t> pinned(extent, 0);
    pinned[0] = pinned[1] = 1;
    for (const auto& span : instructions) {
        for (auto p = span.begin / 64U; p < (span.begin + span.count + 63U) / 64U; ++p) {
            pinned[p] = 1;
        }
    }
    // A boundary page can contain both instructions and media. Its complete
    // media group must then stay resident with the instructions.
    for (const auto& group : groups) {
        if (std::any_of(pinned.begin() + group.begin, pinned.begin() + group.end,
                        [](auto kept) { return kept != 0; })) {
            std::fill(pinned.begin() + group.begin, pinned.begin() + group.end, 1);
        }
    }
    const auto mandatory = static_cast<std::uint32_t>(
        std::count(pinned.begin(), pinned.end(), std::uint8_t{1}));
    if (mandatory + 2U > budget) {
        throw RequestError(
            instructions.empty() ? RequestErrorKind::MediaBudgetExceeded
                                 : RequestErrorKind::ContextLengthExceeded,
            "KVMem window cannot hold system/developer instructions plus sink and tail pages");
    }
    for (const auto& group : groups) {
        const auto extra = static_cast<std::uint32_t>(
            std::count(pinned.begin() + group.begin, pinned.begin() + group.end, std::uint8_t{0}));
        // Reject before admission rather than silently cutting mandatory context.
        if (mandatory + extra + 2U > budget) {
            throw RequestError(
                RequestErrorKind::MediaBudgetExceeded,
                "KVMem window cannot hold a complete media group plus instructions and tail pages");
        }
    }
}

// System/developer instructions and latest visible media are mandatory; other
// media is selected as a whole or omitted. Instruction-bearing requests also
// reserve their two latest pages before spending the remainder on retrieval.
// Only the materialized prefix of an in-progress item exists. No future KV is loaded.
// Requests without semantic instruction ranges retain their existing policy.
inline std::vector<std::uint32_t> context_window_page_set(std::uint32_t mapped, std::uint32_t budget,
                                                        std::span<const std::uint32_t> preferred,
                                                        std::span<const MediaPageGroup> groups,
                                                        std::span<const TokenSpan> instructions = {},
                                                        bool fill_recent = true) {
    if (groups.empty() && instructions.empty()) {
        return decode_window_page_set(mapped, budget, preferred);
    }
    std::vector<std::uint8_t> kept(mapped, 0);
    std::uint32_t used = 0;
    const auto keep    = [&](std::uint32_t page) {
        if (page >= mapped) { return true; }
        if (kept[page]) { return true; }
        std::uint32_t begin = page, end = page + 1;
        const auto it = std::upper_bound(
            groups.begin(), groups.end(), page,
            [](std::uint32_t p, const MediaPageGroup& group) { return p < group.begin; });
        if (it != groups.begin() && page < std::prev(it)->end) {
            begin = std::prev(it)->begin;
            end   = std::min(mapped, std::prev(it)->end);
        }
        std::uint32_t cost = 0;
        for (auto p = begin; p < end; ++p) { cost += !kept[p]; }
        if (cost > budget - used) { return false; }
        for (auto p = begin; p < end; ++p) { kept[p] = 1; }
        used += cost;
        return true;
    };
    for (std::uint32_t p = 0; p < std::min(mapped, 2U); ++p) {
        if (!keep(p)) { throw std::logic_error("KVMem media sink exceeds its admitted window"); }
    }
    for (const auto& span : instructions) {
        const auto end = std::min(mapped, static_cast<std::uint32_t>((span.begin + span.count + 63U) / 64U));
        for (auto p = static_cast<std::uint32_t>(span.begin / 64U); p < end; ++p) {
            if (!keep(p)) { throw std::logic_error("KVMem instructions exceed their admitted window"); }
        }
    }
    for (auto it = groups.rbegin(); it != groups.rend(); ++it) {
        if (it->begin < mapped) {
            if (!keep(it->begin)) {
                throw std::logic_error("KVMem latest media exceeds its admitted window");
            }
            break;
        }
    }
    if (!instructions.empty()) {
        for (auto p = mapped > 2U ? mapped - 2U : 0U; p < mapped; ++p) {
            if (!keep(p)) { throw std::logic_error("KVMem tail exceeds its admitted window"); }
        }
    }
    for (const auto p : preferred) { (void)keep(p); }
    if (fill_recent) {
        for (auto p = mapped; p > 0 && used < budget;) { (void)keep(--p); }
    }
    std::vector<std::uint32_t> pages;
    pages.reserve(used);
    for (std::uint32_t p = 0; p < mapped; ++p) {
        if (kept[p]) { pages.push_back(p); }
    }
    return pages;
}

} // namespace ninfer::models::qwen3_5::detail
