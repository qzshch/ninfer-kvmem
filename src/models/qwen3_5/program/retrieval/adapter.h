#pragma once

#include "kvmem/block_retrieval.h"
#include "kvmem/context_window.h"
#include "kvmem/host_budget.h"
#include "kvmem/query_span.h"
#include "kvmem/window_capacity.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "core/host_context_arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/types.h"

namespace ninfer::models::qwen3_5::detail {

static_assert(kvmem::kPageTokens == kPagedKVPageSize);
using kvmem::RetrievalBlockMeta;
using kvmem::RetrievalIndex;
using kvmem::BlockSelectionConfig;
using kvmem::BlockSelection;
using kvmem::MediaPageGroup;
using kvmem::KvMemQuerySpan;
using kvmem::select_blocks;
using kvmem::prefill_window_page_set;
using kvmem::append_prefill_growth_pages;
using kvmem::block_pages;
using kvmem::decode_window_page_set;
using kvmem::kvmem_lane_page_budget;
using kvmem::kvmem_pool_page_budget;
using kvmem::kvmem_replay_quanta;
using kvmem::mark_unique_host_replica;
using kvmem::sparse_host_budget_occupancy;
using kvmem::sparse_host_budget_fits;

inline kvmem::MetadataBudget retrieval_metadata_budget(HostContextArena* arena) noexcept {
    return {arena, arena ? +[](void* owner, std::size_t bytes) -> std::shared_ptr<void> {
        return static_cast<HostContextArena*>(owner)->charge_metadata(bytes);
    } : nullptr};
}

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

inline void validate_context_window(std::span<const MediaPageGroup> groups,
                                    std::uint32_t budget, std::span<const TokenSpan> instructions = {}) {
    try { kvmem::validate_context_window<TokenSpan>(groups, budget, instructions); }
    catch (const kvmem::WindowCapacityError& e) {
        throw RequestError(e.instructions ? RequestErrorKind::ContextLengthExceeded
                                          : RequestErrorKind::MediaBudgetExceeded, e.what());
    }
}

inline std::vector<std::uint32_t> context_window_page_set(std::uint32_t mapped,
    std::uint32_t budget, std::span<const std::uint32_t> preferred,
    std::span<const MediaPageGroup> groups, std::span<const TokenSpan> instructions = {},
    bool fill_recent = true) {
    return kvmem::context_window_page_set<TokenSpan>(mapped, budget, preferred, groups,
                                                    instructions, fill_recent);
}

inline KvMemQuerySpan kvmem_query_span(std::uint32_t prompt_tokens, std::uint32_t reuse_base,
    const std::optional<TokenSpan>& query, std::span<const VisionItem> media = {}) {
    const auto range = query ? std::optional<kvmem::TokenRange>{{query->begin, query->count}}
                             : std::nullopt;
    auto result = kvmem::kvmem_query_span(prompt_tokens, reuse_base, range);
    for (const auto& item : media) {
        if (item.token_spans.empty()) { continue; }
        const auto first = item.token_spans.front().begin;
        const auto& last = item.token_spans.back();
        const auto consumer = first == 0 ? 0 : first - 1;
        if (consumer < result.begin && result.begin < last.begin + last.count) {
            result.begin = std::max(reuse_base, static_cast<std::uint32_t>(consumer));
            break;
        }
    }
    return result;
}

} // namespace ninfer::models::qwen3_5::detail
