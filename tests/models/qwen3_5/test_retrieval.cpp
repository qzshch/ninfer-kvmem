#include "models/qwen3_5/program/retrieval/adapter.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace r = ninfer::models::qwen3_5::detail;

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

constexpr std::uint32_t kLayers   = 2;
constexpr std::uint32_t kKvHeads  = 2;
constexpr std::uint32_t kHeadDim  = 8;
constexpr std::uint32_t kBlockTok = 128;

r::RetrievalIndex make_index() { return r::RetrievalIndex(kBlockTok, kLayers, kKvHeads, kHeadDim); }

void test_append_and_truncate() {
    auto index = make_index();
    auto full  = index.append(300);
    expect(index.block_count() == 3 && index.total_tokens() == 300,
           "append splits 300 tokens into three blocks");
    expect(full.size() == 2 && full[0].block_id == 0 && full[1].block_id == 1,
           "append reports exactly the newly full blocks");
    expect(index.block(2).n_tokens == 44 && !index.block(2).full,
           "the trailing partial block holds the remainder");

    index.truncate_to(200);
    expect(index.block_count() == 2 && index.total_tokens() == 200,
           "truncate drops fully-past blocks");

    index.append(56);
    expect(index.block_count() == 2 && index.block(1).n_tokens == 128 && index.block(1).full,
           "re-append refills the shrunk block exactly");
}

void test_scoring() {
    auto index = make_index();
    (void)index.append(256); // two full blocks

    // Block 0: every head points along +x. Block 1: along -x.
    std::vector<float> mean(index.kv_heads() * index.head_dim(), 0.0F);
    for (std::uint32_t head = 0; head < index.kv_heads(); ++head) {
        mean[head * index.head_dim()] = 1.0F; // +x on every KV head
    }
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        index.write_block_mean(0, layer, mean);
        std::vector<float> flipped = mean;
        for (std::uint32_t head = 0; head < index.kv_heads(); ++head) {
            flipped[head * index.head_dim()] = -1.0F;
        }
        index.write_block_mean(1, layer, flipped);
    }

    std::vector<float> query(static_cast<std::size_t>(kLayers) * kKvHeads * kHeadDim, 0.0F);
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t head = 0; head < kKvHeads; ++head) {
            query[layer * kKvHeads * kHeadDim + head * kHeadDim] = 1.0F;
        }
    }
    const std::vector<std::uint32_t> counts(kLayers, 7U);

    float aligned = std::numeric_limits<float>::quiet_NaN();
    float opposed = std::numeric_limits<float>::quiet_NaN();
    expect(index.score(0, query, counts, aligned) && aligned > 0.999F,
           "aligned query/block scores near +1");
    expect(index.score(1, query, counts, opposed) && opposed < -0.999F,
           "opposed query/block scores near -1");

    const std::vector<std::uint32_t> zero_counts(kLayers, 0U);
    float unused = 0.0F;
    expect(!index.score(0, query, zero_counts, unused),
           "a layer without query accumulation cannot score");
}

void test_selection() {
    auto index = make_index();
    (void)index.append(6 * kBlockTok); // six full blocks

    const std::vector<float> scores{0.1F, 0.9F, 0.5F, 0.7F, 0.3F, 0.8F};
    r::BlockSelectionConfig config;
    config.block_tokens  = kBlockTok;
    config.budget_blocks = 3;
    config.sink_blocks   = 1;
    config.recent_blocks = 1;

    const r::BlockSelection selection = r::select_blocks(index, scores, config);
    expect(selection.selected == std::vector<std::uint32_t>({0U, 1U, 5U}),
           "selection keeps sink, recent, and the best remaining score");
    expect(selection.scored_blocks == 4 && selection.mandatory_kept == 0,
           "non-structural blocks are ranked by score");

    const r::BlockSelection mandated =
        r::select_blocks(index, scores, config, std::array<const std::uint32_t, 1>{3U});
    expect(mandated.selected == std::vector<std::uint32_t>({0U, 3U, 5U}),
           "a mandatory block displaces the weakest kept score");
    expect(mandated.mandatory_kept == 1, "mandatory keeps are counted");

    config.budget_blocks               = 10;
    const r::BlockSelection everything = r::select_blocks(index, scores, config);
    expect(everything.selected.size() == 6, "an oversized budget selects every block");

    const std::vector<float> with_nan{
        0.1F, std::numeric_limits<float>::quiet_NaN(), 0.5F, 0.7F, 0.3F, 0.8F};
    const r::BlockSelection skipped = r::select_blocks(index, with_nan, config);
    bool one_absent                 = false;
    for (const std::uint32_t block : skipped.selected) { one_absent |= block == 1U; }
    expect(!one_absent && skipped.scored_blocks == 3, "unscored blocks never win a slot");
}

void test_window_helpers() {
    const auto decoded = r::decode_window_page_set(40, 8, std::array{8U, 9U});
    expect(decoded == std::vector<std::uint32_t>({0, 1, 8, 9, 36, 37, 38, 39}),
           "decode retains retrieved history and rolls only its recent share");
    expect(r::decode_window_page_set(3, 8, {}).size() == 3, "short contexts stay dense");
    const auto full = r::prefill_window_page_set(5, 1, 4);
    expect(full == std::vector<std::uint32_t>({0U, 1U, 2U, 3U, 4U}),
           "a covering window keeps every page");

    const auto windowed = r::prefill_window_page_set(20, 1, 4);
    expect(windowed == std::vector<std::uint32_t>({0U, 16U, 17U, 18U, 19U}),
           "a rolling window keeps the sink prefix and newest pages");

    expect(r::prefill_window_page_set(0, 1, 4).empty(), "an empty prefix has no window");

    const auto pages = r::block_pages(std::vector<std::uint32_t>{0U, 2U}, 128);
    expect(pages == std::vector<std::uint32_t>({0U, 1U, 4U, 5U}),
           "blocks expand to their 64-token pages");
}

void test_media_windows() {
    using ninfer::models::qwen3_5::VisionItem;
    using ninfer::models::qwen3_5::TokenSpan;
    VisionItem a, b, c;
    a.token_spans     = {{193, 128}}; // pages 3..5, including predecessor
    b.token_spans     = {{340, 80}};  // shares page 5; must merge
    c.token_spans     = {{641, 120}};
    const auto groups = r::media_page_groups(std::array{a, b, c});
    expect(groups.size() == 2 && groups[0].begin == 3 && groups[0].end == 7,
           "media sharing a physical page cannot be selected independently");
    r::validate_context_window(groups, 10);
    const auto selected = r::context_window_page_set(20, 8, std::array{4U}, groups);
    expect(selected == std::vector<std::uint32_t>({0, 1, 3, 4, 5, 6, 10, 11}),
           "one scored page brings its full media group, with sink and latest image");
    const auto tight = r::context_window_page_set(20, 6, std::array{4U}, groups);
    expect(tight == std::vector<std::uint32_t>({0, 1, 10, 11, 18, 19}),
           "a group that does not fit is omitted as a whole");
    const auto partial = r::context_window_page_set(5, 8, {}, groups);
    expect(partial == std::vector<std::uint32_t>({0, 1, 2, 3, 4}),
           "in-progress image maps no future pages and a fitting prefix stays dense");
    bool rejected = false;
    try {
        r::validate_context_window(groups, 7);
    } catch (const ninfer::RequestError& error) {
        rejected = error.kind() == ninfer::RequestErrorKind::MediaBudgetExceeded;
    }
    expect(rejected, "oversized media fails before execution rather than becoming partial");
    // Independent all-or-none invariant across every materialized prefix/window.
    for (std::uint32_t mapped = 0; mapped < 30; ++mapped) {
        for (std::uint32_t budget = 8; budget <= 16; ++budget) {
            const auto pages =
                r::context_window_page_set(mapped, budget, std::array{4U, 11U}, groups);
            expect(pages.size() <= budget && std::is_sorted(pages.begin(), pages.end()),
                   "media windows stay sorted and within capacity");
            for (const auto& group : groups) {
                std::size_t count = 0;
                for (const auto p : pages) { count += p >= group.begin && p < group.end; }
                const auto visible = std::min(mapped, group.end) > group.begin
                                         ? std::min(mapped, group.end) - group.begin
                                         : 0;
                expect(count == 0 || count == visible,
                       "execution never sees a partial visible media group");
            }
        }
    }
    VisionItem large;
    large.token_spans = {{4500, 900}};
    const auto query  = r::kvmem_query_span(5500, 0, TokenSpan{4000, 1450}, std::array{large});
    expect(query.begin == 4499 && query.end == 5450 && query.exact,
           "512-token query clipping backs up to the complete media consumer span");
    const auto after = r::kvmem_query_span(6500, 0, TokenSpan{6000, 450}, std::array{large});
    expect(after.begin == 6000, "an older image does not expand a later text query");
}

void test_prefill_lookahead_preserves_history() {
    // Independent membership oracle across committed frontiers and outstanding
    // growth. Looking at mapped=13 instead of committed=9 would keep only four
    // historical pages in an eight-page window (the other four are future writes).
    for (std::uint32_t committed = 0; committed < 128; ++committed) {
        for (std::uint32_t ahead = 0; ahead <= 64; ahead += 4) {
            for (std::uint32_t budget = 2; budget <= 32; budget += 5) {
                auto pages = r::prefill_window_page_set(committed, 2, budget - 2);
                r::append_prefill_growth_pages(pages, committed, committed + ahead);
                std::vector<std::uint32_t> expected;
                for (std::uint32_t p = 0; p < committed + ahead; ++p) {
                    const bool history = p < committed && (p < 2 || committed <= budget ||
                                                           p >= committed - (budget - 2));
                    if (history || p >= committed) expected.push_back(p);
                }
                expect(pages == expected, "future writes do not consume committed history slots");
                expect(pages.size() <= budget + ahead,
                       "history plus lookahead remains within the startup physical claim");
            }
        }
    }
    auto replay = r::decode_window_page_set(40, 8, std::array{8U, 9U});
    r::append_prefill_growth_pages(replay, 40, 44);
    expect(replay == std::vector<std::uint32_t>({0, 1, 8, 9, 36, 37, 38, 39, 40, 41, 42, 43}),
           "retrieved history is preserved while replay append pages stay writable");
    const std::array groups{r::MediaPageGroup{3, 7}, r::MediaPageGroup{10, 12}};
    auto media = r::context_window_page_set(11, 8, {}, groups);
    r::append_prefill_growth_pages(media, 11, 14);
    expect(media == std::vector<std::uint32_t>({0, 1, 2, 7, 8, 9, 10, 11, 12, 13}),
           "media history remains atomic while its future continuation stays mapped");
}

void test_instruction_retention() {
    using ninfer::models::qwen3_5::TokenSpan;
    // Several pages of tool definitions, plus a late developer message. An
    // independently enumerated membership oracle includes every intersecting
    // page, even partial boundary pages, and never includes future KV.
    const std::array spans{TokenSpan{70, 250}, TokenSpan{1300, 90}};
    const std::array preferred{8U, 9U, 10U, 11U, 12U, 13U};
    r::validate_context_window({}, 11, spans);
    for (std::uint32_t mapped = 0; mapped < 45; ++mapped) {
        for (std::uint32_t budget = 11; budget < 22; ++budget) {
            const auto chosen = r::context_window_page_set(mapped, budget, preferred, {}, spans);
            expect(chosen.size() <= budget && std::is_sorted(chosen.begin(), chosen.end()) &&
                       std::adjacent_find(chosen.begin(), chosen.end()) == chosen.end(),
                   "instruction window stays sorted, unique, and within capacity");
            for (std::uint32_t p = 0; p < mapped; ++p) {
                bool mandatory = p < 2 || p + 2 >= mapped;
                for (const auto& span : spans) {
                    mandatory |= p * 64 < span.begin + span.count && (p + 1) * 64 > span.begin;
                }
                if (mandatory) {
                    expect(std::binary_search(chosen.begin(), chosen.end(), p),
                           "all visible instructions and tail survive scored history");
                }
            }
            expect(chosen.empty() || chosen.back() < mapped,
                   "unmaterialized instruction pages are never loaded");
            auto with_growth = chosen;
            r::append_prefill_growth_pages(with_growth, mapped, mapped + 4);
            expect(with_growth.size() == chosen.size() + 4 &&
                       std::equal(chosen.begin(), chosen.end(), with_growth.begin()),
                   "prefill lookahead preserves instruction-bearing history");
        }
    }
    bool rejected = false;
    try { r::validate_context_window({}, 8, spans); }
    catch (const ninfer::RequestError& error) {
        rejected = error.kind() == ninfer::RequestErrorKind::ContextLengthExceeded;
    }
    expect(rejected, "instructions too large for the window fail before admission");

    // Instructions sharing a page with image KV pin the whole group. A different
    // latest image also remains atomic, without double charging overlapping pages.
    const std::array groups{r::MediaPageGroup{4, 8}, r::MediaPageGroup{15, 18}};
    const std::array instruction{TokenSpan{64, 220}}; // page 4 intersects the image
    r::validate_context_window(groups, 13, instruction);
    const auto chosen = r::context_window_page_set(25, 13, preferred, groups, instruction);
    expect(chosen == std::vector<std::uint32_t>({0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 23, 24}),
           "shared instruction/media pages are pinned once, with latest media and tail");
    rejected = false;
    try { r::validate_context_window(groups, 12, instruction); }
    catch (const ninfer::RequestError& error) {
        rejected = error.kind() == ninfer::RequestErrorKind::MediaBudgetExceeded;
    }
    expect(rejected, "instruction plus latest-media capacity is checked before execution");
    expect(r::context_window_page_set(30, 8, preferred, {}) ==
               r::decode_window_page_set(30, 8, preferred),
           "raw text without semantic instructions retains its existing policy");
}

} // namespace

void test_checkpoint_copy_on_write() {
    r::RetrievalIndex live(128, 1, 1, 2);
    (void)live.append(129);
    live.write_block_mean(0, 0, std::array{1.0F, 0.0F});
    auto saved = live;
    live.write_block_mean(0, 0, std::array{-1.0F, 0.0F});
    float saved_score = 0.0F, live_score = 0.0F;
    const std::array query{1.0F, 0.0F};
    const std::array counts{1U};
    expect(saved.score(0, query, counts, saved_score) && saved_score > 0.99F &&
               live.score(0, query, counts, live_score) && live_score < -0.99F,
           "updating one lane does not mutate shared checkpoint block features");
    live.truncate_to(64);
    expect(saved.total_tokens() == 129 && saved.block(0).full && !live.block(0).full,
           "truncate does not shrink a retained checkpoint");
    live = saved;
    (void)live.append(127);
    live.write_block_mean(1, 0, query);
    expect(saved.block_count() == 2 && !saved.block(1).full && live.block(1).full,
           "restored partial retrieval block completes independently");
}

void test_shared_metadata_budget() {
    ninfer::HostContextArena arena(4096, 256, ninfer::HostContextMemory::Pageable);
    {
        ninfer::models::qwen3_5::detail::RetrievalIndex live(128, 1, 1, 2);
        live.bind_metadata_budget(ninfer::models::qwen3_5::detail::retrieval_metadata_budget(&arena));
        (void)live.append(128);
        live.write_block_mean(0, 0, std::array{1.0F, 0.0F});
        auto saved = live;
        expect(arena.metadata_bytes() == 256,
               "checkpoint aliases charge one completed MeanK block");
        live.write_block_mean(0, 0, std::array{-1.0F, 0.0F});
        expect(arena.metadata_bytes() == 512,
               "COW detachment reserves an independent physical charge");
        saved.truncate_to(0);
        expect(arena.metadata_bytes() == 256, "the last checkpoint alias returns its MeanK charge");
    }
    expect(arena.metadata_bytes() == 0 && arena.occupied_bytes() == 0,
           "retiring all sparse features returns both Host quotas");
}

int main() {
    try {
        test_append_and_truncate();
        test_scoring();
        test_selection();
        test_window_helpers();
        test_media_windows();
        test_prefill_lookahead_preserves_history();
        test_instruction_retention();
        test_checkpoint_copy_on_write();
        test_shared_metadata_budget();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
