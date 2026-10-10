#include "kvmem/block_retrieval.h"
#include "kvmem/context_window.h"
#include "kvmem/host_budget.h"
#include "kvmem/query_span.h"
#include "kvmem/window_capacity.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Budget {
    std::size_t used = 0;
    std::size_t limit = 256;
    static std::shared_ptr<void> claim(void* owner, std::size_t bytes) {
        auto& b = *static_cast<Budget*>(owner);
        if (bytes > b.limit - b.used) return {};
        b.used += bytes;
        return {new std::size_t(bytes), [&b](void* p) {
            b.used -= *static_cast<std::size_t*>(p);
            delete static_cast<std::size_t*>(p);
        }};
    }
};

int main() {
    try {
        Budget budget;
        {
            kvmem::RetrievalIndex lane(128, 1, 1, 2);
            lane.bind_metadata_budget({&budget, Budget::claim});
            (void)lane.append(128);
            lane.write_block_mean(0, 0, std::array{1.0f, 0.0f});
            auto saved = lane;
            const auto charged = budget.used;
            float score = 0;
            require(saved.score(0, std::array{1.0f, 0.0f}, std::array{1U}, score) &&
                        std::abs(score - 1.0f) < 1e-6f, "independent cosine oracle");
            budget.limit = charged;
            bool refused = false;
            try { lane.write_block_mean(0, 0, std::array{-1.0f, 0.0f}); }
            catch (const std::bad_alloc&) { refused = true; }
            require(refused && budget.used == charged, "refusal must preserve source charge");
            require(lane.score(0, std::array{1.0f, 0.0f}, std::array{1U}, score) && score > .99f,
                    "refused COW must preserve source values");
            budget.limit = 256;
            lane.write_block_mean(0, 0, std::array{-1.0f, 0.0f});
            require(budget.used == charged * 2, "each physical COW block charged once");
            saved.truncate_to(0);
            require(budget.used == charged, "last checkpoint releases its own charge");
        }
        require(budget.used == 0, "lane retirement must return metadata quota");
        std::vector<std::uint32_t> growth{0, 1, 137};
        kvmem::append_prefill_growth_pages(growth, 138, 140);
        require(growth == std::vector<std::uint32_t>{0, 1, 137, 138, 139},
                "uncommitted lookahead must stay resident through unit settlement");
        growth = {0, 1, 139};
        kvmem::append_prefill_growth_pages(growth, 138, 140);
        kvmem::append_prefill_growth_pages(growth, 138, 140);
        require(growth == std::vector<std::uint32_t>{0, 1, 138, 139},
                "growth preservation must merge a selected tail without duplicate pages");
        require(kvmem::kvmem_pool_page_budget(262144, 1024, 576, 2) == 1216,
                "two windows, chunk growth and safety slack must fit the device pool");
        const std::array instructions{kvmem::TokenRange{0, 400}, kvmem::TokenRange{12000, 80}};
        const std::array media{kvmem::MediaPageGroup{20, 25}};
        kvmem::validate_context_window(media, 32, std::span<const kvmem::TokenRange>(instructions));
        const auto pages = kvmem::context_window_page_set(200, 32, {}, media,
                              std::span<const kvmem::TokenRange>(instructions));
        require(pages.size() == 32 && std::is_sorted(pages.begin(), pages.end()), "bounded ordered set");
        for (auto p : {0U, 1U, 6U, 20U, 24U, 187U, 188U, 198U, 199U}) {
            require(std::binary_search(pages.begin(), pages.end(), p), "mandatory semantic coverage");
        }
        bool refused = false;
        try { kvmem::validate_context_window(media, 8, std::span<const kvmem::TokenRange>(instructions)); }
        catch (const kvmem::WindowCapacityError& e) { refused = e.instructions; }
        require(refused, "too-small window cannot silently truncate instructions");
        require(kvmem::sparse_host_budget_occupancy(100, 40, std::array<std::size_t, 2>{80, 120}) == 260,
                "shared aliases credited once before complete future peaks");
        require(!kvmem::sparse_host_budget_occupancy(1, 2, {}), "inconsistent accounting rejected");
        require(!kvmem::sparse_host_budget_occupancy(1, 0,
                    std::array{std::numeric_limits<std::size_t>::max()}), "overflow rejected");
        const auto query = kvmem::kvmem_query_span(8000, 4000, kvmem::TokenRange{7000, 900});
        require(query.begin == 7388 && query.end == 7900 && query.exact, "bounded typed query");
        std::cout << "portable retrieval, retention, quotas and retirement: ok\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
