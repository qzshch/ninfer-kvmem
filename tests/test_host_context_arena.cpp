#include "core/host_context_arena.h"

#include <cuda_runtime_api.h>
#include <array>

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void test_reserve_publish_and_rollback() {
    ninfer::HostContextArena arena(1025, 1);
    expect(arena.capacity_bytes() == 1025 && arena.occupied_bytes() == 0,
           "idle fixed backing remains fully charged");
    auto source = arena.allocate(257);
    expect(source && source->bytes() == 512 && arena.reserved_bytes() == 512,
           "allocation includes alignment padding immediately");
    std::memset(source->data(), 0x51, source->bytes());
    source->publish();
    source->publish();
    expect(arena.live_bytes() == 512 && arena.reserved_bytes() == 0,
           "publication charges one unique live allocation");
    {
        auto target = arena.allocate(257);
        expect(target && target->data() != source->data() && arena.occupied_bytes() == 1024 &&
                   arena.live_bytes() == 512 && arena.reserved_bytes() == 512,
               "whole destination is reserved while the source remains resident");
        expect(!arena.allocate(1) && !arena.can_allocate(1),
               "unused trailing byte cannot satisfy aligned allocation");
    }
    expect(arena.occupied_bytes() == 512 && arena.reserved_bytes() == 0 &&
               arena.peak_occupied_bytes() == 1024,
           "cancelled destination rolls back occupancy without losing peak");
    expect(source->data()[0] == std::byte{0x51} && source->data()[511] == std::byte{0x51},
           "failed destination preserves the pinned source");
    auto moved = std::move(*source);
    expect(!source->valid() && arena.allocation_count() == 1,
           "moving ownership neither duplicates nor releases bytes");
    expect(moved.release() && !moved.release() && arena.occupied_bytes() == 0 &&
               arena.capacity_bytes() == 1025,
           "last owner release frees reusable bytes without hiding fixed backing");
    expect(!arena.allocate(0) && !arena.allocate(std::numeric_limits<std::size_t>::max()),
           "zero and overflowing requests fail without changing accounting");
}

void test_fragmentation_and_split() {
    ninfer::HostContextArena arena(2048, 256);
    auto full = arena.allocate(2048);
    full->publish();
    std::vector<ninfer::HostContextAllocation> pieces;
    pieces.reserve(8);
    while (full->bytes() > 256) {
        auto split = arena.split(std::move(*full), 256);
        pieces.push_back(std::move(split.first));
        *full = std::move(split.second);
    }
    pieces.push_back(std::move(*full));
    expect(arena.allocation_count() == 8 && arena.live_bytes() == 2048,
           "split reaches the smallest geometry without changing charged bytes");
    for (std::size_t index = 0; index < pieces.size(); index += 2) {
        (void)pieces[index].release();
    }
    expect(arena.free_bytes() == 1024 && !arena.can_allocate(512) && !arena.allocate(512),
           "actual contiguous extents, not total free bytes, govern availability");
    const std::array<ninfer::HostPageDemand, 2> mixed = {{{512, 1}, {256, 2}}};
    expect(arena.page_allocation_shortage(mixed) == 1024 &&
               arena.occupied_bytes() == 1024 && arena.reserved_bytes() == 0,
           "complete-page quote rejects fragmented capacity without changing source ownership");
    const std::array<ninfer::HostPageDemand, 1> small = {{{256, 4}}};
    expect(arena.page_allocation_shortage(small) == 0,
           "separate complete-page runs are admissible without one contiguous large allocation");
    (void)pieces[1].release();
    expect(arena.page_allocation_shortage(mixed) == 0 &&
               arena.page_allocation_shortage(mixed, 512) == 256,
           "coalescing and external metadata both affect the complete execution quote");
    auto merged = arena.allocate(768);
    expect(merged.has_value(), "adjacent released split pieces coalesce for another geometry");
    bool invalid_split_rejected = false;
    try {
        (void)arena.split(std::move(*merged), 1);
    } catch (const std::out_of_range&) { invalid_split_rejected = true; }
    expect(invalid_split_rejected && merged->valid() && merged->bytes() == 768,
           "invalid split leaves its source allocation owned");
    pieces.clear();
    merged.reset();
    expect(arena.occupied_bytes() == 0 && arena.allocation_count() == 0 && arena.can_allocate(2048),
           "all split pieces return to one complete reusable extent");
    ninfer::HostContextArena disabled(0, 256);
    expect(!disabled.allocate(256) && disabled.capacity_bytes() == 0,
           "Host zero has no backing or allocations");
}

void test_tiered_resident_budget() {
    ninfer::HostContextArena arena(4096, 256, 1024);
    auto cold = arena.allocate_cold(2048);
    expect(cold && !cold->data() && arena.occupied_bytes() == 2048 && arena.resident_bytes() == 0,
           "cold KV charges the common extent ledger without pins");
    auto state = arena.allocate(1024);
    expect(state && state->data() && !arena.can_allocate(256) && arena.can_allocate_cold(256),
           "cold capacity never satisfies resident State capacity");
    std::memset(state->data(), 0x39, 1024);
    auto pieces = arena.split(std::move(*state), 256);
    (void)pieces.first.release();
    expect(arena.resident_bytes() == 1024 && !arena.allocate(256) &&
               pieces.second.data()[767] == std::byte{0x39},
           "a split reader retains the complete physical backing charge");
    (void)pieces.second.release();
    expect(arena.resident_bytes() == 0 && arena.allocate(1024).has_value(),
           "last resident reader returns physical capacity independently of cold KV");
    cold.reset();
    expect(arena.occupied_bytes() == 0 && arena.allocation_count() == 0,
           "tiered cancellation releases both extent and physical ledgers");
}

void test_cold_file_address_growth() {
    ninfer::HostContextArena arena(4096, 256, 2048);
    auto state = arena.allocate(1024);
    auto cold = arena.allocate_cold(1024);
    auto other_state = arena.allocate(512);
    auto other_cold = arena.allocate_cold(512);
    expect(state && cold && other_state && other_cold &&
               cold->offset() == 0 && other_cold->offset() == cold->bytes() &&
               other_state->offset() >= other_cold->offset() + other_cold->bytes(),
           "resident State must not create unwritten holes before growing cold KV payload");
    state.reset(); other_state.reset(); cold.reset(); other_cold.reset();
    expect(arena.occupied_bytes() == 0 && arena.resident_bytes() == 0 &&
               arena.can_allocate_cold(4096),
           "opposite-end allocations must coalesce under the same complete quota");
}

void test_metadata_charge() {
    ninfer::HostContextArena arena(2048, 256, 1024);
    auto metadata = arena.charge_metadata(257);
    expect(metadata && arena.metadata_bytes() == 512 && arena.occupied_bytes() == 512 &&
               arena.resident_bytes() == 512,
           "metadata charges common and resident quotas once");
    auto alias = metadata;
    metadata.reset();
    expect(arena.metadata_bytes() == 512 && !arena.charge_metadata(513),
           "aliases retain their unique charge and cannot exceed the resident quota");
    auto state = arena.allocate(512);
    auto cold  = arena.allocate_cold(1024);
    expect(state && cold && !arena.allocate_cold(256) && !arena.charge_metadata(1),
           "cold extents and metadata share the common Host admission quota");
    alias.reset();
    expect(arena.metadata_bytes() == 0 && static_cast<bool>(arena.charge_metadata(1)),
           "last alias returns its charge and temporary leases return theirs");
    expect(arena.resident_bytes() == 512, "temporary metadata leases return their charge");
    state.reset();
    cold.reset();
    expect(arena.occupied_bytes() == 0 && arena.resident_bytes() == 0 &&
               !arena.charge_metadata(std::numeric_limits<std::size_t>::max()),
           "metadata overflow and teardown leave reusable ledgers");
}
} // namespace

int main() {
    int count               = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
        (error == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        test_reserve_publish_and_rollback();
        test_fragmentation_and_split();
        test_tiered_resident_budget();
        test_cold_file_address_growth();
        test_metadata_charge();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
