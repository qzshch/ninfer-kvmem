#pragma once

#include "core/arena.h"

#include <cstddef>
#include <optional>
#include <memory>
#include <utility>
#include <vector>

namespace ninfer {

class HostContextArena;
struct HostResidentStorage;
class HostResidentCharge;

// A unique, immovable physical extent. Keeping this owner alive pins its bytes; views never
// allocate or charge a second copy. Transfer destinations remain reserved until publication.
class HostContextAllocation {
public:
    HostContextAllocation() noexcept = default;
    ~HostContextAllocation();
    HostContextAllocation(const HostContextAllocation&)            = delete;
    HostContextAllocation& operator=(const HostContextAllocation&) = delete;
    HostContextAllocation(HostContextAllocation&& other) noexcept;
    HostContextAllocation& operator=(HostContextAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

    [[nodiscard]] bool reserved() const noexcept { return valid() && reserved_; }

    [[nodiscard]] std::byte* data() const noexcept;
    void publish() noexcept;
    bool release() noexcept;

private:
    friend class HostContextArena;

    HostContextAllocation(HostContextArena& owner, std::size_t offset, std::size_t bytes,
                          bool reserved = true) noexcept
        : owner_(&owner), offset_(offset), bytes_(bytes), reserved_(reserved) {}

    void disarm() noexcept;

    HostContextArena* owner_ = nullptr;
    std::size_t offset_      = 0;
    std::size_t bytes_       = 0;
    bool reserved_           = false;
    std::shared_ptr<HostResidentStorage> resident_;
    std::size_t resident_offset_ = 0;
};

// One extent ledger shared by all Host context representations. The default uses
// startup-fixed pinned backing. HiCache uses bounded, allocation-owned pinned State
// buffers and nonresident KV extents; cold bytes never masquerade as pinned capacity.
// minimum_allocation_bytes is the smallest supported complete geometry, not a quota split.
// It bounds metadata at startup; all allocations and split pieces must meet that minimum.
// The arena must outlive its allocations and the typed pools that own them.
class HostContextArena {
public:
    static constexpr std::size_t alignment = 256;

    HostContextArena(std::size_t capacity_bytes, std::size_t minimum_allocation_bytes,
                     std::optional<std::size_t> resident_capacity_bytes = std::nullopt);
    HostContextArena(const HostContextArena&)            = delete;
    HostContextArena& operator=(const HostContextArena&) = delete;
    HostContextArena(HostContextArena&&)                 = delete;
    HostContextArena& operator=(HostContextArena&&)      = delete;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    [[nodiscard]] std::size_t minimum_allocation_bytes() const noexcept { return minimum_bytes_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept {
        return occupied_bytes_ + metadata_bytes_;
    }

    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    [[nodiscard]] std::size_t live_bytes() const noexcept {
        return occupied_bytes_ + metadata_bytes_ - reserved_bytes_;
    }

    [[nodiscard]] std::size_t free_bytes() const noexcept {
        return capacity_bytes_ - occupied_bytes_ - metadata_bytes_;
    }

    [[nodiscard]] std::size_t peak_occupied_bytes() const noexcept { return peak_occupied_bytes_; }

    [[nodiscard]] std::size_t allocation_count() const noexcept { return allocation_count_; }

    [[nodiscard]] bool can_allocate(std::size_t bytes) const noexcept;
    [[nodiscard]] bool can_allocate_cold(std::size_t bytes) const noexcept;
    [[nodiscard]] std::optional<HostContextAllocation> allocate_cold(std::size_t bytes) noexcept;

    [[nodiscard]] std::size_t resident_bytes() const noexcept {
        return (lazy_resident_ ? resident_bytes_ : capacity_bytes_) + metadata_bytes_;
    }

    [[nodiscard]] std::size_t metadata_bytes() const noexcept { return metadata_bytes_; }

    [[nodiscard]] std::shared_ptr<HostResidentCharge> charge_metadata(std::size_t bytes) noexcept;

    [[nodiscard]] std::size_t resident_free_bytes() const noexcept {
        return lazy_resident_ ? resident_capacity_bytes_ - resident_bytes_ - metadata_bytes_
                              : free_bytes();
    }
    [[nodiscard]] std::optional<HostContextAllocation> allocate(std::size_t bytes) noexcept;
    [[nodiscard]] std::pair<HostContextAllocation, HostContextAllocation>
    split(HostContextAllocation&& allocation, std::size_t byte_offset);

private:
    friend class HostContextAllocation;
    friend struct HostResidentStorage;
    friend class HostResidentCharge;

    struct FreeExtent {
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    [[nodiscard]] std::size_t allocation_bytes(std::size_t bytes) const noexcept;
    [[nodiscard]] std::optional<std::size_t> find_free_extent(std::size_t bytes) const noexcept;
    void insert_free_extent(FreeExtent extent) noexcept;
    void release(std::size_t offset, std::size_t bytes, bool reserved) noexcept;

    [[nodiscard]] std::optional<HostContextAllocation> allocate_impl(std::size_t bytes,
                                                                     bool resident) noexcept;
    bool lazy_resident_                  = false;
    std::size_t resident_capacity_bytes_ = 0;
    std::size_t resident_bytes_          = 0;
    std::size_t metadata_bytes_          = 0;
    std::optional<PinnedHostBuffer> backing_;
    std::size_t capacity_bytes_      = 0;
    std::size_t minimum_bytes_       = 0;
    std::size_t occupied_bytes_      = 0;
    std::size_t reserved_bytes_      = 0;
    std::size_t peak_occupied_bytes_ = 0;
    std::size_t allocation_count_    = 0;
    std::vector<FreeExtent> free_extents_;
};

// A CPU representation's unique physical charge. Its owner (e.g. a shared MeanK block)
// owns the actual bytes. Aliases share this lease; no dummy pinned buffer is allocated.
class HostResidentCharge {
public:
    ~HostResidentCharge();
    HostResidentCharge(const HostResidentCharge&)            = delete;
    HostResidentCharge& operator=(const HostResidentCharge&) = delete;
private:
    friend class HostContextArena;

    HostResidentCharge(HostContextArena& owner, std::size_t bytes) noexcept
        : owner_(owner), bytes_(bytes) {}

    HostContextArena& owner_;
    std::size_t bytes_;
    bool charged_ = false;
};

} // namespace ninfer
