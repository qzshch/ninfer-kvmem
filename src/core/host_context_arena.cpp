#include "core/host_context_arena.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer {

HostContextAllocation::~HostContextAllocation() { (void)release(); }

HostContextAllocation::HostContextAllocation(HostContextAllocation&& other) noexcept
    : owner_(other.owner_), offset_(other.offset_), bytes_(other.bytes_),
      reserved_(other.reserved_) {
    other.disarm();
}

HostContextAllocation& HostContextAllocation::operator=(HostContextAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    owner_    = other.owner_;
    offset_   = other.offset_;
    bytes_    = other.bytes_;
    reserved_ = other.reserved_;
    other.disarm();
    return *this;
}

std::byte* HostContextAllocation::data() const noexcept {
    return valid() ? static_cast<std::byte*>(owner_->backing_->data()) + offset_ : nullptr;
}

void HostContextAllocation::publish() noexcept {
    if (!reserved()) { return; }
    owner_->reserved_bytes_ -= bytes_;
    reserved_ = false;
}

bool HostContextAllocation::release() noexcept {
    if (!valid()) { return false; }
    owner_->release(offset_, bytes_, reserved_);
    disarm();
    return true;
}

void HostContextAllocation::disarm() noexcept {
    owner_    = nullptr;
    offset_   = 0;
    bytes_    = 0;
    reserved_ = false;
}

HostContextArena::HostContextArena(std::size_t capacity_bytes, std::size_t minimum_allocation_bytes)
    : capacity_bytes_(capacity_bytes) {
    if (minimum_allocation_bytes == 0 ||
        minimum_allocation_bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::invalid_argument("Host context minimum allocation is invalid");
    }
    minimum_bytes_ = (minimum_allocation_bytes + alignment - 1) & ~(alignment - 1);
    if (capacity_bytes_ == 0) { return; }
    // At most N allocated extents separate N+1 free runs; releases and splits never allocate
    // metadata. The unused trailing bytes remain part of the charged physical backing.
    free_extents_.reserve(capacity_bytes_ / minimum_bytes_ + 1);
    backing_.emplace(capacity_bytes_);
    free_extents_.push_back({0, capacity_bytes_});
}

HostResidentCharge::~HostResidentCharge() {
    if (charged_) { owner_.metadata_bytes_ -= bytes_; }
}

std::shared_ptr<HostResidentCharge> HostContextArena::charge_metadata(std::size_t bytes) noexcept {
    if (!bytes || bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) { return {}; }
    const auto rounded = (bytes + alignment - 1) & ~(alignment - 1);
    if (rounded > free_bytes()) { return {}; }
    try {
        auto charge = std::shared_ptr<HostResidentCharge>(new HostResidentCharge(*this, rounded));
        metadata_bytes_ += rounded;
        charge->charged_     = true;
        peak_occupied_bytes_ = std::max(peak_occupied_bytes_, occupied_bytes());
        return charge;
    } catch (...) { return {}; }
}

std::size_t HostContextArena::allocation_bytes(std::size_t bytes) const noexcept {
    if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        return 0;
    }
    const std::size_t rounded = (bytes + alignment - 1) & ~(alignment - 1);
    return rounded < minimum_bytes_ ? 0 : rounded;
}

std::optional<std::size_t> HostContextArena::find_free_extent(std::size_t bytes) const noexcept {
    for (std::size_t index = 0; index < free_extents_.size(); ++index) {
        if (free_extents_[index].bytes >= bytes) { return index; }
    }
    return std::nullopt;
}

bool HostContextArena::can_allocate(std::size_t bytes) const noexcept {
    const std::size_t rounded = allocation_bytes(bytes);
    return rounded != 0 && rounded <= free_bytes() && find_free_extent(rounded).has_value();
}

std::size_t HostContextArena::page_allocation_shortage(
    std::span<const HostPageDemand> demands, std::size_t metadata_bytes) const {
    std::size_t total = 0;
    for (const auto& demand : demands) {
        if (!demand.pages) { continue; }
        const auto stride = allocation_bytes(demand.page_bytes);
        if (!stride || demand.pages > (std::numeric_limits<std::size_t>::max() - total) / stride) {
            return std::numeric_limits<std::size_t>::max();
        }
        total += stride * demand.pages;
    }
    if (metadata_bytes > std::numeric_limits<std::size_t>::max() - total) {
        return std::numeric_limits<std::size_t>::max();
    }
    if (total + metadata_bytes > free_bytes()) { return total + metadata_bytes - free_bytes(); }
    if (!total || find_free_extent(total)) { return 0; }

    // Match prepare_prefix: take the largest whole-page run that fits, then
    // allocate from the first extent that can hold it. Neither metadata charges
    // nor quote scratch alter the physical extent ledger.
    auto available = free_extents_;
    std::size_t remaining_bytes = total;
    for (const auto& demand : demands) {
        auto remaining = demand.pages;
        if (!remaining) { continue; }
        const auto stride = allocation_bytes(demand.page_bytes);
        while (remaining) {
            std::size_t run = 0;
            for (const auto& extent : available) {
                run = std::max(run, std::min(remaining, extent.bytes / stride));
            }
            if (!run) { return remaining_bytes; }
            const auto bytes = run * stride;
            auto extent = std::find_if(available.begin(), available.end(),
                                      [bytes](const auto& e) { return e.bytes >= bytes; });
            extent->bytes -= bytes;
            remaining -= run;
            remaining_bytes -= bytes;
        }
    }
    return 0;
}

std::optional<HostContextAllocation> HostContextArena::allocate(std::size_t bytes) noexcept {
    const std::size_t rounded = allocation_bytes(bytes);
    if (rounded == 0 || rounded > free_bytes()) { return std::nullopt; }
    const auto free_index = find_free_extent(rounded);
    if (!free_index) { return std::nullopt; }
    FreeExtent& free         = free_extents_[*free_index];
    const std::size_t offset = free.offset;
    free.offset += rounded;
    free.bytes -= rounded;
    if (free.bytes == 0) {
        free_extents_.erase(free_extents_.begin() + static_cast<std::ptrdiff_t>(*free_index));
    }
    occupied_bytes_ += rounded;
    reserved_bytes_ += rounded;
    peak_occupied_bytes_ = std::max(peak_occupied_bytes_, occupied_bytes());
    ++allocation_count_;
    return HostContextAllocation(*this, offset, rounded);
}

std::pair<HostContextAllocation, HostContextAllocation>
HostContextArena::split(HostContextAllocation&& allocation, std::size_t byte_offset) {
    if (allocation.owner_ != this) {
        throw std::invalid_argument("Cannot split a foreign or released Host context allocation");
    }
    if (byte_offset % alignment != 0 || byte_offset < minimum_bytes_ ||
        byte_offset > allocation.bytes_ || allocation.bytes_ - byte_offset < minimum_bytes_) {
        throw std::out_of_range("Host context split must retain two aligned complete geometries");
    }
    HostContextAllocation left(*this, allocation.offset_, byte_offset, allocation.reserved_);
    HostContextAllocation right(*this, allocation.offset_ + byte_offset,
                                allocation.bytes_ - byte_offset, allocation.reserved_);
    allocation.disarm();
    ++allocation_count_;
    return {std::move(left), std::move(right)};
}

void HostContextArena::release(std::size_t offset, std::size_t bytes, bool reserved) noexcept {
    occupied_bytes_ -= bytes;
    if (reserved) { reserved_bytes_ -= bytes; }
    --allocation_count_;
    insert_free_extent({offset, bytes});
}

void HostContextArena::insert_free_extent(FreeExtent extent) noexcept {
    const auto position = std::lower_bound(
        free_extents_.begin(), free_extents_.end(), extent.offset,
        [](const FreeExtent& candidate, std::size_t offset) { return candidate.offset < offset; });
    auto inserted = free_extents_.insert(position, extent);
    if (inserted != free_extents_.begin()) {
        auto previous = inserted - 1;
        if (previous->offset + previous->bytes == inserted->offset) {
            previous->bytes += inserted->bytes;
            free_extents_.erase(inserted);
            inserted = previous;
        }
    }
    const auto next = inserted + 1;
    if (next != free_extents_.end() && inserted->offset + inserted->bytes == next->offset) {
        inserted->bytes += next->bytes;
        free_extents_.erase(next);
    }
}

} // namespace ninfer
