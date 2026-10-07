#pragma once

#include "core/host_kv_arena.h"
#include "models/qwen3_5/program/storage/logical_kv_store.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "models/qwen3_5/program/storage/kv_address_space.h"
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class HostKVExtentStore;

struct HostKVPageReplicaRelease {
    LogicalKVPageStore* pages = nullptr;
    LogicalKVPageHandle page;
};

class HostKVExtentReservation {
public:
    HostKVExtentReservation() noexcept = default;
    ~HostKVExtentReservation();

    HostKVExtentReservation(HostKVExtentReservation&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), descriptor_(other.descriptor_),
          generation_(other.generation_), page_store_(other.page_store_) {}

    HostKVExtentReservation& operator=(HostKVExtentReservation&&)      = delete;
    HostKVExtentReservation(const HostKVExtentReservation&)            = delete;
    HostKVExtentReservation& operator=(const HostKVExtentReservation&) = delete;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

private:
    HostKVExtentStore* owner_       = nullptr;
    std::uint32_t descriptor_       = 0;
    std::uint32_t generation_       = 0;
    LogicalKVPageStore* page_store_ = nullptr;

    friend class HostKVExtentStore;
};

// Owns typed Host extents and the ordered logical pages represented by each allocation. It has no
// checkpoint, retention, or scheduling policy.
class HostKVExtentStore {
public:
    HostKVExtentStore(HostKVArena& arena, std::uint32_t descriptor_capacity)
        : arena_(&arena), extents_(descriptor_capacity), free_(descriptor_capacity),
          free_count_(descriptor_capacity), memberships_(descriptor_capacity),
          free_memberships_(descriptor_capacity), free_membership_count_(descriptor_capacity),
          release_marks_(descriptor_capacity), extent_marks_(descriptor_capacity) {
        if (descriptor_capacity == 0) {
            throw std::invalid_argument("Host KV extent descriptor capacity is zero");
        }
        for (std::uint32_t index = 0; index < descriptor_capacity; ++index) {
            free_[index]             = descriptor_capacity - 1U - index;
            free_memberships_[index] = descriptor_capacity - 1U - index;
        }
        affected_extents_.reserve(descriptor_capacity);
        extent_scan_scratch_.reserve(descriptor_capacity);
        partition_runs_.reserve(descriptor_capacity);
    }

    HostKVExtentStore(const HostKVExtentStore&)            = delete;
    HostKVExtentStore& operator=(const HostKVExtentStore&) = delete;
    HostKVExtentStore(HostKVExtentStore&&)                 = delete;
    HostKVExtentStore& operator=(HostKVExtentStore&&)      = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(extents_.size());
    }

    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }

    [[nodiscard]] std::size_t free_bytes() const noexcept { return arena_->free_bytes(); }

    [[nodiscard]] std::uint32_t max_allocatable_pages(const LogicalKVPageStore& pages,
                                                    std::uint32_t limit) const {
        return arena_->max_allocatable_pages(page_layout(pages), limit);
    }

    [[nodiscard]] const HostKVPageLayout& page_layout(const LogicalKVPageStore& pages) const {
        const HostKVPageLayout* layout = arena_->layout_for(pages.physical_pool().geometry());
        if (layout == nullptr) {
            throw std::invalid_argument("Host KV page store has no arena layout");
        }
        return *layout;
    }

    [[nodiscard]] std::optional<HostKVExtentReservation>
    prepare(LogicalKVPageStore& pages, std::span<const LogicalKVPageHandle> membership,
            bool pin_active_writers = false) {
        if (membership.empty() || free_count_ == 0 || membership.size() > free_membership_count_) {
            return std::nullopt;
        }
        for (const LogicalKVPageHandle page : membership) {
            if ((!pages.can_pin_source(page) &&
                 !(pin_active_writers && pages.can_pin_active_source(page))) ||
                pages.host_resident(page)) {
                return std::nullopt;
            }
        }

        const HostKVPageLayout& layout = page_layout(pages);
        std::optional<HostKVAllocation> allocation =
            arena_->allocate(layout, static_cast<std::uint32_t>(membership.size()));
        if (!allocation) { return std::nullopt; }

        const std::uint32_t descriptor = free_[--free_count_];
        Extent& extent                 = extents_[descriptor];
        if (extent.state != ExtentState::Free) { std::terminate(); }
        extent.state      = ExtentState::Reserved;
        extent.page_store = &pages;
        extent.allocation = std::move(allocation);
        for (const LogicalKVPageHandle page : membership) {
            const std::uint32_t node = take_membership();
            if (node == kInvalidIndex) { std::terminate(); }
            Membership& entry = memberships_[node];
            entry.page        = page;
            entry.epoch       = pages.content_epoch(page);
            entry.coverage    = pages.committed_columns(page);
            entry.extent      = descriptor;
            entry.offset      = extent.page_count;
            entry.next        = kInvalidIndex;
            if (extent.tail == kInvalidIndex) {
                extent.head = node;
            } else {
                memberships_[extent.tail].next = node;
            }
            extent.tail = node;
            ++extent.page_count;
        }
        for (const LogicalKVPageHandle page : membership) { pages.pin_source(page); }

        HostKVExtentReservation reservation;
        reservation.owner_      = this;
        reservation.descriptor_ = descriptor;
        reservation.generation_ = extent.generation;
        reservation.page_store_ = &pages;
        return reservation;
    }

    [[nodiscard]] std::optional<HostKVExtentReservation>
    prepare_prefix(LogicalKVPageStore& pages, std::span<const LogicalKVPageHandle> membership,
                   bool pin_active_writers = false) {
        const auto limit = static_cast<std::uint32_t>(
            std::min<std::size_t>(membership.size(), free_membership_count_));
        if (free_count_ == 0 || limit == 0) { return std::nullopt; }
        const auto count = arena_->max_allocatable_pages(page_layout(pages), limit);
        return count ? prepare(pages, membership.first(count), pin_active_writers) : std::nullopt;
    }

    void check_io_errors() const { arena_->check_io_errors(); }

    [[nodiscard]] HostKVAllocationView writable_view(HostKVExtentReservation& reservation) {
        validate(reservation);
        return arena_->writable_view(*extents_[reservation.descriptor_].allocation);
    }

    [[nodiscard]] std::vector<DeviceKVPageHandle>
    device_sources(const HostKVExtentReservation& reservation) const {
        validate(reservation);
        std::vector<DeviceKVPageHandle> out;
        out.resize(extents_[reservation.descriptor_].page_count);
        device_sources(reservation, out);
        return out;
    }

    void device_sources(const HostKVExtentReservation& reservation,
                        std::span<DeviceKVPageHandle> out) const {
        validate(reservation);
        const Extent& extent = extents_[reservation.descriptor_];
        if (out.size() != extent.page_count) {
            throw std::invalid_argument("Host KV device-source output has the wrong size");
        }
        std::uint32_t node = extent.head;
        for (std::size_t index = 0; index < out.size(); ++index) {
            if (node == kInvalidIndex) { std::terminate(); }
            out[index] = extent.page_store->physical(memberships_[node].page);
            node       = memberships_[node].next;
        }
        if (node != kInvalidIndex) { std::terminate(); }
    }

    [[nodiscard]] std::uint32_t page_count(const HostKVExtentReservation& reservation) const {
        validate(reservation);
        return extents_[reservation.descriptor_].page_count;
    }

    [[nodiscard]] HostKVExtentCapability publish(HostKVExtentReservation&& reservation) noexcept {
        if (!valid(reservation)) { std::terminate(); }
        Extent& extent     = extents_[reservation.descriptor_];
        std::uint32_t node = extent.head;
        for (std::uint32_t index = 0; index < extent.page_count; ++index) {
            if (node == kInvalidIndex) { std::terminate(); }
            const Membership& entry = memberships_[node];
            if (!extent.page_store->can_attach_host_replica(entry.page, entry.epoch,
                                                            entry.coverage)) {
                std::terminate();
            }
            node = entry.next;
        }
        if (node != kInvalidIndex) { std::terminate(); }
        extent.allocation->publish();
        extent.state = ExtentState::Published;
        const HostKVExtentCapability capability(this, reservation.descriptor_, extent.generation);
        node = extent.head;
        for (std::uint32_t index = 0; index < extent.page_count; ++index) {
            Membership& entry = memberships_[node];
            try {
                extent.page_store->attach_host_replica(
                    entry.page, HostKVPageReplica{.extent            = capability,
                                                  .page_offset       = index,
                                                  .membership_node   = node,
                                                  .content_epoch     = entry.epoch,
                                                  .committed_columns = entry.coverage});
            } catch (...) { std::terminate(); }
            extent.page_store->unpin_source(entry.page);
            node = entry.next;
        }
        consume(reservation);
        return capability;
    }

    void abort(HostKVExtentReservation& reservation) noexcept {
        if (!valid(reservation)) {
            consume(reservation);
            return;
        }
        Extent& extent     = extents_[reservation.descriptor_];
        std::uint32_t node = extent.head;
        for (std::uint32_t index = 0; index < extent.page_count; ++index) {
            if (node == kInvalidIndex) { std::terminate(); }
            const LogicalKVPageHandle page = memberships_[node].page;
            if (!extent.page_store->valid(page) || extent.page_store->source_pins(page) == 0) {
                std::terminate();
            }
            try {
                extent.page_store->unpin_source(page);
            } catch (...) { std::terminate(); }
            node = memberships_[node].next;
        }
        release_descriptor(reservation.descriptor_, extent);
        consume(reservation);
    }

    [[nodiscard]] bool valid(HostKVExtentCapability capability) const noexcept {
        return capability.owner_ == this && capability.index_ < extents_.size() &&
               extents_[capability.index_].state == ExtentState::Published &&
               extents_[capability.index_].generation == capability.generation_;
    }

    [[nodiscard]] HostKVAllocationConstView view(HostKVExtentCapability capability) const {
        const Extent& extent = require(capability);
        return arena_->view(*extent.allocation);
    }

    [[nodiscard]] bool release(HostKVExtentCapability capability) noexcept {
        if (!valid(capability)) { return false; }
        Extent& extent     = extents_[capability.index_];
        std::uint32_t node = extent.head;
        for (std::uint32_t index = 0; index < extent.page_count; ++index) {
            if (node == kInvalidIndex) { return false; }
            const Membership& membership   = memberships_[node];
            const LogicalKVPageHandle page = membership.page;
            const HostKVPageReplica replica =
                extent.page_store->valid(page) && extent.page_store->host_resident(page)
                    ? extent.page_store->host_replica(page)
                    : HostKVPageReplica{};
            if (!extent.page_store->valid(page) || !extent.page_store->host_resident(page) ||
                replica.extent != capability || replica.page_offset != index ||
                replica.membership_node != node || membership.extent != capability.index_ ||
                membership.offset != index ||
                (!extent.page_store->device_resident(page) &&
                 extent.page_store->address_references(page) != 0)) {
                return false;
            }
            node = memberships_[node].next;
        }
        if (node != kInvalidIndex) { return false; }
        node = extent.head;
        for (std::uint32_t index = 0; index < extent.page_count; ++index) {
            const LogicalKVPageHandle page = memberships_[node].page;
            if (!extent.page_store->detach_host_replica(page, capability)) { std::terminate(); }
            node = memberships_[node].next;
        }
        release_descriptor(capability.index_, extent);
        return true;
    }

    [[nodiscard]] bool can_release_page_replica(LogicalKVPageStore& pages,
                                                LogicalKVPageHandle page) const noexcept {
        if (!pages.valid(page) || !pages.host_resident(page) || pages.source_pins(page) != 0 ||
            (!pages.device_resident(page) && pages.address_references(page) != 0)) {
            return false;
        }
        const HostKVPageReplica replica = pages.host_replica(page);
        if (!valid(replica.extent)) { return false; }
        const Extent& extent     = extents_[replica.extent.index_];
        const std::uint32_t node = replica.membership_node;
        if (extent.page_store != &pages || replica.page_offset >= extent.page_count ||
            node >= memberships_.size() || memberships_[node].page != page ||
            memberships_[node].extent != replica.extent.index_ ||
            memberships_[node].offset != replica.page_offset) {
            return false;
        }
        return true;
    }

    [[nodiscard]] bool
    can_release_page_replicas(std::span<const HostKVPageReplicaRelease> releases) const noexcept {
        begin_release_marks();
        for (const HostKVPageReplicaRelease& release : releases) {
            if (release.pages == nullptr || !mark_release(*release.pages, release.page)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool
    can_release_page_replicas(LogicalKVPageStore& pages,
                              std::span<const LogicalKVPageHandle> releases) const noexcept {
        begin_release_marks();
        for (const LogicalKVPageHandle release : releases) {
            if (!mark_release(pages, release)) { return false; }
        }
        return true;
    }

    [[nodiscard]] bool release_page_replicas(std::span<const HostKVPageReplicaRelease> releases) {
        if (!can_release_page_replicas(releases)) { return false; }
        release_marked_extents();
        return true;
    }

    [[nodiscard]] bool release_page_replicas(LogicalKVPageStore& pages,
                                             std::span<const LogicalKVPageHandle> releases) {
        if (!can_release_page_replicas(pages, releases)) { return false; }
        release_marked_extents();
        return true;
    }

    // Address-space teardown can leave part of an extent with no logical owner. Partition each
    // affected extent once, release all zero-reference runs, and republish retained runs with
    // generation-checked capabilities. Descriptor capacity is provisioned per Host page.
    [[nodiscard]] std::size_t release_unreferenced() noexcept {
        std::size_t released_bytes = 0;
        try {
            extent_scan_scratch_.clear();
            for (std::uint32_t index = 0; index < extents_.size(); ++index) {
                const Extent& extent = extents_[index];
                if (extent.state == ExtentState::Published && extent.allocation &&
                    extent.page_store != nullptr) {
                    extent_scan_scratch_.push_back(index);
                }
            }
            for (const std::uint32_t index : extent_scan_scratch_) {
                const std::size_t bytes =
                    partition_extent(index, [](const LogicalKVPageStore& pages,
                                               LogicalKVPageHandle page, std::uint32_t) {
                        return pages.address_references(page) == 0 && pages.source_pins(page) == 0;
                    });
                if (bytes > std::numeric_limits<std::size_t>::max() - released_bytes) {
                    std::terminate();
                }
                released_bytes += bytes;
            }
        } catch (...) { std::terminate(); }
        return released_bytes;
    }

private:
    static constexpr std::uint32_t kInvalidIndex = std::numeric_limits<std::uint32_t>::max();

    enum class ExtentState : std::uint8_t {
        Free,
        Reserved,
        Published,
    };

    struct Extent {
        ExtentState state              = ExtentState::Free;
        std::uint32_t generation       = 1;
        LogicalKVPageStore* page_store = nullptr;
        std::optional<HostKVAllocation> allocation;
        std::uint32_t head       = kInvalidIndex;
        std::uint32_t tail       = kInvalidIndex;
        std::uint32_t page_count = 0;
    };

    struct Membership {
        LogicalKVPageHandle page;
        std::uint64_t epoch    = 0;
        std::uint32_t coverage = 0;
        std::uint32_t extent   = kInvalidIndex;
        std::uint32_t offset   = 0;
        std::uint32_t next     = kInvalidIndex;
    };

    struct PartitionRun {
        std::uint32_t begin = 0;
        std::uint32_t head  = kInvalidIndex;
        std::uint32_t tail  = kInvalidIndex;
        std::uint32_t count = 0;
        bool release        = false;
    };

    [[nodiscard]] bool valid(const HostKVExtentReservation& reservation) const noexcept {
        if (reservation.owner_ != this || reservation.descriptor_ >= extents_.size() ||
            reservation.page_store_ == nullptr) {
            return false;
        }
        const Extent& extent = extents_[reservation.descriptor_];
        return extent.state == ExtentState::Reserved &&
               extent.generation == reservation.generation_ &&
               extent.page_store == reservation.page_store_ && extent.allocation &&
               extent.page_count != 0 && extent.head != kInvalidIndex &&
               extent.tail != kInvalidIndex;
    }

    void validate(const HostKVExtentReservation& reservation) const {
        if (!valid(reservation)) { throw std::logic_error("Host KV extent reservation is stale"); }
    }

    [[nodiscard]] Extent& require(HostKVExtentCapability capability) {
        if (!valid(capability)) { throw std::invalid_argument("Host KV extent is stale"); }
        return extents_[capability.index_];
    }

    [[nodiscard]] const Extent& require(HostKVExtentCapability capability) const {
        if (!valid(capability)) { throw std::invalid_argument("Host KV extent is stale"); }
        return extents_[capability.index_];
    }

    static void increment_generation(std::uint32_t& generation) noexcept {
        ++generation;
        if (generation == 0) { ++generation; }
    }

    [[nodiscard]] std::uint32_t take_membership() noexcept {
        if (free_membership_count_ == 0) { return kInvalidIndex; }
        return free_memberships_[--free_membership_count_];
    }

    void begin_release_marks() const noexcept {
        affected_extents_.clear();
        ++release_stamp_;
        if (release_stamp_ == 0) {
            std::fill(release_marks_.begin(), release_marks_.end(), 0);
            std::fill(extent_marks_.begin(), extent_marks_.end(), 0);
            release_stamp_ = 1;
        }
    }

    [[nodiscard]] bool mark_release(LogicalKVPageStore& pages,
                                    LogicalKVPageHandle page) const noexcept {
        if (!can_release_page_replica(pages, page)) { return false; }
        const HostKVPageReplica replica = pages.host_replica(page);
        if (!valid(replica.extent) || replica.membership_node >= memberships_.size()) {
            return false;
        }
        const Membership& membership = memberships_[replica.membership_node];
        if (membership.page != page || membership.extent != replica.extent.index_ ||
            membership.offset != replica.page_offset ||
            replica.page_offset >= extents_[replica.extent.index_].page_count ||
            extents_[replica.extent.index_].page_store != &pages) {
            return false;
        }
        if (release_marks_[replica.membership_node] == release_stamp_) { return false; }
        release_marks_[replica.membership_node] = release_stamp_;
        if (extent_marks_[replica.extent.index_] != release_stamp_) {
            extent_marks_[replica.extent.index_] = release_stamp_;
            affected_extents_.push_back(replica.extent.index_);
        }
        return true;
    }

    void release_marked_extents() {
        for (const std::uint32_t index : affected_extents_) {
            (void)partition_extent(
                index, [&](const LogicalKVPageStore&, LogicalKVPageHandle, std::uint32_t node) {
                    return release_marks_[node] == release_stamp_;
                });
        }
    }

    template <typename Predicate>
    [[nodiscard]] std::size_t partition_extent(std::uint32_t index, Predicate&& should_release) {
        if (index >= extents_.size()) {
            throw std::logic_error("Host KV partition extent index is out of range");
        }
        Extent& original = extents_[index];
        if (original.state != ExtentState::Published || original.page_store == nullptr ||
            !original.allocation || original.page_count == 0 ||
            original.allocation->page_count() != original.page_count) {
            throw std::logic_error("Host KV extent is not partitionable");
        }
        LogicalKVPageStore* const pages = original.page_store;
        const HostKVExtentCapability old(this, index, original.generation);

        partition_runs_.clear();
        std::uint32_t node           = original.head;
        bool any_release             = false;
        std::uint32_t retained_runs  = 0;
        std::uint32_t released_pages = 0;
        for (std::uint32_t offset = 0; offset < original.page_count; ++offset) {
            if (node == kInvalidIndex || node >= memberships_.size()) {
                throw std::logic_error("Host KV extent membership is truncated");
            }
            const Membership& entry = memberships_[node];
            if (!pages->valid(entry.page) || !pages->host_resident(entry.page)) {
                throw std::logic_error("Host KV extent membership is stale");
            }
            const HostKVPageReplica replica = pages->host_replica(entry.page);
            if (entry.extent != index || entry.offset != offset || replica.extent != old ||
                replica.page_offset != offset || replica.membership_node != node ||
                replica.content_epoch != entry.epoch ||
                replica.committed_columns != entry.coverage) {
                throw std::logic_error("Host KV extent membership disagrees with its replica");
            }
            const bool release = should_release(*pages, entry.page, node);
            if (release && (pages->source_pins(entry.page) != 0 ||
                            (!pages->device_resident(entry.page) &&
                             pages->address_references(entry.page) != 0))) {
                throw std::logic_error("Host KV partition contains an unreleasable page");
            }
            if (partition_runs_.empty() || partition_runs_.back().release != release) {
                partition_runs_.push_back(PartitionRun{
                    .begin = offset, .head = node, .tail = node, .count = 1, .release = release});
                if (!release) { ++retained_runs; }
            } else {
                PartitionRun& run = partition_runs_.back();
                run.tail          = node;
                ++run.count;
            }
            if (release) {
                any_release = true;
                ++released_pages;
            }
            node = entry.next;
        }
        if (node != kInvalidIndex) {
            throw std::logic_error("Host KV extent membership exceeds its page count");
        }
        if (!any_release) { return 0; }
        const std::uint32_t additional_extents = retained_runs == 0 ? 0U : retained_runs - 1U;
        if (additional_extents > free_count_) {
            throw std::logic_error("Host KV partition descriptor capacity is exhausted");
        }
        const std::size_t stride = page_layout(*pages).page_stride;
        if (released_pages > std::numeric_limits<std::size_t>::max() / stride) {
            throw std::overflow_error("Host KV released byte count overflow");
        }
        const std::size_t released_bytes = stride * static_cast<std::size_t>(released_pages);

        HostKVAllocation remaining = std::move(*original.allocation);
        original.allocation.reset();
        increment_generation(original.generation);
        bool original_assigned = false;
        for (std::size_t run_index = 0; run_index < partition_runs_.size(); ++run_index) {
            const PartitionRun& run = partition_runs_[run_index];
            HostKVAllocation allocation;
            if (run_index + 1U == partition_runs_.size()) {
                allocation = std::move(remaining);
            } else {
                auto split = arena_->split(std::move(remaining), run.count);
                allocation = std::move(split.first);
                remaining  = std::move(split.second);
            }
            memberships_[run.tail].next = kInvalidIndex;

            if (run.release) {
                std::uint32_t release_node = run.head;
                for (std::uint32_t offset = 0; offset < run.count; ++offset) {
                    if (release_node == kInvalidIndex) { std::terminate(); }
                    Membership& entry        = memberships_[release_node];
                    const std::uint32_t next = entry.next;
                    if (!pages->detach_host_replica(entry.page, old)) { std::terminate(); }
                    entry                                       = {};
                    free_memberships_[free_membership_count_++] = release_node;
                    release_node                                = next;
                }
                if (release_node != kInvalidIndex || !allocation.release()) { std::terminate(); }
                continue;
            }

            const std::uint32_t target_index = !original_assigned ? index : free_[--free_count_];
            Extent& target                   = extents_[target_index];
            if (target_index != index && target.state != ExtentState::Free) { std::terminate(); }
            original_assigned = true;
            target.state      = ExtentState::Published;
            target.page_store = pages;
            target.allocation.emplace(std::move(allocation));
            target.head       = run.head;
            target.tail       = run.tail;
            target.page_count = run.count;
            const HostKVExtentCapability replacement(this, target_index, target.generation);
            std::uint32_t retained_node = run.head;
            for (std::uint32_t offset = 0; offset < run.count; ++offset) {
                if (retained_node == kInvalidIndex) { std::terminate(); }
                Membership& entry        = memberships_[retained_node];
                const std::uint32_t next = entry.next;
                pages->rebind_host_replica(entry.page,
                                           HostKVPageReplica{.extent          = old,
                                                             .page_offset     = run.begin + offset,
                                                             .membership_node = retained_node,
                                                             .content_epoch   = entry.epoch,
                                                             .committed_columns = entry.coverage},
                                           HostKVPageReplica{.extent            = replacement,
                                                             .page_offset       = offset,
                                                             .membership_node   = retained_node,
                                                             .content_epoch     = entry.epoch,
                                                             .committed_columns = entry.coverage});
                entry.extent  = target_index;
                entry.offset  = offset;
                retained_node = next;
            }
            if (retained_node != kInvalidIndex) { std::terminate(); }
        }

        if (!original_assigned) {
            original.state      = ExtentState::Free;
            original.page_store = nullptr;
            original.allocation.reset();
            original.head        = kInvalidIndex;
            original.tail        = kInvalidIndex;
            original.page_count  = 0;
            free_[free_count_++] = index;
        }
        return released_bytes;
    }

    void release_descriptor(std::uint32_t index, Extent& extent) noexcept {
        std::uint32_t node = extent.head;
        for (std::uint32_t offset = 0; offset < extent.page_count; ++offset) {
            if (node == kInvalidIndex) { std::terminate(); }
            const std::uint32_t next                    = memberships_[node].next;
            memberships_[node]                          = {};
            free_memberships_[free_membership_count_++] = node;
            node                                        = next;
        }
        if (node != kInvalidIndex) { std::terminate(); }
        extent.state      = ExtentState::Free;
        extent.page_store = nullptr;
        extent.allocation.reset();
        extent.head       = kInvalidIndex;
        extent.tail       = kInvalidIndex;
        extent.page_count = 0;
        increment_generation(extent.generation);
        free_[free_count_++] = index;
    }

    static void consume(HostKVExtentReservation& reservation) noexcept {
        reservation.owner_      = nullptr;
        reservation.page_store_ = nullptr;
    }

    HostKVArena* arena_ = nullptr;
    std::vector<Extent> extents_;
    std::vector<std::uint32_t> free_;
    std::uint32_t free_count_ = 0;
    std::vector<Membership> memberships_;
    std::vector<std::uint32_t> free_memberships_;
    std::uint32_t free_membership_count_ = 0;
    mutable std::vector<std::uint32_t> release_marks_;
    mutable std::vector<std::uint32_t> extent_marks_;
    mutable std::vector<std::uint32_t> affected_extents_;
    mutable std::uint32_t release_stamp_ = 0;
    std::vector<std::uint32_t> extent_scan_scratch_;
    std::vector<PartitionRun> partition_runs_;
};

inline HostKVExtentReservation::~HostKVExtentReservation() {
    if (owner_ != nullptr) { owner_->abort(*this); }
}

inline void KVAddressSpaceStore::truncate_for_replay(KVAddressSpaceHandle handle,
                                                     std::uint32_t frontier,
                                                     HostKVExtentStore& host) {
    Address& address = require_active(handle);
    if (frontier > address.committed_frontier) {
        throw std::invalid_argument("replay rewind extends KV coverage");
    }
    if (frontier < address.checkpoint_frontier) {
        throw std::logic_error("query rewind crosses a protected checkpoint");
    }
    const auto target = pages_for_tokens(frontier);
    std::vector<LogicalKVPageHandle> removed_host;
    for (auto p = target; p < address.page_count; ++p) {
        const auto logical = membership(address, p);
        if (pages_->address_references(logical) != 1 ||
            !pages_->can_release_reference_after_active_reference(logical)) {
            throw std::logic_error("replay suffix is shared or pinned");
        }
        if (pages_->host_resident(logical)) removed_host.push_back(logical);
    }
    const auto columns   = target == 0 ? 0U : frontier - (target - 1U) * kPagedKVPageSize;
    const auto tail      = target == 0 ? LogicalKVPageHandle{} : membership(address, target - 1U);
    const bool trim_tail = target != 0 && columns != pages_->committed_columns(tail);
    if (trim_tail &&
        (!pages_->can_destructive_truncate(tail, columns, true) ||
         (pages_->host_resident(tail) && !host.can_release_page_replica(*pages_, tail)))) {
        throw std::logic_error("replay partial tail is not a private resident page");
    }
    if (trim_tail && pages_->host_resident(tail)) {
        if (!host.release_page_replicas(*pages_, std::span<const LogicalKVPageHandle>(&tail, 1))) {
            throw std::logic_error("replay could not invalidate the partial tail's Host copy");
        }
    }
    while (address.page_count > target) {
        const auto p       = --address.page_count;
        const auto logical = membership(address, p);
        pages_->release_active_reference(logical);
        if (!pages_->release_reference(logical, true)) {
            throw std::logic_error("replay could not release private suffix ownership");
        }
    }
    if (!host.release_page_replicas(*pages_, removed_host)) {
        throw std::logic_error("replay could not release discarded Host pages");
    }
    if (address.device_working_set) {
        auto& set = *address.device_working_set;
        set.erase(std::lower_bound(set.begin(), set.end(), target), set.end());
    }
    if (trim_tail) pages_->destructive_truncate(tail, columns);
    trim_unique_suffix(address.directory, directory_capacity_, target);
    address.committed_frontier = frontier;
}

inline KVAddressSpaceStore::KVPlacementCounts
KVAddressSpaceStore::apply_device_placement(KVAddressSpaceHandle handle,
                                            HostKVExtentStore& host_kv_extents,
                                            std::span<const std::uint32_t> selected_pages,
                                            cudaStream_t transfer_stream, const char* trace_phase) {
    const bool trace        = std::getenv("NINFER_KVMEM_TRANSFER_TRACE") != nullptr;
    using TraceClock        = std::chrono::steady_clock;
    const auto trace_begin  = TraceClock::now();
    std::uint64_t d2h_bytes = 0, h2d_bytes = 0;
    std::size_t d2h_pages = 0;
    double d2h_ms = 0, h2d_ms = 0;
    Address& address = require_active(handle);
    for (std::size_t index = 0; index < selected_pages.size(); ++index) {
        if (selected_pages[index] >= address.page_count ||
            (index != 0 && selected_pages[index] <= selected_pages[index - 1U])) {
            throw std::invalid_argument("KV device placement selects an invalid page set");
        }
    }
    const auto selected = [&](std::uint32_t page) {
        return std::binary_search(selected_pages.begin(), selected_pages.end(), page);
    };

    const std::uint32_t reserved_before = address.reservation.pages();
    std::uint32_t resident_before       = 0;
    for (std::uint32_t page = 0; page < address.page_count; ++page) {
        if (page_in_working_set(address, page) &&
            pages_->device_resident(membership(address, page))) {
            ++resident_before;
        }
    }

    KVPlacementCounts counts;
    counts.telemetry.calls = 1;
    std::vector<std::uint32_t> outgoing;
    for (std::uint32_t page = 0; page < address.page_count; ++page) {
        if (pages_->device_resident(membership(address, page)) && !selected(page)) {
            outgoing.push_back(page);
        }
    }
    const bool same_selection =
        address.device_working_set
            ? (address.device_working_set->size() == selected_pages.size() &&
               std::equal(address.device_working_set->begin(), address.device_working_set->end(),
                          selected_pages.begin()))
            : selected_pages.size() == address.page_count;
    if (same_selection && outgoing.empty() && resident_before == selected_pages.size()) {
        // ensure_mapped publishes growth on the compute stream; prior placement
        // publication has completed at the preceding worker boundary. With the
        // same fully resident set, this row needs no table rewrite or transfer
        // wait. In particular, do not drain another owner's pending H2D restore.
        if (!address.device_working_set) {
            address.device_working_set.emplace(selected_pages.begin(), selected_pages.end());
        }
        counts.telemetry.no_copy_calls      = 1;
        counts.telemetry.total_host_wall_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(TraceClock::now() - trace_begin)
                .count());
        if (trace) {
            std::fprintf(
                stderr,
                "KVPLACEMENT phase=%s row=%d planes=%zu mapped=%u selected=%zu "
                "demoted=0 promoted=0 d2h_pages=0 d2h_bytes=0 h2d_bytes=0 "
                "d2h_submit_wait_ms=0 h2d_submit_wait_ms=0 total_ms=%.6f unchanged_table=1\n",
                trace_phase, bound_row(handle), pages_->physical_pool().geometry().planes.size(),
                address.page_count, selected_pages.size(),
                static_cast<double>(counts.telemetry.total_host_wall_ns) * 1e-6);
        }
        return counts;
    }
    if (!outgoing.empty()) {
        std::vector<LogicalKVPageHandle> stale;
        for (const std::uint32_t page : outgoing) {
            const LogicalKVPageHandle logical = membership(address, page);
            if (!pages_->host_replica_current(logical)) { stale.push_back(logical); }
        }
        if (!stale.empty()) {
            std::vector<HostKVExtentReservation> backups;
            backups.reserve(stale.size());
            placement_scratch_.reserve(stale.size());
            std::size_t reserved = 0;
            while (reserved < stale.size()) {
                auto backup = host_kv_extents.prepare_prefix(
                    *pages_, std::span(stale).subspan(reserved), true);
                if (!backup) {
                    const auto page = stale[reserved];
                    std::fprintf(stderr,
                                 "KVPLACEMENT capacity phase=%s kind=Host extents=%u/%u "
                                 "remaining=%zu host=%d current=%d writers=%u references=%u "
                                 "pins=%u committed=%u free_bytes=%zu max_pages=%u\n",
                                 trace_phase, host_kv_extents.occupied(),
                                 host_kv_extents.capacity(), stale.size() - reserved,
                                 pages_->host_resident(page), pages_->host_replica_current(page),
                                 pages_->writer_references(page), pages_->address_references(page),
                                 pages_->source_pins(page), pages_->committed_columns(page),
                                 host_kv_extents.free_bytes(),
                                 host_kv_extents.max_allocatable_pages(*pages_, 1));
                    throw std::bad_alloc();
                }
                reserved += host_kv_extents.page_count(*backup);
                backups.push_back(std::move(*backup));
            }
            // All byte ranges and source pins exist before the first DMA. Failed
            // preparation rolls them all back without changing any published replica.
            d2h_pages = stale.size();
            d2h_bytes = pages_->physical_pool()
                            .host_transfer_run_work(static_cast<std::uint32_t>(stale.size()))
                            .payload_bytes;
            const auto copy_begin = TraceClock::now();
            try {
                for (auto& backup : backups) {
                    placement_scratch_.resize(host_kv_extents.page_count(backup));
                    host_kv_extents.device_sources(backup, placement_scratch_);
                    pages_->physical_pool().copy_to_host(
                        placement_scratch_, host_kv_extents.writable_view(backup), transfer_stream);
                }
                if (cudaStreamSynchronize(transfer_stream) != cudaSuccess) {
                    throw std::runtime_error("KV device placement stage-out transfer failed");
                }
                host_kv_extents.check_io_errors();
            } catch (...) {
                // Earlier extents may already be submitted. Keep every reservation
                // and source pin alive until that stream is drained.
                (void)cudaStreamSynchronize(transfer_stream);
                throw;
            }
            counts.telemetry.d2h_submit_wait_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(TraceClock::now() - copy_begin)
                    .count());
            d2h_ms = static_cast<double>(counts.telemetry.d2h_submit_wait_ns) / 1.0e6;
            for (auto& backup : backups) { (void)host_kv_extents.publish(std::move(backup)); }
        }
        for (const std::uint32_t page : outgoing) {
            // A pinned page belongs to a pending active snapshot or transfer: demoting
            // it would break that publication. Keep it resident; the next placement
            // reconsiders once the pin clears.
            if (pages_->source_pins(membership(address, page)) != 0 ||
                !pages_->device_payload_ready(membership(address, page))) {
                continue;
            }
            // A physical replica can back several live rows with different
            // retrieval sets. Keep it until no other row can read it.
            const auto logical = membership(address, page);
            const bool other_reader =
                std::any_of(addresses_.begin(), addresses_.end(), [&](const Address& other) {
                    return &other != &address && other.active && page < other.page_count &&
                           page_in_working_set(other, page) && membership(other, page) == logical;
                });
            if (other_reader) { continue; }
            if (!pages_->drop_device_replica_within_active(membership(address, page))) {
                throw std::logic_error("KV device placement cannot demote an outgoing page");
            }
            ++counts.demoted;
        }
    }

    std::vector<std::uint32_t> incoming;
    for (const std::uint32_t page : selected_pages) {
        if (!pages_->device_resident(membership(address, page))) { incoming.push_back(page); }
    }
    if (!incoming.empty()) {
        // Preserve the already granted unit margin while reserving missing destinations.
        // Promotion consumes only its own added pages; unit growth remains untouched.
        const auto available = pages_->physical_pool().available_pages();
        if (incoming.size() > available) {
            counts.demoted += pages_->reclaim_inactive_device_replicas(
                static_cast<std::uint32_t>(incoming.size() - available));
        }
        if (!pages_->physical_pool().can_resize_reservation(address.reservation,
                                                            reserved_before + incoming.size())) {
            std::fprintf(stderr,
                         "KVPLACEMENT capacity phase=%s kind=Device allocated=%u reserved=%u "
                         "capacity=%u incoming=%zu own_reserved=%u outgoing=%zu demoted=%u\n",
                         trace_phase, pages_->physical_pool().allocated_pages(),
                         pages_->physical_pool().reserved_pages(),
                         pages_->physical_pool().capacity_pages(), incoming.size(), reserved_before,
                         outgoing.size(), counts.demoted);
        }
        pages_->physical_pool().resize_reservation(
            address.reservation, reserved_before + static_cast<std::uint32_t>(incoming.size()));
        std::vector<HostKVPageReplica> sources;
        placement_scratch_.clear();
        try {
            for (const std::uint32_t page : incoming) {
                const LogicalKVPageHandle logical = membership(address, page);
                if (!pages_->host_resident(logical) || !pages_->host_replica_current(logical)) {
                    throw std::logic_error("KV device placement cannot promote a Host-absent page");
                }
                sources.push_back(pages_->host_replica(logical));
                placement_scratch_.push_back(
                    pages_->reserve_device_replica(logical, address.reservation));
            }
            const auto copy_begin = TraceClock::now();
            std::size_t begin     = 0;
            while (begin < sources.size()) {
                std::size_t end = begin + 1;
                while (end < sources.size() && sources[end].extent == sources[begin].extent &&
                       sources[end].page_offset == sources[end - 1U].page_offset + 1U) {
                    ++end;
                }
                const HostKVAllocationConstView source =
                    host_kv_extents.view(sources[begin].extent)
                        .subview(sources[begin].page_offset,
                                 static_cast<std::uint32_t>(end - begin));
                h2d_bytes += pages_->physical_pool()
                                 .host_transfer_run_work(static_cast<std::uint32_t>(end - begin))
                                 .payload_bytes;
                pages_->physical_pool().copy_from_host(
                    source,
                    std::span<const DeviceKVPageHandle>(placement_scratch_.data() + begin,
                                                        end - begin),
                    transfer_stream);
                begin = end;
            }
            if (cudaStreamSynchronize(transfer_stream) != cudaSuccess) {
                throw std::runtime_error("KV device placement stage-in transfer failed");
            }
            host_kv_extents.check_io_errors();
            counts.telemetry.h2d_submit_wait_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(TraceClock::now() - copy_begin)
                    .count());
            h2d_ms = static_cast<double>(counts.telemetry.h2d_submit_wait_ns) / 1.0e6;
            for (const std::uint32_t page : incoming) {
                pages_->publish_device_replica(membership(address, page));
                ++counts.promoted;
            }
        } catch (...) {
            // Host and pending destinations remain leased until all submitted reads settle.
            (void)cudaStreamSynchronize(transfer_stream);
            for (const auto index : incoming) {
                pages_->abort_device_replica(membership(address, index), address.reservation);
            }
            pages_->physical_pool().resize_reservation(address.reservation, reserved_before);
            throw;
        }
    }

    std::uint32_t page = 0;
    while (page < address.page_count) {
        if (selected(page)) {
            std::uint32_t end = page + 1U;
            publish_scratch_.clear();
            publish_scratch_.push_back(pages_->physical(membership(address, page)));
            while (end < address.page_count && selected(end)) {
                publish_scratch_.push_back(pages_->physical(membership(address, end)));
                ++end;
            }
            tables_->publish(address.row->handle(), page, publish_scratch_, transfer_stream);
            page = end;
        } else {
            std::uint32_t end = page + 1U;
            while (end < address.page_count && !selected(end)) { ++end; }
            tables_->publish_holes(address.row->handle(), page, end - page, transfer_stream);
            page = end;
        }
    }

    pages_->physical_pool().resize_reservation(address.reservation, reserved_before);
    // Record the set for the next activation: it defines the restore/verify scope.
    // Freshly mapped growth pages materialize device-resident on ensure_mapped and are
    // outside this list until the next placement or activation republishes residency.
    address.device_working_set =
        std::vector<std::uint32_t>(selected_pages.begin(), selected_pages.end());
    // The next consumer is on the compute stream, not transfer_stream. Its
    // block-table reads must observe this placement's final publication.
    const auto publication_wait_begin = TraceClock::now();
    if (cudaStreamSynchronize(transfer_stream) != cudaSuccess) {
        throw std::runtime_error("KV device placement table publication failed");
    }
    counts.telemetry.publication_wait_ns =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       TraceClock::now() - publication_wait_begin)
                                       .count());
    counts.telemetry.demoted_pages      = counts.demoted;
    counts.telemetry.promoted_pages     = counts.promoted;
    counts.telemetry.d2h_pages          = d2h_pages;
    counts.telemetry.d2h_bytes          = d2h_bytes;
    counts.telemetry.h2d_bytes          = h2d_bytes;
    counts.telemetry.no_copy_calls      = d2h_bytes == 0 && h2d_bytes == 0 ? 1 : 0;
    counts.telemetry.total_host_wall_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(TraceClock::now() - trace_begin)
            .count());
    if (trace) {
        // Payload excludes Host arena padding. Copy times include submission and
        // the existing stream wait; total also includes planning/table publication.
        std::fprintf(
            stderr,
            "KVPLACEMENT phase=%s row=%d planes=%zu mapped=%u selected=%zu "
            "demoted=%u promoted=%u d2h_pages=%zu d2h_bytes=%llu h2d_bytes=%llu "
            "d2h_submit_wait_ms=%.6f h2d_submit_wait_ms=%.6f total_ms=%.6f\n",
            trace_phase, bound_row(handle), pages_->physical_pool().geometry().planes.size(),
            address.page_count, selected_pages.size(), counts.demoted, counts.promoted, d2h_pages,
            static_cast<unsigned long long>(d2h_bytes), static_cast<unsigned long long>(h2d_bytes),
            d2h_ms, h2d_ms,
            std::chrono::duration<double, std::milli>(TraceClock::now() - trace_begin).count());
    }
    return counts;
}


} // namespace ninfer::models::qwen3_5::detail
