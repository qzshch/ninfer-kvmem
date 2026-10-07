#pragma once

#include "core/arena.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace ninfer {

struct FileKVSnapshot {
    std::uint64_t read_bytes            = 0;
    std::uint64_t written_bytes         = 0;
    std::uint64_t read_ns               = 0;
    std::uint64_t write_ns              = 0;
    std::uint64_t staging_wait_ns       = 0;
    std::uint64_t reads                 = 0;
    std::uint64_t writes                = 0;
    std::size_t pinned_bytes            = 0;
    std::size_t integrity_bytes         = 0;
    std::uint64_t pending_reads         = 0;
    std::uint64_t pending_writes        = 0;
    std::uint64_t pending_callbacks     = 0;
    std::size_t ram_capacity_bytes      = 0;
    std::uint64_t ram_resident_bytes    = 0;
    std::uint64_t ram_dirty_bytes       = 0;
    std::uint64_t ram_hit_bytes         = 0;
    std::uint64_t ram_miss_bytes        = 0;
    std::uint64_t disk_read_bytes       = 0;
    std::uint64_t disk_written_bytes    = 0;
    std::uint64_t disk_read_ns          = 0;
    std::uint64_t disk_write_ns         = 0;
    std::uint64_t disk_pwrite_ns        = 0;
    std::uint64_t disk_sync_ns          = 0;
    std::uint64_t disk_sync_calls       = 0;
    std::uint64_t disk_high_water_bytes = 0;
    // Conservative page-rounded charge for unsynced file writes (gauge).
    std::uint64_t filesystem_pending_bytes = 0;
    std::uint64_t filesystem_write_budget_bytes = 0;
    std::uint64_t ram_evictions         = 0;
    std::uint64_t prefetch_bytes        = 0;
    std::uint64_t prefetch_hit_bytes    = 0;
    std::uint64_t prefetch_wasted_bytes = 0;
    std::uint64_t prefetch_dropped_jobs = 0;
    std::uint64_t pending_prefetches    = 0;
    std::uint64_t pending_writebacks    = 0;
};

// Storage only. The IO thread touches disk, managed RAM and fixed pinned staging;
// it never calls CUDA or changes an allocation, logical page or model state.
// The caller retains source/destination leases until its stream has completed,
// then checks errors before publishing any replica. Files are instance-local
// and removed at destruction, not a cross-engine checkpoint format.
class FileKVBacking {
public:
    class Transfer {
    public:
        [[nodiscard]] std::byte* data() const noexcept;
        void enqueue_before(cudaStream_t stream);
        void enqueue_after(cudaStream_t stream);
        // Submission failure: unblock callbacks before draining the stream.
        void abort() noexcept;
        // Call only after the submission stream has drained. A read without an
        // after callback must still release its slot after the IO worker settles.
        void retire_after_drain() noexcept;

    private:
        struct State;

        explicit Transfer(std::shared_ptr<State> state) : state_(std::move(state)) {}

        std::shared_ptr<State> state_;
        friend class FileKVBacking;
    };

    FileKVBacking(const std::filesystem::path& directory, std::size_t capacity_bytes,
                  std::size_t staging_slot_bytes = 16ULL << 20, std::size_t ram_capacity_bytes = 0,
                  bool prefetch = false, bool write_through = false);
    ~FileKVBacking();
    FileKVBacking(const FileKVBacking&)            = delete;
    FileKVBacking& operator=(const FileKVBacking&) = delete;

    [[nodiscard]] std::size_t slot_bytes() const noexcept;
    [[nodiscard]] cudaStream_t stream() const noexcept;
    void order_before(cudaStream_t caller_stream);
    void order_after(cudaStream_t caller_stream);
    [[nodiscard]] Transfer read(std::size_t offset, std::size_t bytes);
    [[nodiscard]] Transfer write(std::size_t offset, std::size_t bytes);
    // Best-effort, bounded L3 -> L2 hints. They never publish a logical replica
    // or retain an allocation: invalidation epochs discard stale hints.
    void prefetch(std::size_t offset, std::size_t bytes);
    void check_errors() const;
    // File transfers must unwind and drain their byte worker on submission
    // failure; the ordinary core CUDA checker deliberately aborts the process.
    static void check_cuda_submission(cudaError_t error);
    // Only after stream synchronization reports failure. CUDA may discard host
    // callbacks: poison this instance and unblock every staging dependency.
    void abort_after_failed_stream() noexcept;
    // Only an unleased/free range may be invalidated by the allocation owner.
    void invalidate(std::size_t offset, std::size_t bytes) noexcept;
    // Only after the CUDA streams are drained. No callbacks may reference us.
    // Also synchronize/evict the bounded cross-job filesystem write batch.
    void drain();
    [[nodiscard]] FileKVSnapshot snapshot() const noexcept;
    [[nodiscard]] const std::filesystem::path& path() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    [[nodiscard]] Transfer submit(std::size_t offset, std::size_t bytes, bool writing);
};

} // namespace ninfer
