#pragma once

#include "ninfer/types.h"

#include <chrono>
#include <cstdint>

namespace ninfer::models::qwen3_5::detail {

inline std::uint64_t kvmem_elapsed_ns(std::chrono::steady_clock::time_point started) noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - started)
                                          .count());
}

inline void add_kvmem_placement(KvmemPlacementStats& destination,
                                const KvmemPlacementStats& source) noexcept {
    destination.calls += source.calls;
    destination.no_copy_calls += source.no_copy_calls;
    destination.demoted_pages += source.demoted_pages;
    destination.promoted_pages += source.promoted_pages;
    destination.d2h_pages += source.d2h_pages;
    destination.d2h_bytes += source.d2h_bytes;
    destination.h2d_bytes += source.h2d_bytes;
    destination.d2h_submit_wait_ns += source.d2h_submit_wait_ns;
    destination.h2d_submit_wait_ns += source.h2d_submit_wait_ns;
    destination.publication_wait_ns += source.publication_wait_ns;
    destination.total_host_wall_ns += source.total_host_wall_ns;
}

// Lifetime is one already-scheduled replay step. Records its inclusive wall time
// without polling, events, waits, or changes to the underlying execution.
class KvmemReplayStepTimer {
public:
    explicit KvmemReplayStepTimer(KvmemDiagnostics* diagnostics) noexcept
        : diagnostics_(diagnostics),
          started_(diagnostics ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{}) {}

    ~KvmemReplayStepTimer() noexcept {
        if (diagnostics_ != nullptr) {
            ++diagnostics_->replay_units;
            diagnostics_->replay_step_host_wall_ns += kvmem_elapsed_ns(started_);
        }
    }
private:
    KvmemDiagnostics* diagnostics_;
    std::chrono::steady_clock::time_point started_;
};

class KvmemHostWallTimer {
public:
    explicit KvmemHostWallTimer(std::uint64_t& elapsed) noexcept
        : elapsed_(elapsed), started_(std::chrono::steady_clock::now()) {}

    ~KvmemHostWallTimer() noexcept { elapsed_ += kvmem_elapsed_ns(started_); }
private:
    std::uint64_t& elapsed_;
    std::chrono::steady_clock::time_point started_;
};

} // namespace ninfer::models::qwen3_5::detail
