#include "core/file_kv_backing.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <list>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace ninfer {
namespace {
using Clock                                   = std::chrono::steady_clock;
constexpr std::size_t kSector                 = 256;
constexpr std::size_t kMaximumQueuedTransfers = 8192;
constexpr std::size_t kHotBlock               = 1ULL << 20;
constexpr std::size_t kHotSectors             = kHotBlock / kSector;
constexpr std::size_t kMaximumPrefetches      = 128;
constexpr std::size_t kFilesystemWriteBudget  = 64ULL << 20;
constexpr auto kIdleWritebackDelay = std::chrono::milliseconds(50);

std::uint64_t elapsed(Clock::time_point begin) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
}

// Non-cryptographic integrity check over every byte, including canonical padding.
std::uint32_t checksum(const std::byte* bytes) {
    std::uint64_t a = 0x9e3779b185ebca87ULL;
    std::uint64_t b = 0xc2b2ae3d27d4eb4fULL;
    for (std::size_t i = 0; i < kSector; i += 16) {
        std::uint64_t x, y;
        std::memcpy(&x, bytes + i, sizeof(x));
        std::memcpy(&y, bytes + i + 8, sizeof(y));
        a = std::rotl(a ^ x, 27) * 0x9e3779b185ebca87ULL;
        b = std::rotl(b ^ y, 31) * 0xc2b2ae3d27d4eb4fULL;
    }
    a ^= b + (a >> 33);
    a *= 0xff51afd7ed558ccdULL;
    return static_cast<std::uint32_t>(a ^ (a >> 32));
}

[[noreturn]] void io_error(const char* operation) {
    throw std::runtime_error(std::string("file KV ") + operation + ": " + std::strerror(errno));
}
} // namespace

struct FileKVBacking::Transfer::State {
    FileKVBacking::Impl* owner = nullptr;
    std::byte* buffer          = nullptr;
    std::size_t offset         = 0;
    std::size_t bytes          = 0;
    bool writing               = false;
    std::shared_ptr<State> predecessor;
    std::mutex mutex;
    std::condition_variable cv;
    bool io_ready    = false;
    bool io_complete = false;
    bool gpu_ready   = false;
    bool done        = false;
    bool aborted     = false;

    void wait_predecessor() {
        std::shared_ptr<State> previous;
        {
            std::lock_guard lock(mutex);
            previous = predecessor;
        }
        if (previous) { previous->wait_done(); }
        {
            std::lock_guard lock(mutex);
            predecessor.reset();
        }
    }

    void wait_done() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return done; });
    }

    void finish() noexcept {
        {
            std::lock_guard lock(mutex);
            done = true;
        }
        cv.notify_all();
    }

    void before() {
        wait_predecessor();
        if (writing) {
            {
                std::lock_guard lock(mutex);
                if (aborted) { return; }
            }
            std::memset(buffer, 0, bytes);
        } else {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return io_ready || aborted; });
        }
    }

    void after() {
        if (!writing) {
            finish();
            return;
        }
        {
            std::lock_guard lock(mutex);
            gpu_ready = true;
        }
        cv.notify_all();
        // The stream completion must include the write and its integrity record.
        wait_done();
    }
};

struct FileKVBacking::Impl {
    struct Callback;
    using CallbackList = std::list<std::unique_ptr<Callback>>;

    struct Callback {
        Impl* owner = nullptr;
        std::shared_ptr<Transfer::State> state;
        bool before = false;
        CallbackList::iterator position;
    };

    std::filesystem::path path;
    int fd                        = -1;
    std::size_t capacity          = 0;
    std::size_t slot_bytes        = 0;
    std::size_t system_page_bytes = 0;
    std::array<std::optional<PinnedHostBuffer>, 2> slots;
    std::array<std::shared_ptr<Transfer::State>, 2> tails;
    std::size_t next_slot = 0;
    std::vector<std::uint32_t> checksums;
    std::vector<std::uint8_t> written;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::shared_ptr<Transfer::State>> queue;
    std::exception_ptr error;
    // Allocate the error before any work, so failed-context cleanup allocates
    // neither a traversal vector nor an exception object.
    std::exception_ptr failed_stream_error =
        std::make_exception_ptr(std::runtime_error("file KV CUDA stream failed"));
    CallbackList callbacks;
    bool stopping = false;
    bool flush_requested = false;
    std::thread worker;
    cudaStream_t stream       = nullptr;
    cudaEvent_t source_ready  = nullptr;
    cudaEvent_t transfer_done = nullptr;
    std::atomic<std::uint64_t> read_bytes{0}, written_bytes{0}, read_ns{0}, write_ns{0};
    std::atomic<std::uint64_t> staging_wait_ns{0}, reads{0}, writes{0};
    std::atomic<std::uint64_t> pending_reads{0}, pending_writes{0};
    std::atomic<std::uint64_t> pending_callbacks{0};

    // The RAM tier is owned memory, not the filesystem page cache or the two
    // pinned transfer slots. A protected read set survives streaming writes.
    struct HotSlot {
        std::size_t block = std::numeric_limits<std::size_t>::max();
        int previous = -1, next = -1;
        bool protected_set = false;
        std::array<std::uint8_t, kHotSectors> valid{}, prefetched{};
    };

    struct Prefetch {
        std::size_t block;
        std::uint64_t epoch;
    };

    std::size_t ram_capacity = 0;
    bool prefetch_enabled    = false;
    bool write_through       = false;
    std::unique_ptr<std::byte[]> ram;
    std::vector<HotSlot> hot;
    std::vector<int> free_hot;
    std::vector<int> block_slots;
    std::vector<std::uint64_t> block_epochs;
    std::array<int, 2> heads{-1, -1}, ends{-1, -1};
    std::size_t protected_count = 0, next_hot = 0;
    // Sector written values: 0 absent, 1 readable file copy, 2 dirty RAM owner.
    // A readable file copy need not be crash-durable: this is ephemeral storage.
    mutable std::mutex hot_mutex;
    std::deque<Prefetch> prefetch_queue;
    std::atomic<std::uint64_t> ram_resident_bytes{0}, ram_dirty_bytes{0};
    std::atomic<std::uint64_t> ram_hit_bytes{0}, ram_miss_bytes{0};
    std::atomic<std::uint64_t> disk_read_bytes{0}, disk_written_bytes{0};
    std::atomic<std::uint64_t> disk_read_ns{0}, disk_write_ns{0}, ram_evictions{0};
    std::atomic<std::uint64_t> disk_pwrite_ns{0}, disk_sync_ns{0}, disk_sync_calls{0};
    std::atomic<std::uint64_t> disk_high_water_bytes{0};
    std::atomic<std::uint64_t> prefetch_bytes{0}, prefetch_hit_bytes{0};
    std::atomic<std::uint64_t> prefetch_wasted_bytes{0}, prefetch_dropped_jobs{0};
    std::atomic<std::uint64_t> pending_prefetches{0}, pending_writebacks{0};
    struct WrittenRange { std::size_t offset, bytes; };
    // At most 64 MiB of newly written filesystem pages, plus the current read,
    // can be pending eviction. Keep this fixed rather than scaling with cold capacity.
    std::array<WrittenRange, 64> written_ranges{};
    std::size_t written_range_count = 0;
    std::atomic<std::uint64_t> filesystem_pending_bytes{0};

    void unlink_hot(int index) {
        auto& slot    = hot[index];
        const int set = slot.protected_set ? 1 : 0;
        if (slot.previous >= 0)
            hot[slot.previous].next = slot.next;
        else
            heads[set] = slot.next;
        if (slot.next >= 0)
            hot[slot.next].previous = slot.previous;
        else
            ends[set] = slot.previous;
        if (slot.protected_set) --protected_count;
        slot.previous = slot.next = -1;
    }

    void link_hot(int index, bool protect) {
        auto& slot         = hot[index];
        const int set      = protect ? 1 : 0;
        slot.protected_set = protect;
        slot.previous      = -1;
        slot.next          = heads[set];
        if (heads[set] >= 0)
            hot[heads[set]].previous = index;
        else
            ends[set] = index;
        heads[set] = index;
        if (protect) ++protected_count;
    }

    void touch_hot(int index, bool demand_read) {
        const bool protect = demand_read || hot[index].protected_set;
        unlink_hot(index);
        link_hot(index, protect);
        const auto limit = std::max<std::size_t>(1, hot.size() * 3 / 4);
        if (protected_count > limit) {
            const int demote = ends[1];
            unlink_hot(demote);
            link_hot(demote, false);
        }
    }

    void disk_io(std::size_t offset, std::byte* buffer, std::size_t bytes, bool writing) {
        const auto begin_time = Clock::now();
        std::size_t completed = 0;
        while (completed != bytes) {
            const auto count =
                writing ? ::pwrite(fd, buffer + completed, bytes - completed, offset + completed)
                        : ::pread(fd, buffer + completed, bytes - completed, offset + completed);
            if (count < 0) {
                if (errno == EINTR) continue;
                io_error(writing ? "pwrite" : "pread");
            }
            if (!count) throw std::runtime_error("file KV short positional IO");
            completed += static_cast<std::size_t>(count);
        }
        if (writing) {
            disk_written_bytes += bytes;
            const auto duration = elapsed(begin_time);
            disk_write_ns += duration;
            disk_pwrite_ns += duration;
            disk_high_water_bytes.store(std::max(disk_high_water_bytes.load(),
                                                 static_cast<std::uint64_t>(offset + bytes)));
        } else {
            disk_read_bytes += bytes;
            disk_read_ns += elapsed(begin_time);
        }
    }

    void evict_file_pages(std::size_t offset, std::size_t bytes) {
        const auto begin = offset / system_page_bytes * system_page_bytes;
        const auto end =
            (offset + bytes + system_page_bytes - 1) / system_page_bytes * system_page_bytes;
        const auto result = ::posix_fadvise(fd, begin, end - begin, POSIX_FADV_DONTNEED);
        if (result)
            throw std::runtime_error(std::string("file KV cache eviction: ") +
                                     std::strerror(result));
    }

    void flush_disk_batch() {
        if (!written_range_count) return;
        // Ephemeral live storage needs completed page writeback, not metadata or
        // device-cache crash durability. Range writeback keeps the fixed dirty
        // budget bounded without an expensive whole-file flush across DrvFs.
        std::sort(written_ranges.begin(), written_ranges.begin() + written_range_count,
                  [](const WrittenRange& a, const WrittenRange& b) { return a.offset < b.offset; });
        std::size_t i = 0;
        while (i < written_range_count) {
            const auto begin = written_ranges[i].offset / system_page_bytes * system_page_bytes;
            auto end = (written_ranges[i].offset + written_ranges[i].bytes + system_page_bytes - 1) /
                       system_page_bytes * system_page_bytes;
            ++i;
            while (i < written_range_count && written_ranges[i].offset <= end) {
                end = std::max(end, (written_ranges[i].offset + written_ranges[i].bytes +
                                    system_page_bytes - 1) / system_page_bytes * system_page_bytes);
                ++i;
            }
            const auto start = Clock::now();
            while (::sync_file_range(fd, static_cast<off_t>(begin), static_cast<off_t>(end - begin),
                                     SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE |
                                         SYNC_FILE_RANGE_WAIT_AFTER) != 0) {
                if (errno == EINTR) continue;
                io_error("sync_file_range");
            }
            const auto duration = elapsed(start);
            disk_write_ns += duration;
            disk_sync_ns += duration;
            ++disk_sync_calls;
            evict_file_pages(begin, end - begin);
        }
        written_range_count = 0;
        filesystem_pending_bytes = 0;
    }

    std::size_t rounded_file_bytes(std::size_t offset, std::size_t bytes) const {
        return ((offset + bytes + system_page_bytes - 1) / system_page_bytes -
                offset / system_page_bytes) * system_page_bytes;
    }

    // Single byte worker, with hot_mutex held. Include page alignment in the
    // charge, and sync BEFORE exceeding the fixed budget. The extra page allows
    // one maximum-size staging transfer whose starting sector is unaligned.
    void reserve_disk_batch(std::size_t offset, std::size_t bytes) {
        const auto charge = rounded_file_bytes(offset, bytes);
        if (written_range_count == written_ranges.size() ||
            filesystem_pending_bytes.load() + charge > kFilesystemWriteBudget + system_page_bytes)
            flush_disk_batch();
    }

    void record_disk_batch(std::size_t offset, std::size_t bytes) {
        written_ranges[written_range_count++] = {offset, bytes};
        filesystem_pending_bytes += rounded_file_bytes(offset, bytes);
    }

    // Called with hot_mutex, only on the byte worker. Clean-sector publication
    // follows successful writeback. A disk failure poisons the whole instance.
    void flush_hot(int index) {
        auto& slot = hot[index];
        if (slot.block == std::numeric_limits<std::size_t>::max()) { return; }
        const auto base    = slot.block * kHotSectors;
        const auto sectors = std::min(kHotSectors, written.size() - base);
        bool wrote         = false;
        reserve_disk_batch(slot.block * kHotBlock, sectors * kSector);
        for (std::size_t s = 0; s < sectors;) {
            if (!slot.valid[s] || written[base + s] != 2) {
                ++s;
                continue;
            }
            auto end = s + 1;
            while (end < sectors && slot.valid[end] && written[base + end] == 2) ++end;
            auto* data = ram.get() + static_cast<std::size_t>(index) * kHotBlock + s * kSector;
            disk_io((base + s) * kSector, data, (end - s) * kSector, true);
            s     = end;
            wrote = true;
        }
        if (!wrote) return;
        record_disk_batch(slot.block * kHotBlock, sectors * kSector);
        for (std::size_t s = 0; s < sectors; ++s) {
            if (slot.valid[s] && written[base + s] == 2) {
                written[base + s] = 1;
                ram_dirty_bytes -= kSector;
            }
        }
    }

    int acquire_hot(std::size_t block, bool demand_read) {
        if (const int found = block_slots[block]; found >= 0) {
            touch_hot(found, demand_read);
            return found;
        }
        int index;
        if (!free_hot.empty()) {
            index = free_hot.back();
            free_hot.pop_back();
        } else if (next_hot < hot.size())
            index = static_cast<int>(next_hot++);
        else {
            index        = ends[0] >= 0 ? ends[0] : ends[1];
            auto& victim = hot[index];
            flush_hot(index);
            for (std::size_t s = 0; s < kHotSectors; ++s) {
                if (victim.valid[s]) ram_resident_bytes -= kSector;
                if (victim.prefetched[s]) prefetch_wasted_bytes += kSector;
            }
            block_slots[victim.block] = -1;
            unlink_hot(index);
            victim.valid.fill(0);
            victim.prefetched.fill(0);
            ++ram_evictions;
        }
        hot[index].block   = block;
        block_slots[block] = index;
        link_hot(index, false);
        if (demand_read) touch_hot(index, true);
        return index;
    }

    void cached_io(const Transfer::State& state) {
        std::lock_guard lock(hot_mutex);
        for (std::size_t done = 0; done < state.bytes;) {
            const auto offset = state.offset + done;
            const auto block  = offset / kHotBlock;
            const auto local  = offset % kHotBlock;
            const auto bytes  = std::min(state.bytes - done, kHotBlock - local);
            const int index   = acquire_hot(block, !state.writing);
            auto& slot        = hot[index];
            auto* data        = ram.get() + static_cast<std::size_t>(index) * kHotBlock + local;
            const auto first  = local / kSector;
            const auto base   = offset / kSector;
            if (state.writing) {
                std::memcpy(data, state.buffer + done, bytes);
                for (std::size_t s = 0; s < bytes / kSector; ++s) {
                    if (!slot.valid[first + s]) ram_resident_bytes += kSector;
                    if (written[base + s] != 2) ram_dirty_bytes += kSector;
                    if (slot.prefetched[first + s]) prefetch_wasted_bytes += kSector;
                    slot.valid[first + s]      = 1;
                    slot.prefetched[first + s] = 0;
                    checksums[base + s]        = checksum(data + s * kSector);
                    written[base + s]          = 2;
                }
            } else {
                for (std::size_t s = 0; s < bytes / kSector;) {
                    if (!written[base + s]) throw std::runtime_error("HiCache missing payload");
                    if (slot.valid[first + s]) {
                        if (checksum(data + s * kSector) != checksums[base + s])
                            throw std::runtime_error("HiCache corrupt RAM payload");
                        ram_hit_bytes += kSector;
                        if (slot.prefetched[first + s]) {
                            prefetch_hit_bytes += kSector;
                            slot.prefetched[first + s] = 0;
                        }
                        ++s;
                        continue;
                    }
                    auto end = s + 1;
                    while (end < bytes / kSector && !slot.valid[first + end] &&
                           written[base + end] == 1)
                        ++end;
                    if (written[base + s] != 1)
                        throw std::logic_error("HiCache lost dirty RAM owner");
                    disk_io((base + s) * kSector, data + s * kSector, (end - s) * kSector, false);
                    evict_file_pages((base + s) * kSector, (end - s) * kSector);
                    for (; s < end; ++s) {
                        if (checksum(data + s * kSector) != checksums[base + s])
                            throw std::runtime_error("HiCache corrupt disk payload");
                        slot.valid[first + s] = 1;
                        ram_resident_bytes += kSector;
                        ram_miss_bytes += kSector;
                    }
                }
                std::memcpy(state.buffer + done, data, bytes);
            }
            done += bytes;
        }
        // Publish readable bytes and checksums. Sync/eviction batches span
        // demand jobs; their fixed charge is independent of logical capacity.
    }

    void prefetch_block(const Prefetch& hint) {
        std::lock_guard lock(hot_mutex);
        if (block_epochs[hint.block] != hint.epoch) {
            ++prefetch_dropped_jobs;
            return;
        }
        const auto base    = hint.block * kHotSectors;
        const auto sectors = std::min(kHotSectors, written.size() - base);
        // A hint must not replace a dirty/absent sector or evict an entire hot
        // block merely to discover there is no published cold payload.
        const int prior = block_slots[hint.block];
        bool needed     = false;
        for (std::size_t s = 0; s < sectors; ++s) {
            if (written[base + s] == 1 && (prior < 0 || !hot[prior].valid[s])) {
                needed = true;
                break;
            }
        }
        if (!needed) return;
        const int index = acquire_hot(hint.block, false);
        auto& slot      = hot[index];
        auto* data      = ram.get() + static_cast<std::size_t>(index) * kHotBlock;
        for (std::size_t s = 0; s < sectors;) {
            if (slot.valid[s] || written[base + s] != 1) {
                ++s;
                continue;
            }
            auto end = s + 1;
            while (end < sectors && !slot.valid[end] && written[base + end] == 1) ++end;
            disk_io((base + s) * kSector, data + s * kSector, (end - s) * kSector, false);
            evict_file_pages((base + s) * kSector, (end - s) * kSector);
            for (; s < end; ++s) {
                if (checksum(data + s * kSector) != checksums[base + s])
                    throw std::runtime_error("HiCache corrupt prefetched payload");
                slot.valid[s] = slot.prefetched[s] = 1;
                ram_resident_bytes += kSector;
                prefetch_bytes += kSector;
            }
        }
    }

    void background_writeback() {
        std::lock_guard lock(hot_mutex);
        // At most one block between demand jobs. No additional dirty buffers.
        for (std::size_t index = 0; index < next_hot; ++index) {
            const auto before = ram_dirty_bytes.load();
            flush_hot(static_cast<int>(index));
            if (ram_dirty_bytes.load() != before) return;
        }
    }

    void enqueue_callback(const std::shared_ptr<Transfer::State>& state, cudaStream_t target,
                          bool before) {
        auto context    = std::make_unique<Callback>();
        context->owner  = this;
        context->state  = state;
        context->before = before;
        auto* handle    = context.get();
        {
            std::lock_guard lock(mutex);
            handle->position = callbacks.insert(callbacks.end(), std::move(context));
            ++pending_callbacks;
        }
        const auto result = cudaLaunchHostFunc(
            target,
            [](void* pointer) {
                auto* handle = static_cast<Callback*>(pointer);
                std::unique_ptr<Callback> context;
                {
                    std::lock_guard lock(handle->owner->mutex);
                    context = std::move(*handle->position);
                    handle->owner->callbacks.erase(handle->position);
                    --handle->owner->pending_callbacks;
                }
                // The local owner keeps both userdata and State alive while running.
                // The registry owns callbacks CUDA never executes after an error.
                if (context->before)
                    context->state->before();
                else
                    context->state->after();
            },
            handle);
        if (result != cudaSuccess) {
            // A launch can report an earlier asynchronous error. Keep userdata
            // owned until the caller drains/fails the stream, rather than
            // guessing whether CUDA could have accepted the callback.
            FileKVBacking::check_cuda_submission(result);
        }
    }

    void retire_callbacks(const std::shared_ptr<Transfer::State>& state) noexcept {
        std::lock_guard lock(mutex);
        for (auto it = callbacks.begin(); it != callbacks.end();) {
            if ((*it)->state == state) {
                it = callbacks.erase(it);
                --pending_callbacks;
            } else
                ++it;
        }
    }

    void remember_error() noexcept {
        std::lock_guard lock(mutex);
        if (!error) { error = std::current_exception(); }
    }

    void io(const Transfer::State& state) {
        if (ram_capacity) {
            cached_io(state);
            return;
        }
        std::lock_guard lock(hot_mutex);
        if (state.writing) reserve_disk_batch(state.offset, state.bytes);
        disk_io(state.offset, state.buffer, state.bytes, state.writing);
        for (std::size_t i = 0; i < state.bytes; i += kSector) {
            const auto sector = (state.offset + i) / kSector;
            const auto value  = checksum(state.buffer + i);
            if (state.writing) {
                checksums[sector] = value;
                written[sector]   = 1;
            } else if (!written[sector] || checksums[sector] != value) {
                throw std::runtime_error("file KV missing or corrupt payload");
            }
        }
        if (state.writing)
            record_disk_batch(state.offset, state.bytes);
        else
            evict_file_pages(state.offset, state.bytes);
    }

    void run() noexcept {
        for (;;) {
            std::shared_ptr<Transfer::State> state;
            std::optional<Prefetch> hint;
            bool writeback = false, flush = false;
            {
                std::unique_lock lock(mutex);
                auto ready = [&] {
                    return stopping || !queue.empty() || !prefetch_queue.empty() ||
                           (!error && (flush_requested ||
                               (write_through && ram_dirty_bytes.load() != 0)));
                };
                if (!error && filesystem_pending_bytes.load() != 0)
                    flush = !cv.wait_for(lock, kIdleWritebackDelay, ready);
                else
                    cv.wait(lock, ready);
                if (!queue.empty()) {
                    state = std::move(queue.front());
                    queue.pop_front();
                } else if (!prefetch_queue.empty()) {
                    hint = prefetch_queue.front();
                    prefetch_queue.pop_front();
                } else if (!error && write_through && ram_dirty_bytes.load() != 0) {
                    writeback = true;
                    ++pending_writebacks;
                } else if (!error && (flush || flush_requested ||
                                      (stopping && filesystem_pending_bytes.load() != 0))) {
                    flush = true;
                    flush_requested = false;
                    ++pending_writebacks;
                } else if (stopping)
                    return;
            }
            cv.notify_all();
            if (hint || writeback || flush) {
                try {
                    {
                        std::lock_guard lock(mutex);
                        if (error) std::rethrow_exception(error);
                    }
                    if (hint)
                        prefetch_block(*hint);
                    else if (writeback)
                        background_writeback();
                    else {
                        std::lock_guard lock(hot_mutex);
                        flush_disk_batch();
                    }
                } catch (...) { remember_error(); }
                if (hint)
                    --pending_prefetches;
                else
                    --pending_writebacks;
                cv.notify_all();
                continue;
            }
            if (!state) continue;
            const auto waiting = Clock::now();
            state->wait_predecessor();
            if (state->writing) {
                std::unique_lock lock(state->mutex);
                state->cv.wait(lock, [&] { return state->gpu_ready || state->aborted; });
            }
            staging_wait_ns.fetch_add(elapsed(waiting), std::memory_order_relaxed);
            bool aborted;
            {
                std::lock_guard lock(state->mutex);
                aborted = state->aborted;
            }
            const auto started = Clock::now();
            if (!aborted) {
                try {
                    {
                        std::lock_guard lock(mutex);
                        if (error) std::rethrow_exception(error);
                    }
                    io(*state);
                    if (state->writing) {
                        written_bytes += state->bytes;
                        ++writes;
                    } else {
                        read_bytes += state->bytes;
                        ++reads;
                    }
                } catch (...) { remember_error(); }
            }
            if (state->writing) {
                write_ns.fetch_add(elapsed(started), std::memory_order_relaxed);
                --pending_writes;
                {
                    std::lock_guard lock(state->mutex);
                    state->io_complete = true;
                }
                state->finish();
            } else {
                read_ns.fetch_add(elapsed(started), std::memory_order_relaxed);
                --pending_reads;
                {
                    std::lock_guard lock(state->mutex);
                    state->io_ready    = true;
                    state->io_complete = true;
                }
                state->cv.notify_all();
                if (aborted) { state->finish(); }
            }
            cv.notify_all();
        }
    }
};

std::byte* FileKVBacking::Transfer::data() const noexcept { return state_->buffer; }

void FileKVBacking::Transfer::enqueue_before(cudaStream_t stream) {
    state_->owner->enqueue_callback(state_, stream, true);
}

void FileKVBacking::Transfer::enqueue_after(cudaStream_t stream) {
    state_->owner->enqueue_callback(state_, stream, false);
}

void FileKVBacking::Transfer::abort() noexcept {
    {
        std::lock_guard lock(state_->mutex);
        state_->aborted   = true;
        state_->gpu_ready = true;
        state_->io_ready  = true;
    }
    state_->cv.notify_all();
}

void FileKVBacking::Transfer::retire_after_drain() noexcept {
    {
        std::unique_lock lock(state_->mutex);
        state_->cv.wait(lock, [&] { return state_->io_complete; });
    }
    state_->finish();
    state_->owner->retire_callbacks(state_);
}

FileKVBacking::FileKVBacking(const std::filesystem::path& directory, std::size_t capacity_bytes,
                             std::size_t staging_slot_bytes, std::size_t ram_capacity_bytes,
                             bool prefetch, bool write_through)
    : impl_(std::make_unique<Impl>()) {
    if (capacity_bytes == 0 || capacity_bytes % kSector || capacity_bytes > (64ULL << 30) ||
        staging_slot_bytes == 0 || staging_slot_bytes % kSector ||
        staging_slot_bytes > (64ULL << 20) || ram_capacity_bytes > capacity_bytes ||
        ram_capacity_bytes % kHotBlock ||
        ((prefetch || write_through) && ram_capacity_bytes == 0)) {
        throw std::invalid_argument("file KV capacity/staging geometry is invalid");
    }
    std::filesystem::create_directories(directory);
    auto pattern = (directory / "ninfer-kv-XXXXXX").string();
    std::vector<char> filename(pattern.begin(), pattern.end());
    filename.push_back('\0');
    impl_->fd = ::mkstemp(filename.data());
    if (impl_->fd < 0) { io_error("mkstemp"); }
    impl_->path = filename.data();
    try {
        const auto system_page = ::sysconf(_SC_PAGESIZE);
        if (system_page <= 0) {
            throw std::runtime_error("file KV system page size is unavailable");
        }
        impl_->system_page_bytes = static_cast<std::size_t>(system_page);
        if (::fcntl(impl_->fd, F_SETFD, FD_CLOEXEC) < 0) { io_error("fcntl"); }
        // Capacity is a logical bound, not an instruction to extend scratch storage before
        // any payload exists. Positional writes grow the private file on demand. In
        // particular DrvFs can physically allocate a whole ftruncate extent on Windows.
        // Read-ahead is explicitly bounded by the two staging slots. Avoid a
        // second, implicit filesystem read-ahead window.
        const auto advice = ::posix_fadvise(impl_->fd, 0, 0, POSIX_FADV_RANDOM);
        if (advice != 0) {
            throw std::runtime_error(std::string("file KV read-ahead policy: ") +
                                     std::strerror(advice));
        }
        impl_->capacity   = capacity_bytes;
        impl_->slot_bytes = std::min(staging_slot_bytes, capacity_bytes);
        for (auto& slot : impl_->slots) { slot.emplace(impl_->slot_bytes); }
        impl_->checksums.resize(capacity_bytes / kSector);
        impl_->written.resize(capacity_bytes / kSector);
        impl_->ram_capacity     = ram_capacity_bytes;
        impl_->prefetch_enabled = prefetch;
        impl_->write_through    = write_through;
        if (ram_capacity_bytes) {
            impl_->ram = std::make_unique<std::byte[]>(ram_capacity_bytes);
            impl_->hot.resize(ram_capacity_bytes / kHotBlock);
            impl_->free_hot.reserve(impl_->hot.size());
            impl_->block_slots.resize((capacity_bytes + kHotBlock - 1) / kHotBlock, -1);
            impl_->block_epochs.resize(impl_->block_slots.size());
        }
        check_cuda_submission(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking));
        check_cuda_submission(
            cudaEventCreateWithFlags(&impl_->source_ready, cudaEventDisableTiming));
        check_cuda_submission(
            cudaEventCreateWithFlags(&impl_->transfer_done, cudaEventDisableTiming));
        impl_->worker = std::thread([this] { impl_->run(); });
    } catch (...) {
        const auto discarded = ::ftruncate(impl_->fd, 0);
        (void)discarded;
        ::close(impl_->fd);
        impl_->fd = -1;
        if (impl_->transfer_done) cudaEventDestroy(impl_->transfer_done);
        if (impl_->source_ready) cudaEventDestroy(impl_->source_ready);
        if (impl_->stream) cudaStreamDestroy(impl_->stream);
        std::filesystem::remove(impl_->path);
        throw;
    }
}

FileKVBacking::~FileKVBacking() {
    // Model owners drain their streams first; outstanding jobs must have a
    // corresponding completion callback, including on cancellation.
    if (cudaStreamSynchronize(impl_->stream) != cudaSuccess) { abort_after_failed_stream(); }
    drain();
    {
        std::lock_guard lock(impl_->mutex);
        impl_->stopping = true;
    }
    impl_->cv.notify_all();
    impl_->worker.join();
    // This is private scratch storage, with no surviving checkpoint namespace. Release its
    // extents before unlink as well: filesystem observers may retain a handle to a deleted
    // large file on Windows/DrvFs. All stream readers and byte jobs have already retired.
    const auto discarded = ::ftruncate(impl_->fd, 0);
    (void)discarded;
    ::close(impl_->fd);
    cudaEventDestroy(impl_->transfer_done);
    cudaEventDestroy(impl_->source_ready);
    cudaStreamDestroy(impl_->stream);
    std::error_code ignored;
    std::filesystem::remove(impl_->path, ignored);
}

std::size_t FileKVBacking::slot_bytes() const noexcept { return impl_->slot_bytes; }

cudaStream_t FileKVBacking::stream() const noexcept { return impl_->stream; }

void FileKVBacking::order_before(cudaStream_t caller_stream) {
    check_cuda_submission(cudaEventRecord(impl_->source_ready, caller_stream));
    check_cuda_submission(cudaStreamWaitEvent(impl_->stream, impl_->source_ready));
}

void FileKVBacking::order_after(cudaStream_t caller_stream) {
    check_cuda_submission(cudaEventRecord(impl_->transfer_done, impl_->stream));
    check_cuda_submission(cudaStreamWaitEvent(caller_stream, impl_->transfer_done));
}

const std::filesystem::path& FileKVBacking::path() const noexcept { return impl_->path; }

FileKVBacking::Transfer FileKVBacking::submit(std::size_t offset, std::size_t bytes, bool writing) {
    check_errors();
    if (bytes == 0 || offset % kSector || bytes % kSector || bytes > slot_bytes() ||
        offset > impl_->capacity || bytes > impl_->capacity - offset) {
        throw std::out_of_range("file KV transfer exceeds backing/staging range");
    }
    auto state     = std::make_shared<Transfer::State>();
    state->owner   = impl_.get();
    state->offset  = offset;
    state->bytes   = bytes;
    state->writing = writing;
    std::unique_lock lock(impl_->mutex);
    impl_->cv.wait(lock, [&] { return impl_->queue.size() < kMaximumQueuedTransfers; });
    const auto slot    = impl_->next_slot++ % impl_->slots.size();
    state->buffer      = static_cast<std::byte*>(impl_->slots[slot]->data());
    state->predecessor = impl_->tails[slot];
    impl_->queue.push_back(state);
    if (writing) {
        ++impl_->pending_writes;
    } else {
        ++impl_->pending_reads;
    }
    impl_->tails[slot] = state;
    lock.unlock();
    impl_->cv.notify_all();
    return Transfer(std::move(state));
}

FileKVBacking::Transfer FileKVBacking::read(std::size_t offset, std::size_t bytes) {
    return submit(offset, bytes, false);
}

FileKVBacking::Transfer FileKVBacking::write(std::size_t offset, std::size_t bytes) {
    return submit(offset, bytes, true);
}

void FileKVBacking::prefetch(std::size_t offset, std::size_t bytes) {
    if (!impl_->prefetch_enabled || !bytes) return;
    check_errors();
    if (offset > impl_->capacity || bytes > impl_->capacity - offset)
        throw std::out_of_range("HiCache prefetch exceeds backing");
    // Read ahead no further than two transfer chunks, independent of the
    // logical request length. Hints have lower priority than demand transfers.
    const auto end = offset + std::min({bytes, 2 * impl_->slot_bytes, impl_->ram_capacity});
    for (auto block = offset / kHotBlock; block <= (end - 1) / kHotBlock; ++block) {
        std::uint64_t epoch;
        {
            std::lock_guard lock(impl_->hot_mutex);
            epoch = impl_->block_epochs[block];
        }
        std::lock_guard lock(impl_->mutex);
        if (impl_->prefetch_queue.size() == kMaximumPrefetches) {
            ++impl_->prefetch_dropped_jobs;
            break;
        }
        const auto duplicate = std::any_of(
            impl_->prefetch_queue.begin(), impl_->prefetch_queue.end(),
            [&](const auto& hint) { return hint.block == block && hint.epoch == epoch; });
        if (!duplicate) {
            impl_->prefetch_queue.push_back({block, epoch});
            ++impl_->pending_prefetches;
        }
    }
    impl_->cv.notify_all();
}

void FileKVBacking::check_errors() const {
    std::lock_guard lock(impl_->mutex);
    if (impl_->error) { std::rethrow_exception(impl_->error); }
}

void FileKVBacking::check_cuda_submission(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("file KV CUDA submission: ") +
                                 cudaGetErrorString(error));
    }
}

void FileKVBacking::abort_after_failed_stream() noexcept {
    // Synchronization has returned: callbacks have finished or been discarded.
    // No state may be reused or published after this instance is poisoned.
    auto abort_state = [](const std::shared_ptr<Transfer::State>& state) {
        std::shared_ptr<Transfer::State> previous;
        {
            std::lock_guard lock(state->mutex);
            previous         = state->predecessor;
            state->aborted   = true;
            state->gpu_ready = true;
            state->io_ready  = true;
            state->done      = true;
        }
        state->cv.notify_all();
        return previous;
    };
    std::lock_guard lock(impl_->mutex);
    if (!impl_->error) impl_->error = impl_->failed_stream_error;
    for (auto state : impl_->tails) {
        while (state) state = abort_state(state);
    }
    // Also own/release discarded callback userdata even if the worker already
    // retired its predecessor. This prevents a failed-context shared_ptr leak.
    for (const auto& context : impl_->callbacks) (void)abort_state(context->state);
    impl_->callbacks.clear();
    impl_->pending_callbacks = 0;
}

void FileKVBacking::invalidate(std::size_t offset, std::size_t bytes) noexcept {
    if (offset % kSector || bytes % kSector || offset > impl_->capacity ||
        bytes > impl_->capacity - offset) {
        std::terminate();
    }
    std::lock_guard lock(impl_->hot_mutex);
    if (impl_->ram_capacity && bytes) {
        for (auto block = offset / kHotBlock; block <= (offset + bytes - 1) / kHotBlock; ++block) {
            ++impl_->block_epochs[block];
            const int index = impl_->block_slots[block];
            if (index < 0) continue;
            auto& slot       = impl_->hot[index];
            const auto begin = std::max(offset, block * kHotBlock);
            const auto end   = std::min(offset + bytes, (block + 1) * kHotBlock);
            for (auto p = begin; p < end; p += kSector) {
                const auto s = (p % kHotBlock) / kSector;
                if (slot.valid[s]) impl_->ram_resident_bytes -= kSector;
                if (slot.prefetched[s]) impl_->prefetch_wasted_bytes += kSector;
                if (impl_->written[p / kSector] == 2) impl_->ram_dirty_bytes -= kSector;
                slot.valid[s] = slot.prefetched[s] = 0;
            }
            if (std::none_of(slot.valid.begin(), slot.valid.end(),
                             [](std::uint8_t valid) { return valid != 0; })) {
                impl_->unlink_hot(index);
                impl_->block_slots[block] = -1;
                slot.block                = std::numeric_limits<std::size_t>::max();
                impl_->free_hot.push_back(index);
            }
        }
    }
    std::fill_n(impl_->written.begin() + offset / kSector, bytes / kSector, 0);
    impl_->cv.notify_all();
}

void FileKVBacking::drain() {
    for (const auto& tail : impl_->tails) {
        if (tail) tail->wait_done();
    }
    std::unique_lock lock(impl_->mutex);
    impl_->flush_requested = true;
    impl_->cv.notify_all();
    impl_->cv.wait(lock, [&] {
        return (impl_->error || impl_->filesystem_pending_bytes.load() == 0) &&
               impl_->pending_prefetches.load() == 0 && impl_->pending_writebacks.load() == 0 &&
               (!impl_->write_through || impl_->error || impl_->ram_dirty_bytes.load() == 0);
    });
}

FileKVSnapshot FileKVBacking::snapshot() const noexcept {
    return {.read_bytes      = impl_->read_bytes.load(),
            .written_bytes   = impl_->written_bytes.load(),
            .read_ns         = impl_->read_ns.load(),
            .write_ns        = impl_->write_ns.load(),
            .staging_wait_ns = impl_->staging_wait_ns.load(),
            .reads           = impl_->reads.load(),
            .writes          = impl_->writes.load(),
            .pinned_bytes    = impl_->slot_bytes * impl_->slots.size(),
            .integrity_bytes = impl_->checksums.size() * sizeof(std::uint32_t) +
                               impl_->written.size() + impl_->hot.size() * sizeof(Impl::HotSlot) +
                               impl_->free_hot.capacity() * sizeof(int) +
                               impl_->block_slots.size() * sizeof(int) +
                               impl_->block_epochs.size() * sizeof(std::uint64_t),
            .pending_reads         = impl_->pending_reads.load(),
            .pending_writes        = impl_->pending_writes.load(),
            .pending_callbacks     = impl_->pending_callbacks.load(),
            .ram_capacity_bytes    = impl_->ram_capacity,
            .ram_resident_bytes    = impl_->ram_resident_bytes.load(),
            .ram_dirty_bytes       = impl_->ram_dirty_bytes.load(),
            .ram_hit_bytes         = impl_->ram_hit_bytes.load(),
            .ram_miss_bytes        = impl_->ram_miss_bytes.load(),
            .disk_read_bytes       = impl_->disk_read_bytes.load(),
            .disk_written_bytes    = impl_->disk_written_bytes.load(),
            .disk_read_ns          = impl_->disk_read_ns.load(),
            .disk_write_ns         = impl_->disk_write_ns.load(),
            .disk_pwrite_ns        = impl_->disk_pwrite_ns.load(),
            .disk_sync_ns          = impl_->disk_sync_ns.load(),
            .disk_sync_calls       = impl_->disk_sync_calls.load(),
            .disk_high_water_bytes = impl_->disk_high_water_bytes.load(),
            .filesystem_pending_bytes = impl_->filesystem_pending_bytes.load(),
            .filesystem_write_budget_bytes = kFilesystemWriteBudget + impl_->system_page_bytes,
            .ram_evictions         = impl_->ram_evictions.load(),
            .prefetch_bytes        = impl_->prefetch_bytes.load(),
            .prefetch_hit_bytes    = impl_->prefetch_hit_bytes.load(),
            .prefetch_wasted_bytes = impl_->prefetch_wasted_bytes.load(),
            .prefetch_dropped_jobs = impl_->prefetch_dropped_jobs.load(),
            .pending_prefetches    = impl_->pending_prefetches.load(),
            .pending_writebacks    = impl_->pending_writebacks.load()};
}

} // namespace ninfer
