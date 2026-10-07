#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <csignal>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
bool discard_callbacks          = false;
bool fail_callback_submission   = false;
bool fail_copy_submission       = false;
cudaStream_t failed_sync_stream = nullptr;
} // namespace

extern "C" cudaError_t CUDARTAPI __real_cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t, void*);

extern "C" cudaError_t CUDARTAPI __wrap_cudaLaunchHostFunc(cudaStream_t stream,
                                                           cudaHostFn_t function, void* userdata) {
    // Simulate CUDA accepting callbacks and then discarding them after a context
    // fault. The desktop GPU itself remains usable by the subsequent tests.
    if (discard_callbacks) return cudaSuccess;
    if (fail_callback_submission) return cudaErrorUnknown;
    return __real_cudaLaunchHostFunc(stream, function, userdata);
}

extern "C" cudaError_t CUDARTAPI __real_cudaMemcpy2DAsync(void*, std::size_t, const void*,
                                                          std::size_t, std::size_t, std::size_t,
                                                          cudaMemcpyKind, cudaStream_t);

extern "C" cudaError_t CUDARTAPI __wrap_cudaMemcpy2DAsync(void* destination, std::size_t dpitch,
                                                          const void* source, std::size_t spitch,
                                                          std::size_t width, std::size_t height,
                                                          cudaMemcpyKind kind,
                                                          cudaStream_t stream) {
    if (fail_copy_submission) return cudaErrorUnknown;
    return __real_cudaMemcpy2DAsync(destination, dpitch, source, spitch, width, height, kind,
                                    stream);
}

extern "C" cudaError_t CUDARTAPI __real_cudaStreamSynchronize(cudaStream_t);

extern "C" cudaError_t CUDARTAPI __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    if (stream && stream == failed_sync_stream) return cudaErrorUnknown;
    return __real_cudaStreamSynchronize(stream);
}

namespace {
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void require_small_file_cache(const std::filesystem::path& path,
                              std::size_t maximum_bytes = 8 * 4096) {
    // mincore observes residency without faulting the file into RAM. This
    // independently checks that successful writes/reads requested clean-page
    // eviction on the test filesystem, not merely that staging is bounded.
    const auto page = ::sysconf(_SC_PAGESIZE);
    require(page > 0, "system page size unavailable");
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(fd >= 0, "could not open residency fixture");

    struct stat info {};

    const auto stated = ::fstat(fd, &info);
    if (stated != 0) {
        ::close(fd);
        require(false, "could not stat residency fixture");
    }
    const auto size = static_cast<std::size_t>(info.st_size);
    auto* mapping   = ::mmap(nullptr, size, PROT_NONE, MAP_PRIVATE, fd, 0);
    ::close(fd);
    require(mapping != MAP_FAILED, "could not map residency fixture");
    std::vector<unsigned char> residency((size + page - 1) / page);
    const auto queried = ::mincore(mapping, size, residency.data());
    ::munmap(mapping, size);
    require(queried == 0, "could not query file residency");
    const auto cached = std::count_if(residency.begin(), residency.end(),
                                      [](unsigned char value) { return value & 1; });
    require(cached * page <= maximum_bytes, "file tier retained an unbounded kernel page cache");
}

std::vector<ninfer::DeviceKVPageLease> allocate(ninfer::DeviceKVPagePool& pool, unsigned pages) {
    auto reservation = pool.reserve(pages);
    require(reservation.has_value(), "physical reservation failed");
    std::vector<ninfer::DeviceKVPageLease> result;
    result.reserve(pages);
    pool.materialize(*reservation, pages, result);
    return result;
}

void roundtrip(ninfer::DeviceContext& context, ninfer::PagedKVPlaneOrder order, bool file = true,
               std::size_t hot_bytes = 0) {
    constexpr unsigned pages = 97;
    const ninfer::KVPageGeometry geometry{.device_plane_order = order,
                                          .planes = {{ninfer::DType::FP8_E4M3FN, 256, 2, 256},
                                                     {ninfer::DType::U8, 128, 2, 256},
                                                     {ninfer::DType::FP16, 1, 2, 256},
                                                     {ninfer::DType::U8, 16, 2, 256}}};
    ninfer::LayoutBuilder builder;
    const auto layout = ninfer::plan_device_kv_page_pool(
        builder, {.page_group_count = pages, .geometry = geometry});
    ninfer::DeviceBuffer source_storage(builder.finish(256));
    ninfer::DeviceBuffer destination_storage(source_storage.bytes);
    ninfer::DeviceKVPagePool source({source_storage.p, source_storage.bytes}, layout);
    ninfer::DeviceKVPagePool destination({destination_storage.p, destination_storage.bytes},
                                         layout);
    auto sources      = allocate(source, pages);
    auto destinations = allocate(destination, pages);
    std::vector<ninfer::DeviceKVPageHandle> input, output;
    for (unsigned i = 0; i < pages; ++i) {
        input.push_back(sources[(i * 37) % pages].handle());
        output.push_back(destinations[i].handle());
    }
    std::vector<std::vector<std::byte>> oracle;
    for (unsigned plane = 0; plane < source.plane_count(); ++plane) {
        const auto& tensor = source.plane(plane);
        auto& bytes        = oracle.emplace_back(tensor.bytes());
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = std::byte((i * 29 + (i >> 13) * 17 + plane * 61) & 255);
        }
        CUDA_CHECK(cudaMemcpyAsync(tensor.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice,
                                   context.stream));
    }
    context.synchronize();
    const auto host_layout = ninfer::plan_host_kv_page_layout(geometry);
    const std::array layouts{host_layout};
    std::filesystem::path file_path;
    {
        // Small slots force repeated slot reuse across fragmented copies.
        ninfer::HostContextArena ledger(host_layout.page_stride * pages * 2, 256,
                                        file ? std::optional<std::size_t>(0) : std::nullopt);
        ninfer::HostKVArena arena(ledger, layouts,
                                  file ? std::filesystem::temp_directory_path()
                                       : std::filesystem::path{},
                                  512ULL << 10, hot_bytes);
        auto stored = arena.allocate(host_layout, pages);
        require(stored.has_value(), "file allocation failed");
        auto view = arena.writable_view(*stored);
        require(file == (view.data() == nullptr && view.file_backing()),
                "Host backing mode differs from requested storage");
        if (file) { file_path = view.file_backing()->path(); }
        source.copy_to_host(input, view, context.transfer_stream);
        CUDA_CHECK(cudaStreamSynchronize(context.transfer_stream));
        arena.check_io_errors();
        if (file) { view.file_backing()->drain(); require_small_file_cache(file_path); }
        destination.zero_pages(output, context.stream);
        const auto transfer_work =
            destination.copy_from_host(arena.view(*stored), output, context.stream);
        require(transfer_work.payload_bytes > 0 && transfer_work.copy_operations > 0,
                "coalesced restore must report actual payload and operations");
        context.synchronize();
        arena.check_io_errors();
        if (file) { view.file_backing()->drain(); require_small_file_cache(file_path); }
        for (unsigned p = 0; p < source.plane_count(); ++p) {
            const auto& tensor = destination.plane(p);
            std::vector<std::byte> actual(tensor.bytes());
            CUDA_CHECK(
                cudaMemcpy(actual.data(), tensor.data, actual.size(), cudaMemcpyDeviceToHost));
            const auto& plane = host_layout.planes[p];
            for (unsigned page = 0; page < pages; ++page) {
                const auto original = (page * 37) % pages;
                for (int head = 0; head < geometry.planes[p].head_extent; ++head) {
                    const auto offset = [&](unsigned index) {
                        return order == ninfer::PagedKVPlaneOrder::PageMajor
                                   ? index * tensor.nb[3] + head * plane.head_payload_bytes
                                   : head * tensor.nb[3] + index * tensor.nb[2];
                    };
                    require(std::memcmp(actual.data() + offset(page),
                                        oracle[p].data() + offset(original),
                                        plane.head_payload_bytes) == 0,
                            "file restore changed device payload bytes");
                }
            }
        }
        if (!file) { return; }
        const auto stats = arena.file_snapshot();
        require(stats.reads > 2 && stats.writes > 2 &&
                    stats.read_bytes == host_layout.page_stride * pages &&
                    stats.written_bytes == stats.read_bytes && stats.pinned_bytes == (1ULL << 20),
                "file IO/staging accounting is not bounded or exact");
        // Corruption must complete all callbacks, then fail before publication.
        const int fd = ::open(file_path.c_str(), O_RDWR);
        require(fd >= 0, "failed to open corruption fixture");
        const std::byte corrupt{0xff};
        require(::pwrite(fd, &corrupt, 1, view.file_offset()) == 1, "corruption injection failed");
        ::close(fd);
        destination.copy_from_host(arena.view(*stored).subview(0, 1), std::span(output).first(1),
                                   context.transfer_stream);
        CUDA_CHECK(cudaStreamSynchronize(context.transfer_stream));
        bool failed = false;
        try {
            arena.check_io_errors();
        } catch (const std::runtime_error&) { failed = true; }
        require(failed, "corrupt file payload was accepted");
    }
    require(!std::filesystem::exists(file_path), "instance-local backing file leaked");
}

void stale_and_short(ninfer::DeviceContext& context, bool truncate_file) {
    const ninfer::KVPageGeometry geometry{.planes = {{ninfer::DType::I8, 8, 2, 256}}};
    ninfer::LayoutBuilder builder;
    const auto layout =
        ninfer::plan_device_kv_page_pool(builder, {.page_group_count = 1, .geometry = geometry});
    ninfer::DeviceBuffer storage(builder.finish(256));
    ninfer::DeviceKVPagePool pool({storage.p, storage.bytes}, layout);
    auto pages = allocate(pool, 1);
    const std::array handles{pages[0].handle()};
    const auto host_layout = ninfer::plan_host_kv_page_layout(geometry);
    const std::array layouts{host_layout};
    bool small_slot = false;
    try {
        ninfer::HostContextArena ledger(host_layout.page_stride, 256, 0);
        ninfer::HostKVArena invalid(ledger, layouts, std::filesystem::temp_directory_path(), 256);
    } catch (const std::invalid_argument&) { small_slot = true; }
    require(small_slot, "oversized page accepted by staging geometry");
    ninfer::HostContextArena ledger(host_layout.page_stride, 256, 0);
    ninfer::HostKVArena arena(ledger, layouts, std::filesystem::temp_directory_path(),
                              host_layout.page_stride);
    auto stored = arena.allocate(host_layout, 1);
    pool.zero_pages(handles, context.stream);
    pool.copy_to_host(handles, arena.writable_view(*stored), context.stream);
    context.synchronize();
    arena.check_io_errors();
    if (truncate_file) {
        const auto file = arena.view(*stored).file_backing()->path();
        require(::truncate(file.c_str(), 0) == 0, "short-read injection failed");
    } else {
        const auto stale = arena.view(*stored);
        stored->release();
        require(!stale.valid(), "released generation retained a valid view");
        stored = arena.allocate(host_layout, 1);
    }
    pool.copy_from_host(arena.view(*stored), handles, context.stream);
    context.synchronize();
    bool failed = false;
    try {
        arena.check_io_errors();
    } catch (const std::runtime_error&) { failed = true; }
    require(failed, "short/unwritten replacement payload was accepted");
}

void failed_submission_and_write(ninfer::DeviceContext& context) {
    // A read submitted without its trailing CUDA callback must not strand its
    // staging slot or destruction. Cancellation drains before slot retirement.
    ninfer::FileKVBacking file(std::filesystem::temp_directory_path(), 4096, 256);
    auto write = file.write(0, 256);
    write.enqueue_before(file.stream());
    write.enqueue_after(file.stream());
    CUDA_CHECK(cudaStreamSynchronize(file.stream()));
    file.check_errors();
    auto abandoned = file.read(0, 256);
    abandoned.enqueue_before(file.stream());
    abandoned.abort();
    CUDA_CHECK(cudaStreamSynchronize(file.stream()));
    abandoned.retire_after_drain();
    auto next = file.read(0, 256);
    next.enqueue_before(file.stream());
    next.enqueue_after(file.stream());
    CUDA_CHECK(cudaStreamSynchronize(file.stream()));
    file.check_errors();

    auto rejected_callback   = file.read(0, 256);
    fail_callback_submission = true;
    bool submission_failed   = false;
    try {
        rejected_callback.enqueue_before(file.stream());
    } catch (const std::runtime_error&) { submission_failed = true; }
    fail_callback_submission = false;
    rejected_callback.abort();
    CUDA_CHECK(cudaStreamSynchronize(file.stream()));
    rejected_callback.retire_after_drain();
    require(submission_failed && file.snapshot().pending_callbacks == 0,
            "failed callback submission retained its userdata after stream drain");

    struct rlimit prior {};

    require(file.snapshot().pending_reads == 0 && file.snapshot().pending_writes == 0,
            "drained normal and abandoned jobs retained pending IO claims");
    require(::getrlimit(RLIMIT_FSIZE, &prior) == 0, "could not read file size limit");
    const auto signal = std::signal(SIGXFSZ, SIG_IGN);

    const struct rlimit zero {
        0, prior.rlim_max
    };

    require(::setrlimit(RLIMIT_FSIZE, &zero) == 0, "could not inject write failure");
    auto failure = file.write(256, 256);
    failure.enqueue_before(file.stream());
    failure.enqueue_after(file.stream());
    const auto result   = cudaStreamSynchronize(file.stream());
    const auto restored = ::setrlimit(RLIMIT_FSIZE, &prior);
    std::signal(SIGXFSZ, signal);
    CUDA_CHECK(result);
    require(restored == 0, "file size limit was not restored");
    bool failed = false;
    try {
        file.check_errors();
    } catch (const std::runtime_error&) { failed = true; }
    require(failed, "failed write was published as ready");
    require(file.snapshot().pending_reads == 0 && file.snapshot().pending_writes == 0,
            "failed write retained a pending IO claim");
}

void discarded_callback_chain() {
    std::filesystem::path path;
    {
        ninfer::FileKVBacking file(std::filesystem::temp_directory_path(), 4096, 256);
        path = file.path();

        struct FaultCleanup {
            ninfer::FileKVBacking& file;

            ~FaultCleanup() {
                discard_callbacks = false;
                (void)cudaStreamSynchronize(file.stream());
                file.abort_after_failed_stream();
            }
        } cleanup{file};

        discard_callbacks = true;
        std::vector<ninfer::FileKVBacking::Transfer> transfers;
        for (unsigned i = 0; i < 8; ++i) {
            auto transfer = i % 2 ? file.read(i * 256, 256) : file.write(i * 256, 256);
            transfer.enqueue_before(file.stream());
            transfer.enqueue_after(file.stream());
            transfers.push_back(std::move(transfer));
        }
        discard_callbacks = false;
        CUDA_CHECK(cudaStreamSynchronize(file.stream()));
        require(file.snapshot().pending_callbacks == 16, "discarded callback ownership was lost");
        // Retiring only the last transfer used to deadlock behind a predecessor
        // whose callback would never execute. Poison the whole chain first.
        transfers.back().abort();
        file.abort_after_failed_stream();
        transfers.back().retire_after_drain();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (file.snapshot().pending_reads || file.snapshot().pending_writes) {
            require(std::chrono::steady_clock::now() < deadline,
                    "discarded callback chain stranded the IO worker");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require(file.snapshot().pending_callbacks == 0,
                "discarded callback userdata was not released");
        for (bool writing : {false, true}) {
            bool rejected = false;
            try {
                if (writing)
                    (void)file.write(0, 256);
                else
                    (void)file.read(0, 256);
            } catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "poisoned file instance accepted new transfers");
        }
        file.abort_after_failed_stream(); // cleanup is idempotent
    }
    require(!std::filesystem::exists(path), "failed-stream backing file leaked");
}

void pool_failed_stream(ninfer::DeviceContext& context, bool writing) {
    const ninfer::KVPageGeometry geometry{.planes = {{ninfer::DType::I8, 8, 2, 256}}};
    ninfer::LayoutBuilder builder;
    const auto layout =
        ninfer::plan_device_kv_page_pool(builder, {.page_group_count = 1, .geometry = geometry});
    ninfer::DeviceBuffer storage(builder.finish(256));
    ninfer::DeviceKVPagePool pool({storage.p, storage.bytes}, layout);
    auto pages = allocate(pool, 1);
    const std::array handles{pages[0].handle()};
    pool.zero_pages(handles, context.stream);
    context.synchronize();
    const auto host_layout = ninfer::plan_host_kv_page_layout(geometry);
    const std::array layouts{host_layout};
    std::filesystem::path path;
    {
        ninfer::HostContextArena ledger(host_layout.page_stride, 256, 0);
        ninfer::HostKVArena arena(ledger, layouts, std::filesystem::temp_directory_path(),
                                  host_layout.page_stride);
        auto stored = arena.allocate(host_layout, 1);
        auto* file  = arena.writable_view(*stored).file_backing();
        path        = file->path();

        struct FaultCleanup {
            ninfer::FileKVBacking& file;

            ~FaultCleanup() {
                discard_callbacks    = false;
                fail_copy_submission = false;
                failed_sync_stream   = nullptr;
                (void)cudaStreamSynchronize(file.stream());
                file.abort_after_failed_stream();
            }
        } cleanup{*file};

        discard_callbacks = true;
        // Both staging slots have a predecessor whose accepted callbacks are
        // missing. Exercise the actual pool catch path, not just the helper.
        for (unsigned i = 0; i < 2; ++i) {
            auto prior = file->write(i * 256, 256);
            prior.enqueue_before(file->stream());
            prior.enqueue_after(file->stream());
        }
        fail_copy_submission = true;
        failed_sync_stream   = file->stream();
        bool rejected        = false;
        try {
            if (writing)
                pool.copy_to_host(handles, arena.writable_view(*stored), context.stream);
            else
                pool.copy_from_host(arena.view(*stored), handles, context.stream);
        } catch (const std::runtime_error&) { rejected = true; }
        fail_copy_submission = false;
        failed_sync_stream   = nullptr;
        discard_callbacks    = false;
        CUDA_CHECK(cudaStreamSynchronize(file->stream()));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (file->snapshot().pending_reads || file->snapshot().pending_writes) {
            require(std::chrono::steady_clock::now() < deadline,
                    "pool failed-stream catch stranded a predecessor");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require(rejected && file->snapshot().pending_callbacks == 0,
                "pool failed-stream catch did not release callbacks");
        bool poisoned = false;
        try {
            arena.check_io_errors();
        } catch (const std::runtime_error&) { poisoned = true; }
        require(poisoned, "failed-stream pool payload could be published ready");
    }
    require(!std::filesystem::exists(path), "pool failed-stream file leaked");
}

} // namespace

namespace {
void partial_promotion_error_counters() {
    constexpr std::size_t block = 1ULL << 20, bytes = 512;
    ninfer::FileKVBacking store(std::filesystem::temp_directory_path(), 2 * block,
                                bytes, block);
    ninfer::DeviceBuffer payload(bytes);
    CUDA_CHECK(cudaMemset(payload.p, 0x5a, bytes));
    for (const auto offset : {std::size_t{0}, block}) {
        auto writing = store.write(offset, bytes);
        writing.enqueue_before(store.stream());
        CUDA_CHECK(cudaMemcpyAsync(writing.data(), payload.p, bytes, cudaMemcpyDeviceToHost,
                                   store.stream()));
        writing.enqueue_after(store.stream());
        CUDA_CHECK(cudaStreamSynchronize(store.stream()));
        store.check_errors();
    }
    store.drain();
    const int fd = ::open(store.path().c_str(), O_RDWR | O_CLOEXEC);
    require(fd >= 0, "partial-promotion corruption fixture could not open");
    const std::byte corrupt{0xff};
    const auto changed = ::pwrite(fd, &corrupt, 1, 256);
    ::close(fd);
    require(changed == 1, "partial-promotion corruption injection failed");
    const auto before = store.snapshot();
    auto reading = store.read(0, bytes);
    reading.enqueue_before(store.stream());
    reading.enqueue_after(store.stream());
    CUDA_CHECK(cudaStreamSynchronize(store.stream()));
    bool rejected = false;
    try {
        store.check_errors();
    } catch (const std::runtime_error&) { rejected = true; }
    const auto after = store.snapshot();
    require(rejected && after.ram_miss_bytes - before.ram_miss_bytes == 256 &&
                after.ram_resident_bytes == 256 && after.ram_dirty_bytes == 0 &&
                after.disk_read_bytes - before.disk_read_bytes == bytes &&
                after.pending_reads == 0 && after.pending_callbacks == 0,
            "partial promotion lost completed-sector counters or accepted corruption");
}

void hierarchical_cache(ninfer::DeviceContext& context, bool write_through) {
    constexpr std::size_t block = 1ULL << 20;
    ninfer::DeviceBuffer device(block);
    std::vector<std::byte> oracle(block), actual(block);
    for (std::size_t i = 0; i < block; ++i) oracle[i] = std::byte((i * 31 + (i >> 11) * 17) & 255);
    CUDA_CHECK(cudaMemcpy(device.p, oracle.data(), block, cudaMemcpyHostToDevice));
    std::filesystem::path path;
    {
        ninfer::FileKVBacking store(std::filesystem::temp_directory_path(), 8 * block, block,
                                    2 * block, true, write_through);
        path       = store.path();
        auto write = [&](std::size_t offset, std::size_t bytes = 1ULL << 20) {
            auto transfer = store.write(offset, bytes);
            transfer.enqueue_before(store.stream());
            CUDA_CHECK(cudaMemcpyAsync(transfer.data(), device.p, bytes, cudaMemcpyDeviceToHost,
                                       store.stream()));
            transfer.enqueue_after(store.stream());
            CUDA_CHECK(cudaStreamSynchronize(store.stream()));
            store.check_errors();
        };
        auto read = [&](std::size_t offset, std::size_t bytes = 1ULL << 20) {
            auto transfer = store.read(offset, bytes);
            transfer.enqueue_before(store.stream());
            CUDA_CHECK(cudaMemcpyAsync(device.p, transfer.data(), bytes, cudaMemcpyHostToDevice,
                                       store.stream()));
            transfer.enqueue_after(store.stream());
            CUDA_CHECK(cudaStreamSynchronize(store.stream()));
            store.check_errors();
            CUDA_CHECK(cudaMemcpy(actual.data(), device.p, bytes, cudaMemcpyDeviceToHost));
            require(std::equal(oracle.begin(), oracle.begin() + bytes, actual.begin()),
                    "HiCache RAM/disk promotion differs from exact byte oracle");
        };
        for (unsigned i = 0; i < 6; ++i) {
            write(i * block);
            if (!write_through && i == 0) {
                store.drain();
                require(store.snapshot().disk_written_bytes == 0 &&
                            store.snapshot().ram_dirty_bytes == block,
                        "write-back performed eager SSD writes or lost the RAM owner");
            }
        }
        store.drain();
        require(store.snapshot().ram_evictions >= 4 &&
                    store.snapshot().ram_resident_bytes <= 2 * block,
                "HiCache did not bound the RAM tier");
        auto before = store.snapshot();
        read(0);
        require(store.snapshot().disk_read_bytes - before.disk_read_bytes == block,
                "cold HiCache restore did not read exactly the disk payload");
        before = store.snapshot();
        read(0);
        require(store.snapshot().disk_read_bytes == before.disk_read_bytes &&
                    store.snapshot().ram_hit_bytes - before.ram_hit_bytes == block,
                "warm HiCache restore touched disk or missed managed RAM");
        // Streaming writes must not flush out a repeatedly read prefix.
        write(6 * block);
        write(7 * block);
        write(5 * block);
        before = store.snapshot();
        read(0);
        require(store.snapshot().disk_read_bytes == before.disk_read_bytes,
                "streaming writes evicted the protected read prefix");
        store.prefetch(2 * block, block);
        store.drain();
        before = store.snapshot();
        read(2 * block);
        require(store.snapshot().prefetch_hit_bytes - before.prefetch_hit_bytes == block &&
                    store.snapshot().disk_read_bytes == before.disk_read_bytes,
                "prefetched prefix was not consumed from RAM");
        // Free/reallocate a subrange sharing a cache block. No neighbour loss,
        // no stale bytes and no queued hint can resurrect the old generation.
        store.invalidate(2 * block + 256, 256);
        write(2 * block + 256, 256);
        store.drain();
        read(2 * block, 256);
        read(2 * block + 256, 256);
        store.drain();
        // Completely freed protected blocks must become reusable hot slots.
        // Otherwise streaming writes evict each other while an empty protected
        // slot permanently reduces the usable RAM tier.
        store.invalidate(0, 8 * block);
        before = store.snapshot();
        require(before.ram_resident_bytes == 0 && before.ram_dirty_bytes == 0,
                "freed HiCache allocations retained physical hot ownership");
        write(6 * block);
        write(7 * block);
        store.drain();
        if (!write_through)
            require(store.snapshot().disk_written_bytes == before.disk_written_bytes,
                    "empty protected slots forced unnecessary dirty eviction");
        read(6 * block);
        read(7 * block);
        const auto final = store.snapshot();
        require((!write_through || final.ram_dirty_bytes == 0) &&
                    final.ram_dirty_bytes <= 2 * block && final.pending_callbacks == 0 &&
                    final.pending_prefetches == 0 && final.pending_writebacks == 0 &&
                    final.pending_reads == 0 && final.pending_writes == 0,
                "HiCache did not drain transfers/writeback/prefetch leases");
        require_small_file_cache(path);
        std::cout << "HiCache exact byte oracle / protected hot set / disk promotion / prefetch / "
                     "generations passed\n";
    }
    require(!std::filesystem::exists(path), "HiCache file was not removed");
}

void large_fragmented_hot_eviction(ninfer::DeviceContext& context) {
    constexpr std::size_t bytes = 64ULL << 20;
    constexpr std::size_t offset = 256;
    // A sector-aligned transfer spans 65 hot blocks and exceeds the one-block
    // RAM tier. Check every byte and kernel residency at both completion boundaries.
    ninfer::FileKVBacking store(std::filesystem::temp_directory_path(), 2 * bytes,
                                bytes, 1ULL << 20, false, false);
    std::vector<std::byte> expected(bytes), observed(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        expected[i] = static_cast<std::byte>((i * 17 + (i >> 8)) & 255);
    }
    ninfer::DeviceBuffer payload(bytes);
    CUDA_CHECK(cudaMemcpyAsync(payload.p, expected.data(), bytes, cudaMemcpyHostToDevice,
                               context.stream));
    context.synchronize();
    auto writing = store.write(offset, bytes);
    writing.enqueue_before(store.stream());
    CUDA_CHECK(cudaMemcpyAsync(writing.data(), payload.p, bytes, cudaMemcpyDeviceToHost,
                               store.stream()));
    writing.enqueue_after(store.stream());
    CUDA_CHECK(cudaStreamSynchronize(store.stream()));
    store.check_errors();
    store.drain();
    require_small_file_cache(store.path());
    auto reading = store.read(offset, bytes);
    reading.enqueue_before(store.stream());
    CUDA_CHECK(cudaMemcpyAsync(payload.p, reading.data(), bytes, cudaMemcpyHostToDevice,
                               store.stream()));
    reading.enqueue_after(store.stream());
    CUDA_CHECK(cudaStreamSynchronize(store.stream()));
    store.check_errors();
    store.drain();
    require_small_file_cache(store.path());
    CUDA_CHECK(cudaMemcpy(observed.data(), payload.p, bytes, cudaMemcpyDeviceToHost));
    require(observed == expected, "large fragmented eviction changed bytes");
    const auto stats = store.snapshot();
    require(stats.ram_resident_bytes <= (1ULL << 20) && stats.pending_reads == 0 &&
                stats.pending_writes == 0 && stats.pending_callbacks == 0,
            "large fragmented eviction retained work or exceeded RAM quota");
    require(stats.disk_pwrite_ns > 0 && stats.disk_sync_ns > 0 && stats.disk_sync_calls > 0 &&
                stats.disk_write_ns == stats.disk_pwrite_ns + stats.disk_sync_ns &&
                stats.disk_sync_calls < stats.writes + 65 &&
                stats.disk_high_water_bytes <= offset + bytes,
            "bounded disk batches lost IO attribution or exceeded actual payload extent");
}

// Exact public transfer oracle across many small demand jobs. File residency
// stays bounded at CUDA completion, and explicit drain still requests complete
// sync/eviction. This catches the previous per-job sync throughput regression.
void cross_job_writeback(ninfer::DeviceContext& context) {
    constexpr std::size_t chunk = 1ULL << 20, count = 80;
    ninfer::FileKVBacking store(std::filesystem::temp_directory_path(), count * chunk,
                                chunk);
    ninfer::DeviceBuffer payload(chunk);
    std::vector<std::byte> expected(chunk), observed(chunk);
    for (std::size_t job = 0; job < count; ++job) {
        for (std::size_t i = 0; i < chunk; ++i)
            expected[i] = static_cast<std::byte>((i * 17 + job * 23) & 255);
        CUDA_CHECK(cudaMemcpyAsync(payload.p, expected.data(), chunk, cudaMemcpyHostToDevice,
                                   context.stream));
        context.synchronize();
        auto writing = store.write(job * chunk, chunk);
        writing.enqueue_before(store.stream());
        CUDA_CHECK(cudaMemcpyAsync(writing.data(), payload.p, chunk, cudaMemcpyDeviceToHost,
                                   store.stream()));
        writing.enqueue_after(store.stream());
        CUDA_CHECK(cudaStreamSynchronize(store.stream()));
        store.check_errors();
        const auto stats = store.snapshot();
        require(stats.filesystem_pending_bytes <= stats.filesystem_write_budget_bytes,
                "cross-job writeback exceeded its fixed filesystem budget");
        require_small_file_cache(store.path(), stats.filesystem_write_budget_bytes + 8 * 4096);
    }
    store.drain();
    store.check_errors();
    require_small_file_cache(store.path());
    const auto written = store.snapshot();
    require(written.filesystem_pending_bytes == 0 && written.pending_writebacks == 0 &&
                written.disk_sync_calls < count / 2 && written.disk_sync_calls > 0,
            "small demand jobs were not batched or did not settle");
    for (std::size_t job = 0; job < count; ++job) {
        auto reading = store.read(job * chunk, chunk);
        reading.enqueue_before(store.stream());
        CUDA_CHECK(cudaMemcpyAsync(payload.p, reading.data(), chunk, cudaMemcpyHostToDevice,
                                   store.stream()));
        reading.enqueue_after(store.stream());
        CUDA_CHECK(cudaStreamSynchronize(store.stream()));
        store.check_errors();
        CUDA_CHECK(cudaMemcpy(observed.data(), payload.p, chunk, cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < chunk; ++i)
            require(observed[i] == static_cast<std::byte>((i * 17 + job * 23) & 255),
                    "cross-job writeback changed payload bytes");
    }
    store.drain();
    require_small_file_cache(store.path());
    std::cout << "cross-job bounded filesystem batching / exact bytes / drain passed\n";
}

void cleanup_with_external_reader(ninfer::DeviceContext& context) {
    int observer = -1;
    std::filesystem::path path;
    {
        ninfer::FileKVBacking store(std::filesystem::temp_directory_path(), 16ULL << 20,
                                    1ULL << 20);
        path     = store.path();
        observer = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        require(observer >= 0, "cleanup fixture observer could not open the scratch file");

        struct stat initial {};

        require(::fstat(observer, &initial) == 0 && initial.st_size == 0,
                "logical capacity eagerly extended an empty scratch file");
        ninfer::DeviceBuffer payload(4096);
        CUDA_CHECK(cudaMemsetAsync(payload.p, 0x5a, payload.bytes, context.stream));
        context.synchronize();
        auto writing = store.write(0, payload.bytes);
        writing.enqueue_before(store.stream());
        CUDA_CHECK(cudaMemcpyAsync(writing.data(), payload.p, payload.bytes, cudaMemcpyDeviceToHost,
                                   store.stream()));
        writing.enqueue_after(store.stream());
        CUDA_CHECK(cudaStreamSynchronize(store.stream()));
        store.check_errors();
        require(::fstat(observer, &initial) == 0 && initial.st_size == 4096,
                "on-demand write grew beyond its actual payload extent");
    }

    struct stat info {};

    const int stated = ::fstat(observer, &info);
    ::close(observer);
    require(stated == 0 && info.st_size == 0 && !std::filesystem::exists(path),
            "scratch teardown retained payload extents through an external reader");
}
} // namespace

int main() {
    try {
        int devices       = 0;
        const auto result = cudaGetDeviceCount(&devices);
        if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver || devices == 0) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(result);
        ninfer::DeviceContext context;
        partial_promotion_error_counters();
        cleanup_with_external_reader(context);
        cross_job_writeback(context);
        large_fragmented_hot_eviction(context);
        hierarchical_cache(context, true);
        hierarchical_cache(context, false);
        roundtrip(context, ninfer::PagedKVPlaneOrder::PageMajor);
        roundtrip(context, ninfer::PagedKVPlaneOrder::HeadMajor);
        roundtrip(context, ninfer::PagedKVPlaneOrder::PageMajor, true, 2ULL << 20);
        roundtrip(context, ninfer::PagedKVPlaneOrder::HeadMajor, true, 2ULL << 20);
        roundtrip(context, ninfer::PagedKVPlaneOrder::PageMajor, false);
        roundtrip(context, ninfer::PagedKVPlaneOrder::HeadMajor, false);
        stale_and_short(context, false);
        stale_and_short(context, true);
        failed_submission_and_write(context);
        discarded_callback_chain();
        pool_failed_stream(context, false);
        pool_failed_stream(context, true);
        std::cout << "file KV fragmented transfer, slot reuse, corruption, short reads, "
                     "generations, discarded callback chains and cleanup passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
