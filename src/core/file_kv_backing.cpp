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
    std::thread worker;
    cudaStream_t stream       = nullptr;
    cudaEvent_t source_ready  = nullptr;
    cudaEvent_t transfer_done = nullptr;
    std::atomic<std::uint64_t> read_bytes{0}, written_bytes{0}, read_ns{0}, write_ns{0};
    std::atomic<std::uint64_t> staging_wait_ns{0}, reads{0}, writes{0};
    std::atomic<std::uint64_t> pending_reads{0}, pending_writes{0};
    std::atomic<std::uint64_t> pending_callbacks{0};

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
        std::size_t completed = 0;
        while (completed != state.bytes) {
            const auto count = state.writing
                                   ? ::pwrite(fd, state.buffer + completed, state.bytes - completed,
                                              static_cast<off_t>(state.offset + completed))
                                   : ::pread(fd, state.buffer + completed, state.bytes - completed,
                                             static_cast<off_t>(state.offset + completed));
            if (count < 0) {
                if (errno == EINTR) continue;
                io_error(state.writing ? "pwrite" : "pread");
            }
            if (count == 0) { throw std::runtime_error("file KV short positional IO"); }
            completed += static_cast<std::size_t>(count);
        }
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
        // This is a capacity tier, not an additional unbounded RAM cache. In
        // particular WSL charges the Linux page cache to Windows commit while
        // reporting those pages as reclaimable MemAvailable. Complete writeback
        // before requesting eviction: DONTNEED alone preserves dirty pages.
        // The instance is ephemeral; the flush is for bounded residency rather
        // than a promise of crash-durable checkpoints.
        if (state.writing) {
            while (::fdatasync(fd) != 0) {
                if (errno == EINTR) { continue; }
                io_error("fdatasync");
            }
        }
        const auto page   = system_page_bytes;
        const auto begin  = state.offset / page * page;
        const auto end    = (state.offset + state.bytes + page - 1) / page * page;
        const auto advice = ::posix_fadvise(fd, static_cast<off_t>(begin),
                                            static_cast<off_t>(end - begin), POSIX_FADV_DONTNEED);
        if (advice != 0) {
            throw std::runtime_error(std::string("file KV cache eviction: ") +
                                     std::strerror(advice));
        }
    }

    void run() noexcept {
        for (;;) {
            std::shared_ptr<Transfer::State> state;
            {
                std::unique_lock lock(mutex);
                cv.wait(lock, [&] { return stopping || !queue.empty(); });
                if (queue.empty()) { return; }
                state = std::move(queue.front());
                queue.pop_front();
            }
            cv.notify_all();
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
                             std::size_t staging_slot_bytes)
    : impl_(std::make_unique<Impl>()) {
    if (capacity_bytes == 0 || capacity_bytes % kSector || capacity_bytes > (64ULL << 30) ||
        staging_slot_bytes == 0 || staging_slot_bytes % kSector ||
        staging_slot_bytes > (64ULL << 20)) {
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
        if (::ftruncate(impl_->fd, static_cast<off_t>(capacity_bytes))) { io_error("ftruncate"); }
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
        check_cuda_submission(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking));
        check_cuda_submission(
            cudaEventCreateWithFlags(&impl_->source_ready, cudaEventDisableTiming));
        check_cuda_submission(
            cudaEventCreateWithFlags(&impl_->transfer_done, cudaEventDisableTiming));
        impl_->worker = std::thread([this] { impl_->run(); });
    } catch (...) {
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
    std::fill_n(impl_->written.begin() + offset / kSector, bytes / kSector, 0);
}

void FileKVBacking::drain() {
    for (const auto& tail : impl_->tails) {
        if (tail) tail->wait_done();
    }
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
            .integrity_bytes =
                impl_->checksums.size() * sizeof(std::uint32_t) + impl_->written.size(),
            .pending_reads     = impl_->pending_reads.load(),
            .pending_writes    = impl_->pending_writes.load(),
            .pending_callbacks = impl_->pending_callbacks.load()};
}

} // namespace ninfer
