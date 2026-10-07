#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::artifact::detail {

struct DirectReadBlock {
    std::size_t file       = 0;
    std::uint64_t source   = 0;
    std::uint64_t remaining = 0;
    std::size_t request    = 0;
};

struct CompletedRead {
    DirectReadBlock block;
    std::size_t received = 0;
};

// Startup-only CPU reads into caller-owned staging slots. There is at most one
// job per slot and no payload allocation here. The caller waits for the slot's
// upload event before submit; only the caller performs CUDA work.
class BoundedPrefetch {
public:
    using Read = std::function<std::size_t(std::size_t, const DirectReadBlock&)>;

    BoundedPrefetch(std::size_t slot_count, std::size_t workers, Read read)
        : jobs_(slot_count), read_(std::move(read)) {
        if (!slot_count || slot_count > 4 || !workers || workers > slot_count || !read_) {
            throw std::invalid_argument("invalid bounded weight prefetch configuration");
        }
        try {
            threads_.reserve(workers);
            for (std::size_t i = 0; i < workers; ++i) {
                threads_.emplace_back([this] { run(); });
            }
        } catch (...) {
            close(); // std::thread construction may fail after earlier workers start.
            throw;
        }
    }

    ~BoundedPrefetch() { close(); }
    BoundedPrefetch(const BoundedPrefetch&)            = delete;
    BoundedPrefetch& operator=(const BoundedPrefetch&) = delete;

    void submit(std::size_t slot, const DirectReadBlock& block) {
        {
            std::lock_guard lock(mutex_);
            if (stopping_) { throw std::logic_error("weight prefetch is closed"); }
            auto& job = jobs_.at(slot);
            if (job.state != State::Free) {
                throw std::logic_error("weight staging slot already has a read");
            }
            queue_.push_back(slot); // Allocate before changing state: exceptions leave it free.
            job.block = block;
            job.error = nullptr;
            job.state = State::Queued;
        }
        work_.notify_one();
    }

    [[nodiscard]] CompletedRead take(std::size_t slot) {
        std::unique_lock lock(mutex_);
        auto& job = jobs_.at(slot);
        if (job.state == State::Free) { throw std::logic_error("no read for staging slot"); }
        ready_.wait(lock, [&] { return job.state == State::Ready || stopping_; });
        if (stopping_) { throw std::logic_error("weight prefetch cancelled"); }
        const auto result = CompletedRead{job.block, job.received};
        auto error       = job.error;
        job.state        = State::Free;
        lock.unlock();
        if (error) { std::rethrow_exception(error); }
        return result;
    }

    // Drop queued jobs and join active reads before caller-owned buffers disappear.
    // A blocking pread is not forcibly interrupted; it must return before join.
    void close() noexcept {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            queue_.clear();
        }
        work_.notify_all();
        ready_.notify_all();
        for (auto& thread : threads_) {
            if (thread.joinable()) { thread.join(); }
        }
    }

private:
    enum class State { Free, Queued, Reading, Ready };
    struct Job {
        DirectReadBlock block;
        std::size_t received = 0;
        std::exception_ptr error;
        State state = State::Free;
    };

    void run() {
        while (true) {
            std::size_t slot;
            DirectReadBlock block;
            {
                std::unique_lock lock(mutex_);
                work_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
                if (stopping_) { return; }
                slot = queue_.front();
                queue_.pop_front();
                auto& job = jobs_[slot];
                job.state = State::Reading;
                block = job.block;
            }
            std::size_t received = 0;
            std::exception_ptr error;
            try { received = read_(slot, block); }
            catch (...) { error = std::current_exception(); }
            {
                std::lock_guard lock(mutex_);
                auto& job = jobs_[slot];
                job.received = received;
                job.error = error;
                job.state = State::Ready;
            }
            ready_.notify_all();
        }
    }

    std::vector<Job> jobs_;
    Read read_;
    std::mutex mutex_;
    std::condition_variable work_, ready_;
    std::deque<std::size_t> queue_;
    bool stopping_ = false;
    std::vector<std::thread> threads_;
};

} // namespace ninfer::artifact::detail
