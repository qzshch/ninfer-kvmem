#include "artifact/bounded_prefetch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using ninfer::artifact::detail::BoundedPrefetch;
using ninfer::artifact::detail::DirectReadBlock;

namespace {

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

template <class Error, class Function>
void requires_error(Function action, const char* message) {
    try { action(); }
    catch (const Error&) { return; }
    throw std::runtime_error(message);
}

void ordered_blocks(std::size_t workers) {
    constexpr std::size_t bytes = 4096;
    std::array<std::array<std::byte, bytes>, 4> buffers;
    std::vector<DirectReadBlock> blocks;
    for (std::size_t i = 0; i < 11; ++i) {
        blocks.push_back({i < 6 ? 0U : 1U, (i < 6 ? i : i - 6) * bytes,
                          i == 10 ? 2731U : bytes, bytes});
    }
    std::mutex mutex;
    std::condition_variable ready;
    bool second_finished = false;
    std::vector<std::uint64_t> completed;
    BoundedPrefetch prefetch(4, workers, [&](std::size_t slot, const DirectReadBlock& block) {
        const auto count = std::min<std::uint64_t>(block.request, block.remaining);
        const auto fill  = static_cast<std::byte>(block.file * 16 + block.source / bytes);
        std::fill(buffers[slot].begin(), buffers[slot].end(), std::byte{0x7F});
        std::fill_n(buffers[slot].begin(), count, fill);
        // Force two reads to complete out of order, without a timing-dependent sleep.
        std::unique_lock lock(mutex);
        if (workers > 1 && block.file == 0 && block.source == 0) {
            ready.wait(lock, [&] { return second_finished; });
        }
        completed.push_back(block.file * 100 + block.source / bytes);
        if (block.file == 0 && block.source == bytes) {
            second_finished = true;
            ready.notify_all();
        }
        return static_cast<std::size_t>(count);
    });
    std::size_t submitted = 0;
    for (; submitted < std::min<std::size_t>(4, blocks.size()); ++submitted) {
        prefetch.submit(submitted, blocks[submitted]);
    }
    for (std::size_t consumed = 0; consumed < blocks.size(); ++consumed) {
        const auto slot = consumed % 4;
        const auto read = prefetch.take(slot);
        const auto& expected = blocks[consumed];
        require(read.block.file == expected.file && read.block.source == expected.source,
                "consumer lost file/block ordering");
        require(read.received == expected.remaining, "short final read was not preserved");
        const auto fill = static_cast<std::byte>(expected.file * 16 + expected.source / bytes);
        require(std::all_of(buffers[slot].begin(), buffers[slot].begin() + read.received,
                            [&](std::byte value) { return value == fill; }),
                "slot content was overwritten before consumption");
        if (read.received < bytes) {
            require(buffers[slot][read.received] == std::byte{0x7F},
                    "short read touched bytes after EOF");
        }
        if (submitted < blocks.size()) {
            prefetch.submit(slot, blocks[submitted++]);
        }
    }
    if (workers > 1) {
        const auto first = std::find(completed.begin(), completed.end(), 0);
        const auto second = std::find(completed.begin(), completed.end(), 1);
        require(second < first, "test did not force out-of-order completion");
    }
}

void errors_and_reuse() {
    BoundedPrefetch prefetch(2, 2, [](std::size_t, const DirectReadBlock& block) {
        if (block.source == 13) { throw std::runtime_error("synthetic pread failure"); }
        return block.request;
    });
    requires_error<std::logic_error>([&] { (void)prefetch.take(0); }, "take of free slot");
    prefetch.submit(0, {0, 13, 4096, 4096});
    requires_error<std::logic_error>([&] { prefetch.submit(0, {0, 0, 4096, 4096}); },
                                     "duplicate slot submit was accepted");
    requires_error<std::runtime_error>([&] { (void)prefetch.take(0); },
                                       "worker error did not reach caller");
    prefetch.submit(0, {1, 0, 4096, 4096});
    require(prefetch.take(0).received == 4096, "failed slot could not be reused");
    prefetch.close();
    requires_error<std::logic_error>([&] { prefetch.submit(0, {}); },
                                     "submit after close was accepted");
}

void cancel_queued_join_active() {
    std::mutex mutex;
    std::condition_variable ready;
    bool started = false;
    bool release = false;
    std::atomic<int> calls = 0;
    BoundedPrefetch prefetch(2, 1, [&](std::size_t, const DirectReadBlock&) {
        ++calls;
        std::unique_lock lock(mutex);
        started = true;
        ready.notify_all();
        ready.wait(lock, [&] { return release; });
        return 4096;
    });
    prefetch.submit(0, {0, 0, 4096, 4096});
    {
        std::unique_lock lock(mutex);
        ready.wait(lock, [&] { return started; });
    }
    prefetch.submit(1, {0, 4096, 4096, 4096});
    std::thread closing([&] { prefetch.close(); });
    // This returns only after close has set stopping; the active read still waits.
    requires_error<std::logic_error>([&] { (void)prefetch.take(1); },
                                     "queued read was not cancelled");
    {
        std::lock_guard lock(mutex);
        release = true;
    }
    ready.notify_all();
    closing.join();
    require(calls == 1, "close ran a queued job instead of dropping it");
}

} // namespace

int main() {
    try {
        requires_error<std::invalid_argument>(
            [] { BoundedPrefetch invalid(0, 1, [](auto, const auto&) { return 0; }); },
            "zero slots accepted");
        requires_error<std::invalid_argument>(
            [] { BoundedPrefetch invalid(4, 5, [](auto, const auto&) { return 0; }); },
            "more workers than slots accepted");
        ordered_blocks(1);
        ordered_blocks(2);
        ordered_blocks(4);
        errors_and_reuse();
        cancel_queued_join_active();
        std::cout << "bounded prefetch Host checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
