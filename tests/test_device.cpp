#include "core/device.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_throws_device(int device_id) {
    try {
        ninfer::DeviceContext invalid(device_id);
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const ninfer::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.transfer_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.compute_capability() <= 0) {
        std::cerr << label << " compute capability is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--invalid-sync") {
        try {
            ninfer::DeviceContext ctx(0);
        } catch (const std::invalid_argument& error) {
            return std::string_view(error.what()).find("NINFER_CUDA_SYNC") != std::string_view::npos
                       ? 0
                       : fail("invalid sync setting has no configuration diagnostic");
        }
        return fail("invalid sync setting did not fail before CUDA initialization");
    }
    const unsigned int expected_flags =
        argc == 2 ? static_cast<unsigned int>(std::stoul(argv[1])) : cudaDeviceScheduleSpin;
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;

    ninfer::DeviceContext ctx(0);
    unsigned int actual_flags = 0;
    CUDA_CHECK(cudaGetDeviceFlags(&actual_flags));
    if ((actual_flags & cudaDeviceScheduleMask) != expected_flags) {
        return fail("CUDA did not apply the requested synchronization schedule");
    }
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    int* device_value = nullptr;
    int* host_value   = nullptr;
    CUDA_CHECK(cudaMalloc(&device_value, sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&host_value, sizeof(int)));
    *host_value = 0;
    CUDA_CHECK(cudaMemsetAsync(device_value, 0x5a, sizeof(int), ctx.stream));
    CUDA_CHECK(
        cudaMemcpyAsync(host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
    ctx.synchronize();
    const bool transfer_complete = *host_value == 0x5a5a5a5a;
    CUDA_CHECK(cudaFreeHost(host_value));
    CUDA_CHECK(cudaFree(device_value));
    if (!transfer_complete) {
        return fail("stream synchronization returned before transfer completed");
    }

    struct WaitProof {
        std::atomic<bool> release{false};
        bool called     = false;
        bool unfinished = false;
        ninfer::CudaCompletionEvent completion;

        explicit WaitProof(ninfer::DeviceContext& device) : completion(device) {}
    } proof(ctx);

    // A bounded stream callback makes the unfinished-unit state deterministic.
    // This is an ownership / event-order test, not a GPU performance measurement.
    CUDA_CHECK(cudaLaunchHostFunc(
        ctx.stream,
        [](void* owner) {
            auto& p             = *static_cast<WaitProof*>(owner);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (!p.release.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
        },
        &proof));
    const ninfer::DeviceWaitWork work{.owner = &proof, .run = [](void* owner, cudaStream_t stream) {
                                          auto& p = *static_cast<WaitProof*>(owner);
                                          p.completion.record(stream);
                                          p.unfinished = !p.completion.ready();
                                          p.called     = true;
                                          p.release.store(true, std::memory_order_release);
                                      }};
    {
        ninfer::ScopedDeviceWaitWork outer(ctx, &work);
        {
            ninfer::ScopedDeviceWaitWork disabled(ctx, nullptr);
            if (ctx.wait_work != nullptr) { return fail("nested work disable failed"); }
        }
        ctx.synchronize();
    }
    if (!proof.called || !proof.unfinished || ctx.wait_work != nullptr) {
        return fail("CPU wait work did not run before stream completion / restore scope");
    }
    try {
        ninfer::ScopedDeviceWaitWork restore(ctx, &work);
        throw std::runtime_error("scope-unwind fixture");
    } catch (const std::runtime_error&) {}
    if (ctx.wait_work != nullptr) { return fail("CPU work escaped exception scope"); }
    proof.called = false;
    ctx.synchronize();
    if (proof.called) { return fail("completed scope invoked stale CPU work"); }

    const cudaStream_t original_stream = ctx.stream;
    ninfer::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.transfer_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    failures += expect_throws_device(count);

    ninfer::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    return failures == 0 ? 0 : fail("device test failed");
}
