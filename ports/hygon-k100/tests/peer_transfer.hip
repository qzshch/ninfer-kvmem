// Exact transport oracle for three Hygon devices; not model tensor parallelism.
#include "core/arena.h"
#include "core/device.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kCount = 262147; // deliberately not a block-size multiple

void check(hipError_t result) {
    if (result != hipSuccess) { throw std::runtime_error(hipGetErrorString(result)); }
}

__global__ void write_pattern(std::uint32_t* out, std::size_t count, unsigned salt) {
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = static_cast<std::uint32_t>(i) * 2654435761u ^ salt; }
    return;
}

struct Storage {
    ninfer::DeviceContext context;
    ninfer::DeviceBuffer source;
    ninfer::DeviceBuffer target;
    ninfer::CudaCompletionEvent produced;
    ninfer::CudaCompletionEvent copied;

    explicit Storage(int device)
        : context(device), source(kCount * sizeof(std::uint32_t)),
          target(kCount * sizeof(std::uint32_t)), produced(context), copied(context) {}

    // Members are released while their owning device is selected. Production
    // multi-device allocation ownership remains a separate port requirement.
    ~Storage() { context.bind_to_current_thread_noexcept(); }
};

void verify(const std::vector<std::uint32_t>& values, unsigned salt) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto expected = static_cast<std::uint32_t>(i) * 2654435761u ^ salt;
        if (values[i] != expected) {
            throw std::runtime_error("peer transfer byte oracle differs at element " + std::to_string(i));
        }
    }
}
} // namespace

int main() try {
    int count = 0;
    check(hipGetDeviceCount(&count));
    if (count != 3) { throw std::runtime_error("qualification requires exactly 3 visible K100 devices"); }
    std::array<std::unique_ptr<Storage>, 3> devices;
    for (int device = 0; device < 3; ++device) {
        hipDeviceProp_t props{};
        check(hipGetDeviceProperties(&props, device));
        if (std::string(props.gcnArchName).find("gfx928") != 0) {
            throw std::runtime_error("expected gfx928, got " + std::string(props.gcnArchName));
        }
        std::printf("device %d: %s %s wave=%d bytes=%zu\n", device, props.name,
                    props.gcnArchName, props.warpSize, props.totalGlobalMem);
        devices[device] = std::make_unique<Storage>(device);
    }
    std::vector<std::uint32_t> host(kCount);
    unsigned transfers = 0;
    for (int src = 0; src < 3; ++src) {
        auto& from = *devices[src];
        for (int dst = 0; dst < 3; ++dst) {
            if (src == dst) { continue; }
            auto& to = *devices[dst];
            int can_access = 0;
            check(hipDeviceCanAccessPeer(&can_access, dst, src));
            if (!can_access) { throw std::runtime_error("required peer direction unavailable"); }
            to.context.bind_to_current_thread();
            auto status = hipDeviceEnablePeerAccess(src, 0);
            if (status != hipSuccess && status != hipErrorPeerAccessAlreadyEnabled) { check(status); }
            // Fresh data every iteration prevents an old destination from hiding
            // a missing event/copy. Both streams are non-blocking NInfer streams.
            for (unsigned repetition = 0; repetition < 8; ++repetition) {
                const unsigned salt = 1009u * src + 7919u * dst + 104729u * repetition;
                from.context.bind_to_current_thread();
                write_pattern<<<(kCount + 255) / 256, 256, 0, from.context.stream>>>(
                    static_cast<std::uint32_t*>(from.source.p), kCount, salt);
                check(hipGetLastError());
                from.produced.record(from.context.stream);
                to.context.bind_to_current_thread();
                from.produced.wait(to.context.transfer_stream);
                check(hipMemcpyPeerAsync(to.target.p, dst, from.source.p, src,
                                         from.source.bytes, to.context.transfer_stream));
                to.copied.record(to.context.transfer_stream);
                to.copied.wait(to.context.stream);
                check(hipMemcpyAsync(host.data(), to.target.p, to.target.bytes,
                                     hipMemcpyDeviceToHost, to.context.stream));
                to.context.synchronize();
                verify(host, salt);
                ++transfers;
            }
        }
    }
    std::printf("PASS: %u exact peer copies, 6 directions, non-blocking producer/transfer/consumer ordering\n", transfers);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
