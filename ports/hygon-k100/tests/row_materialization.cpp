#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/views.h"
#include "core/device.h"
#include <cstdio>

int main() try {
    using namespace ninfer;
    using namespace ninfer::artifact;
    using namespace ninfer::test::artifact_fixture;
    DeviceContext device;
    for (const auto format : {"q8_g32_fp16_block", "bf16", "q6_k_g256_block", "iq4_xs_g256_block"}) {
        Fixture fixture;
        const std::string_view name(format);
        const bool q8 = name == "q8_g32_fp16_block", q6 = name == "q6_k_g256_block", iq4 = name == "iq4_xs_g256_block";
        const int k = q6 || iq4 ? 256 : 32, n = q6 ? 3 : iq4 ? 5 : 12;
        const int row_bytes = q8 ? 34 : q6 ? 210 : iq4 ? 136 : 64;
        const int begin = q6 ? 1 : iq4 ? 2 : 3, count = q6 ? 2 : iq4 ? 3 : 7;
        auto& object = fixture.root["objects"][1];
        object["shape"] = {n, k}; object["format"] = format;
        object["layout"] = name == "bf16" ? "contiguous_le_v1" : "block_interleaved_v1";
        object["bytes"] = n * row_bytes;
        fixture.root["bindings"].erase("row"); fixture.root["uses"] = Json::array();
        fixture.write(true);
        Reader reader(fixture.entry);
        Binder binder(reader);
        const auto ref = binder.parameter_rows("matrix", {static_cast<unsigned>(n), static_cast<unsigned>(k)}, begin, count);
        auto storage = materialize(reader, std::move(binder).finish(), device);
        const auto w = native_weight(bind_view(ref, storage));
        require(w.n == count && w.k == k && storage.stats().h2d_bytes == static_cast<std::uint64_t>(count) * row_bytes &&
                    storage.stats().device_capacity_bytes == static_cast<std::uint64_t>(count) * row_bytes,
                "selected rows allocated/uploaded a complete parent");
        std::vector<std::byte> actual(count * row_bytes);
        CUDA_CHECK(cudaMemcpy(actual.data(), w.qdata, actual.size(), cudaMemcpyDeviceToHost));
        require(std::equal(actual.begin(), actual.end(), fixture.payload.begin() + 256 + begin * row_bytes),
                "cross-file row upload changed original encoded bytes");
    }
    std::puts("PASS: compact direct/interleaved row uploads preserve original bytes");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
