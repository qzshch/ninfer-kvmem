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
    for (const auto format : {"q8_g32_fp16_block", "bf16"}) {
        Fixture fixture;
        const bool q8 = std::string_view(format) == "q8_g32_fp16_block";
        const int row_bytes = q8 ? 34 : 64;
        auto& object = fixture.root["objects"][1];
        object["shape"] = {16, 32}; object["format"] = format;
        object["layout"] = q8 ? "block_interleaved_v1" : "contiguous_le_v1";
        object["bytes"] = 16 * row_bytes;
        // Keep the fixture disjoint even for wider direct rows.
        fixture.root["objects"][1]["shape"] = {12, 32};
        fixture.root["objects"][1]["bytes"] = 12 * row_bytes;
        fixture.root["bindings"].erase("row"); fixture.root["uses"] = Json::array();
        fixture.write(true);
        Reader reader(fixture.entry);
        Binder binder(reader);
        const auto ref = binder.parameter_rows("matrix", {12, 32}, 3, 7);
        auto storage = materialize(reader, std::move(binder).finish(), device);
        const auto w = native_weight(bind_view(ref, storage));
        require(w.n == 7 && w.k == 32 && storage.stats().h2d_bytes == 7 * row_bytes &&
                    storage.stats().device_capacity_bytes == 7 * row_bytes,
                "selected rows allocated/uploaded a complete parent");
        std::vector<std::byte> actual(7 * row_bytes);
        CUDA_CHECK(cudaMemcpy(actual.data(), w.qdata, actual.size(), cudaMemcpyDeviceToHost));
        require(std::equal(actual.begin(), actual.end(), fixture.payload.begin() + 256 + 3 * row_bytes),
                "cross-file row upload changed original encoded bytes");
    }
    std::puts("PASS: compact direct/interleaved row uploads preserve original bytes");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
