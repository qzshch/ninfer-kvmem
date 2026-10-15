#include "ninfer/ops/indexed_linear.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
float bf(double x) {
    const auto u = std::bit_cast<unsigned>(static_cast<float>(x));
    return std::bit_cast<float>((u + 0x7fffu + ((u >> 16) & 1)) & 0xffff0000u);
}
double half(unsigned u) {
    const int exponent = (u >> 10) & 31;
    return (u & 32768 ? -1. : 1.) * (exponent ? std::ldexp(1024. + (u & 1023), exponent - 25)
                                              : std::ldexp(double(u & 1023), -24));
}

// Independently represented Q8 weights. The exact dyadic first dots avoid
// manufacturing a compound error allowance for moved intermediate boundaries.
struct Bank {
    std::vector<unsigned char> encoded;
    std::vector<double> values;
    DeviceBuffer device;
    Weight weight;
    Bank(int n, int k, int seed) : encoded(static_cast<std::size_t>(n) * (k / 32) * 34),
        values(static_cast<std::size_t>(n) * k), device(encoded.size()) {
        for (int row = 0; row < n; ++row) for (int c = 0; c < k; c += 32) {
            const auto at = (static_cast<std::size_t>(row) * (k / 32) + c / 32) * 34;
            const unsigned scale = 0x2400 + ((row + c / 32 + seed) % 3) * 0x400;
            encoded[at] = scale & 255; encoded[at + 1] = scale >> 8;
            for (int j = 0; j < 32; ++j) {
                const int code = ((row * 17 + (c + j) * 7 + seed * 3) % 15) - 7;
                encoded[at + 2 + j] = static_cast<unsigned char>(code);
                values[static_cast<std::size_t>(row) * k + c + j] = half(scale) * code;
            }
        }
        device.copy_from_host(encoded.data(), encoded.size());
        weight.qdata = device.p; weight.n = n; weight.k = k;
        weight.qtype = QType::Q8_G32_FP16_BLOCK; weight.layout = QuantLayout::BlockInterleaved;
    }
    void unchanged() const {
        std::vector<unsigned char> after(encoded.size()); device.copy_to_host(after.data(), after.size());
        require(after == encoded, "indexed MLP mutated original encoded weights");
    }
};

void run(DeviceContext& ctx, int tokens, bool physical_bf16, bool wide, float limit, bool graph_mode) {
    const int k = wide ? 256 : 32, f = wide ? 256 : 32, n = wide ? 129 : 17;
    constexpr int banks = 3, choices = 8, first = 96;
    const auto storage = physical_bf16 ? DType::BF16 : DType::FP32;
    const int word = physical_bf16 ? 2 : 4, count = tokens * choices;
    Bank gate(banks * f, k, 1), up(banks * f, k, 5), down(banks * n, f, 9);
    DeviceBuffer input(static_cast<std::size_t>(tokens) * k * word), ids(count * sizeof(int)),
        output(static_cast<std::size_t>(count) * n * word + 32);
    Tensor x(input.p, storage, {k, tokens});
    Tensor y(static_cast<unsigned char*>(output.p) + 16, storage, {n, count});
    const auto quoted = ops::indexed_swiglu_linear_workspace_capacity_bytes(storage, banks, f, choices, tokens);
    DeviceBuffer backing(quoted + 768); backing.fill(0xa5);
    WorkspaceArena work(DeviceSpan{static_cast<unsigned char*>(backing.p) + 256, quoted + 256});
    (void)work.alloc_bytes(3);
    const auto invoke = [&] {
        ops::indexed_swiglu_linear(x, gate.weight, up.weight, down.weight, static_cast<const int*>(ids.p),
            y, f, n, choices, tokens, first, limit, work, ctx.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    if (graph_mode) { definition.capture(ctx.stream, invoke); executable.instantiate(definition); }
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<int> selected(count);
        for (int t = 0; t < tokens; ++t) for (int c = 0; c < choices; ++c)
            selected[t * choices + c] = pass == 2 || (t == 0 && tokens > 1) ? first - 1
                : pass == 1 ? first + 1 : (t + c) % 5 + first - 1;
        if (pass == 0 && tokens > 1) {
            selected[choices] = std::numeric_limits<int>::max();
            selected[choices + 1] = -1;
        }
        std::vector<float> values(static_cast<std::size_t>(tokens) * k);
        for (int t = 0; t < tokens; ++t) for (int c = 0; c < k; ++c)
            values[t * k + c] = pass == 2 || (t == 0 && tokens > 1) ? std::numeric_limits<float>::quiet_NaN()
                : float(((t * 11 + c * 5) % 31) - 15) / (limit > 0 ? 1.f : 64.f);
        std::vector<unsigned char> input_bytes(input.bytes);
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (physical_bf16) {
                const auto u = static_cast<std::uint16_t>(std::bit_cast<unsigned>(bf(values[i])) >> 16);
                std::memcpy(input_bytes.data() + i * 2, &u, 2);
            } else std::memcpy(input_bytes.data() + i * 4, &values[i], 4);
        }
        input.copy_from_host(input_bytes.data(), input_bytes.size());
        ids.copy_from_host(selected.data(), ids.bytes); output.fill(0xa5);
        if (graph_mode) executable.launch(ctx.stream); else invoke();
        ctx.synchronize();
        require(work.used() == 3 && work.peak_used() <= quoted + 3, "indexed MLP workspace quote/lifetime");
        std::vector<unsigned char> result(output.bytes); output.copy_to_host(result.data(), result.size());
        for (int i = 0; i < 16; ++i)
            require(result[i] == 0xa5 && result[result.size() - 1 - i] == 0xa5, "indexed MLP output guard");
        for (int t = 0; t < tokens; ++t) for (int choice = 0; choice < choices; ++choice) {
            const auto bank = static_cast<long long>(selected[t * choices + choice]) - first;
            std::vector<double> hidden(f);
            if (bank >= 0 && bank < banks) for (int row = 0; row < f; ++row) {
                double g = 0, u = 0;
                for (int c = 0; c < k; ++c) {
                    const auto at = (bank * f + row) * k + c;
                    g += gate.values[at] * values[t * k + c];
                    u += up.values[at] * values[t * k + c];
                }
                g = bf(g); u = bf(u);
                if (limit > 0) { g = std::min(g, double(limit)); u = std::clamp(u, -double(limit), double(limit)); }
                hidden[row] = bf(double(bf(g / (1. + std::exp(-g)))) * u);
            }
            for (int row = 0; row < n; ++row) {
                double sum = 0, absolute = 0;
                if (bank >= 0 && bank < banks) for (int c = 0; c < f; ++c) {
                    const auto product = down.values[(bank * n + row) * f + c] * hidden[c];
                    sum += product; absolute += std::abs(product);
                }
                const auto at = 16 + (static_cast<std::size_t>(t * choices + choice) * n + row) * word;
                float actual;
                if (physical_bf16) {
                    std::uint16_t bits; std::memcpy(&bits, result.data() + at, 2);
                    actual = std::bit_cast<float>(static_cast<unsigned>(bits) << 16);
                } else std::memcpy(&actual, result.data() + at, 4);
                require(std::isfinite(actual) && actual == bf(actual), "indexed MLP output BF16 boundary");
                // Same complete-dot bound as the independent projection tests.
                double floor = std::abs(double(bf(sum)) - sum);
                const int bits = std::bit_cast<unsigned>(bf(sum)) >> 16;
                for (int delta : {-1, 1}) if (bits + delta >= 0 && bits + delta <= 65535) {
                    const auto neighbor = std::bit_cast<float>(static_cast<unsigned>(bits + delta) << 16);
                    if (std::isfinite(neighbor)) floor = std::min(floor, std::abs(double(neighbor) - sum));
                }
                require(std::abs(actual - sum) <= floor + 4e-6 * absolute + 1e-30,
                        "indexed MLP independent FP64 compound oracle");
                if (bank < 0 || bank >= banks) require(std::bit_cast<unsigned>(actual) == 0, "unowned MLP exact zero");
            }
        }
        std::vector<unsigned char> after(input.bytes); input.copy_to_host(after.data(), after.size());
        require(after == input_bytes, "indexed MLP input mutated");
    }
    std::vector<unsigned char> guard(backing.bytes); backing.copy_to_host(guard.data(), guard.size());
    for (int i = 0; i < 256; ++i)
        require(guard[i] == 0xa5 && guard[guard.size() - 1 - i] == 0xa5, "indexed MLP workspace guard");
    gate.unchanged(); up.unchanged(); down.unchanged();
}
} // namespace

int main() try {
    DeviceContext ctx;
    for (bool physical_bf16 : {false, true}) {
        for (int tokens : {1, 8, 15, 16, 17, 33}) run(ctx, tokens, physical_bf16, false, 0.f, tokens == 17);
        for (int tokens : {16, 17, 32}) run(ctx, tokens, physical_bf16, true, 10.f, tokens == 17);
    }
    std::puts("indexed MLP independent FP64, ownership, route reuse and Graph checks passed");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
