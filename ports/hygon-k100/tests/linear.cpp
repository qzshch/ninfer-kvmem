#include "ninfer/ops/linear.h"
#include "core/device.h"
#include "core/weight_view.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace ninfer;

void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}

std::uint16_t bf16(float x) {
    auto u = std::bit_cast<std::uint32_t>(x);
    return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}

double unbf16(std::uint16_t u) { return std::bit_cast<float>(static_cast<std::uint32_t>(u) << 16); }

double nearest_error(double x) {
    auto candidate = bf16(static_cast<float>(x));
    double result  = std::abs(unbf16(candidate) - x);
    for (int delta : {-1, 1}) {
        const int word = static_cast<int>(candidate) + delta;
        if (word >= 0 && word <= 65535 && std::isfinite(unbf16(word))) {
            result = std::min(result, std::abs(unbf16(word) - x));
        }
    }
    return result;
}

double half(std::uint16_t u) {
    const int exponent = (u >> 10) & 31;
    const double sign  = u & 32768 ? -1.0 : 1.0;
    return sign * (exponent ? std::ldexp(1024.0 + (u & 1023), exponent - 25)
                            : std::ldexp(static_cast<double>(u & 1023), -24));
}

template <class T>
void append(std::vector<std::byte>& bytes, T value) {
    const auto before = bytes.size();
    bytes.resize(before + sizeof(T));
    std::memcpy(bytes.data() + before, &value, sizeof(T));
}

std::vector<std::byte> fixture(QType type, std::size_t count) {
    std::vector<std::byte> out;
    if (type == QType::MXFP4_G32_E8M0 || type == QType::Q8_G32_FP16_BLOCK) {
        for (std::size_t block = 0; block < count / 32; ++block) {
            if (type == QType::MXFP4_G32_E8M0) {
                append(out, static_cast<std::uint8_t>(121 + block % 7));
                for (unsigned i = 0; i < 16; ++i) {
                    append(out, static_cast<std::uint8_t>(((i + block) & 15) |
                                                          (((15 - i + block) & 15) << 4)));
                }
            } else {
                append(out, static_cast<std::uint16_t>(0x2800 + (block % 16) * 32));
                for (unsigned i = 0; i < 32; ++i) {
                    append(out, static_cast<std::int8_t>((i * 17 + block) % 256 - 128));
                }
            }
        }
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            const float value = static_cast<float>(static_cast<int>((i * 73) % 511) - 255) / 128.0f;
            if (type == QType::FP32) {
                append(out, value);
            } else if (type == QType::BF16) {
                append(out, bf16(value));
            } else {
                append(out,
                       static_cast<std::uint16_t>(0x3000 + (i % 1024) + ((i & 1) ? 0x8000 : 0)));
            }
        }
    }
    return out;
}

std::vector<double> decode(const std::vector<std::byte>& bytes, QType type, std::size_t count) {
    const auto* u = reinterpret_cast<const unsigned char*>(bytes.data());
    std::vector<double> out(count);
    constexpr double e2m1[] = {0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6};
    for (std::size_t i = 0; i < count; ++i) {
        if (type == QType::MXFP4_G32_E8M0) {
            auto* block     = u + (i / 32) * 17;
            const auto lane = i % 32;
            out[i]          = std::ldexp(e2m1[(block[1 + lane % 16] >> (lane < 16 ? 0 : 4)) & 15],
                                         static_cast<int>(block[0]) - 127);
        } else if (type == QType::Q8_G32_FP16_BLOCK) {
            auto* block    = u + (i / 32) * 34;
            const int code = block[2 + i % 32] < 128 ? block[2 + i % 32]
                                                     : static_cast<int>(block[2 + i % 32]) - 256;
            out[i]         = half(static_cast<std::uint16_t>(block[0] | (block[1] << 8))) * code;
        } else if (type == QType::FP32) {
            float v;
            std::memcpy(&v, u + i * 4, 4);
            out[i] = v;
        } else {
            std::uint16_t v;
            std::memcpy(&v, u + i * 2, 2);
            out[i] = type == QType::BF16 ? unbf16(v) : half(v);
        }
    }
    return out;
}

double run(DeviceContext& ctx, QType type, int n, int k, int tokens, bool bf16_io,
           const std::vector<std::byte>* real = nullptr, bool rounded = false) {
    const int row_offset  = real ? 0 : 1;
    const int parent_rows = real ? n : n + 2;
    auto weights          = real ? *real : fixture(type, static_cast<std::size_t>(parent_rows) * k);
    const auto host       = decode(weights, type, static_cast<std::size_t>(parent_rows) * k);
    const auto layout     = (type == QType::MXFP4_G32_E8M0 || type == QType::Q8_G32_FP16_BLOCK)
                                ? QuantLayout::BlockInterleaved
                                : QuantLayout::Contiguous;
    DeviceBuffer w(weights.size());
    w.copy_from_host(weights.data(), weights.size());
    const std::array<std::uint64_t, 2> shape{static_cast<unsigned>(parent_rows),
                                             static_cast<unsigned>(k)};
    WeightParent parent{weight_geometry(type, layout, shape), static_cast<const std::byte*>(w.p)};
    auto weight = native_weight(WeightView{{static_cast<unsigned>(n), static_cast<unsigned>(k)},
                                           {{&parent, static_cast<std::uint64_t>(row_offset) * k,
                                             static_cast<std::uint64_t>(row_offset + n) * k}}});
    const std::size_t input_count  = static_cast<std::size_t>(k) * tokens,
                      output_count = static_cast<std::size_t>(n) * tokens;
    std::vector<float> x(input_count);
    std::vector<std::uint16_t> xb(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        // Deterministic full-mantissa represented inputs, rather than exact small
        // multiples of a power of two that conceal accumulation error.
        std::uint32_t bits = static_cast<std::uint32_t>(i) * 747796405u + 2891336453u;
        bits               = ((bits >> ((bits >> 28u) + 4u)) ^ bits) * 277803737u;
        bits               = (bits >> 22u) ^ bits;
        x[i]  = std::bit_cast<float>((bits & 0x807fffffu) | ((124u + (bits % 6u)) << 23));
        xb[i] = bf16(x[i]);
        if (bf16_io) { x[i] = static_cast<float>(unbf16(xb[i])); }
    }
    const std::size_t word = bf16_io ? 2 : 4;
    DeviceBuffer input(input_count * word), output(output_count * word + 32);
    input.copy_from_host(bf16_io ? static_cast<const void*>(xb.data()) : x.data(), input.bytes);
    output.fill(0xa5);
    const auto dtype = bf16_io ? DType::BF16 : DType::FP32;
    Tensor a(input.p, dtype, {k, tokens}),
        y(static_cast<std::byte*>(output.p) + 16, dtype, {n, tokens});
    if (bf16_io) {
        ops::linear(a, weight, y, ctx.stream);
    } else {
        ops::linear_fp32(a, weight, y, ctx.stream, rounded);
    }
    ctx.synchronize();
    std::vector<std::byte> result(output.bytes);
    output.copy_to_host(result.data(), result.size());
    for (std::size_t i = 0; i < 16; ++i) {
        require(result[i] == std::byte{0xa5} && result[result.size() - 1 - i] == std::byte{0xa5},
                "Linear output guard changed");
    }
    std::vector<std::byte> after(weights.size());
    w.copy_to_host(after.data(), after.size());
    require(after == weights, "Linear modified encoded weight bytes");
    if (rounded) {
        ops::linear_fp32(a, weight, y, ctx.stream, false);
        ctx.synchronize();
        std::vector<std::byte> raw_result(output.bytes);
        output.copy_to_host(raw_result.data(), raw_result.size());
        for (std::size_t i = 0; i < output_count; ++i) {
            float raw_value, fused_value;
            std::memcpy(&raw_value, raw_result.data() + 16 + i * 4, 4);
            std::memcpy(&fused_value, result.data() + 16 + i * 4, 4);
            require(fused_value == unbf16(bf16(raw_value)),
                    "direct-format exact fused output cast");
        }
    }
    double worst = 0;
    for (int t = 0; t < tokens; ++t)
        for (int row = 0; row < n; ++row) {
            double expected = 0, absolute_sum = 0;
            for (int c = 0; c < k; ++c) {
                const double product = host[static_cast<std::size_t>(row + row_offset) * k + c] *
                                       x[static_cast<std::size_t>(t) * k + c];
                expected += product;
                absolute_sum += std::abs(product);
            }
            double observed;
            const auto index = static_cast<std::size_t>(t) * n + row;
            if (bf16_io) {
                std::uint16_t v;
                std::memcpy(&v, result.data() + 16 + index * 2, 2);
                observed = unbf16(v);
            } else {
                float v;
                std::memcpy(&v, result.data() + 16 + index * 4, 4);
                observed = v;
            }
            const double floor     = (bf16_io || rounded) ? nearest_error(expected) : 0;
            const double allowance = 4e-6 * absolute_sum + 1e-30;
            const double ratio = std::max(0.0, std::abs(observed - expected) - floor) / allowance;
            worst              = std::max(worst, ratio);
            if (!std::isfinite(observed) || ratio > 1) {
                std::fprintf(
                    stderr, "q=%d N=%d K=%d T=%d BF16=%d ratio=%g expected=%g observed=%g\n",
                    static_cast<int>(type), n, k, tokens, bf16_io, ratio, expected, observed);
                throw std::runtime_error("FP64 complete-dot oracle failed");
            }
        }
    return worst;
}
} // namespace
int main(int argc, char** argv) try {
    ninfer::DeviceContext ctx;
    unsigned cases = 0;
    double worst   = 0;
    for (auto type :
         {QType::MXFP4_G32_E8M0, QType::Q8_G32_FP16_BLOCK, QType::FP32, QType::BF16, QType::FP16})
        for (int tokens : {1, 5, 6, 32})
            for (bool bf16_io : {false, true}) {
                worst = std::max(worst, run(ctx, type, 17, 64, tokens, bf16_io));
                ++cases;
            }
    for (auto [n, k] : {std::pair{2304, 5120}, std::pair{5120, 2304}}) {
        worst = std::max(worst, run(ctx, QType::MXFP4_G32_E8M0, n, k, 1, false));
        ++cases;
    }
    for (auto type :
         {QType::MXFP4_G32_E8M0, QType::Q8_G32_FP16_BLOCK, QType::FP32, QType::BF16, QType::FP16})
        for (int tokens : {1, 8})
            for (bool bf16_io : {false, true}) {
                worst = std::max(worst, run(ctx, type, 513, 1024, tokens, bf16_io));
                ++cases;
                worst = std::max(worst, run(ctx, type, 513, 544, tokens, bf16_io));
                ++cases;
            }
    for (auto type :
         {QType::MXFP4_G32_E8M0, QType::Q8_G32_FP16_BLOCK, QType::FP32, QType::BF16, QType::FP16}) {
        for (int tokens : {1, 3, 8}) {
            worst = std::max(worst, run(ctx, type, 513, 1024, tokens, false, nullptr, true));
            ++cases;
        }
        worst = std::max(worst, run(ctx, type, 17, 64, 5, false, nullptr, true));
        ++cases;
    }
    if (argc == 2) {
        std::ifstream stream(argv[1], std::ios::binary);
        require(static_cast<bool>(stream), "missing real MXFP4 sample");
        std::vector<std::byte> bytes(64 * 5120 / 32 * 17);
        stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        require(static_cast<std::size_t>(stream.gcount()) == bytes.size(),
                "truncated real MXFP4 sample");
        for (int tokens : {1, 5, 6}) {
            worst =
                std::max(worst, run(ctx, QType::MXFP4_G32_E8M0, 64, 5120, tokens, false, &bytes));
            ++cases;
        }
    }
    std::printf(
        "PASS: %u Linear cases; complete FP64 dot oracle; max_arithmetic_budget_fraction=%.9g; "
        "exact weight preservation and output guards; real_sample=%s\n",
        cases, worst, argc == 2 ? "yes" : "no");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
