#include "ninfer/ops/linear.h"
#include "ninfer/ops/indexed_linear.h"
#include "core/device.h"
#include "core/arena.h"
#include "core/weight_view.h"
#include "artifact/formats.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;

void require(bool ok, const char* error) {
    if (!ok) throw std::runtime_error(error);
}

template <class T>
std::vector<T> read(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(bool(f), "missing CPU codec fixture");
    auto bytes = static_cast<std::size_t>(f.tellg());
    require(bytes % sizeof(T) == 0, "unaligned fixture bytes");
    std::vector<T> out(bytes / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), bytes);
    require(bool(f), "truncated fixture read");
    return out;
}

std::uint16_t bf16(float v) {
    auto b = std::bit_cast<unsigned>(v);
    return (b + 0x7fffu + ((b >> 16) & 1)) >> 16;
}

double unbf16(std::uint16_t b) { return std::bit_cast<float>(static_cast<unsigned>(b) << 16); }

void run_indexed(DeviceContext& ctx, const std::filesystem::path& root, const nlohmann::json& item,
                 int t, bool shared, bool implicit, bool rounded, bool owned = false, bool physical_bf16 = false) {
    const auto raw    = read<std::byte>(root / item.at("encoded").get<std::string>());
    const auto oracle = read<double>(root / item.at("oracle").get<std::string>());
    const int k = item.at("k"), rows = item.at("parent_rows");
    const auto type = artifact::parse_format(item.at("format").get<std::string>());
    const int n = rows % 2 == 0 ? rows / 2 : 1, banks = rows / n, choices = std::min(banks, 3);
    const std::array<std::uint64_t, 2> shape{static_cast<unsigned>(rows), static_cast<unsigned>(k)};
    const auto geometry = weight_geometry(type, QuantLayout::BlockInterleaved, shape);
    DeviceBuffer weights(raw.size());
    weights.copy_from_host(raw.data(), raw.size());
    WeightParent parent{geometry, static_cast<const std::byte*>(weights.p)};
    auto w = native_weight(WeightView{{static_cast<unsigned>(rows), static_cast<unsigned>(k)},
                                      {{&parent, 0, static_cast<std::uint64_t>(rows) * k}}});
    std::vector<std::int32_t> ids(t * choices);
    for (int token = 0; token < t; ++token)
        for (int c = 0; c < choices; ++c)
            ids[token * choices + c] = implicit ? c : (banks - 1 - c + token) % banks;
    if (owned) for (int token = 0; token < t; ++token)
        for (int c = 0; c < choices; ++c) {
            auto& id = ids[token * choices + c];
            id = (token + c) % 3 == 0 ? 95 : (token + c) % 3 == 1 ? 96 + id : 96 + banks;
        }
    std::vector<float> x(static_cast<std::size_t>(t) * (shared ? 1 : choices) * k);
    for (std::size_t i = 0; i < x.size(); ++i) {
        unsigned b = static_cast<unsigned>(i) * 747796405u + 2891336453u;
        b          = ((b >> ((b >> 28) + 4)) ^ b) * 277803737u;
        b ^= b >> 22;
        x[i] = std::bit_cast<float>((b & 0x807fffffu) | ((124 + b % 6) << 23));
    }
    std::vector<std::uint16_t> words(x.size());
    if (physical_bf16) for (unsigned i = 0; i < x.size(); ++i) {
        words[i] = bf16(x[i]); x[i] = unbf16(words[i]);
    }
    DeviceBuffer dx(x.size() * (physical_bf16 ? 2 : 4)), di(ids.size() * 4),
        dy(static_cast<std::size_t>(t) * choices * n * (physical_bf16 ? 2 : 4) + 32);
    dx.copy_from_host(physical_bf16 ? static_cast<const void*>(words.data()) : x.data(), dx.bytes);
    di.copy_from_host(ids.data(), di.bytes);
    dy.fill(0xa5);
    auto* output = reinterpret_cast<float*>(static_cast<std::byte*>(dy.p) + 16);
    WorkspaceArena scratch(ops::indexed_linear_grouped_workspace_capacity_bytes(banks, choices, t));
    auto project = [&](bool round) {
        if (physical_bf16) {
            Tensor input(dx.p, DType::BF16, {k, t * (shared ? 1 : choices)});
            Tensor result(output, DType::BF16, {n, t * choices});
            const auto* selected = implicit ? nullptr : static_cast<const int*>(di.p);
            if (owned && t >= 16) ops::indexed_linear_grouped(input, w, selected, result,
                n, choices, t, shared, owned ? 96 : 0, scratch, ctx.stream);
            else ops::indexed_linear_mixed(input, w, selected, result,
                n, choices, t, shared, owned ? 96 : 0, ctx.stream);
        } else if (owned && t >= 16) ops::indexed_linear_grouped_fp32(static_cast<float*>(dx.p), w,
            static_cast<std::int32_t*>(di.p), output, n, choices, t, shared, 96, scratch, ctx.stream, round);
        else if (owned) ops::indexed_linear_owned_fp32(static_cast<float*>(dx.p), w,
            static_cast<std::int32_t*>(di.p), output, n, choices, t, shared, 96, ctx.stream, round);
        else ops::indexed_linear_fp32(static_cast<float*>(dx.p), w,
            implicit ? nullptr : static_cast<std::int32_t*>(di.p), output, n, choices, t, shared, ctx.stream, round);
    };
    project(rounded);
    ctx.synchronize();
    std::vector<float> y(static_cast<std::size_t>(t) * choices * n);
    if (physical_bf16) {
        std::vector<std::uint16_t> result_words(y.size());
        CUDA_CHECK(cudaMemcpy(result_words.data(), output, y.size() * 2, cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < y.size(); ++i) y[i] = unbf16(result_words[i]);
    } else CUDA_CHECK(cudaMemcpy(y.data(), output, y.size() * 4, cudaMemcpyDeviceToHost));
    if (rounded && !physical_bf16) {
        project(false);
        ctx.synchronize();
        std::vector<float> raw_y(y.size());
        CUDA_CHECK(cudaMemcpy(raw_y.data(), output, raw_y.size() * 4, cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < y.size(); ++i)
            require(y[i] == unbf16(bf16(raw_y[i])),
                    "indexed fused boundary differs from exact CPU cast of raw output");
    }
    for (int token = 0; token < t; ++token)
        for (int c = 0; c < choices; ++c)
            for (int row = 0; row < n; ++row) {
                double sum = 0, absolute = 0;
                const int bank = ids[token * choices + c] - (owned ? 96 : 0);
                for (int col = 0; bank >= 0 && bank < banks && col < k; ++col) {
                    double p = oracle[(static_cast<std::size_t>(bank) * n + row) * k + col] *
                               x[(static_cast<std::size_t>(token) * (shared ? 1 : choices) +
                                  (shared ? 0 : c)) *
                                     k +
                                 col];
                    sum += p;
                    absolute += std::abs(p);
                }
                double value = y[(token * choices + c) * n + row];
                double floor = 0;
                if (rounded || physical_bf16) {
                    const int word = bf16(static_cast<float>(sum));
                    floor          = std::abs(unbf16(word) - sum);
                    for (int d : {-1, 1})
                        if (word + d >= 0 && word + d <= 65535)
                            floor = std::min(floor, std::abs(unbf16(word + d) - sum));
                    require(value == unbf16(bf16(static_cast<float>(value))),
                            "indexed fused BF16 output representation");
                }
                require(std::isfinite(value) &&
                            std::abs(value - sum) <= floor + 4e-6 * absolute + 1e-30,
                        "indexed projection full FP64 dot criterion");
            }
    std::vector<std::byte> guard(dy.bytes);
    dy.copy_to_host(guard.data(), guard.size());
    for (int i = 0; i < 16; ++i)
        require(guard[i] == std::byte{0xa5} && guard[guard.size() - 1 - i] == std::byte{0xa5},
                "indexed projection guard changed");
    if (!implicit && shared) {
        std::vector<int> tokens{rows - 1, 0, std::min(3, rows - 1)};
        DeviceBuffer itokens(12), embedding(static_cast<std::size_t>(3) * k * 4);
        itokens.copy_from_host(tokens.data(), 12);
        ops::embedding_fp32(w, static_cast<std::int32_t*>(itokens.p),
                            static_cast<float*>(embedding.p), 3, ctx.stream);
        ctx.synchronize();
        std::vector<float> got(3 * k);
        embedding.copy_to_host(got.data(), embedding.bytes);
        for (int token = 0; token < 3; ++token)
            for (int col = 0; col < k; ++col)
                require(got[token * k + col] ==
                            unbf16(bf16(static_cast<float>(
                                oracle[static_cast<std::size_t>(tokens[token]) * k + col]))),
                        "embedding exact decoded BF16 boundary");
    }
    std::vector<std::byte> after(raw.size());
    weights.copy_to_host(after.data(), after.size());
    require(after == raw, "indexed projection mutated encoded banks");
}

double run(DeviceContext& ctx, const std::filesystem::path& root, const nlohmann::json& item,
           int tokens, bool bf, bool rounded = false, bool partial = false) {
    const auto raw    = read<std::byte>(root / item.at("encoded").get<std::string>());
    const auto oracle = read<double>(root / item.at("oracle").get<std::string>());
    const int n = item.at("n"), k = item.at("k"), rows = item.at("parent_rows"),
              begin = item.at("row_begin");
    const auto type = artifact::parse_format(item.at("format").get<std::string>());
    const std::array<std::uint64_t, 2> shape{static_cast<unsigned>(rows), static_cast<unsigned>(k)};
    const auto geometry = weight_geometry(type, QuantLayout::BlockInterleaved, shape);
    require(geometry.bytes == raw.size() && oracle.size() == static_cast<std::size_t>(rows) * k,
            "codec/geometry mismatch");
    DeviceBuffer weights(raw.size());
    weights.copy_from_host(raw.data(), raw.size());
    WeightParent parent{geometry, static_cast<const std::byte*>(weights.p)};
    auto w = native_weight(WeightView{{static_cast<unsigned>(n), static_cast<unsigned>(k)},
                                      {{&parent, static_cast<std::uint64_t>(begin) * k,
                                        static_cast<std::uint64_t>(begin + n) * k}}});
    const int input_k = partial ? k / 2 : k;
    const int k_begin = partial ? k / 2 : 0;
    std::vector<float> x(static_cast<std::size_t>(input_k) * tokens);
    std::vector<std::uint16_t> xb(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        unsigned bits = static_cast<unsigned>(i) * 747796405u + 2891336453u;
        bits          = ((bits >> ((bits >> 28) + 4)) ^ bits) * 277803737u;
        bits ^= bits >> 22;
        x[i]  = std::bit_cast<float>((bits & 0x807fffffu) | ((124 + (bits % 6)) << 23));
        xb[i] = bf16(x[i]);
        if (bf) x[i] = static_cast<float>(unbf16(xb[i]));
    }
    const auto word = bf ? 2 : 4;
    DeviceBuffer input(x.size() * word), output(static_cast<std::size_t>(n) * tokens * word + 32);
    input.copy_from_host(bf ? static_cast<const void*>(xb.data()) : x.data(), input.bytes);
    output.fill(0xa5);
    const auto dtype = bf ? DType::BF16 : DType::FP32;
    Tensor a(input.p, dtype, {input_k, tokens}),
        y(static_cast<std::byte*>(output.p) + 16, dtype, {n, tokens});
    if (partial)
        ops::linear_partial_fp32(static_cast<const float*>(a.data), w, static_cast<float*>(y.data),
                                 k_begin, input_k, tokens, ctx.stream);
    else if (bf)
        ops::linear(a, w, y, ctx.stream);
    else
        ops::linear_fp32(a, w, y, ctx.stream, rounded);
    ctx.synchronize();
    auto observed = std::vector<std::byte>(output.bytes);
    output.copy_to_host(observed.data(), observed.size());
    if (rounded) {
        ops::linear_fp32(a, w, y, ctx.stream, false);
        ctx.synchronize();
        std::vector<std::byte> raw_y(output.bytes);
        output.copy_to_host(raw_y.data(), raw_y.size());
        for (int i = 0; i < n * tokens; ++i) {
            float raw_value, fused_value;
            std::memcpy(&raw_value, raw_y.data() + 16 + i * 4, 4);
            std::memcpy(&fused_value, observed.data() + 16 + i * 4, 4);
            require(fused_value == unbf16(bf16(raw_value)),
                    "dense fused boundary differs from exact CPU cast of raw output");
        }
    }
    for (int i = 0; i < 16; ++i)
        require(observed[i] == std::byte{0xa5} &&
                    observed[observed.size() - 1 - i] == std::byte{0xa5},
                "block projection output guard");
    std::vector<std::byte> after(raw.size());
    weights.copy_to_host(after.data(), after.size());
    require(after == raw, "encoded parent changed");
    double worst = 0;
    for (int t = 0; t < tokens; ++t)
        for (int row = 0; row < n; ++row) {
            double expected = 0, absolute = 0;
            for (int c = 0; c < input_k; ++c) {
                auto p = oracle[static_cast<std::size_t>(begin + row) * k + k_begin + c] *
                         x[static_cast<std::size_t>(t) * input_k + c];
                expected += p;
                absolute += std::abs(p);
            }
            double value, floor = 0;
            const auto offset = 16 + (static_cast<std::size_t>(t) * n + row) * word;
            if (bf) {
                std::uint16_t v;
                std::memcpy(&v, observed.data() + offset, 2);
                value = unbf16(v);
            } else {
                float v;
                std::memcpy(&v, observed.data() + offset, 4);
                value = v;
            }
            if (bf || rounded) {
                const auto approx = static_cast<int>(bf16(static_cast<float>(expected)));
                floor             = std::abs(unbf16(approx) - expected);
                for (int d : {-1, 1})
                    if (approx + d >= 0 && approx + d <= 65535)
                        floor = std::min(floor, std::abs(unbf16(approx + d) - expected));
            }
            if (rounded)
                require(value == unbf16(bf16(static_cast<float>(value))),
                        "dense fused BF16 output representation");
            const auto ratio =
                std::max(0., std::abs(value - expected) - floor) / (4e-6 * absolute + 1e-30);
            worst = std::max(worst, ratio);
            if (!std::isfinite(value) || ratio > 1) {
                std::fprintf(stderr, "format=%s row=%d token=%d expected=%g value=%g ratio=%g\n",
                             item.at("format").get<std::string>().c_str(), row, t, expected, value,
                             ratio);
                throw std::runtime_error("block projection FP64 full-dot criterion");
            }
        }
    return worst;
}
} // namespace
int main(int argc, char** argv) try {
    require(argc == 2, "fixture directory required");
    std::filesystem::path root(argv[1]);
    std::ifstream f(root / "manifest.json");
    nlohmann::json cases;
    f >> cases;
    DeviceContext ctx;
    unsigned count = 0;
    double worst   = 0;
    for (const auto& item : cases)
        for (int tokens : {1, 3, 8})
            for (bool bf : {false, true}) {
                worst = std::max(worst, run(ctx, root, item, tokens, bf));
                ++count;
                if (!bf) {
                    worst = std::max(worst, run(ctx, root, item, tokens, false, true));
                    ++count;
                }
            }
    for (const auto& item : cases) if (item.at("k").get<int>() % 512 == 0)
        for (int tokens : {1, 8, 17}) {
            worst = std::max(worst, run(ctx, root, item, tokens, false, false, true));
            ++count;
        }
    unsigned indexed = 0;
    for (const auto& item : cases)
        for (int tokens : {1, 3, 8})
            for (bool shared : {false, true})
                for (bool implicit : {false, true}) {
                    for (bool rounded : {false, true}) {
                        run_indexed(ctx, root, item, tokens, shared, implicit, rounded);
                        ++indexed;
                    }
                }
    for (const auto& item : cases) for (int tokens : {1, 3, 32, 33}) for (bool shared : {false, true}) {
        run_indexed(ctx, root, item, tokens, shared, false, true, true, true);
        run_indexed(ctx, root, item, tokens, shared, true, true, false, true);
    }
    for (const auto& item : cases)
        for (int tokens : {1, 8, 33})
            for (bool shared : {false, true})
                for (bool rounded : {false, true}) {
                    run_indexed(ctx, root, item, tokens, shared, false, rounded, true);
                    ++indexed;
                }
    std::printf("PASS: %u indexed bank/head projections, plus exact BF16 embedding checks\n",
                indexed);
    std::printf("PASS: %u block projection cases; six source formats; independent scalar CPU "
                "block decode and full FP64 dot; max_arithmetic_budget_fraction=%.9g\n",
                count, worst);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
}
