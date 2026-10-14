// Independent FP64 oracle at the public NInfer RMSNorm/GatedRMSNorm boundary.
// Test codecs use host bit_cast; they do not share device helpers.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "core/arena.h"
#include "core/device.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
std::uint16_t encode(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    // Fixtures are finite. RNE follows the represented input's bit pattern.
    return static_cast<std::uint16_t>((bits + 32767u + ((bits >> 16) & 1u)) >> 16);
}
double decode(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}
std::uint16_t nearest_bf16(double value) {
    const auto approximate = encode(static_cast<float>(value));
    auto best = approximate;
    double distance = std::abs(decode(best) - value);
    // Inspect adjacent values to avoid double rounding at an FP32 midpoint.
    for (int delta : {-1, 1}) {
        const int candidate = static_cast<int>(approximate) + delta;
        if (candidate < 0 || candidate > 65535) { continue; }
        const double decoded = decode(static_cast<std::uint16_t>(candidate));
        if (!std::isfinite(decoded)) { continue; }
        const double error = std::abs(decoded - value);
        if (error < distance || (error == distance && (candidate & 1) == 0)) {
            best = static_cast<std::uint16_t>(candidate);
            distance = error;
        }
    }
    return best;
}
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct Input {
    std::vector<std::uint16_t> represented;
    ninfer::DeviceBuffer device;
    explicit Input(std::size_t count) : represented(count + 2, 0x5a5a), device((count + 2) * 2) {}
    void upload() { device.copy_from_host(represented.data(), represented.size() * 2); }
    void* data() { return static_cast<std::uint16_t*>(device.p) + 1; }
    void verify() {
        std::vector<std::uint16_t> actual(represented.size());
        device.copy_to_host(actual.data(), actual.size() * 2);
        require(actual == represented, "RMSNorm modified represented inputs or their guards");
    }
};

struct Result {
    double relative_l2;
    double gross_ratio;
    double arithmetic_budget_fraction;
    bool original_statistical_cutoff_exceeded;
};

Result run_case(ninfer::DeviceContext& context, int d, int rows, int mode, float scale,
                 bool zeros = false) {
    constexpr float eps = 1.0e-6f;
    const std::size_t count = static_cast<std::size_t>(d) * rows;
    Input x(count), weight(d), z(count);
    for (std::size_t i = 0; i < count; ++i) {
        x.represented[i + 1] = encode(zeros ? 0.0f : scale * static_cast<float>(static_cast<int>((i * 73u + 19u) % 2003u) - 1001) / 1001.0f);
        z.represented[i + 1] = encode(static_cast<float>(static_cast<int>((i * 31u) % 1001u) - 500) / 50.0f);
    }
    for (int i = 0; i < d; ++i) {
        weight.represented[i + 1] = encode(static_cast<float>((i * 37) % 257 - 128) / 85.0f);
    }
    if (d > 1) { weight.represented[1] = encode(mode == 1 ? -1.0f : 0.0f); }
    z.represented[1] = 0; // exact zero SiLU gate
    x.upload(); weight.upload(); z.upload();
    ninfer::DeviceBuffer storage((count + 2) * 2);
    storage.fill(0xa5);
    // Two-byte offset deliberately violates BF16x2 alignment.
    ninfer::Tensor input(x.data(), ninfer::DType::BF16, {d, rows});
    ninfer::Tensor gains(weight.data(), ninfer::DType::BF16, {d});
    ninfer::Tensor gate(z.data(), ninfer::DType::BF16, {d, rows});
    ninfer::Tensor output(static_cast<std::uint16_t*>(storage.p) + 1, ninfer::DType::BF16, {d, rows});
    if (mode == 2) {
        ninfer::ops::gated_rmsnorm(input, gains, gate, eps, output, context.execution_view());
    } else {
        ninfer::ops::rmsnorm(input, gains, eps, mode == 1, output, context.stream);
    }
    context.synchronize();
    std::vector<std::uint16_t> actual(count + 2);
    storage.copy_to_host(actual.data(), actual.size() * 2);
    require(actual.front() == 0xa5a5 && actual.back() == 0xa5a5, "RMSNorm overwrote output guards");
    x.verify(); weight.verify(); z.verify();
    double error2 = 0.0, reference2 = 0.0, max_error = 0.0, max_reference = 0.0;
    double storage_error2 = 0.0, max_storage_error = 0.0, max_arithmetic_fraction = 0.0;
    for (int row = 0; row < rows; ++row) {
        const auto base = static_cast<std::size_t>(row) * d;
        double sum = 0.0;
        for (int c = 0; c < d; ++c) {
            const double value = decode(x.represented[base + c + 1]);
            sum += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum / d + static_cast<double>(eps));
        for (int c = 0; c < d; ++c) {
            const double value = decode(x.represented[base + c + 1]);
            const double gain = decode(weight.represented[c + 1]) + (mode == 1 ? 1.0 : 0.0);
            double expected = value * inverse * gain;
            if (mode == 2) {
                const double gate_value = decode(z.represented[base + c + 1]);
                expected *= gate_value / (1.0 + std::exp(-gate_value));
            }
            const double observed = decode(actual[base + c + 1]);
            require(std::isfinite(observed), "RMSNorm returned non-finite output for finite fixture");
            // A statistical relative-L2 ceiling is not a representability bound
            // for 1/3-element rows. Require the exact best BF16 result instead;
            // do not drop these adversarial cases or loosen wide-row criteria.
            if (d <= 3) {
                require(actual[base + c + 1] == nearest_bf16(expected),
                        "small RMSNorm output is not the exact nearest FP64-oracle BF16 value");
            }
            const double error = std::abs(observed - expected);
            const double minimum_storage_error = std::abs(decode(nearest_bf16(expected)) - expected);
            // Direct comparison to FP64 ideal, with storage error in the
            // criterion. This is not a rounded-output mathematical oracle.
            // The extra FP32 arithmetic allowance is 4e-6 relative + 1e-35.
            const double arithmetic_budget = 4.0e-6 * std::abs(expected) + 1.0e-35;
            const double excess = std::max(0.0, error - minimum_storage_error);
            max_arithmetic_fraction = std::max(max_arithmetic_fraction, excess / arithmetic_budget);
            require(excess <= arithmetic_budget, "RMSNorm exceeds BF16 storage floor plus FP32 arithmetic budget");
            storage_error2 += minimum_storage_error * minimum_storage_error;
            max_storage_error = std::max(max_storage_error, minimum_storage_error);
            error2 += error * error;
            reference2 += expected * expected;
            max_error = std::max(max_error, error);
            max_reference = std::max(max_reference, std::abs(expected));
        }
    }
    const double relative_l2 = reference2 == 0 ? std::sqrt(error2) : std::sqrt(error2 / reference2);
    const double gross_bound = (mode == 2 ? 4.5e-5 : 1.0e-5) + (mode == 2 ? 2.8e-3 : 3.95e-3) * max_reference;
    const double floor_l2 = reference2 == 0 ? 0.0 : std::sqrt(storage_error2 / reference2);
    const bool original_exceeded = relative_l2 > 1.85e-3 || max_error > gross_bound;
    // Retain original aggregate limits where feasible; an independently proven
    // BF16 representability floor cannot be made smaller by changing kernels.
    const double l2_limit = std::max(1.85e-3, floor_l2 + 4.0e-6);
    const double gross_limit = std::max(gross_bound, max_storage_error + 4.0e-6 * max_reference + 1.0e-35);
    if (relative_l2 > l2_limit || max_error > gross_limit) {
        std::fprintf(stderr, "d=%d rows=%d mode=%d scale=%g L2=%g max=%g bound=%g\n",
                     d, rows, mode, scale, relative_l2, max_error, gross_bound);
        throw std::runtime_error("RMSNorm failed storage-aware aggregate numerical criterion");
    }
    return {relative_l2, max_error / gross_bound, max_arithmetic_fraction, original_exceeded};
}

void validate_rejection(ninfer::DeviceContext& context) {
    ninfer::DeviceBuffer input(16), weights(16), output(16);
    ninfer::Tensor x(input.p, ninfer::DType::BF16, {8});
    ninfer::Tensor w(weights.p, ninfer::DType::BF16, {8});
    ninfer::Tensor y(output.p, ninfer::DType::BF16, {8});
    for (float eps : {0.0f, -1.0f, std::numeric_limits<float>::infinity()}) {
        bool rejected = false;
        try { ninfer::ops::rmsnorm(x, w, eps, false, y, context.stream); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "RMSNorm accepted an invalid epsilon");
    }
    bool rejected = false;
    try { ninfer::ops::gated_rmsnorm(x, w, x, 1e-6f, y, {context.stream, 0}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "gated RMSNorm accepted an invalid execution view");
}
} // namespace

int main() try {
    ninfer::DeviceContext context;
    unsigned cases = 0;
    double max_l2 = 0.0, max_wide_l2 = 0.0, max_gross_ratio = 0.0;
    double max_arithmetic_fraction = 0.0;
    unsigned original_cutoff_exceedances = 0;
    for (int mode : {0, 1, 2}) {
        for (int d : {1, 3, 127, 128, 256, 2880, 4096, 5120, 6144, 8192}) {
            for (int rows : {1, 8, 33}) {
                auto result = run_case(context, d, rows, mode, 4.0f);
                max_l2 = std::max(max_l2, result.relative_l2);
                if (d > 3) { max_wide_l2 = std::max(max_wide_l2, result.relative_l2); }
                max_gross_ratio = std::max(max_gross_ratio, result.gross_ratio);
                max_arithmetic_fraction = std::max(max_arithmetic_fraction, result.arithmetic_budget_fraction);
                original_cutoff_exceedances += result.original_statistical_cutoff_exceeded;
                ++cases;
            }
        }
        for (float scale : {1.0e-20f, 128.0f}) {
            auto result = run_case(context, 6144, 257, mode, scale);
            max_l2 = std::max(max_l2, result.relative_l2);
            max_wide_l2 = std::max(max_wide_l2, result.relative_l2);
            max_gross_ratio = std::max(max_gross_ratio, result.gross_ratio);
            max_arithmetic_fraction = std::max(max_arithmetic_fraction, result.arithmetic_budget_fraction);
            original_cutoff_exceedances += result.original_statistical_cutoff_exceeded;
            ++cases;
        }
        (void)run_case(context, 128, 8, mode, 0.0f, true);
        ++cases;
    }
    validate_rejection(context);
    std::printf("PASS: %u RMSNorm cases; FP64 oracle max_relative_l2_all=%.9g max_relative_l2_wide=%.9g max_original_gross_bound_ratio=%.9g max_arithmetic_budget_fraction=%.9g original_cutoff_exceedances=%u; 18 small cases exact nearest BF16; guards and represented inputs exact\n",
                cases, max_l2, max_wide_l2, max_gross_ratio, max_arithmetic_fraction, original_cutoff_exceedances);
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
