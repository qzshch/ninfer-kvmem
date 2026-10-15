#include "ops/linear/portable/mmac_mapping.h"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using ninfer::ops::detail::MmacF32Mapping;
int main() try {
    for (int width : {1, 2}) for (int m : {1, 3, 8, 16}) for (int n : {1, 15, 16}) {
        constexpr int k = 132, parent_k = 192, begin = 32;
        std::vector<double> a(n * parent_k), b(m * k), output(n * m);
        std::vector<int> writes(n * m);
        for (unsigned i = 0; i < a.size(); ++i) a[i] = std::sin(i * .319);
        for (unsigned i = 0; i < b.size(); ++i) b[i] = std::cos(i * .713);
        std::array<std::array<double, 4>, 64> accum{};
        for (int base = 0; base < k; base += 4 * width) {
            std::array<std::array<double, 2>, 64> lhs{}, rhs{};
            for (int lane = 0; lane < 64; ++lane) for (int j = 0; j < width; ++j) {
                const int row = MmacF32Mapping::row(lane);
                const int c = base + MmacF32Mapping::contraction(lane, width, j);
                if (row < n && c < k) lhs[lane][j] = a[row * parent_k + begin + c];
                if (row < m && c < k) rhs[lane][j] = b[row * k + c];
            }
            // Independent ideal of one hardware instruction, from its full
            // 16x16x(4*width) contraction, on the represented operand registers.
            for (int lane = 0; lane < 64; ++lane) for (int reg = 0; reg < 4; ++reg) {
                const int row = lane % 16, col = reg * 4 + lane / 16;
                for (int c = 0; c < 4 * width; ++c)
                    accum[lane][reg] += lhs[row + (c / width) * 16][c % width] *
                                        rhs[col + (c / width) * 16][c % width];
            }
        }
        for (int lane = 0; lane < 64; ++lane) for (int reg = 0; reg < 4; ++reg) {
            const int row = MmacF32Mapping::row(lane), col = MmacF32Mapping::column(lane, reg);
            if (row < n && col < m) { output[col * n + row] = accum[lane][reg]; ++writes[col * n + row]; }
        }
        for (int t = 0; t < m; ++t) for (int r = 0; r < n; ++r) {
            double oracle = 0;
            for (int c = 0; c < k; ++c) oracle += a[r * parent_k + begin + c] * b[t * k + c];
            if (writes[t * n + r] != 1 || std::abs(output[t * n + r] - oracle) > 1e-12)
                throw std::runtime_error("MMAC operand/register mapping or masked K interval");
        }
    }
    std::cout << "PASS: CPU MMAC register contraction/layout, masked rows/columns and K intervals; no ISA execution\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
