#include "ops/linear_attention/kimi_delta_attention/chunked_dtk.h"
#include <iostream>
#include <random>
#include <vector>

int main() try {
    using namespace ninfer::ops::detail::kimi_delta_attention;
    std::mt19937 rng(30271);
    for (int sequences = 1; sequences <= 8; ++sequences)
        for (int rows = sequences; rows <= 1024; ++rows) {
            const auto bytes = dtk_packed_workspace_bytes(24, rows, sequences);
            if (rows < 16) {
                if (bytes) throw std::runtime_error("recurrent route requested packet storage");
                continue;
            }
            for (int sample = 0; sample < 16; ++sample) {
                std::vector<bool> used(bytes / sizeof(DtkChunk) / 24);
                int begin = 0;
                for (int seq = 0; seq < sequences; ++seq) {
                    const int remaining = rows - begin - (sequences - seq - 1);
                    const int count = seq == sequences - 1 ? rows - begin : 1 + rng() % remaining;
                    for (int t = 0; t < count; t += 16) {
                        const auto index = static_cast<unsigned>(dtk_packet_base(begin, seq) + t / 16);
                        if (index >= used.size() || used[index])
                            throw std::runtime_error("ragged sequence packets overlap or exceed the resource quote");
                        used[index] = true;
                    }
                    begin += count;
                }
                if (begin != rows) throw std::runtime_error("incomplete test span partition");
            }
        }
    std::cout << "PASS: ragged KDA packet disjointness and bounded resident workspace\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
