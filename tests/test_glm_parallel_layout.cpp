#include "models/glm5_next/parallel_layout.h"
#include <algorithm>
#include <iostream>
#include <vector>
using namespace ninfer::models::glm5_next;

template<class Partition> void coverage(int total, Partition get) {
    std::vector<int> owner(total, -1);
    for (int rank = 0; rank < 3; ++rank) {
        const auto part = get(rank);
        if (part.begin < 0 || part.count < 1 || part.begin + part.count > total)
            throw std::runtime_error("row partition outside the original matrix");
        for (int row = part.begin; row < part.begin + part.count; ++row) {
            if (owner[row] != -1) throw std::runtime_error("duplicated physical row");
            owner[row] = rank;
        }
    }
    if (std::find(owner.begin(), owner.end(), -1) != owner.end()) throw std::runtime_error("unbound row");
}
int main() try {
    coverage(64, head_partition); coverage(154880, vocabulary_partition);
    for (int n : {2048, 12288}) coverage(n, [n](int r) { return mlp_partition(n, r); });
    for (int rank = 0; rank < 3; ++rank) {
        const auto h = head_partition(rank);
        if ((h.begin * 128) % 256 || (h.count * 128) % 256)
            throw std::runtime_error("KDA column interval violates block256 alignment");
    }
    for (int width : {128,256,512})
        coverage(64 * width, [width](int rank) {
            const auto h = head_partition(rank); return RowPartition{h.begin * width, h.count * width};
        });
    bool rejected = false;
    try { (void)mlp_partition(8192, 0); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("unsupported MLP geometry accepted");
    std::cout << "PASS: TP rows cover original heads, vocabulary and MLP exactly once; codec alignment\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
