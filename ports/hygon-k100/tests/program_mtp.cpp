#include "artifact/reader.h"
#include "models/glm5_next/frontend.h"
#include "models/glm5_next/program.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace models::glm5_next;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F function) {
    bool rejected = false;
    try { function(); } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "invalid MTP transaction admitted");
}
int prefill(Program& program, const std::vector<int>& history, unsigned chunk) {
    int anchor = -1;
    const std::span<const int> all(history);
    for (std::size_t offset = 0; offset < all.size();) {
        const auto count = std::min<std::size_t>(chunk, all.size() - offset);
        const bool final = offset + count == all.size();
        anchor = program.mtp_prefill_chunk(all.subspan(offset, count),
            all.subspan(offset + 1, count - (final ? 1 : 0)), final);
        offset += count;
    }
    return anchor;
}

int main(int argc, char** argv) try {
    require(argc >= 2, "explicit v3 artifact required; optional --graph --fp32-activations");
    ProgramOptions options;
    options.max_context = 512; options.mtp = true; options.prefill_chunk = 128;
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--graph") options.graph = true;
        else if (flag == "--fp32-activations") options.bf16_activations = false;
        else throw std::invalid_argument("unknown MTP transaction check flag");
    }
    std::array<std::unique_ptr<DeviceContext>, 3> owned;
    std::array<DeviceContext*, 3> devices{};
    for (int d = 0; d < 3; ++d) { owned[d] = std::make_unique<DeviceContext>(d); devices[d] = owned[d].get(); }
    artifact::Reader reader(argv[1]);
    Model model(reader, devices, nullptr, false, true);
    Frontend frontend(model.tokenizer());
    std::string text = "Remember RIVER 83. ";
    while (frontend.encode(text).size() < 512) text += "alpha beta gamma. ";
    const auto tokens = frontend.encode(text);
    const std::array<int, 6> lengths{2, 3, 4, 126, 127, 128};
    std::array<std::vector<int>, 6> expected;
    {
        auto ordinary = options; ordinary.mtp = false;
        Program baseline(model, devices, ordinary);
        for (unsigned c = 0; c < lengths.size(); ++c) {
            baseline.reset();
            int anchor = -1;
            for (int i = 0; i < lengths[c]; ++i) anchor = baseline.target(tokens[i]);
            expected[c].push_back(anchor);
            for (int i = 0; i < 12; ++i) { anchor = baseline.target(anchor); expected[c].push_back(anchor); }
        }
    }
    Program program(model, devices, options);
    for (unsigned c = 0; c < lengths.size(); ++c) for (unsigned chunk : {1u, 7u, 128u}) {
        std::vector<int> history(tokens.begin(), tokens.begin() + lengths[c]);
        program.reset();
        require(prefill(program, history, chunk) == expected[c][0], "MTP packed prefill differs from ordinary");
        unsigned generated = 0;
        std::vector<std::pair<unsigned, int>> draft_checkpoints;
        // Forced candidates exercise every path regardless of natural MTP accuracy.
        for (auto [match, count] : {std::pair{true, 0u}, {false, 0u}, {false, 1u},
                                    {true, 1u}, {true, 2u}, {false, 1u}, {true, 2u}}) {
            const int anchor = expected[c][generated];
            const int real_draft = program.mtp_forward(anchor, true);
            draft_checkpoints.emplace_back(generated, real_draft);
            const int candidate = (expected[c][generated + 1] + (match ? 0 : 1)) % 154880;
            const auto before = program.frontier();
            const auto result = program.verify_mtp(anchor, candidate);
            require(program.lane_busy(0) && program.frontier() == before, "verification published before prefix decision");
            require(result.predictions[0] == expected[c][generated + 1] && result.valid_prefix == (match ? 2u : 1u),
                    "K1 first target output or valid prefix differs from ordinary");
            if (match) require(result.predictions[1] == expected[c][generated + 2], "K1 bonus differs from ordinary");
            rejects([&] { program.reset(); });
            rejects([&] { program.target(anchor); });
            if (!match) rejects([&] { program.commit_mtp(result.ticket, 2); });
            program.commit_mtp(result.ticket, count);
            require(!program.lane_busy(0) && program.frontier() == before + static_cast<int>(count), "wrong accepted input frontier");
            rejects([&] { program.commit_mtp(result.ticket, count); });
            generated += count;
        }
        // Rebuild each accepted history independently, then compare the native
        // proposal. This observes teacher retention and MTP cache rollback as
        // well as the main recurrent/index state tested by continuation above.
        for (auto [offset, proposal] : draft_checkpoints) {
            auto clean = history;
            clean.insert(clean.end(), expected[c].begin(), expected[c].begin() + offset);
            program.reset();
            require(prefill(program, clean, 7) == expected[c][offset], "reconstructed target teacher mismatch");
            require(program.mtp_forward(expected[c][offset], true) == proposal, "MTP teacher/cache differs after prefix or abort");
        }
    }
    // Final context positions: proposal admission must fail without consuming
    // its cache when only C1 fits. A complete K1 may exactly fill the capacity.
    program.reset();
    const std::vector<int> tail(tokens.begin(), tokens.begin() + 511);
    int anchor = prefill(program, tail, 128);
    rejects([&] { program.mtp_forward(anchor, true); });
    program.target(anchor);
    require(program.frontier() == 512, "final ordinary input did not fit");
    rejects([&] { program.target(anchor); });
    auto pair_tail = tail; pair_tail.pop_back();
    program.reset(); anchor = prefill(program, pair_tail, 128);
    const int first = program.target(anchor), bonus = program.target(first);
    program.reset(); anchor = prefill(program, pair_tail, 128);
    program.mtp_forward(anchor, true);
    const auto last = program.verify_mtp(anchor, first);
    require(last.valid_prefix == 2 && last.predictions[0] == first && last.predictions[1] == bonus,
            "last two target positions differ from ordinary");
    program.commit_mtp(last.ticket, 2);
    require(program.frontier() == 512, "K1 failed to fill exact context capacity");
    rejects([&] { program.mtp_forward(bonus, true); });
    std::cout << "PASS: native two-position verifier, forced accept/reject/abort, prefix truncation, "
                 "KDA/conv/index boundaries, teacher/cache continuation, graph geometry and chunked prefill\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
