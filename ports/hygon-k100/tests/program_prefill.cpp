#include "artifact/reader.h"
#include "models/glm5_next/frontend.h"
#include "models/glm5_next/program.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using models::glm5_next::Program;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main(int argc, char** argv) try {
    require(argc == 2, "explicit v3 artifact required; uses three resident devices");
    std::array<std::unique_ptr<DeviceContext>, 3> owned;
    std::array<DeviceContext*, 3> devices{};
    for (int d = 0; d < 3; ++d) {
        owned[d] = std::make_unique<DeviceContext>(d);
        devices[d] = owned[d].get();
    }
    artifact::Reader reader(argv[1]);
    models::glm5_next::Model model(reader, devices, nullptr, false, false);
    models::glm5_next::Frontend frontend(model.tokenizer());
    models::glm5_next::ProgramOptions options;
    options.max_context = 1024;
    options.max_lanes = 4;
    options.draft = false;
    options.graph = true;
    options.kvmem_window_pages = 0;
    Program program(model, devices, options);
    std::string history = "Remember marker RIVER PINE 27. ";
    while (frontend.encode(history).size() < 400) history += "alpha beta gamma. ";
    const std::array<std::vector<TokenId>, 3> prompts{
        frontend.encode("The sky is"), frontend.encode(history),
        frontend.encode("Write the next number: 7, 8, 9,")};
    const std::array<int, 3> slots{3, 0, 2}, lengths{8, 5, 9};
    for (int r = 0; r < 3; ++r)
        require(!prompts[r].empty() && prompts[r].size() + lengths[r] <= static_cast<std::size_t>(options.max_context),
                "fixture prompt capacity");
    std::array<std::vector<TokenId>, 3> reference;
    for (int r = 0; r < 3; ++r) {
        program.reset_lane(slots[r]);
        int anchor = 0;
        for (auto id : prompts[r]) anchor = program.target_batch({&id, 1}, {&slots[r], 1})[0];
        reference[r].push_back(anchor);
        while (reference[r].size() < static_cast<unsigned>(lengths[r])) {
            anchor = program.target_batch({&anchor, 1}, {&slots[r], 1})[0];
            reference[r].push_back(anchor);
        }
    }
    for (unsigned chunk : {1, 7, 64, 128, 1024}) {
        program.reset();
        const auto before = program.stats();
        std::array<unsigned, 3> cursor{};
        std::array<std::vector<int>, 3> actual;
        while (cursor[0] < prompts[0].size() || cursor[1] < prompts[1].size() ||
               cursor[2] < prompts[2].size()) {
            std::vector<int> input, outputs;
            std::vector<unsigned> finishing;
            std::vector<runtime::TokenSpan> spans;
            for (unsigned r = 0; r < 3; ++r) {
                const unsigned count = std::min<std::size_t>(
                    std::min(chunk, static_cast<unsigned>(program.input_capacity()) / 3),
                    prompts[r].size() - cursor[r]);
                if (!count) continue;
                const auto begin = static_cast<unsigned>(input.size());
                spans.push_back({static_cast<unsigned>(slots[r]), cursor[r], begin, count});
                input.insert(input.end(), prompts[r].begin() + cursor[r],
                             prompts[r].begin() + cursor[r] + count);
                if (cursor[r] + count == prompts[r].size()) {
                    outputs.push_back(begin + count - 1);
                    finishing.push_back(r);
                }
                cursor[r] += count;
            }
            auto predicted = program.execute({runtime::TokenBatchKind::Prefill, input, spans, outputs});
            require(predicted.size() == finishing.size(), "prefill output row mapping");
            for (unsigned j = 0; j < finishing.size(); ++j)
                actual[finishing[j]].push_back(predicted[j]);
        }
        for (;;) {
            std::vector<int> input, lanes;
            std::vector<unsigned> rows;
            for (int r = 2; r >= 0; --r)
                if (actual[r].size() < static_cast<unsigned>(lengths[r])) {
                    input.push_back(actual[r].back());
                    lanes.push_back(slots[r]);
                    rows.push_back(r);
                }
            if (input.empty()) break;
            auto predicted = program.target_batch(input, lanes);
            for (unsigned j = 0; j < rows.size(); ++j) actual[rows[j]].push_back(predicted[j]);
        }
        unsigned expected_outputs = 0;
        for (int r = 0; r < 3; ++r) {
            require(actual[r] == reference[r], "temporal prefill changed ordinary token IDs");
            require(program.frontier(slots[r]) == static_cast<int>(prompts[r].size()) + lengths[r] - 1,
                    "prefill/decode input frontier mismatch");
            expected_outputs += lengths[r];
        }
        require(program.stats().predicted_rows - before.predicted_rows == expected_outputs,
                "intermediate prompt rows executed the output head");
    }
    std::cout << "PASS native temporal prefill vs single-input generation; ragged spans, "
                 "remapped slots, context boundaries, native DSA, output selection, Decode Graphs; KVMem disabled\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
