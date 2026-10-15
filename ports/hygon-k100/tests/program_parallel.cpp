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
using namespace ninfer::models::glm5_next;

int main(int argc, char** argv) try {
    if (argc != 2) throw std::invalid_argument("explicit GLM v3 artifact required; loads PP then TP/EP serially");
    std::array<std::unique_ptr<DeviceContext>, 3> owned;
    std::array<DeviceContext*, 3> devices{};
    for (int d = 0; d < 3; ++d) { owned[d] = std::make_unique<DeviceContext>(d); devices[d] = owned[d].get(); }
    std::array<std::vector<int>, 3> prompts, expected;
    const std::array<int, 3> slots{3, 0, 2};
    for (auto plan : {Parallelism::Pipeline, Parallelism::TensorExpert}) {
        artifact::Reader reader(argv[1]);
        Model model(reader, devices, nullptr, false, false, plan);
        Frontend frontend(model.tokenizer());
        if (plan == Parallelism::Pipeline) {
            prompts[0] = frontend.encode("Complete: 1, 2, 3,");
            prompts[1] = frontend.encode("Remember the marker PINE 27. Summarize it in one sentence.");
            std::string text = "Remember marker RIVER 83. ";
            while (frontend.encode(text).size() < 400) text += "alpha beta gamma. ";
            prompts[2] = frontend.encode(text);
        }
        for (bool graph : {false, true}) for (unsigned frames : {1u, 2u}) for (bool compact : {false, true}) {
            ProgramOptions options;
            options.max_context = 1024; options.max_lanes = 4; options.draft = false;
            options.bf16_activations = compact; options.decode_frames = frames; options.graph = graph; options.kvmem_window_pages = 6; options.prefill_chunk = 128;
            Program program(model, devices, options);
            for (unsigned chunk : {1u, 17u, 128u}) {
                program.reset();
                std::array<unsigned, 3> positions{};
                std::array<int, 3> anchors{};
                while (positions[0] < prompts[0].size() || positions[1] < prompts[1].size() || positions[2] < prompts[2].size()) {
                    std::vector<int> input, output_rows, finishing;
                    std::vector<runtime::TokenSpan> spans;
                    for (int r = 0; r < 3; ++r) {
                        const auto count = std::min<std::size_t>(std::min<unsigned>(chunk, 42), prompts[r].size() - positions[r]);
                        if (!count) continue;
                        spans.push_back({static_cast<unsigned>(slots[r]), positions[r], static_cast<unsigned>(input.size()), static_cast<unsigned>(count)});
                        input.insert(input.end(), prompts[r].begin() + positions[r], prompts[r].begin() + positions[r] + count);
                        positions[r] += count;
                        if (positions[r] == prompts[r].size()) { output_rows.push_back(input.size() - 1); finishing.push_back(r); }
                    }
                    const auto ids = program.execute({runtime::TokenBatchKind::Prefill, input, spans, output_rows});
                    for (unsigned j = 0; j < ids.size(); ++j) anchors[finishing[j]] = ids[j];
                }
                std::array<std::vector<int>, 3> generated;
                const std::array<int, 3> lengths{8, 5, 9};
                for (int step = 0; step < 9; ++step) {
                    std::vector<int> input, lanes, owners;
                    for (int r = 0; r < 3; ++r) if (step < lengths[r]) {
                        generated[r].push_back(anchors[r]);
                        input.push_back(anchors[r]); lanes.push_back(slots[r]); owners.push_back(r);
                    }
                    std::vector<int> next;
                    if (frames == 1 || input.size() == 1) next = program.target_batch(input, lanes);
                    else {
                        const unsigned split = 1;
                        auto enqueue = [&](unsigned begin, unsigned end) {
                            std::vector<runtime::TokenSpan> spans;
                            std::vector<int> rows;
                            for (unsigned j = begin; j < end; ++j) {
                                spans.push_back({static_cast<unsigned>(lanes[j]),
                                    static_cast<unsigned>(program.frontier(lanes[j])), j - begin, 1});
                                rows.push_back(j - begin);
                            }
                            return program.submit({runtime::TokenBatchKind::Decode,
                                {input.data() + begin, end - begin}, spans, rows});
                        };
                        const auto first = enqueue(0, split);
                        bool rejected = false;
                        try { program.reset_lane(lanes[0]); } catch (const std::logic_error&) { rejected = true; }
                        if (!rejected) throw std::runtime_error("reset accepted a pending sequence writer");
                        const auto second = enqueue(split, input.size());
                        // Caller input may disappear immediately after submission.
                        std::fill(input.begin(), input.end(), -123);
                        auto tail = program.finish(second);
                        next = program.finish(first);
                        next.insert(next.end(), tail.begin(), tail.end());
                        rejected = false;
                        try { program.finish(first); } catch (const std::logic_error&) { rejected = true; }
                        if (!rejected) throw std::runtime_error("retired frame ticket accepted");
                    }
                    for (unsigned j = 0; j < next.size(); ++j) anchors[owners[j]] = next[j];
                }
                if (plan == Parallelism::Pipeline && !graph && chunk == 1 && frames == 1 && !compact) expected = generated;
                else if (generated != expected) throw std::runtime_error("native PP/TP/EP scalar/temporal/Graph generated token mismatch");
                if (graph && (!program.stats().graph_captures ||
                              program.stats().graph_replays <= program.stats().graph_captures))
                    throw std::runtime_error("ordinary Graph mode failed to reuse captured execution");
                program.reset_lane(slots[1]);
                if (program.frontier(slots[1]) != 0 || program.frontier(slots[0]) == 0)
                    throw std::runtime_error("TP physical-slot reset/frontier isolation");
            }
            if (frames == 2) {
                std::array<std::array<int, 2>, 2> serial{};
                program.reset();
                for (int r = 0; r < 2; ++r) {
                    int token = prompts[r][0], lane = slots[r];
                    for (int step = 0; step < 2; ++step) {
                        token = program.target_batch({&token, 1}, {&lane, 1})[0];
                        serial[r][step] = token;
                    }
                }
                program.reset();
                const auto enqueue = [&](int token, int lane, runtime::TokenBatchKind kind = runtime::TokenBatchKind::Decode) {
                    const runtime::TokenSpan span{static_cast<unsigned>(lane),
                        static_cast<unsigned>(program.frontier(lane)), 0, 1};
                    const int row = 0;
                    return program.submit({kind, {&token, 1}, {&span, 1}, {&row, 1}});
                };
                const auto first = enqueue(prompts[0][0], slots[0]);
                const auto second = enqueue(prompts[1][0], slots[1], runtime::TokenBatchKind::Prefill);
                const auto a = program.finish(first)[0];
                // Refill a completed frame while the other request still owns
                // its Prefill ticket/state reservation; no common round barrier.
                const auto continuation = enqueue(a, slots[0]);
                const auto a2 = program.finish(continuation)[0];
                const auto b = program.finish(second)[0];
                const auto b2 = program.finish(enqueue(b, slots[1]))[0];
                if (std::array{a, a2} != serial[0] || std::array{b, b2} != serial[1])
                    throw std::runtime_error("independent frame refill changed request state/results");
            }
        }
    }
    std::cout << "PASS: native PP versus TP/EP; ragged prefill, remapped slots, KVMem, eager/Graph ordinary Decode\n";
    return 0;
} catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
