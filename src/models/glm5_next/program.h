#pragma once
#include "models/glm5_next/model.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "runtime/contract/token_batch.h"
#include "models/glm5_next/frame_ledger.h"
#include <map>
#include <functional>
#include <span>
#include "kvmem/block_retrieval.h"
#include <array>
#include <memory>
#include <vector>

namespace ninfer::models::glm5_next {
struct ProgramOptions {
    int max_context             = 32768;
    unsigned kvmem_window_pages = 0;
    int max_lanes               = 1;
    bool graph                  = false;
    bool draft                  = true;
    bool mtp                    = false;
    unsigned prefill_chunk      = 1024;
    unsigned decode_frames      = 1;
    bool bf16_activations       = true;
};

struct ProgramStats {
    std::uint64_t target_steps = 0, kvmem_selections = 0, kvmem_restricted_steps = 0,
                  graph_captures = 0, graph_replays = 0, batch_rounds = 0, projection_columns = 0,
                  peer_bytes = 0, draft_calls = 0, drafted_tokens = 0, draft_context_tokens = 0,
                  predicted_rows = 0;
    std::array<std::size_t, 3> persistent_bytes{}, workspace_peak{};
};

class Program {
public:
    Program(const Model& model, std::array<DeviceContext*, 3> devices, ProgramOptions options);
    ~Program();
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    // One native target token, advancing recurrent/KV/index/feature frontier.
    // snapshot_slot=1..8 preserves the resulting recurrent state on GPU.
    std::int32_t target(std::int32_t token, int snapshot_slot = 0);
    std::vector<std::int32_t> propose(std::int32_t anchor, int width = 8);
    void reset();
    void reset_lane(int lane);
    std::int32_t mtp_forward(std::int32_t next_token, bool proposal);
    std::vector<std::int32_t> target_batch(std::span<const std::int32_t> tokens,
                                           std::span<const std::int32_t> lanes);
    std::vector<std::int32_t> execute(const runtime::TokenBatch& batch);
    // Single host worker; tickets own GPU work until finish() joins and commits.
    FrameTicket submit(const runtime::TokenBatch& batch);
    bool ready(FrameTicket ticket) const;
    std::vector<std::int32_t> finish(FrameTicket ticket);
    void drain() noexcept;
    int input_capacity() const noexcept { return frames_[0]->capacity; }
    unsigned frame_capacity() const noexcept { return options_.decode_frames; }
    bool primary_frame_available() const { return !ledger_.active(0); }
    bool lane_busy(unsigned lane) const { return ledger_.busy(lane); }
    std::size_t workspace_capacity_bytes() const noexcept;

    int frontier(int lane) const { return ledger_.frontier(lane); }

    void restore(int frontier, int snapshot_slot);

    int frontier() const noexcept { return position_; }

    const ProgramStats& stats() const noexcept { return stats_; }
private:
    struct DeviceState {
        DeviceBuffer streams, metadata, hidden;
        DeviceBuffer partial, received, coefficients, normalized;
        std::unique_ptr<CudaCompletionEvent> ready, copied, done;
        std::unique_ptr<WorkspaceArena> workspace;
    };

    struct LayerState {
        float* recurrent = nullptr;
        float* conv[3]{};
        DeviceBuffer latent, keys, gates, pools, means;
        std::unique_ptr<kvmem::RetrievalIndex> retrieval;
    };

    const Model& model_;
    const Parameters& parameters_;
    ProgramOptions options_;
    FrameLedger ledger_;
    bool tensor_parallel_ = false;
    int position_ = 0;
    std::array<std::array<DeviceBuffer, 9>, 3> recurrent_;
    struct Frame {
        // Destruction is explicitly ordered by Program, on each owning device.
        std::array<std::unique_ptr<DeviceContext>, 3> owned_devices;
        std::array<DeviceContext*, 3> devices{};
        std::array<DeviceState, 3> state;
        std::map<int, std::array<DecodeGraphExecutable, 4>> graphs;
        std::map<std::array<int, 4>, DecodeGraphExecutable> tensor_graphs;
        std::map<std::array<int, 2>, std::uint64_t> tensor_graph_families;
        std::uint64_t tensor_graph_clock = 0;
        int capacity = 1, batch = 1, sequences = 1, outputs = 1, tier = 128;
        runtime::TokenBatchKind kind = runtime::TokenBatchKind::Decode;
        std::vector<int> host_metadata;
        std::unique_ptr<PinnedHostBuffer> upload, result;
        std::unique_ptr<CudaCompletionEvent> completed;
        std::vector<runtime::TokenSpan> spans;
    };
    std::array<std::unique_ptr<Frame>, 2> frames_;
    Frame* frame_ = nullptr;
    std::array<std::array<std::unique_ptr<CudaCompletionEvent>, 3>, 8> sequence_ready_;
    enum MetadataField { TokenIds, StateSlots, Positions, OutputRows, PredictedIds, SequenceOffsets };
    int* metadata(int device, MetadataField field) const noexcept;
    void batch_stage(int device);
    void batch_head();
    ProgramStats stats_;
    std::array<LayerState, 45> layers_;
    std::array<std::array<LayerState, 45>, 2> parallel_layers_;
    LayerState& layer_state(int device, int layer) {
        return tensor_parallel_ && device ? parallel_layers_.at(device - 1).at(layer) : layers_.at(layer);
    }
    void tensor_phase(int layer, int device, bool feed_forward);
    void tensor_launch(int category, int component, const std::function<void()>& body);
    void tensor_finalize(int device);
    void tensor_reduce(bool routed);
    void tensor_execute();
    DeviceBuffer features_;
    LayerState mtp_state_;
    int mtp_frontier_ = 0;

    struct DraftCache {
        DeviceBuffer key, value;
    };

    std::array<DraftCache, 5> draft_cache_;
    int draft_frontier_ = 0;
    float* floats(WorkspaceArena& work, std::size_t count);
    void projection(const float* x, const Weight& w, float* y, int tokens, int device,
                    bool round = true);
    void mlp(const float* x, const MlpParameters& weights, float* out, int intermediate, int tokens,
             int device, float limit);
    void hyper_prepare(const float* streams, const HyperParameters& hc, float* coeff,
                       float* collapsed, int device, int tokens = 1);
    void kda(const float* input, const KdaParameters& weights, LayerState& state, float* out,
             int device, int tokens = 1);
    void mla(const float* input, const MlaParameters& weights, LayerState& state, float* out,
             int device, int tokens = 1);
    void feed_forward(const float* input, const LayerParameters& weights, float* out,
                      int tokens = 1);
    void mlp(const Tensor& input, const MlpParameters& weights, float* out, int tokens, int device);
    void kda(const Tensor& input, const KdaParameters& weights, LayerState& state,
             float* out, int device, int tokens);
    void mla(const Tensor& input, const MlaParameters& weights, LayerState& state,
             float* out, int device, int tokens);
    void feed_forward(const Tensor& input, const LayerParameters& weights, float* out, int tokens);
    void projection(const Tensor& input, const Weight& weights, Tensor& output, int device);
    void output_projection(const Tensor& input, const Weight& weights, float* output,
                           int begin, int device, int tokens);
    void peer(void* destination, int dst, const void* source, int src, std::size_t bytes);
    void capture(const float* streams, int layer, int device);
    void publish_retrieval_mean(LayerState& layer, int block, int device);
};
} // namespace ninfer::models::glm5_next
