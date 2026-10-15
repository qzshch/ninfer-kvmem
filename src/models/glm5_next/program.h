#pragma once
#include "models/glm5_next/model.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "runtime/contract/token_batch.h"
#include <map>
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
    int input_capacity() const noexcept { return max_rows_; }

    int frontier(int lane) const { return frontiers_.at(lane); }

    void restore(int frontier, int snapshot_slot);

    int frontier() const noexcept { return position_; }

    const ProgramStats& stats() const noexcept { return stats_; }
private:
    struct DeviceState {
        std::array<DeviceBuffer, 9> recurrent;
        DeviceBuffer streams, metadata, hidden;
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
    std::array<DeviceContext*, 3> devices_;
    ProgramOptions options_;
    int position_ = 0;
    std::array<int, 8> frontiers_{};
    std::map<int, std::array<DecodeGraphExecutable, 4>> graphs_;
    int max_rows_ = 1, batch_ = 1, sequences_ = 1, outputs_ = 1, tier_ = 128;
    runtime::TokenBatchKind kind_ = runtime::TokenBatchKind::Decode;
    enum MetadataField { TokenIds, StateSlots, Positions, OutputRows, PredictedIds, SequenceOffsets };
    std::vector<int> host_metadata_;
    int* metadata(int device, MetadataField field) const noexcept;
    void batch_stage(int device);
    void batch_head();
    ProgramStats stats_;
    std::array<DeviceState, 3> state_;
    std::array<LayerState, 45> layers_;
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
    void peer(void* destination, int dst, const void* source, int src, std::size_t bytes);
    void capture(const float* streams, int layer, int device);
    void publish_retrieval_mean(LayerState& layer, int block, int device);
};
} // namespace ninfer::models::glm5_next
