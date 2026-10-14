#pragma once
#include "models/glm5_next/model.h"
#include "core/arena.h"
#include "kvmem/block_retrieval.h"
#include <array>
#include <memory>
#include <vector>

namespace ninfer::models::glm5_next {
struct ProgramOptions {
    int max_context             = 32768;
    unsigned kvmem_window_pages = 0;
};

struct ProgramStats {
    std::uint64_t target_steps = 0, kvmem_selections = 0, kvmem_restricted_steps = 0,
                  peer_bytes = 0, draft_calls = 0, drafted_tokens = 0, draft_context_tokens = 0;
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
    void restore(int frontier, int snapshot_slot);

    int frontier() const noexcept { return position_; }

    const ProgramStats& stats() const noexcept { return stats_; }
private:
    struct DeviceState {
        std::array<DeviceBuffer, 9> recurrent;
        DeviceBuffer streams;
        std::unique_ptr<WorkspaceArena> workspace;
    };

    struct LayerState {
        float* recurrent = nullptr;
        float* conv[3]{};
        DeviceBuffer latent, keys, gates, pools;
        std::unique_ptr<kvmem::RetrievalIndex> retrieval;
    };

    const Model& model_;
    const Parameters& parameters_;
    std::array<DeviceContext*, 3> devices_;
    ProgramOptions options_;
    int position_ = 0;
    ProgramStats stats_;
    std::array<DeviceState, 3> state_;
    std::array<LayerState, 45> layers_;
    DeviceBuffer features_;

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
                       float* collapsed, int device);
    void kda(const float* input, const KdaParameters& weights, LayerState& state, float* out,
             int device);
    void mla(const float* input, const MlaParameters& weights, LayerState& state, float* out,
             int device);
    void feed_forward(const float* input, const LayerParameters& weights, float* out);
    void peer(void* destination, int dst, const void* source, int src, std::size_t bytes);
    void capture(const float* streams, int layer, int device);
    void publish_retrieval_mean(LayerState& layer, int block, int device);
};
} // namespace ninfer::models::glm5_next
