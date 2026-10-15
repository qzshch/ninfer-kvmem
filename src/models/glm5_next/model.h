#pragma once
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/weight.h"
#include <array>
#include <cstdint>

namespace ninfer::models::glm5_next {
enum class Parallelism { Pipeline, TensorExpert };
struct HyperParameters {
    Weight projection;
    Tensor base, scale;
};

struct MlpParameters {
    Weight gate, up, down;
    int column_begin = 0;
};

struct KdaParameters {
    int head_begin = 0;
    Weight query, key, value;
    Tensor query_conv, key_conv, value_conv, negative_decay_rate, decay_bias;
    Weight input_gate, forget_in, forget_out, output_gate_in, output_gate_out, output;
    Tensor output_norm;
};

struct IndexParameters {
    Weight query, key, head_weights, pool_gate;
    Tensor key_norm, key_norm_bias, position_bias;
};

struct MlaParameters {
    int head_begin = 0;
    Weight query_lowrank, query_heads, kv_latent, key_absorb, value_expand, output;
    Tensor query_norm, kv_norm;
    IndexParameters index;
};

struct LayerParameters {
    int device  = 0;
    int expert_begin = 0;
    bool linear = false, dense = false;
    Tensor attention_norm, ffn_norm;
    HyperParameters attention_hc, ffn_hc;
    KdaParameters kda;
    MlaParameters mla;
    MlpParameters ffn, experts;
    Weight router;
    Tensor router_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    Weight kernel_projection;
};

struct DraftLayerParameters {
    Weight query, key, value, output;
    Tensor attention_norm, ffn_norm, query_norm, key_norm;
    MlpParameters ffn;
    DynamicConvParameters attention_conv, mlp_conv;
};

struct DraftParameters {
    Weight feature_projection, selector_projection, predecessor, successor;
    Tensor context_norm, final_norm;
    std::array<DraftLayerParameters, 5> layers;
};

struct MtpParameters {
    Weight fusion;
    Tensor embedding_norm, hidden_norm, output_norm;
    LayerParameters layer;
};

struct Parameters {
    Weight embedding, output_head;
    int output_begin = 0;
    Tensor output_norm;
    std::array<LayerParameters, 45> layers;
    DraftParameters draft;
    MtpParameters mtp;
};

// Immutable model owns original encoded device backing and tokenizer metadata.
// No host weight residency, streaming experts, runtime repacking or external engine.
class Model {
public:
    Model(const artifact::Reader& reader, std::array<DeviceContext*, 3> devices,
          const StartupObserver* observer = nullptr, bool dflash = false, bool mtp = false,
          Parallelism parallelism = Parallelism::Pipeline);
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    const Parameters& parameters() const noexcept { return parameters_; }
    const Parameters& parameters(int device) const {
        return device == 0 || parallelism_ == Parallelism::Pipeline ? parameters_ : replicas_.at(device - 1);
    }
    Parallelism parallelism() const noexcept { return parallelism_; }

    const artifact::Json& config() const noexcept { return config_; }

    const artifact::Json& draft_config() const noexcept { return draft_config_; }

    const artifact::Json& tokenizer() const noexcept { return tokenizer_; }

    const artifact::MaterializationStats& stats(int device) const {
        return backing_.at(device).stats();
    }
private:
    std::array<DeviceContext*, 3> devices_;
    std::array<artifact::MaterializedArtifact, 3> backing_;
    Parameters parameters_;
    std::array<Parameters, 2> replicas_;
    Parallelism parallelism_ = Parallelism::Pipeline;
    artifact::Json config_, draft_config_, tokenizer_;
};
} // namespace ninfer::models::glm5_next
