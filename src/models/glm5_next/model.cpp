#include "models/glm5_next/model.h"
#include "models/glm5_next/parallel_layout.h"
#include "artifact/binder.h"
#include "artifact/views.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <variant>

namespace ninfer::models::glm5_next {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw artifact::ArtifactError(message);
}

void shard_parameters(Parameters& p, int rank) {
    p.output_begin = vocabulary_partition(rank).begin;
    const auto partition = head_partition(rank);
    const int h = partition.count, first = partition.begin, channels = h * 128;
    for (auto& l : p.layers) {
        if (l.linear) {
            auto& a = l.kda;
            a.head_begin = first;
            for (auto* conv : {&a.query_conv, &a.key_conv, &a.value_conv})
                *conv = Tensor(static_cast<float*>(conv->data) + first * 128 * 4,
                               DType::FP32, {4, 1, channels});
            a.negative_decay_rate = Tensor(static_cast<float*>(a.negative_decay_rate.data) + first,
                                           DType::FP32, {h});
            a.decay_bias = Tensor(static_cast<float*>(a.decay_bias.data) + first * 128,
                                  DType::FP32, {channels});
        } else {
            auto& a = l.mla;
            a.head_begin = first;
        }
        l.ffn.column_begin = mlp_partition(l.dense ? 12288 : 2048, rank).begin;
    }
}

void config_contract(const artifact::Json& c, const artifact::Json& d) {
    require(c.at("architectures") == artifact::Json::array({"Glm5NextForConditionalGeneration"}) &&
                c.at("hidden_size") == 4096 && c.at("num_hidden_layers") == 45 &&
                c.at("vocab_size") == 154880,
            "GLM text architecture mismatch");
    require(c.at("hc_mult") == 4 && c.at("hc_sinkhorn_iters") == 20 && c.at("hc_eps") == 1e-6 &&
                c.at("n_routed_experts") == 288 && c.at("num_experts_per_tok") == 8 &&
                c.at("first_k_dense_replace") == 3 && c.at("n_shared_experts") == 1,
            "GLM HC/MoE config mismatch");
    require(c.at("q_lora_rank") == 1536 && c.at("kv_lora_rank") == 512 &&
                c.at("num_attention_heads") == 64 && c.at("qk_nope_head_dim") == 256 &&
                c.at("v_head_dim") == 256 && c.at("qk_rope_head_dim") == 0,
            "GLM MLA config mismatch");
    require(c.at("index_head_dim") == 128 && c.at("index_n_heads") == 32 &&
                c.at("index_kpool") == 4 && c.at("index_topk") == 2048 &&
                c.at("index_kpool_compress") == true &&
                c.at("index_kpool_always_select_tail") == true,
            "GLM DSA config mismatch");
    require(c.at("rms_norm_eps") == 1e-5 && c.at("scoring_func") == "sigmoid" &&
                c.at("n_group") == 1 && c.at("norm_topk_prob") == true &&
                c.at("routed_scaling_factor") == 2.5 && c.at("swiglu_limit") == 10.0 &&
                c.at("moe_intermediate_size") == 2048 && c.at("intermediate_size") == 12288,
            "GLM numeric config mismatch");
    const auto& l = c.at("linear_attn_config");
    require(l.at("num_heads") == 64 && l.at("head_dim") == 128 &&
                l.at("gate_lower_bound") == -5.0 && l.at("short_conv_kernel_size") == 4,
            "GLM KDA config mismatch");
    require(c.at("layer_types").size() == 45, "GLM layer coverage mismatch");
    for (int i = 0; i < 45; ++i)
        require(c.at("layer_types").at(i) ==
                    (i % 4 == 3 ? "deepseek_sparse_attention" : "linear_attention"),
                "GLM layer mathematics mismatch");
    require(d.at("architectures") == artifact::Json::array({"DFlash2DraftModel"}) &&
                d.at("hidden_size") == 4096 && d.at("num_hidden_layers") == 5 &&
                d.at("vocab_size") == 154880 && d.at("num_attention_heads") == 32 &&
                d.at("num_key_value_heads") == 8 && d.at("head_dim") == 128 &&
                d.at("intermediate_size") == 12288 && d.at("rms_norm_eps") == 1e-5 &&
                d.at("is_causal") == false && d.at("sliding_window") == 2048,
            "GLM DFlash2 architecture mismatch");
    const auto& dc = d.at("dflash_config");
    require(dc.at("block_size") == 8 &&
                dc.at("target_layer_ids") == artifact::Json::array({5, 14, 24, 33, 42}) &&
                dc.at("conv_group_size") == 16 && dc.at("conv_kernel_size") == 2 &&
                dc.at("selector_rank") == 256 && dc.at("selector_top_k") == 16 &&
                dc.at("mask_token_id") == 154856,
            "GLM DFlash2 companion mismatch");
    require(d.at("rope_parameters").at("rope_theta") == 10000.0, "GLM DFlash2 RoPE mismatch");
}
} // namespace

Model::Model(const artifact::Reader& reader, std::array<DeviceContext*, 3> devices,
             const StartupObserver* observer, bool dflash, bool mtp, Parallelism parallelism)
    : devices_(devices), parallelism_(parallelism) {
    require(parallelism != Parallelism::TensorExpert || (!dflash && !mtp),
            "GLM TP/EP currently accepts ordinary execution only");
    const auto& dir = reader.directory();
    config_         = dir.component("text").config;
    draft_config_   = dir.component("dflash2").config;
    config_contract(config_, draft_config_);
    std::array<std::unique_ptr<artifact::Binder>, 3> binders;
    for (auto& b : binders) b = std::make_unique<artifact::Binder>(reader);

    struct Demand {
        int device;
        std::variant<Weight*, Tensor*> destination;
        artifact::ParameterReference reference;
    };

    std::vector<Demand> demands;
    auto add = [&](auto& destination, std::string name, artifact::Shape shape, int device,
                   std::optional<QType> exact = {}) {
        demands.push_back({device, &destination,
                           binders[device]->parameter(name, std::move(shape),
                                                      artifact::Residency::Device, exact)});
    };
    auto norm = [&](Tensor& w, const std::string& name, int size, int dev, QType type) {
        add(w, name, {static_cast<unsigned>(size)}, dev, type);
    };
    const auto row_weight = [&](Weight& destination, const std::string& name,
                                 int n, int k, RowPartition part, int dev) {
        demands.push_back({dev, &destination, binders[dev]->parameter_rows(name,
            {static_cast<unsigned>(n), static_cast<unsigned>(k)}, part.begin, part.count)});
    };
    const auto head_weight = [&](Weight& destination, const std::string& name,
                                  int n, int k, int per_head, int dev) {
        if (parallelism == Parallelism::TensorExpert) {
            const auto part = head_partition(dev);
            row_weight(destination, name, n, k, {part.begin * per_head, part.count * per_head}, dev);
        } else add(destination, name, {static_cast<unsigned>(n), static_cast<unsigned>(k)}, dev);
    };
    auto mlp = [&](MlpParameters& m, const std::string& p, int intermediate, int dev,
                   int banks = 1) {
        if (parallelism == Parallelism::TensorExpert && banks == 288) {
            const auto bind = [&](Weight& w, std::string name, int n, int k, int rows_per_bank) {
                demands.push_back({dev, &w, binders[dev]->parameter_rows(
                    name, {static_cast<unsigned>(n), static_cast<unsigned>(k)},
                    static_cast<unsigned>(dev * 96 * rows_per_bank),
                    static_cast<unsigned>(96 * rows_per_bank))});
            };
            bind(m.gate, p + "gate", banks * intermediate, 4096, intermediate);
            bind(m.up, p + "up", banks * intermediate, 4096, intermediate);
            bind(m.down, p + "down", banks * 4096, intermediate, 4096);
            return;
        }
        if (parallelism == Parallelism::TensorExpert) {
            const auto part = mlp_partition(intermediate, dev);
            row_weight(m.gate, p + "gate", intermediate, 4096, part, dev);
            row_weight(m.up, p + "up", intermediate, 4096, part, dev);
        } else {
            add(m.gate, p + "gate", {static_cast<unsigned>(banks * intermediate), 4096}, dev);
            add(m.up, p + "up", {static_cast<unsigned>(banks * intermediate), 4096}, dev);
        }
        add(m.down, p + "down",
            {static_cast<unsigned>(banks * 4096), static_cast<unsigned>(intermediate)}, dev);
    };
    auto bind_mla = [&](MlaParameters& a, const std::string& m, int dev) {
        add(a.query_lowrank, m + "query_lowrank", {1536, 4096}, dev);
        norm(a.query_norm, m + "query_norm", 1536, dev, QType::FP32);
        head_weight(a.query_heads, m + "query_heads", 16384, 1536, 256, dev);
        add(a.kv_latent, m + "kv_latent", {512, 4096}, dev);
        norm(a.kv_norm, m + "kv_norm", 512, dev, QType::FP32);
        head_weight(a.key_absorb, m + "key_absorb", 32768, 256, 512, dev);
        head_weight(a.value_expand, m + "value_expand", 16384, 512, 256, dev);
        add(a.output, m + "output", {4096, 16384}, dev);
        auto& x      = a.index;
        const auto z = m + "index/";
        add(x.query, z + "query", {4096, 1536}, dev);
        add(x.key, z + "key", {128, 4096}, dev);
        norm(x.key_norm, z + "key_norm", 128, dev, QType::FP32);
        norm(x.key_norm_bias, z + "key_norm_bias", 128, dev, QType::FP32);
        add(x.head_weights, z + "head_weights", {32, 4096}, dev, QType::FP32);
        add(x.position_bias, z + "position_bias", {4, 128}, dev, QType::FP32);
        add(x.pool_gate, z + "pool_gate", {128, 4096}, dev);
    };
    for (int rank = 0; rank < (parallelism == Parallelism::TensorExpert ? 3 : 1); ++rank) {
    auto& local = rank == 0 ? parameters_ : replicas_[rank - 1];
    if (rank == 0) add(local.embedding, "text/token_embedding", {154880, 4096}, 0);
    if (parallelism == Parallelism::TensorExpert)
        row_weight(local.output_head, "text/output_head", 154880, 4096, vocabulary_partition(rank), rank);
    else add(local.output_head, "text/output_head", {154880, 4096}, rank);
    norm(local.output_norm, "text/output_norm", 4096, rank, QType::FP32);
    for (int i = 0; i < 45; ++i) {
        auto& l       = local.layers[i];
        l.device      = parallelism == Parallelism::TensorExpert ? rank : i < 16 ? 0 : i < 31 ? 1 : 2;
        l.expert_begin = parallelism == Parallelism::TensorExpert ? rank * 96 : 0;
        l.linear      = i % 4 != 3;
        l.dense       = i < 3;
        const int dev = l.device;
        const auto p  = "text/layers/" + std::to_string(i) + "/";
        norm(l.attention_norm, p + "attention_norm", 4096, dev, QType::FP32);
        norm(l.ffn_norm, p + "ffn_norm", 4096, dev, QType::FP32);
        for (auto pair :
             {std::pair{&l.attention_hc, "attention_hc/"}, std::pair{&l.ffn_hc, "ffn_hc/"}}) {
            auto& a      = *pair.first;
            const auto h = p + pair.second;
            add(a.projection, h + "projection", {24, 16384}, dev);
            norm(a.base, h + "base", 24, dev, QType::FP32);
            norm(a.scale, h + "scale", 3, dev, QType::FP32);
        }
        if (l.linear) {
            auto& a      = l.kda;
            const auto k = p + "kda/";
            head_weight(a.query, k + "query", 8192, 4096, 128, dev);
            head_weight(a.key, k + "key", 8192, 4096, 128, dev);
            head_weight(a.value, k + "value", 8192, 4096, 128, dev);
            add(a.query_conv, k + "query_conv", {8192, 1, 4}, dev, QType::FP32);
            add(a.key_conv, k + "key_conv", {8192, 1, 4}, dev, QType::FP32);
            add(a.value_conv, k + "value_conv", {8192, 1, 4}, dev, QType::FP32);
            norm(a.negative_decay_rate, k + "negative_decay_rate", 64, dev, QType::FP32);
            norm(a.decay_bias, k + "decay_bias", 8192, dev, QType::FP32);
            head_weight(a.input_gate, k + "input_gate", 64, 4096, 1, dev);
            add(a.forget_in, k + "forget_in", {128, 4096}, dev);
            head_weight(a.forget_out, k + "forget_out", 8192, 128, 128, dev);
            add(a.output_gate_in, k + "output_gate_in", {128, 4096}, dev);
            head_weight(a.output_gate_out, k + "output_gate_out", 8192, 128, 128, dev);
            norm(a.output_norm, k + "output_norm", 128, dev, QType::FP32);
            add(a.output, k + "output", {4096, 8192}, dev);
        } else {
            auto& a      = l.mla;
            const auto m = p + "mla/";
            bind_mla(a, m, dev);
        }
        if (l.dense)
            mlp(l.ffn, p + "ffn/dense/", 12288, dev);
        else {
            mlp(l.ffn, p + "ffn/shared/", 2048, dev);
            mlp(l.experts, p + "ffn/experts/", 2048, dev, 288);
            add(l.router, p + "ffn/router", {288, 4096}, dev, QType::FP32);
            norm(l.router_bias, p + "ffn/router_bias", 288, dev, QType::FP32);
        }
    }
    }
    if (mtp) {
        require(dir.component("mtp").config.at("num_hidden_layers") == 1 &&
                    config_.at("num_nextn_predict_layers") == 1,
                "GLM native MTP layer config mismatch");
        auto& p = parameters_.mtp;
        add(p.fusion, "mtp/fusion", {4096, 8192}, 2);
        norm(p.embedding_norm, "mtp/embedding_norm", 4096, 2, QType::FP32);
        norm(p.hidden_norm, "mtp/hidden_norm", 4096, 2, QType::FP32);
        norm(p.output_norm, "mtp/output_norm", 4096, 2, QType::FP32);
        auto& l  = p.layer;
        l.device = 2;
        l.linear = false;
        l.dense  = false;
        norm(l.attention_norm, "mtp/attention_norm", 4096, 2, QType::FP32);
        norm(l.ffn_norm, "mtp/ffn_norm", 4096, 2, QType::FP32);
        bind_mla(l.mla, "mtp/mla/", 2);
        mlp(l.ffn, "mtp/ffn/shared/", 2048, 2);
        mlp(l.experts, "mtp/ffn/experts/", 2048, 2, 288);
        add(l.router, "mtp/ffn/router", {288, 4096}, 2, QType::FP32);
        norm(l.router_bias, "mtp/ffn/router_bias", 288, 2, QType::FP32);
    }
    if (dflash) {
        auto& d             = parameters_.draft;
        const std::string p = "dflash2/";
        add(d.feature_projection, p + "feature_projection", {4096, 20480}, 2, QType::BF16);
        norm(d.context_norm, p + "context_norm", 4096, 2, QType::BF16);
        norm(d.final_norm, p + "final_norm", 4096, 2, QType::BF16);
        add(d.selector_projection, p + "selector/hidden_projection", {256, 4096}, 2, QType::BF16);
        add(d.predecessor, p + "selector/predecessor_codebook", {154880, 256}, 2, QType::BF16);
        add(d.successor, p + "selector/successor_codebook", {154880, 256}, 2, QType::BF16);
        for (int i = 0; i < 5; ++i) {
            auto& l      = d.layers[i];
            const auto z = p + "layers/" + std::to_string(i) + "/";
            norm(l.attention_norm, z + "attention_norm", 4096, 2, QType::BF16);
            norm(l.ffn_norm, z + "ffn_norm", 4096, 2, QType::BF16);
            add(l.query, z + "attention/query", {4096, 4096}, 2, QType::BF16);
            add(l.key, z + "attention/key", {1024, 4096}, 2, QType::BF16);
            add(l.value, z + "attention/value", {1024, 4096}, 2, QType::BF16);
            add(l.output, z + "attention/output", {4096, 4096}, 2, QType::BF16);
            norm(l.query_norm, z + "attention/query_norm", 128, 2, QType::BF16);
            norm(l.key_norm, z + "attention/key_norm", 128, 2, QType::BF16);
            mlp(l.ffn, z + "ffn/", 12288, 2);
            for (auto pair : {std::pair{&l.attention_conv, "attention_conv/"},
                              std::pair{&l.mlp_conv, "mlp_conv/"}}) {
                add(pair.first->base_kernel, z + pair.second + "base_kernel", {2, 2, 4096}, 2,
                    QType::BF16);
                add(pair.first->kernel_projection, z + pair.second + "kernel_projection",
                    {1024, 4096}, 2, QType::BF16);
            }
        }
    }
    const auto tokenizer = binders[0]->resource("text", "gguf_tokenizer");
    const auto bytes     = binders[0]->host_object(tokenizer);
    tokenizer_           = artifact::Json::parse(
        std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    for (int dev = 0; dev < 3; ++dev) {
        require(devices_[dev] != nullptr, "GLM missing device context");
        devices_[dev]->bind_to_current_thread();
        backing_[dev] = artifact::materialize(reader, std::move(*binders[dev]).finish(),
                                              *devices_[dev], observer);
    }
    for (const auto& demand : demands) {
        const auto view = artifact::bind_view(demand.reference, backing_[demand.device]);
        if (auto w = std::get_if<Weight*>(&demand.destination))
            **w = native_weight(view);
        else {
            int dims[4] = {1, 1, 1, 1};
            require(view.shape.size() <= 4, "GLM direct parameter rank");
            for (unsigned i = 0; i < view.shape.size(); ++i)
                dims[i] = view.shape[view.shape.size() - 1 - i];
            *std::get<Tensor*>(demand.destination) =
                weight_tensor(view, {dims[0], dims[1], dims[2], dims[3]});
        }
    }
    if (parallelism == Parallelism::TensorExpert)
        for (int rank = 0; rank < 3; ++rank)
            shard_parameters(rank == 0 ? parameters_ : replicas_[rank - 1], rank);
}

Model::~Model() {
    for (int dev = 0; dev < 3; ++dev) {
        devices_[dev]->bind_to_current_thread_noexcept();
        backing_[dev] = {};
    }
}
} // namespace ninfer::models::glm5_next
