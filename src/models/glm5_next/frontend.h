#pragma once
#include "artifact/schema.h"
#include "ninfer/types.h"
#include "text/jinja.h"
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::models::glm5_next {
class Frontend {
public:
    struct Output {
        std::string content, reasoning;
        std::uint32_t reasoning_tokens = 0;
    };

    explicit Frontend(const artifact::Json& metadata);
    ~Frontend();
    Frontend(const Frontend&)            = delete;
    Frontend& operator=(const Frontend&) = delete;
    std::vector<TokenId> encode(std::string_view text) const;
    std::string decode(std::span<const TokenId> tokens, bool skip_control = true) const;
    std::vector<TokenId> prepare(const PromptInput& input, const PreparationControl& control) const;
    bool starts_reasoning(std::span<const TokenId> prompt) const;
    bool default_stop(TokenId token) const;
    Output output(std::span<const TokenId> tokens, bool starts_reasoning,
                  bool preserve_control) const;
private:
    struct Merge {
        int rank, result;
    };

    std::vector<std::string> tokens_;
    std::vector<int> types_;
    std::unordered_map<std::string, int> ids_;
    std::unordered_map<std::uint64_t, Merge> merges_;
    std::array<std::string, 256> byte_symbols_;
    std::unordered_map<std::uint32_t, unsigned char> byte_values_;
    std::vector<int> added_;
    std::vector<bool> special_;
    std::vector<TokenId> stop_;
    TokenId think_open_, think_close_;
    void* regex_ = nullptr;
    std::unique_ptr<text::JinjaTemplate> chat_;
    void encode_ordinary(std::string_view input, std::vector<TokenId>& out) const;
};
} // namespace ninfer::models::glm5_next
