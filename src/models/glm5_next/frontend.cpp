#include "models/glm5_next/frontend.h"
#include "text/unicode.h"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace ninfer::models::glm5_next {
namespace {
std::string utf8(unsigned c) {
    if (c < 128) return std::string(1, static_cast<char>(c));
    std::string s;
    s.push_back(static_cast<char>(0xc0 | (c >> 6)));
    s.push_back(static_cast<char>(0x80 | (c & 63)));
    return s;
}

std::uint64_t pair(int a, int b) {
    return (static_cast<std::uint64_t>(a) << 32) | static_cast<unsigned>(b);
}

constexpr auto Pattern =
    R"glm((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)glm";

void checkpoint(const PreparationControl& c) {
    if (c.cancellation.requested()) throw std::runtime_error("GLM preparation cancelled");
    if (c.deadline != std::chrono::steady_clock::time_point{} &&
        std::chrono::steady_clock::now() >= c.deadline)
        throw std::runtime_error("GLM preparation deadline exceeded");
}

std::string lossy_utf8(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (std::size_t i = 0; i < input.size();) {
        const auto a = static_cast<unsigned char>(input[i]);
        if (a < 128) {
            out.push_back(input[i++]);
            continue;
        }
        const int need = a >= 0xc2 && a <= 0xdf   ? 2
                         : a >= 0xe0 && a <= 0xef ? 3
                         : a >= 0xf0 && a <= 0xf4 ? 4
                                                  : 0;
        if (!need) {
            out += "\xef\xbf\xbd";
            ++i;
            continue;
        }
        int valid = 1;
        for (; valid < need && i + valid < input.size(); ++valid) {
            auto b = static_cast<unsigned char>(input[i + valid]);
            if (b < 0x80 || b > 0xbf) break;
            if (valid == 1 && ((a == 0xe0 && b < 0xa0) || (a == 0xed && b >= 0xa0) ||
                               (a == 0xf0 && b < 0x90) || (a == 0xf4 && b >= 0x90)))
                break;
        }
        if (valid == need)
            out.append(input.substr(i, need));
        else
            out += "\xef\xbf\xbd";
        i += valid;
    }
    return out;
}
} // namespace

Frontend::Frontend(const artifact::Json& m) {
    if (m.at("tokenizer.ggml.model") != "gpt2" || m.at("tokenizer.ggml.pre") != "glm4")
        throw std::invalid_argument("GLM requires its stored glm4 byte BPE tokenizer");
    tokens_ = m.at("tokenizer.ggml.tokens").get<std::vector<std::string>>();
    types_  = m.at("tokenizer.ggml.token_type").get<std::vector<int>>();
    if (tokens_.size() != 154880 || types_.size() != tokens_.size())
        throw std::invalid_argument("GLM tokenizer coverage mismatch");
    special_.resize(tokens_.size());
    for (int i = 0; i < static_cast<int>(tokens_.size()); ++i)
        if (!ids_.emplace(tokens_[i], i).second)
            throw std::invalid_argument("GLM duplicate tokenizer word");
    for (const auto& a : m.at("tokenizer.native.added_tokens")) {
        int id = a.at("id");
        if (id < 0 || id >= 154880 || a.at("content") != tokens_[id])
            throw std::invalid_argument("GLM added-token identity mismatch");
        added_.push_back(id);
        special_[id] = a.at("special").get<bool>();
    }
    think_open_  = ids_.at("<think>");
    think_close_ = ids_.at("</think>");
    for (auto key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                     "tokenizer.ggml.eom_token_id"})
        stop_.push_back(m.at(key).get<TokenId>());
    unsigned next = 256;
    for (unsigned b = 0; b < 256; ++b) {
        unsigned c       = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 ? b : next++;
        byte_symbols_[b] = utf8(c);
        byte_values_[c]  = b;
        if (!ids_.contains(byte_symbols_[b]))
            throw std::invalid_argument("GLM tokenizer lacks byte fallback");
    }
    int rank = 0;
    for (const auto& item : m.at("tokenizer.ggml.merges")) {
        const auto s   = item.get<std::string>();
        const auto pos = s.find(' ');
        if (pos == std::string::npos) throw std::invalid_argument("GLM invalid merge words");
        const auto a = ids_.find(s.substr(0, pos)), b = ids_.find(s.substr(pos + 1)),
                   result = ids_.find(s.substr(0, pos) + s.substr(pos + 1));
        if (a == ids_.end() || b == ids_.end() || result == ids_.end())
            throw std::invalid_argument("GLM merge lacks vocabulary word");
        if (!merges_.emplace(pair(a->second, b->second), Merge{rank++, result->second}).second)
            throw std::invalid_argument("GLM duplicate merge pair");
    }
    int error;
    PCRE2_SIZE offset;
    regex_ = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(Pattern), PCRE2_ZERO_TERMINATED,
                           PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr);
    if (!regex_) throw std::runtime_error("GLM tokenizer regex failed to compile");
    chat_ = std::make_unique<text::JinjaTemplate>(
        m.at("tokenizer.chat_template").get<std::string>(), "artifact:GLM chat template");
}

Frontend::~Frontend() {
    if (regex_) pcre2_code_free(static_cast<pcre2_code*>(regex_));
}

void Frontend::encode_ordinary(std::string_view input, std::vector<TokenId>& out) const {
    auto* code = static_cast<pcre2_code*>(regex_);
    std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
        pcre2_match_data_create_from_pattern(code, nullptr), pcre2_match_data_free);
    if (!match) throw std::bad_alloc();
    std::size_t offset = 0;
    while (offset < input.size()) {
        int rc = pcre2_match(code, reinterpret_cast<PCRE2_SPTR>(input.data()), input.size(), offset,
                             PCRE2_ANCHORED, match.get(), nullptr);
        if (rc < 0) throw std::invalid_argument("GLM tokenizer invalid UTF-8 or unmatched text");
        const auto* bounds = pcre2_get_ovector_pointer(match.get());
        if (bounds[0] != offset || bounds[1] <= offset)
            throw std::runtime_error("GLM tokenizer split made no progress");
        std::vector<int> words;
        words.reserve(bounds[1] - offset);
        for (std::size_t i = offset; i < bounds[1]; ++i)
            words.push_back(ids_.at(byte_symbols_[static_cast<unsigned char>(input[i])]));
        while (words.size() > 1) {
            int best = std::numeric_limits<int>::max(), at = -1, result = -1;
            for (std::size_t i = 0; i + 1 < words.size(); ++i) {
                const auto found = merges_.find(pair(words[i], words[i + 1]));
                if (found != merges_.end() && found->second.rank < best) {
                    best   = found->second.rank;
                    at     = i;
                    result = found->second.result;
                }
            }
            if (at < 0) break;
            words[at] = result;
            words.erase(words.begin() + at + 1);
        }
        out.insert(out.end(), words.begin(), words.end());
        offset = bounds[1];
    }
}

std::vector<TokenId> Frontend::encode(std::string_view input) const {
    std::vector<TokenId> out;
    std::size_t begin = 0;
    while (begin < input.size()) {
        std::size_t first = input.size();
        int id            = -1;
        for (int token : added_) {
            auto p = input.find(tokens_[token], begin);
            if (p < first ||
                (p == first && id >= 0 && tokens_[token].size() > tokens_[id].size())) {
                first = p;
                id    = token;
            }
        }
        encode_ordinary(input.substr(begin, first - begin), out);
        if (id < 0) break;
        out.push_back(id);
        begin = first + tokens_[id].size();
    }
    return out;
}

std::string Frontend::decode(std::span<const TokenId> ids, bool skip) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= static_cast<int>(tokens_.size()))
            throw std::invalid_argument("GLM decode token outside vocabulary");
        if (types_[id] != 1) {
            if ((!skip || !special_[id]) && types_[id] != 5) out += tokens_[id];
            continue;
        }
        const auto& s = tokens_[id];
        for (std::size_t i = 0; i < s.size();) {
            auto c = text::unicode_internal::utf8_codepoint_at(s, i, "GLM byte BPE token");
            const auto found = byte_values_.find(c.value);
            if (found == byte_values_.end())
                throw std::invalid_argument("GLM token is not byte encoded");
            out.push_back(static_cast<char>(found->second));
            i += c.length;
        }
    }
    return lossy_utf8(out);
}

bool Frontend::starts_reasoning(std::span<const TokenId> prompt) const {
    bool open = false;
    for (auto id : prompt) {
        if (id == think_open_) open = true;
        if (id == think_close_) open = false;
    }
    return open;
}

bool Frontend::default_stop(TokenId token) const {
    return std::find(stop_.begin(), stop_.end(), token) != stop_.end();
}

Frontend::Output Frontend::output(std::span<const TokenId> tokens, bool thinking,
                                  bool preserve) const {
    Output out;
    std::vector<TokenId> content, reasoning;
    for (auto id : tokens) {
        if (id == think_open_) {
            thinking = true;
            if (preserve) reasoning.push_back(id);
            continue;
        }
        if (id == think_close_) {
            if (preserve) reasoning.push_back(id);
            thinking = false;
            continue;
        }
        if (thinking) {
            reasoning.push_back(id);
            ++out.reasoning_tokens;
        } else
            content.push_back(id);
    }
    out.content   = decode(content, !preserve);
    out.reasoning = decode(reasoning, !preserve);
    return out;
}

std::vector<TokenId> Frontend::prepare(const PromptInput& input,
                                       const PreparationControl& control) const {
    checkpoint(control);
    if (input.messages.empty()) throw std::invalid_argument("GLM chat requires messages");
    nlohmann::ordered_json c = {{"messages", nlohmann::ordered_json::array()},
                                {"tools", nlohmann::ordered_json::array()},
                                {"add_generation_prompt", true}};
    for (const auto& message : input.messages) {
        std::string role;
        switch (message.role) {
        case ChatRole::System:
            role = "system";
            break;
        case ChatRole::Developer:
            role = "developer";
            break;
        case ChatRole::User:
            role = "user";
            break;
        case ChatRole::Assistant:
            role = "assistant";
            break;
        default:
            throw std::invalid_argument("GLM text profile does not admit tool messages");
        }
        if (!message.tool_calls.empty() || !message.tool_call_id.empty())
            throw std::invalid_argument("GLM text profile does not admit tool calls");
        std::string content;
        for (const auto& part : message.parts) {
            if (part.kind != MessagePartKind::Text)
                throw std::invalid_argument("GLM text profile does not admit media");
            content += part.text;
        }
        c["messages"].push_back({{"role", role},
                                 {"content", content},
                                 {"reasoning_content", message.reasoning_content}});
    }
    if (!input.options.chat_template_kwargs_json.empty()) {
        auto extra = nlohmann::ordered_json::parse(input.options.chat_template_kwargs_json);
        if (!extra.is_object())
            throw std::invalid_argument("GLM template arguments must be an object");
        for (const auto& item : extra.items()) {
            if (item.key() == "messages" || item.key() == "tools")
                throw std::invalid_argument("GLM template arguments cannot replace messages/tools");
            c[item.key()] = item.value();
        }
    }
    if (input.options.reasoning_effort)
        c["reasoning_effort"] = reasoning_effort_name(*input.options.reasoning_effort);
    if (input.options.enable_thinking) c["enable_thinking"] = *input.options.enable_thinking;
    if (input.options.preserve_thinking) c["clear_thinking"] = !*input.options.preserve_thinking;
    if (input.options.continuation != PromptContinuationMode::NewAssistantTurn ||
        !input.options.tool_jsons.empty())
        throw std::invalid_argument("GLM text profile requires a new assistant turn without tools");
    text::TemplateRenderOptions options;
    options.checkpoint = [&] { checkpoint(control); };
    auto rendered      = chat_->render(c, options);
    checkpoint(control);
    auto tokens = encode(rendered.text);
    checkpoint(control);
    return tokens;
}
} // namespace ninfer::models::glm5_next
