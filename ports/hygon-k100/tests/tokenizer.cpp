#include "models/glm5_next/frontend.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("independent tokenizer fixture required");
        std::ifstream f(argv[1]);
        nlohmann::json fixture;
        f >> fixture;
        ninfer::models::glm5_next::Frontend frontend(fixture.at("metadata"));
        unsigned count = 0;
        for (const auto& item : fixture.at("cases")) {
            const auto expected = item.at("ids").get<std::vector<ninfer::TokenId>>();
            auto actual         = frontend.encode(item.at("text").get<std::string>());
            if (actual != expected) {
                std::fprintf(stderr, "fixture %u actual=%s expected=%s\n", count,
                             nlohmann::json(actual).dump().c_str(),
                             nlohmann::json(expected).dump().c_str());
                throw std::runtime_error("GLM BPE differs from pinned official tokenizer");
            }
            if (frontend.decode(actual) != item.at("decoded")) {
                std::fprintf(stderr, "decode fixture %u actual=%s expected=%s\n", count,
                             nlohmann::json(frontend.decode(actual)).dump().c_str(),
                             item.at("decoded").dump().c_str());
                throw std::runtime_error(
                    "GLM decoder/control-token semantics differ from official tokenizer");
            }
            ++count;
        }
        for (const auto& item : fixture.at("chats")) {
            ninfer::PromptInput prompt;
            prompt.options.chat_template_kwargs_json = item.at("kwargs").dump();
            for (const auto& m : item.at("messages")) {
                ninfer::ChatMessage message;
                const auto role = m.at("role").get<std::string>();
                message.role    = role == "system"      ? ninfer::ChatRole::System
                                  : role == "assistant" ? ninfer::ChatRole::Assistant
                                                        : ninfer::ChatRole::User;
                message.parts.push_back(
                    {ninfer::MessagePartKind::Text, m.at("content").get<std::string>(), {}});
                message.reasoning_content = m.value("reasoning_content", std::string{});
                prompt.messages.push_back(std::move(message));
            }
            if (frontend.prepare(prompt, {}) != item.at("ids").get<std::vector<ninfer::TokenId>>())
                throw std::runtime_error(
                    "GLM chat differs from pinned template plus official tokenizer");
            ++count;
        }
        for (const auto& item : fixture.at("decode_cases")) {
            if (frontend.decode(item.at("ids").get<std::vector<ninfer::TokenId>>()) !=
                item.at("decoded"))
                throw std::runtime_error(
                    "GLM partial/invalid UTF-8 replacement differs from official decoder");
            ++count;
        }
        std::printf("PASS: %u exact GLM BPE/decode/chat cases against official CPU tokenizers and "
                    "Jinja2; text/control/Unicode/number boundaries\n",
                    count);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
