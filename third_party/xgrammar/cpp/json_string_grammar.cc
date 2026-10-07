#include "json_string_grammar.h"

#include <picojson.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "grammar_functor.h"
#include "regex_converter.h"
#include "support/encoding.h"
#include "support/json_parse.h"

namespace xgrammar {
namespace {
std::string encode(const std::string& value) {
  auto json = picojson::value(value).serialize(false);
  return json.substr(1, json.size() - 2);
}

class JSONEncoder : public GrammarMutator {
  int32_t VisitByteString(const GrammarExpr& expr) final {
    std::string value;
    for (int byte : expr) value += static_cast<char>(byte);
    return builder_->AddByteString(encode(value));
  }

  int32_t VisitCharacterClass(const GrammarExpr& expr) final {
    std::vector<std::pair<int32_t, int32_t>> ranges;
    for (int i = 1; i < expr.size(); i += 2) ranges.emplace_back(expr[i], expr[i + 1]);
    if (expr[0]) {
      std::sort(ranges.begin(), ranges.end());
      std::vector<std::pair<int32_t, int32_t>> complement;
      int next = 0;
      for (const auto& [lo, hi] : ranges) {
        if (next < lo) complement.emplace_back(next, lo - 1);
        next = std::max(next, hi + 1);
      }
      if (next <= 0x10ffff) complement.emplace_back(next, 0x10ffff);
      ranges = std::move(complement);
    }
    std::vector<GrammarBuilder::CharacterClassElement> raw;
    std::vector<int32_t> choices;
    // ASCII control bytes, quote and backslash need JSON escapes. Surrogates are not scalar values.
    const std::pair<int, int> safe[] = {
        {0x20, 0x21}, {0x23, 0x5b}, {0x5d, 0xd7ff}, {0xe000, 0x10ffff}};
    for (const auto& [lo, hi] : ranges) {
      for (const auto& [a, b] : safe)
        if (std::max(lo, a) <= std::min(hi, b)) raw.emplace_back(std::max(lo, a), std::min(hi, b));
      for (int c = std::max(lo, 0); c <= std::min(hi, 0x5c); ++c) {
        if (c < 0x20 || c == '"' || c == '\\')
          choices.push_back(builder_->AddByteString(encode(std::string(1, static_cast<char>(c)))));
      }
    }
    if (!raw.empty()) choices.push_back(builder_->AddCharacterClass(raw));
    if (choices.empty()) return builder_->AddCharacterClass({{0, 0x10ffff}}, true);
    return choices.size() == 1 ? choices[0] : builder_->AddChoices(choices);
  }

  int32_t VisitCharacterClassStar(const GrammarExpr& expr) final {
    return builder_->AddRepeatFromExpr("json_char", VisitCharacterClass(expr), 0, -1);
  }
};
}  // namespace

std::string SchemaStringPattern(const std::string& pattern) {
  return NormalizeRegexPattern(pattern, false);
}

Grammar JSONStringPattern(const std::string& pattern) {
  return GrammarNormalizer::Apply(
      JSONEncoder().Apply(Grammar::FromRegex(SchemaStringPattern(pattern))));
}

Grammar JSONStringLength(int minimum, int maximum) {
  std::string count = "{" + std::to_string(minimum) + ",";
  if (maximum >= 0) count += std::to_string(maximum);
  count += "}";
  return GrammarNormalizer::Apply(JSONEncoder().Apply(Grammar::FromEBNF("root ::= [^]" + count)));
}
Grammar JSONStringExcept(const std::vector<std::string>& excluded) {
  struct Node {
    bool terminal = false;
    std::map<int, Node> next;
  } root;
  for (const auto& key : excluded) {
    Node* node = &root;
    for (size_t offset = 0; offset < key.size();) {
      const auto [cp, length] = ParseNextUTF8(key.c_str() + offset);
      XGRAMMAR_CHECK(length > 0) << "invalid property name UTF-8";
      node = &node->next[cp];
      offset += length;
    }
    node->terminal = true;
  }
  GrammarBuilder builder;
  const auto any_char = builder.AddCharacterClass({{0, 0x10ffff}});
  const auto any = builder.AddRepeatFromExpr("tail", any_char, 0, -1);
  const std::function<int32_t(const Node&)> build = [&](const Node& node) {
    std::vector<int32_t> alternatives;
    if (!node.terminal) alternatives.push_back(builder.AddEmptyStr());
    std::vector<GrammarBuilder::CharacterClassElement> excluded_chars;
    for (const auto& [cp, next] : node.next) excluded_chars.push_back({cp, cp});
    alternatives.push_back(
        builder.AddSequence({builder.AddCharacterClass(excluded_chars, true), any}));
    for (const auto& [cp, next] : node.next) {
      auto ref = builder.AddRuleRef(builder.AddRuleWithHint("key", build(next)));
      alternatives.push_back(builder.AddSequence({builder.AddByteString(CharToUTF8(cp)), ref}));
    }
    return builder.AddChoices(alternatives);
  };
  const auto root_id = builder.AddRule("root", build(root));
  return GrammarNormalizer::Apply(JSONEncoder().Apply(builder.Get(root_id)));
}
}  // namespace xgrammar
