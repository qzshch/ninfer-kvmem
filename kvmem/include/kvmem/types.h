#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace kvmem {

inline constexpr std::uint32_t kPageTokens = 64;

struct TokenRange {
    std::size_t begin = 0;
    std::size_t count = 0;
};

class WindowCapacityError : public std::invalid_argument {
public:
    WindowCapacityError(bool has_instructions, const std::string& message)
        : std::invalid_argument(message), instructions(has_instructions) {}
    const bool instructions;
};

} // namespace kvmem
