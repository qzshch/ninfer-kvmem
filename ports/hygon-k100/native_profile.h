#pragma once
#include <stdexcept>

namespace ninfer::k100 {
enum class Companion { None, Mtp, DFlash2 };

// Native launcher choices. Engine/Program enforce their own execution contract.
struct NativeProfile {
    Companion companion = Companion::None;
    int kvmem_window_pages = 0, lanes = 1, decode_frames = 1, repeat = 1;
    bool graph = false, tensor_parallel = false, verify = false;

    void select(Companion value) {
        if (selected_) throw std::invalid_argument("choose one of --ordinary, --mtp or --dflash2");
        companion = value;
        selected_ = true;
    }
    void validate() const {
        if (kvmem_window_pages < 0 ||
            (kvmem_window_pages && (kvmem_window_pages < 4 || kvmem_window_pages % 2)))
            throw std::invalid_argument("KVMem window is zero (disabled) or an even page count >=4");
        if (lanes < 1 || lanes > 8 || decode_frames < 1 || decode_frames > 2 || decode_frames > lanes || repeat < 1)
            throw std::invalid_argument("invalid native lanes, frames or repeat count");
        if (companion != Companion::None && (lanes != 1 || decode_frames != 1 || tensor_parallel))
            throw std::invalid_argument("native companions currently require one lane/frame and PP execution");
        if (companion == Companion::DFlash2 && (graph || repeat != 1))
            throw std::invalid_argument("DFlash2 currently requires eager single-run execution");
        if (verify && (companion == Companion::None || repeat != 1))
            throw std::invalid_argument("greedy verification requires --mtp or --dflash2 and --repeat 1");
    }
private:
    bool selected_ = false;
};
} // namespace ninfer::k100
