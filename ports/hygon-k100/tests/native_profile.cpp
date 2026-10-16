#include "native_profile.h"
#include <cstdio>
#include <stdexcept>

using namespace ninfer::k100;
void require(bool ok) { if (!ok) throw std::runtime_error("native profile contract"); }
template<class F> void rejects(F function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
}
int main() try {
    NativeProfile ordinary;
    ordinary.validate();
    require(ordinary.companion == Companion::None && ordinary.kvmem_window_pages == 0);
    ordinary.tensor_parallel = ordinary.graph = true;
    ordinary.lanes = 8; ordinary.decode_frames = 2; ordinary.repeat = 3;
    ordinary.validate();
    for (int window : {-1, 1, 2, 3, 5, 7}) rejects([&] {
        auto profile = ordinary; profile.kvmem_window_pages = window; profile.validate();
    });
    for (int window : {0, 4, 6, 512}) {
        auto profile = ordinary; profile.kvmem_window_pages = window; profile.validate();
    }
    rejects([] { NativeProfile p; p.verify = true; p.validate(); });
    NativeProfile mtp;
    mtp.select(Companion::Mtp); mtp.verify = mtp.graph = true; mtp.validate();
    require(mtp.kvmem_window_pages == 0);
    rejects([&] { auto p = mtp; p.kvmem_window_pages = 4; p.validate(); });
    rejects([&] { mtp.select(Companion::DFlash2); });
    rejects([&] { auto p = mtp; p.tensor_parallel = true; p.validate(); });
    rejects([&] { auto p = mtp; p.lanes = 2; p.decode_frames = 2; p.validate(); });
    rejects([&] { auto p = mtp; p.repeat = 2; p.validate(); });
    NativeProfile draft;
    draft.select(Companion::DFlash2); draft.verify = true; draft.validate();
    rejects([&] { auto p = draft; p.graph = true; p.validate(); });
    NativeProfile selected;
    selected.select(Companion::None);
    rejects([&] { selected.select(Companion::Mtp); });
    std::puts("PASS: ordinary/no-KVMem defaults, explicit companion verification and supported execution choices");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
