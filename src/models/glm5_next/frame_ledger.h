#pragma once
#include "runtime/contract/token_batch.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::models::glm5_next {
struct FrameTicket {
    const void* owner = nullptr;
    unsigned slot = 0;
    std::uint64_t generation = 0;
};

// Program-owned sequence write reservations. Completion means that all device
// writes and result transfers have finished, including for cancelled requests.
class FrameLedger {
public:
    FrameLedger(unsigned lanes, unsigned frames) : lanes_(lanes), frames_(frames) {
        if (!lanes || lanes > 8 || !frames || frames > 2 || frames > lanes)
            throw std::invalid_argument("GLM frame/sequence capacity");
    }
    FrameTicket reserve(const runtime::TokenBatch& batch) {
        if (failed_) throw std::logic_error("GLM frame ledger failed");
        for (const auto& span : batch.spans)
            if (span.lane >= lanes_ || busy_[span.lane] ||
                span.input_frontier != static_cast<unsigned>(frontiers_[span.lane]))
                throw std::logic_error("GLM sequence has pending writes or stale frontier");
        for (unsigned index = 0; index < frames_; ++index) {
            // Keep the temporal-capable primary frame available when only one
            // independent Decode unit is ready.
            const unsigned f = batch.kind == runtime::TokenBatchKind::Decode ? frames_ - 1 - index : index;
            if (slots_[f].active || (f && batch.kind != runtime::TokenBatchKind::Decode)) continue;
            auto& slot = slots_[f];
            // Prepare the owning record before changing any reservation.
            slot.spans.assign(batch.spans.begin(), batch.spans.end());
            slot.active = true;
            ++slot.generation;
            for (const auto& span : slot.spans) busy_[span.lane] = true;
            return {this, f, slot.generation};
        }
        throw std::logic_error("GLM execution frames exhausted");
    }
    void validate(FrameTicket ticket) const {
        if (failed_ || ticket.owner != this || ticket.slot >= frames_ ||
            !slots_[ticket.slot].active || slots_[ticket.slot].generation != ticket.generation)
            throw std::logic_error("GLM invalid or retired execution frame");
    }
    void complete(FrameTicket ticket) {
        validate(ticket);
        auto& slot = slots_[ticket.slot];
        for (const auto& span : slot.spans) {
            frontiers_[span.lane] += span.count;
            busy_[span.lane] = false;
        }
        slot.active = false;
    }
    bool active(unsigned frame) const { return slots_.at(frame).active; }
    bool busy(unsigned lane) const { return busy_.at(lane); }
    int frontier(unsigned lane) const { return frontiers_.at(lane); }
    void reset_lane(unsigned lane) {
        if (failed_ || lane >= lanes_ || busy_[lane])
            throw std::logic_error("GLM cannot reset a sequence with pending writes");
        frontiers_[lane] = 0;
    }
    void reset() {
        for (unsigned lane = 0; lane < lanes_; ++lane)
            if (busy_[lane]) throw std::logic_error("GLM cannot reset pending execution");
        if (failed_) throw std::logic_error("GLM cannot reset failed execution");
        frontiers_.fill(0);
    }
    void fail() noexcept { failed_ = true; }
private:
    struct Slot {
        bool active = false;
        std::uint64_t generation = 0;
        std::vector<runtime::TokenSpan> spans;
    };
    unsigned lanes_, frames_;
    bool failed_ = false;
    std::array<int, 8> frontiers_{};
    std::array<bool, 8> busy_{};
    std::array<Slot, 2> slots_;
};
} // namespace ninfer::models::glm5_next
