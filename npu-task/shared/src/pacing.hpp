#pragma once
#include <algorithm>
#include <chrono>

// Deadlines are based on actual starts, never an accumulated schedule that bursts
// to catch up after a slow inference. Waiting happens before acquiring a grant.
class FramePacer {
public:
    using Clock = std::chrono::steady_clock;
    explicit FramePacer(double fps) : period_(std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(fps > 0 ? 1.0 / fps : 0))) {}
    Clock::time_point deadline(Clock::time_point now) const { return std::max(now, next_); }
    void started(Clock::time_point now) { next_ = now + period_; }
private:
    Clock::duration period_;
    Clock::time_point next_{};
};
