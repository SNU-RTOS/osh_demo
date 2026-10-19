#include "pacing.hpp"
#include <cassert>
int main() {
    using namespace std::chrono;
    FramePacer p(10);
    FramePacer::Clock::time_point t{};
    assert(p.deadline(t)==t);
    p.started(t);
    assert(p.deadline(t+milliseconds(20))==t+milliseconds(100));
    assert(p.deadline(t+milliseconds(500))==t+milliseconds(500));
    p.started(t+milliseconds(500));
    assert(p.deadline(t+milliseconds(510))==t+milliseconds(600));
}
