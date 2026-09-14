#include "shm_ring.hpp"
#include "ipc_protocol.hpp"
#include <cassert>
#include <memory>
#include <thread>
#include <chrono>
#include <sys/mman.h>
#include <unistd.h>

int main() {
    const std::string name = "/osh_ring_test_" + std::to_string(getpid());
    {
        comm::ShmRingProducer producer({name, 24, 64});
        comm::ShmRingConsumer consumer(name);
        for (uint32_t slot = 0; slot < 24; ++slot) {
            producer.slot_ptr(slot)[0] = static_cast<uint8_t>(slot);
            producer.publish_slot_seq(slot, 100 + slot);
        }
        for (uint32_t slot = 0; slot < 24; ++slot) {
            assert(consumer.read_slot_seq(slot) == 100 + slot);
            assert(consumer.slot_ptr(slot)[0] == slot);
        }
        assert(consumer.slot_ptr(24) == nullptr);
    }
    shm_unlink(name.c_str());
    std::unique_ptr<comm::ShmRingProducer> late;
    std::thread creator([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        late = std::make_unique<comm::ShmRingProducer>(comm::ShmRingConfig{name, 24, 64});
    });
    comm::ShmRingConsumer waiting(name, 1000);
    creator.join();
    assert(waiting.slots() == 24);
    shm_unlink(name.c_str());
}
