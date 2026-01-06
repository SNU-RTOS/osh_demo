// dds_common/src/shm_ring.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <atomic>

namespace comm {

static constexpr uint32_t SHM_MAGIC = 0x4F53484D;     // "OSHM"
static constexpr uint32_t SHM_VERSION = 1;
static constexpr size_t   SHM_HDR_BYTES = 4096;       // one page, aligned

inline size_t align64(size_t x) { return (x + 63u) & ~size_t(63u); }

struct ShmHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t slots;
    uint32_t slot_bytes;
    std::atomic<uint64_t> slot_seq[16]; // supports up to 16 slots without changing header
    uint8_t _pad[SHM_HDR_BYTES - 4*4 - 16*8];
};
static_assert(sizeof(ShmHeader) == SHM_HDR_BYTES, "ShmHeader must be 4096 bytes");

struct ShmRingConfig {
    std::string name;     // e.g. "/comm_rgb_shm"
    uint32_t slots;       // 3..16
    uint32_t slot_bytes;  // aligned bytes per slot
};

class ShmRingProducer {
public:
    explicit ShmRingProducer(const ShmRingConfig& cfg);
    ~ShmRingProducer();

    ShmRingProducer(const ShmRingProducer&) = delete;
    ShmRingProducer& operator=(const ShmRingProducer&) = delete;

    uint8_t* slot_ptr(uint32_t slot);
    void publish_slot_seq(uint32_t slot, uint64_t seq);

    uint32_t slots() const { return hdr_->slots; }
    uint32_t slot_bytes() const { return hdr_->slot_bytes; }

private:
    int fd_{-1};
    void* base_{nullptr};
    size_t bytes_{0};
    ShmHeader* hdr_{nullptr};
    uint8_t* data_{nullptr};
};

class ShmRingConsumer {
public:
    explicit ShmRingConsumer(const std::string& name);
    ~ShmRingConsumer();

    ShmRingConsumer(const ShmRingConsumer&) = delete;
    ShmRingConsumer& operator=(const ShmRingConsumer&) = delete;

    const uint8_t* slot_ptr(uint32_t slot) const;
    uint64_t read_slot_seq(uint32_t slot) const;

    uint32_t slots() const { return hdr_->slots; }
    uint32_t slot_bytes() const { return hdr_->slot_bytes; }

private:
    int fd_{-1};
    void* base_{nullptr};
    size_t bytes_{0};
    ShmHeader* hdr_{nullptr};
    uint8_t* data_{nullptr};
};

} // namespace comm
