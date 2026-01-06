// comm/src/shm_ring.cpp
#include "shm_ring.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace comm {

static void throw_sys(const char* what) {
    const int e = errno;
    throw std::runtime_error(std::string(what) + ": " + std::strerror(e));
}

ShmRingProducer::ShmRingProducer(const ShmRingConfig& cfg) {
    if (cfg.name.empty() || cfg.name[0] != '/') {
        throw std::runtime_error("ShmRingProducer: name must start with '/' (POSIX shm)");
    }
    if (cfg.slots < 1 || cfg.slots > 16) {
        throw std::runtime_error("ShmRingProducer: slots must be 1..16");
    }
    if (cfg.slot_bytes == 0 || (cfg.slot_bytes % 64) != 0) {
        throw std::runtime_error("ShmRingProducer: slot_bytes must be nonzero and 64-byte aligned");
    }

    bytes_ = SHM_HDR_BYTES + size_t(cfg.slots) * size_t(cfg.slot_bytes);

    fd_ = shm_open(cfg.name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ < 0) throw_sys("shm_open");

    if (ftruncate(fd_, static_cast<off_t>(bytes_)) != 0) throw_sys("ftruncate");

    base_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (base_ == MAP_FAILED) throw_sys("mmap");

    hdr_ = reinterpret_cast<ShmHeader*>(base_);
    data_ = reinterpret_cast<uint8_t*>(base_) + SHM_HDR_BYTES;

    // Initialize header each time (simple and robust)
    hdr_->magic = SHM_MAGIC;
    hdr_->version = SHM_VERSION;
    hdr_->slots = cfg.slots;
    hdr_->slot_bytes = cfg.slot_bytes;
    for (uint32_t i = 0; i < 16; ++i) {
        hdr_->slot_seq[i].store(0, std::memory_order_relaxed);
    }
}

ShmRingProducer::~ShmRingProducer() {
    if (base_ && base_ != MAP_FAILED) {
        munmap(base_, bytes_);
    }
    if (fd_ >= 0) close(fd_);
}

uint8_t* ShmRingProducer::slot_ptr(uint32_t slot) {
    if (!hdr_) return nullptr;
    if (slot >= hdr_->slots) return nullptr;
    return data_ + size_t(slot) * size_t(hdr_->slot_bytes);
}

void ShmRingProducer::publish_slot_seq(uint32_t slot, uint64_t seq) {
    if (!hdr_) return;
    if (slot >= hdr_->slots) return;
    hdr_->slot_seq[slot].store(seq, std::memory_order_release);
}

ShmRingConsumer::ShmRingConsumer(const std::string& name) {
    if (name.empty() || name[0] != '/') {
        throw std::runtime_error("ShmRingConsumer: name must start with '/' (POSIX shm)");
    }

    fd_ = shm_open(name.c_str(), O_RDWR, 0666);
    if (fd_ < 0) throw_sys("shm_open(existing)");

    struct stat st{};
    if (fstat(fd_, &st) != 0) throw_sys("fstat");
    bytes_ = static_cast<size_t>(st.st_size);

    base_ = mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (base_ == MAP_FAILED) throw_sys("mmap");

    hdr_ = reinterpret_cast<ShmHeader*>(base_);
    if (hdr_->magic != SHM_MAGIC) {
        throw std::runtime_error("ShmRingConsumer: bad magic (producer not initialized?)");
    }
    if (hdr_->version != SHM_VERSION) {
        throw std::runtime_error("ShmRingConsumer: version mismatch");
    }
    if (hdr_->slots < 1 || hdr_->slots > 16) {
        throw std::runtime_error("ShmRingConsumer: invalid slots");
    }

    data_ = reinterpret_cast<uint8_t*>(base_) + SHM_HDR_BYTES;
}

ShmRingConsumer::~ShmRingConsumer() {
    if (base_ && base_ != MAP_FAILED) {
        munmap(base_, bytes_);
    }
    if (fd_ >= 0) close(fd_);
}

const uint8_t* ShmRingConsumer::slot_ptr(uint32_t slot) const {
    if (!hdr_) return nullptr;
    if (slot >= hdr_->slots) return nullptr;
    return data_ + size_t(slot) * size_t(hdr_->slot_bytes);
}

uint64_t ShmRingConsumer::read_slot_seq(uint32_t slot) const {
    if (!hdr_) return 0;
    if (slot >= hdr_->slots) return 0;
    return hdr_->slot_seq[slot].load(std::memory_order_acquire);
}

} // namespace comm
