#pragma once

#include <string>
#include <cstddef>
#include <cstdint>

namespace comm {

class UdsDgram {
public:
    // bind_path: local socket file to bind (will unlink first)
    explicit UdsDgram(const std::string& bind_path);
    ~UdsDgram();

    UdsDgram(const UdsDgram&) = delete;
    UdsDgram& operator=(const UdsDgram&) = delete;

    // Send a datagram to dest_path
    bool send_to(const std::string& dest_path, const void* data, size_t bytes) const;

    // Receive a datagram (non-blocking if you call set_nonblocking(true))
    // Returns number of bytes received, 0 if none (EAGAIN), -1 on error.
    int recv(void* out, size_t max_bytes) const;

    // Set socket O_NONBLOCK
    bool set_nonblocking(bool on);

private:
    int fd_{-1};
    std::string bind_path_;
};

} // namespace comm
