#pragma once

#include <string>
#include <cstddef>

namespace comm {

class UdsDgram {
public:
    explicit UdsDgram(const std::string& bind_path);
    ~UdsDgram();

    UdsDgram(const UdsDgram&) = delete;
    UdsDgram& operator=(const UdsDgram&) = delete;

    bool send_to(const std::string& dest_path, const void* data, size_t bytes) const;

    // Returns:
    //  >0 : bytes received
    //   0 : no data (EAGAIN/EWOULDBLOCK if nonblocking)
    //  -1 : error
    int recv(void* out, size_t max_bytes) const;

    bool set_nonblocking(bool on);

private:
    int fd_{-1};
    std::string bind_path_;
};

} // namespace comm
