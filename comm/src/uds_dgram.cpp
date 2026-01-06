#include "uds_dgram.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace comm {

static void throw_sys(const char* what) {
    const int e = errno;
    throw std::runtime_error(std::string(what) + ": " + std::strerror(e));
}

UdsDgram::UdsDgram(const std::string& bind_path) : bind_path_(bind_path) {
    fd_ = ::socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd_ < 0) throw_sys("socket(AF_UNIX,SOCK_DGRAM)");

    ::unlink(bind_path_.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", bind_path_.c_str());

    if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        throw_sys("bind(unix dgram)");
    }
}

UdsDgram::~UdsDgram() {
    if (fd_ >= 0) ::close(fd_);
    if (!bind_path_.empty()) ::unlink(bind_path_.c_str());
}

bool UdsDgram::set_nonblocking(bool on) {
    const int flags = ::fcntl(fd_, F_GETFL, 0);
    if (flags < 0) return false;
    const int new_flags = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return (::fcntl(fd_, F_SETFL, new_flags) == 0);
}

bool UdsDgram::send_to(const std::string& dest_path, const void* data, size_t bytes) const {
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", dest_path.c_str());

    const ssize_t n = ::sendto(fd_, data, bytes, 0,
                              reinterpret_cast<sockaddr*>(&addr),
                              sizeof(addr));
    return (n == static_cast<ssize_t>(bytes));
}

int UdsDgram::recv(void* out, size_t max_bytes) const {
    const ssize_t n = ::recvfrom(fd_, out, max_bytes, 0, nullptr, nullptr);
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        return -1;
    }
    return static_cast<int>(n);
}

} // namespace comm
