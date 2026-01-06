# =====================
# ==== Build Stage ====
# =====================
FROM ubuntu:22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates openssl \
    git build-essential pkg-config cmake libopencv-dev libjsoncpp-dev wget \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build HailoRT from source
## Set work directory, note this is a directory inside a container
WORKDIR /hailo_source 
## Clone HailoRT source from GitHub
RUN git clone https://github.com/hailo-ai/hailort.git . \
    && git checkout v4.21.0
## Run cmake to build HailoRT
### Include -DHAILO_BUILD_SERVICE=1 if you want to use multi-process service
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config release --target install

# ==========================
# ==== Deployment Stage ====
# ==========================
FROM ubuntu:22.04
# ros:humble-ros-base
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git \
    ca-certificates curl gnupg lsb-release \
    fastddsgen default-jre \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy shared objects of HailoRT from builder
COPY --from=builder /usr/local/lib/libhailort.so* /usr/lib/
## Uncomment the following if you want to use multi-process service
## COPY --from=builder /usr/local/bin/hailort_service /usr/bin/
# Copy header files of HailoRT from builder
COPY --from=builder /usr/local/include/hailo /usr/include/hailo

# Add ROS 2 apt repo key
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS 2 repository (Jammy)
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

# Install ROS base + FastRTPS RMW (this brings FastDDS / FastCDR headers/libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*

# Make interactive shells source ROS
SHELL ["/bin/bash", "-lc"]
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:/usr/lib:$LD_LIBRARY_PATH

WORKDIR /work
CMD ["/bin/bash"]
