cmake_minimum_required(VERSION 3.16)
project(comm CXX)

add_library(comm STATIC
    src/shm_ring.cpp
    src/uds_dgram.cpp
)

target_include_directories(comm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src      # for msgs.h, msgsPubSubTypes.h
)

# FastDDS / FastRTPS / FastCDR headers (choose what exists in your container)
target_include_directories(comm PUBLIC
    /usr/include
)

# Link Fast RTPS & Fast CDR (FastDDS API headers come via fastrtps in many installs)
target_link_directories(comm PUBLIC
    /usr/lib
)

# target_link_libraries(comm PUBLIC
#     fastrtps
#     fastcdr
#     pthread
# )
