cmake_minimum_required(VERSION 3.16)
project(comm CXX)

if(BUILD_TESTING)
    find_package(Threads REQUIRED)
    add_executable(shm_ring_test src/shm_ring_test.cpp src/shm_ring.cpp)
    target_link_libraries(shm_ring_test PRIVATE Threads::Threads)
    add_test(NAME shm_ring COMMAND shm_ring_test)
endif()

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
