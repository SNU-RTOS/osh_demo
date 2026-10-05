find_package(Threads REQUIRED)
add_executable(image_stream_bridge bridge.cpp)
target_link_libraries(image_stream_bridge PRIVATE comm Threads::Threads)
if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME image_stream_ipc
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/serve.py
                --bridge $<TARGET_FILE:image_stream_bridge> --smoke-test)
    endif()
endif()
