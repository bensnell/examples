cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(simple_test)

find_package(Torch REQUIRED)

add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test "${TORCH_LIBRARIES}")
set_property(TARGET simple_test PROPERTY CXX_STANDARD 14)
