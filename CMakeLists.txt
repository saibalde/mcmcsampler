cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(mcmcsampler VERSION 0.1.0)

find_package(Eigen3 REQUIRED)

add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
endif()
