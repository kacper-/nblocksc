cmake_minimum_required(VERSION 3.20)
project(nblocksc)

set(CMAKE_CXX_STANDARD 14)

add_executable(main main.cpp)

add_executable(nblocksc nblocksc.cpp)

add_executable(testnn testnn.cpp)
