cmake_minimum_required(VERSION 3.1)

include_directories(${CMAKE_HOME_DIRECTORY}/include)

add_library(qlearning qlearning.cpp)