cmake_minimum_required(VERSION 3.1)

include_directories(${CMAKE_HOME_DIRECTORY}/include)

add_executable(Example1 example1.cpp)
target_link_libraries(Example1 qlearning)