cmake_minimum_required(VERSION 3.10)
project(shock_wave_tube)

set(CMAKE_CXX_STANDARD 11)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(shock_wave_tube main.cpp)
target_link_libraries (shock_wave_tube Eigen3::Eigen)