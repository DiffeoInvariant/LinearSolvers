cmake_minimum_required (VERSION 3.14.3)
project (jacobi)
add_library(jacobi SHARED jacobi.hpp)
include_directories("/usr/local/Cellar/eigen/3.3.7/include/eigen3")
include_directories("~/jacobi/include")

SET(CMAKE_CXX "clang++")
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "-Wall -O3")

set_target_properties(jacobi PROPERTIES LINKER_LANGUAGE CXX)

install(TARGETS jacobi DESTINATION lib)
install(FILES jacobi.hpp DESTINATION include)