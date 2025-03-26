cmake_minimum_required(VERSION 3.29)
project(Terrain)

set(CMAKE_CXX_STANDARD 23)

include_directories(include)

set(parallel-sat-sources
   lib/sat.cpp)

add_library(parallel-sat SHARED ${parallel-sat-sources})

set(solve-sources
   src/solve.cpp)

add_executable(solve ${solve-sources})
target_link_libraries(solve PRIVATE parallel-sat)
