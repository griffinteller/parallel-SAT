cmake_minimum_required(VERSION 3.22)
project(Terrain)

set(CMAKE_CXX_STANDARD 23)

include_directories(include)

set(parallel-sat-sources
   lib/sat.cpp
   lib/sat_parallel.cpp)

add_library(parallel-sat SHARED ${parallel-sat-sources})

set(solve-sources
   src/solve.cpp)

add_executable(solve ${solve-sources})
target_link_libraries(solve PRIVATE parallel-sat)
target_link_libraries(solve PRIVATE pthread)

set(CNF_PATH "" CACHE STRING "Path to the CNF file benchmark input")
if(CNF_PATH)
    message(STATUS "CNF_PATH is set to: ${CNF_PATH}")
    target_compile_definitions(solve PRIVATE CNF_PATH=\"${CNF_PATH}\")
endif()