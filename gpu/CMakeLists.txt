cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 20)

project(gpu CXX CUDA)


add_executable(hdn host_directed_naive.cu)
add_executable(hddp host_directed_decision_parallel.cu)

foreach(target hdn hddp)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        >)
endforeach()
