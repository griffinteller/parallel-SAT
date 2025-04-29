cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 20)

project(gpu CXX CUDA)


add_executable(iter1 iter1.cu)
add_executable(iter2 iter2.cu)
add_executable(iter3 iter3.cu)

foreach(target iter1 iter2 iter3)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        >)
endforeach()
