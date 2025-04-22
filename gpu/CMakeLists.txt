cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 20)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")

project(gpu CXX CUDA)


add_executable(gpu_solver host_directed_naive.cu)


