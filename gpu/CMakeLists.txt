cmake_minimum_required(VERSION 3.30)
project(gpu CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

add_executable(gpu_solver main.cu)


