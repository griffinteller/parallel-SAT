cmake_minimum_required(VERSION 3.30)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

project(gpu CXX CUDA)


add_executable(gpu_solver main.cu)


