cmake_minimum_required(VERSION 3.18)
project(cuda_examples LANGUAGES CUDA CXX)

add_executable(layernorm layernorm_base.cu)

set_target_properties(layernorm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
