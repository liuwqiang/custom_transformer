cmake_minimum_required(VERSION 3.18)

# 创建静态库 layernorm_base
add_library(layernorm_base STATIC layernorm_base.cu)
set_target_properties(layernorm_base PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# 创建可执行文件 layernorm_checksum
add_executable(layernorm_checksum checksum.cu)
target_link_libraries(layernorm_checksum PRIVATE layernorm_base)
set_target_properties(layernorm_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 创建可执行文件 benchmark
add_executable(layernorm_benchmark benchmark.cu)
target_link_libraries(layernorm_benchmark PRIVATE layernorm_base)
set_target_properties(layernorm_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)