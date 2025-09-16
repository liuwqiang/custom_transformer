cmake_minimum_required(VERSION 3.18)

# 创建静态库
add_library(rope_base STATIC rope_base.cu)
set_target_properties(rope_base PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
# 创建可执行文件
add_executable(rope_checksum checksum.cu)
target_link_libraries(rope_checksum PRIVATE rope_base)
set_target_properties(rope_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# 创建可执行文件
add_executable(rope_benchmark benchmark.cu)
target_link_libraries(rope_benchmark PRIVATE rope_base)
set_target_properties(rope_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)