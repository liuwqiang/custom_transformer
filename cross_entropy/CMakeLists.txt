cmake_minimum_required(VERSION 3.18)

# 创建静态库
add_library(cross_entropy_base STATIC cross_entropy_base.cu)
set_target_properties(cross_entropy_base PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
# 创建可执行文件
add_executable(cross_entropy_checksum checksum.cu)
target_link_libraries(cross_entropy_checksum PRIVATE cross_entropy_base)
set_target_properties(cross_entropy_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# 创建可执行文件
add_executable(cross_entropy_benchmark benchmark.cu)
target_link_libraries(cross_entropy_benchmark PRIVATE cross_entropy_base)
set_target_properties(cross_entropy_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)