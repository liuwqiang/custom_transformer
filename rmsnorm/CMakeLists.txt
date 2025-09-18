cmake_minimum_required(VERSION 3.18)

# 创建静态库
add_library(rmsnorm_base STATIC rmsnorm_base.cu)
set_target_properties(rmsnorm_base PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
# 创建可执行文件
add_executable(rmsnorm_checksum checksum.cu)
target_link_libraries(rmsnorm_checksum PRIVATE rmsnorm_base)
set_target_properties(rmsnorm_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# 创建可执行文件
add_executable(rmsnorm_benchmark benchmark.cu)
target_link_libraries(rmsnorm_benchmark PRIVATE rmsnorm_base)
set_target_properties(rmsnorm_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)