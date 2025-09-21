cmake_minimum_required(VERSION 3.18)

## 创建静态库
add_library(attention_base STATIC attention_base.cu)
set_target_properties(attention_base PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
# 创建可执行文件
add_executable(attention_checksum checksum.cu)
target_link_libraries(attention_checksum PRIVATE attention_base)
set_target_properties(attention_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# 创建可执行文件
add_executable(attention_benchmark benchmark.cu)
target_link_libraries(attention_benchmark PRIVATE attention_base)
set_target_properties(attention_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)