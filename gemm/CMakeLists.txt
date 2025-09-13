cmake_minimum_required(VERSION 3.18)

## 创建静态库
#add_library(gemm_base STATIC gemm_base.cu)
#set_target_properties(gemm_base PROPERTIES
#    POSITION_INDEPENDENT_CODE ON
#    CUDA_SEPARABLE_COMPILATION ON
#)
## 创建可执行文件
#add_executable(gemm_checksum checksum.cu)
#target_link_libraries(gemm_checksum PRIVATE gemm_base)
#set_target_properties(gemm_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
## 创建可执行文件
#add_executable(gemm_benchmark benchmark.cu)
#target_link_libraries(gemm_benchmark PRIVATE gemm_base)
#set_target_properties(gemm_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 创建静态库
add_library(gemm_v1 STATIC gemm_v1.cu)
set_target_properties(gemm_v1 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
# 创建可执行文件
add_executable(gemm_checksum checksum.cu)
target_link_libraries(gemm_checksum PRIVATE gemm_v1)
set_target_properties(gemm_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# 创建可执行文件
add_executable(gemm_benchmark benchmark.cu)
target_link_libraries(gemm_benchmark PRIVATE gemm_v1)
set_target_properties(gemm_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)