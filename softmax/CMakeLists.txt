cmake_minimum_required(VERSION 3.18)

## 创建静态库
#add_library(softmax_base STATIC softmax_base.cu)
#set_target_properties(softmax_base PROPERTIES
#    POSITION_INDEPENDENT_CODE ON
#    CUDA_SEPARABLE_COMPILATION ON
#)
## 创建可执行文件
#add_executable(softmax_checksum checksum.cu)
#target_link_libraries(softmax_checksum PRIVATE softmax_base)
#set_target_properties(softmax_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
## 创建可执行文件
#add_executable(softmax_benchmark benchmark.cu)
#target_link_libraries(softmax_benchmark PRIVATE softmax_base)
#set_target_properties(softmax_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

## 创建静态库
#add_library(softmax_v1 STATIC softmax_v1.cu)
#set_target_properties(softmax_v1 PROPERTIES
#    POSITION_INDEPENDENT_CODE ON
#    CUDA_SEPARABLE_COMPILATION ON
#)
## 创建可执行文件
#add_executable(softmax_checksum checksum.cu)
#target_link_libraries(softmax_checksum PRIVATE softmax_v1)
#set_target_properties(softmax_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
## 创建可执行文件
#add_executable(softmax_benchmark benchmark.cu)
#target_link_libraries(softmax_benchmark PRIVATE softmax_v1)
#set_target_properties(softmax_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

## 创建静态库
add_library(softmax_v2 STATIC softmax_v2.cu)
set_target_properties(softmax_v2 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
# 创建可执行文件
add_executable(softmax_checksum checksum.cu)
target_link_libraries(softmax_checksum PRIVATE softmax_v2)
set_target_properties(softmax_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# 创建可执行文件
add_executable(softmax_benchmark benchmark.cu)
target_link_libraries(softmax_benchmark PRIVATE softmax_v2)
set_target_properties(softmax_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)