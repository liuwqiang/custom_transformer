cmake_minimum_required(VERSION 3.18)

# 创建静态库
add_library(embedding_base STATIC embedding_base.cu rope_base.cu)
set_target_properties(embedding_base PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
# 创建可执行文件
add_executable(embedding_checksum checksum.cu)
target_link_libraries(embedding_checksum PRIVATE embedding_base)
set_target_properties(embedding_checksum PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# 创建可执行文件
add_executable(embedding_benchmark benchmark.cu)
target_link_libraries(embedding_benchmark PRIVATE embedding_base)
set_target_properties(embedding_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)