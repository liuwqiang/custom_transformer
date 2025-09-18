cmake_minimum_required(VERSION 3.18)
project(transformer LANGUAGES CUDA CXX)

# 添加 layernorm 子目录
#add_subdirectory(layernorm)
# 添加 softmax 子目录
#add_subdirectory(softmax)
# 添加cross_entropy 子目录
#add_subdirectory(cross_entropy)
# 添加gemm 子目录
#add_subdirectory(gemm)
# 添加embedding 子目录
#add_subdirectory(embedding)
# 添加rope 子目录
#add_subdirectory(rope)
# 添加rmsnorm 子目录
add_subdirectory(rmsnorm)