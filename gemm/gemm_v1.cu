#include "cuda_runtime.h"
#include "gemm.cuh"
#include <stdio.h>

/**
 *  a:[M,K] [3,5]
 *  b:[K,N] [5,3]
 *  out:[M,N]
 *  优化点：
 *  1.基线代码中最大的问题是访存的问题，a中的每一行需要读取m次，b中的每一列需要读取n次
 *  2.采用共享内存可以缓解memory bound的问题
 *  3.共享内存的大小是有限的，所以不可能将整个矩阵放进来，可以采用分块的方式进行计算
 *  4.将矩阵out切分为[TILE_WIDTH,TILE_WIDTH]大小的多个矩阵块，那么a和b矩阵也需要安装TILE_WIDTH大小进行切分，形成了N个小的矩阵快
 *  5.将a和b中矩阵块的元素加载到共享内存，执行乘加操作
 */

#define TILE_WIDTH 16
__global__ void matmul_kernel(const float* a, const float* b, float* out, int M, int N, int K) {
    //线程索引
    int tx = threadIdx.x; //[0, blockDim.x]
    int ty = threadIdx.y; //[0, blockDim.y]

    //线程块起始位置,blockIdx.x [0, gridDim.x]=[0, M/blockDim.x]
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;
    //每个线程块中的共享内存
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    //分批次加载矩阵a和矩阵b到共享内存并计算
    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        if (row < M && m * TILE_WIDTH + tx < K) {
            sharedA[ty][tx] = a[row * K + m * TILE_WIDTH + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }

        if (col < N && m * TILE_WIDTH + ty < K) {
            sharedB[ty][tx] = b[(m * TILE_WIDTH + ty)  * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }

        // 同步，确保每个线程块中的所有线程都加载完数据
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += sharedA[ty][k] * sharedA[k][tx];
        }
        // 同步，确保线程在加载下一批数据之前完成计算
        __syncthreads();
    }
    if (row < M && col < N) {
        out[row * N + col] = sum;
    }
}