#include "cuda_runtime.h"
#include "gemm.cuh"

/**
 *  a:[M,K] [3,5]
 *  b:[K,N] [5,3]
 *  out:[M,N]
    CUDA 的 Shared Memory 被分成 32 个 bank（每个 bank 一次只能处理一个线程的请求）。
    共享内存sharedA[ty][tx]，存储是 row-major，
    内部布局大概是：sharedA[0][0], sharedA[0][1], …, sharedA[1][0], sharedA[1][1], …。
    当线程访问 sharedA[ty][k] 时（k 在循环里变化），一个 warp（32 线程）里 不同的 threadIdx.x / threadIdx.y 可能会落到 相同的 bank。
 优化点：
    每一行之间的 stride 不再是 TILE_WIDTH，而是 TILE_WIDTH+1。
    因为 stride 和 32 不再是倍数，warp 在访问 sharedA[ty][k] 时，每个线程访问的数据会分布在不同 bank 上，从而消除 bank conflict。
 */
__global__ void matmul_kernel(const float* a, const float* b, float* out, int M, int N, int K) {
    int bx = blockIdx.x; //代表在C分块矩阵中的一列
    int by = blockIdx.y; //代表在C分块矩阵中的一行

    int tx = threadIdx.x; //代表C分块矩阵中列坐标的偏移
    int ty = threadIdx.y; //代表C分块矩阵中行坐标的偏移

    int row = by * TILE_WIDTH + ty; //代表C分块中的行索引
    int col = bx * TILE_WIDTH + tx; //代表C分块中的列索引

    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH + 1];

    float sum = 0.0f;
    for (int tile_index = 0; tile_index < (K + TILE_WIDTH - 1) / TILE_WIDTH; tile_index++) {
        int aCol = tile_index * TILE_WIDTH + tx;
        int bRow = tile_index * TILE_WIDTH + ty;

        if (row < M && aCol < K) {
            sharedA[ty][tx] = a[row * K + aCol];
        } else {
            sharedA[ty][tx] = 0.0f;
        }

        if (col < N && bRow < K) {
            sharedB[ty][tx] = b[bRow * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += sharedA[ty][k] * sharedB[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        out[row * N + col] = sum;
    }
}