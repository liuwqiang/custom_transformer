#include "cuda_runtime.h"
#include "gemm.cuh"

/**
 *  a:[M,K] [3,5]
 *  b:[K,N] [5,3]
 *  out:[M,N]
 */
__global__ void matmul_kernel(const float* a, const float* b, float* out, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row * col < M * N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            //a的M行中K个元素和b的N列中的K个元素相乘并相加
            sum += a[row * K + k] * b[k * N + col];
        }
        out[row * N + col] = sum;
    }
}