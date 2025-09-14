#include "cuda_runtime.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>

#include "gemm.cuh"

#define CUDA_CHECK(call) \
if ((call) != cudaSuccess) { \
fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(call), __FILE__, __LINE__); \
exit(1); \
}

void rand(float* elements, int size) {
    for(int i = 0; i < size; i++){
        elements[i] = (float)rand() / RAND_MAX;
    }
}

bool check(float *out, float *res, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(out[i] - res[i]) > 1e-4) {
            return false;
        }
    }
    return true;
}

/**
 *  a:[M,K] [3,5]
 *  b:[K,N] [5,3]
 *  out:[M,N]
 */
void matmul_cpu(float* a, float* b, float* out, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                //将a中M行的K个元素和b中N列的K个元素相乘并相加
                sum += a[m * K + k] * b[k * N + n];
            }
            out[m * N + n] = sum;
        }
    }
}

void matmul_gpu_base(float* a, float* b, float* out, int M, int N, int K) {
    //分配显存
    float* d_a;
    cudaMalloc(&d_a, M * K * sizeof(float));
    float* d_b;
    cudaMalloc(&d_b, K * N * sizeof(float));
    float* d_out;
    cudaMalloc(&d_out, M * N * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    matmul_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

void matmul_gpu_v1(float* a, float* b, float* out, int M, int N, int K) {
    //分配显存
    float* d_a;
    cudaMalloc(&d_a, M * K * sizeof(float));
    float* d_b;
    cudaMalloc(&d_b, K * N * sizeof(float));
    float* d_out;
    cudaMalloc(&d_out, M * N * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    matmul_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

void matmul_gpu_v2(float* a, float* b, float* out, int M, int N, int K) {
    //分配显存
    float* d_a;
    cudaMalloc(&d_a, M * K * sizeof(float));
    float* d_b;
    cudaMalloc(&d_b, K * N * sizeof(float));
    float* d_out;
    cudaMalloc(&d_out, M * N * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    matmul_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

int main(int argc, char *argv[]) {
    int M = 1024, N = 1024, K = 1024;
    float* a = (float*) malloc(M * K * sizeof(float));
    rand(a, M * K);

    float* b = (float*) malloc(K * N * sizeof(float));
    rand(b, K * N);

    float* out = (float*) malloc(M * N * sizeof(float));
    float* d_out = (float*) malloc(M * N * sizeof(float));

    matmul_cpu(a, b, out, M, N, K);
    matmul_gpu_v2(a, b , d_out, M, N, K);

    if (check(out, d_out, M * N)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    free(a);
    free(b);
    free(out);
    free(d_out);
}
