#include "cuda_runtime.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include "gemm.cuh"
#include <cublas_v2.h>

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

/**
 * 基于cublas的矩阵运算
 * cublas的数据布局是以列主序进行存储，而C++中的数组、列表等都是以行主序存储，所以需要对运算值进行一个转置
 * 转置并不会带来性能上的开销，而是利用 cuBLAS 的列主序解释 + 转置语义，使得内存中的行主序矩阵在数学上被正确地当成行主序矩阵使用
 * a:[M,K]
 * b:[K,N]
 * out:[M,N]
 */
void matmul_cublas(const float* a, const float* b, float* out, int M, int N, int K) {
    //创建handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    //分配显存
    float* d_a;
    cudaMalloc(&d_a, M * K * sizeof(float));
    float* d_b;
    cudaMalloc(&d_b, K * N * sizeof(float));
    float* d_out;
    cudaMalloc(&d_out, M * N * sizeof(float));

    cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * sizeof(int), cudaMemcpyHostToDevice);

    //在行主序布局下，a 矩阵是 (m, k)，但在列主序下会被解释为 (k, m)。
    //我们想计算 out = a * b。
    //通过告诉 cublasSgemv 使用 a 的转置（CUBLAS_OP_T），它就能把内存中按行排布的 a 正确地当作行主序矩阵来处理。
    //out = alpha * (a @ b) + beta * out
    float alpha = 1.0f;
    float beta  = 0.0f;

    cublasStatus_t stat = cublasGemmEx(
        handle,
        CUBLAS_OP_T, //a是行主序存储的，需要转置
        CUBLAS_OP_T, //b也是行主序存储的，也需要转置
        M, //输出out的行
        N, //输出out的列
        K, //a和b的公共维度k
        &alpha,
        d_a, CUDA_R_32F, K, //lda K这个我需要解释下，可以简单的理解为我从列主序的存储中获取一行的值需要跨越多少列，这里a的列维度是K
        d_b, CUDA_R_32F, N, //ldb N这个我需要解释下，可以简单的理解为我从列主序的存储中获取一行的值需要跨越多少列，这里a的列维度是N
        &beta,
        d_out, CUDA_R_32F, M,
        //ldc M这个我更需要解释下了，因为本身计算完out就是列主序存储的，维度是[N,M]，
        //为了得到正确的输出我们需要让其按照行主序存储[M,N]，那么每得到一个值我们需要跨越M行才能得到正确的位置
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    );

    cudaMemcpy(out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasGemmEx failed: %d\n", stat);
    }
    cublasDestroy(handle);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
}

void transpose_cpu(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
}

int main(int argc, char *argv[]) {
    int M = 512, N = 512, K = 256;
    float* a = (float*) malloc(M * K * sizeof(float));
    rand(a, M * K);

    float* b = (float*) malloc(K * N * sizeof(float));
    rand(b, K * N);

    float* out = (float*) malloc(M * N * sizeof(float));
    float* d_out = (float*) malloc(M * N * sizeof(float));

    matmul_cpu(a, b, out, M, N, K);
    matmul_cublas(a, b , d_out, M, N, K);

    float* final_out = (float*) malloc(M * N * sizeof(float));
    transpose_cpu(d_out, final_out, M, N);

    if (check(out, final_out, M * N)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    free(a);
    free(b);
    free(out);
    free(d_out);
}
