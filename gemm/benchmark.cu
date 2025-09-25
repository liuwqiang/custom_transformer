#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "gemm.cuh"
#include <cublas_v2.h>

// Benchmark 测试类
class Benchmark {
public:
    // 运行单轮测试
    template <typename Fn, typename... Args>
    static double run_once(Fn&& kernel, Args&&... args) {
        return kernel(std::forward<Args>(args)...);
    }

    // 运行多轮测试并统计结果
    template <typename Fn, typename... Args>
    static void run_benchmark(int rounds, size_t totalBytes, const long flops, Fn&& kernel, Args&&... args) {
        std::vector<double> timings;
        timings.reserve(rounds);

        //首次启动不计入benchmark内
        run_once(kernel, std::forward<Args>(args)...);

        // 正式测试
        for (size_t i = 0; i < rounds; ++i) {
            double t = run_once(kernel, std::forward<Args>(args)...);
            timings.push_back(t);
            std::cout << "Round " << i + 1 << ": " << t << " ms" << std::endl;
        }

        // 计算统计结果
        double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
        double mean = sum / rounds;

        // 计算带宽 (GB/s)
        double bandwidth = totalBytes / (mean * 1e6);
        // 计算GFlops
        double gflops = flops / (mean * 1e6);
        double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / rounds - mean * mean);

        printf("\n=== Benchmark Results ===\n");
        printf("Rounds: %d\n", rounds);
        printf("Average: %.3f ms\n", mean);
        printf("StdDev: %.3f ms\n", stddev);
        printf("Min: %.3f ms\n", *std::min_element(timings.begin(), timings.end()));
        printf("Max: %.3f ms\n", *std::max_element(timings.begin(), timings.end()));
        printf("Bandwidth: %.2f GB/s\n", bandwidth);
        printf("GFlops: %.2f GFlops/s", gflops);
    }
};

void rand(float* elements, int size) {
    for(int i = 0; i < size; i++){
        elements[i] = (float)rand() / RAND_MAX;
    }
}

float matmul_gpu_base(float* a, float* b, float* out, int M, int N, int K) {
    //分配block和grid
    dim3 dimBlock(8, 8);
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

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

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start);

    matmul_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, M, N, K);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

float matmul_gpu_v1(float* a, float* b, float* out, int M, int N, int K) {
    //分配block和grid
    dim3 dimBlock(8, 8);
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

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

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start);

    matmul_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, M, N, K);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

float matmul_gpu_v2(float* a, float* b, float* out, int M, int N, int K) {
    //分配block和grid
    dim3 dimBlock(8, 8);
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

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

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start);

    matmul_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, M, N, K);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

/**
 * 基于cublas的矩阵运算
 * cublas的数据布局是以列主序进行存储，而C++中的数组、列表等都是以行主序存储，所以需要对运算值进行一个转置
 * 转置并不会带来性能上的开销，而是利用 cuBLAS 的列主序解释 + 转置语义，使得内存中的行主序矩阵在数学上被正确地当成行主序矩阵使用
 * a:[M,K]
 * b:[K,N]
 * out:[M,N]
 */
float matmul_cublas(const float* a, const float* b, float* out, int M, int N, int K) {
    //创建handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    //在行主序布局下，a 矩阵是 (m, k)，但在列主序下会被解释为 (k, m)。
    //我们想计算 out = a * b。
    //通过告诉 cublasSgemv 使用 a 的转置（CUBLAS_OP_T），它就能把内存中按行排布的 a 正确地当作行主序矩阵来处理。
    //out = alpha * (a @ b) + beta * out
    float alpha = 1.0f;
    float beta  = 0.0f;

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

    // ====== 创建 CUDA Events ======
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ====== 记录开始时间 ======
    cudaEventRecord(start, 0);

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

    // ====== 记录结束时间 ======
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime = 0.0f;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasGemmEx failed: %d\n", stat);
    }
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    return elapsedTime;
}

int main() {
    int M = 512, N = 512, K = 256, round = 10;
    float* a = (float*) malloc(M * K * sizeof(float));
    rand(a, M * K);

    float* b = (float*) malloc(K * N * sizeof(float));
    rand(b, K * N);

    float* out = (float*) malloc(M * N * sizeof(float));

    const size_t totalBytes = M * K * sizeof(float) + K * N * sizeof(float) + M * N * sizeof(float);
    const int flops = 2 * K * M * N;
    Benchmark::run_benchmark(
            round, totalBytes, flops,
            matmul_gpu_v2,
            a, b, out, M, N, K
    );
    return 0;
}