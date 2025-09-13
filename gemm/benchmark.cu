#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "gemm.cuh"

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

int main() {
    unsigned long M = 1024UL, N = 1024UL, K = 768UL, round = 10;
    float* a = (float*) malloc(M * K * sizeof(float));
    rand(a, M * K);

    float* b = (float*) malloc(K * N * sizeof(float));
    rand(b, K * N);

    float* out = (float*) malloc(M * N * sizeof(float));

    const size_t totalBytes = M * K * sizeof(float) + K * N * sizeof(float) + M * N * sizeof(float);
    const long flops = 2 * K * M * N;
    Benchmark::run_benchmark(
            round, totalBytes, flops,
            matmul_gpu_v1,
            a, b, out, M, N, K
    );
    return 0;
}