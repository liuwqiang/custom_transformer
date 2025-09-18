#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "rms_norm.cuh"

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
    static void run_benchmark(int rounds, size_t totalBytes, Fn&& kernel, Args&&... args) {
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
        double sq_sum = std::inner_product(timings.begin(), timings.end(), timings.begin(), 0.0);
        double stddev = std::sqrt(sq_sum / rounds - mean * mean);

        printf("\n=== Benchmark Results ===\n");
        printf("Rounds: %d\n", rounds);
        printf("Average: %.3f ms\n", mean);
        printf("StdDev: %.3f ms\n", stddev);
        printf("Min: %.3f ms\n", *std::min_element(timings.begin(), timings.end()));
        printf("Max: %.3f ms\n", *std::max_element(timings.begin(), timings.end()));
        printf("Bandwidth: %.2f GB/s\n", bandwidth);
    }
};

void rand(float* elements, int size) {
    for(int i = 0; i < size; i++){
        elements[i] = (float)rand() / RAND_MAX;
    }
}

float rotate_qk_gpu(float* q, float* k, int pos) {
    //分配显存
    float* d_q;
    cudaMalloc(&d_q, Q_DIM * sizeof(float));
    float* d_k;
    cudaMalloc(&d_k, KV_DIM * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_q, q, Q_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, KV_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start);

    dim3 dimBlock(32);
    rotate_qk_kernel<<<(Q_DIM / 2 + dimBlock.x - 1) / dimBlock.x, dimBlock>>>(d_q, d_k, pos);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(q, d_q, Q_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(k, d_k, KV_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}


int main() {
    int round = 10;
    float* q = (float*) malloc(Q_DIM * sizeof(float));
    rand(q, Q_DIM);

    float* k = (float*) malloc(KV_DIM * sizeof(float));
    rand(k, KV_DIM);

    const size_t totalBytes = Q_DIM * 2 * sizeof(int) + KV_DIM * 2 * sizeof(float);
    Benchmark::run_benchmark(
            round, totalBytes,
            rotate_qk_gpu,
            q, k, 0
    );
    return 0;
}