#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "cross_entropy.cuh"

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
    static void run_benchmark(int rounds, size_t totalBytes, int block_size, Fn&& kernel, Args&&... args) {
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
        printf("BlockSize: %d\n", block_size);
        printf("Rounds: %d\n", rounds);
        printf("Average: %.3f ms\n", mean);
        printf("StdDev: %.3f ms\n", stddev);
        printf("Min: %.3f ms\n", *std::min_element(timings.begin(), timings.end()));
        printf("Max: %.3f ms\n", *std::max_element(timings.begin(), timings.end()));
        printf("Bandwidth: %.2f GB/s", bandwidth);
    }
};

void rand(float* elements, int size) {
    for(int i = 0; i < size; i++){
        elements[i] = (float)rand() / RAND_MAX;
    }
}

float cross_entropy_gpu_base(float* loss, const float* probs, const int* targets, int B, int T, int V, int block_size) {
    //计算grid
    double N = B * T;
    //分配显存
    float* d_loss;
    cudaMalloc(&d_loss, B * T * sizeof(float));
    float* d_probs;
    cudaMalloc(&d_probs, B * T * V * sizeof(float));
    int* d_targets;
    cudaMalloc(&d_targets, B * T * sizeof(int));

    //拷贝数据到显存
    cudaMemcpy(d_probs, probs, B * T * V * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start);

    cross_entropy_kernel<<<(N + block_size -1) / block_size, block_size>>>(d_loss, d_probs, d_targets, B, T, V);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(loss, d_loss, B * T * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_loss);
    cudaFree(d_probs);
    cudaFree(d_targets);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

int main() {
    int B = 64, T = 1024, V = 768, block_size = 128, round = 10;
    float* probs = (float*) malloc(B * T * V * sizeof(float));
    rand(probs, B * T * V);

    float* losses = (float*) malloc(B * T * sizeof(float));

    int* targets = new int[B * T];
    for (int i = 0; i < B * T; ++i) {
        targets[i] = std::rand() % V;
    }

    // 计算总数据量 (bytes)
    const size_t totalBytes = B * T * V * sizeof(float) + B * T * sizeof(float) + B * T * sizeof(int);
    Benchmark::run_benchmark(
            round, totalBytes,block_size,
            cross_entropy_gpu_base,
            probs, losses,targets,
            B, T, V, block_size
    );
    return 0;
}