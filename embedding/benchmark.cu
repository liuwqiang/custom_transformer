#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "embedding.cuh"

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

float  embedding_gpu_base(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C){
    //分配显存
    float* d_out;
    cudaMalloc(&d_out, B * T * C * sizeof(float));
    int* d_inp;
    cudaMalloc(&d_inp, B * T * sizeof(int));
    float* d_wte;
    cudaMalloc(&d_wte, V * C * sizeof(float));
    float* d_wpe;
    cudaMalloc(&d_wpe, L * C * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_inp, inp, B * T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wte, wte, V * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wpe, wpe, L * C * sizeof(float), cudaMemcpyHostToDevice);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start);

    dim3 dimBlock(128);
    embedding_kernel<<<(B * T * C + dimBlock.x - 1) / dimBlock.x, dimBlock>>>(d_out, d_inp, d_wte, d_wpe, B, T, C);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_inp);
    cudaFree(d_wte);
    cudaFree(d_wpe);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}


int main() {
    int B = 1024, T = 1024, C = 128, round = 10;
    int* inp = (int*) malloc(B * T * sizeof(int));

    srand((unsigned int)time(NULL));
    for (int i = 0; i < B * T; i++) {
        inp[i] = rand() % V;
    }

    float* wte = (float*) malloc(V * C * sizeof(float));
    rand(wte, V * C);

    float* wpe = (float*) malloc(L * C * sizeof(float));
    rand(wpe, L* C);

    float* out = (float*) malloc(B * T * C * sizeof(float));

    const size_t totalBytes = B * T * sizeof(int) + V * C * sizeof(float) + L * C * sizeof(float) + B * T * C * sizeof(float);
    Benchmark::run_benchmark(
            round, totalBytes,
            embedding_gpu_base,
            out, inp, wte, wpe, B, T, C
    );
    return 0;
}