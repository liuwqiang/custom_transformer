#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "softmax.cuh"

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

float softmax_gpu_base(const float *inp, float *out, int B, int T, int C, int block_size) {
    //计算grid
    double N = B * T;
    int grid_size = ceil((N + block_size - 1)/ block_size);

    //分配显存
    float* d_inp;
    cudaMalloc(&d_inp, B * T * C * sizeof(float));
    float* d_out;
    cudaMalloc(&d_out, B * T * C * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start);

    softmax_kernel<<<grid_size, block_size>>>(d_inp, d_out, B, T, C);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inp);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

float softmax_gpu_v1(const float *inp, float *out, int B, int T, int C, int block_size) {
    //计算grid
    double N = B * T;

    //分配显存
    float* d_inp;
    cudaMalloc(&d_inp, B * T * C * sizeof(float));
    float* d_out;
    cudaMalloc(&d_out, B * T * C * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start);

    softmax_kernel<<<N, block_size, (C + block_size + 1/ block_size) * sizeof(float)>>>(d_inp, d_out, B, T, C);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inp);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

float softmax_gpu_v2(const float *inp, float *out, int B, int T, int C, int block_size) {
    //计算grid
    double N = B * T;
    int grid_size = ceil((N + block_size - 1)/ block_size);

    //分配显存
    float* d_inp;
    cudaMalloc(&d_inp, B * T * C * sizeof(float));
    float* d_out;
    cudaMalloc(&d_out, B * T * C * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start);

    softmax_kernel<<<grid_size, block_size>>>(d_inp, d_out, B, T, C);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inp);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}

int main() {
    int B = 64, T = 1024, C = 768, block_size = 128, round = 10;
    float* inp = (float*) malloc(B * T * C * sizeof(float));
    rand(inp, B * T * C);

    float* out = (float*) malloc(B * T * C * sizeof(float));

    // 计算总数据量 (bytes)
    size_t totalBytes = (B * T * C * sizeof(float) * 2); // inp + out

    Benchmark::run_benchmark(
            round, totalBytes,block_size,
            softmax_gpu_v2,
            inp, out,
            B, T, C, block_size
    );

    return 0;
}