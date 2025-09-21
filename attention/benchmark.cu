#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "attention.cuh"

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

float attention_gpu_base(float* out, const float* q, const float* k_cache, const float* v_cache, int pos) {
    float* d_out;
    cudaMalloc(&d_out, N_HEADS * HEAD_DIM * sizeof(float));

    float* d_att;
    cudaMalloc(&d_att, N_HEADS * SEQ_LEN * sizeof(float));

    float* d_q;
    cudaMalloc(&d_q, N_HEADS * HEAD_DIM * sizeof(float));

    float* d_k_cache;
    cudaMalloc(&d_k_cache, SEQ_LEN * KV_DIM * sizeof(float));

    float* d_v_cache;
    cudaMalloc(&d_v_cache, SEQ_LEN * KV_DIM * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_q, q, N_HEADS * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_cache, k_cache, SEQ_LEN * KV_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_cache, v_cache, SEQ_LEN * KV_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 记录开始时间
    cudaEventRecord(start);

    //计算kv注意力得分
    attention_qk_kernel<<<N_HEADS, pos + 1>>>(d_att, d_q, d_k_cache, pos);
    //计算softmax
    attention_softmax_kernel<<<N_HEADS, 1>>>(d_att, pos);
    //计算输出
    attention_v_kernel<<<N_HEADS, HEAD_DIM>>>(d_out, d_att, d_v_cache, pos);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, N_HEADS * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_att);
    cudaFree(d_q);
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}


int main() {
    int pos = 128, round = 10;
    float* out = (float*) malloc(N_HEADS * HEAD_DIM * sizeof(float));

    float* q = (float*) malloc(N_HEADS * HEAD_DIM * sizeof(float));
    rand(q, N_HEADS * HEAD_DIM);

    float* k_cache = (float*) malloc(SEQ_LEN * KV_DIM * sizeof(float));
    float* v_cache = (float*) malloc(SEQ_LEN * KV_DIM * sizeof(float));
    rand(k_cache, SEQ_LEN * KV_DIM);
    rand(v_cache, SEQ_LEN * KV_DIM);

    const size_t totalBytes = 2 * N_HEADS * HEAD_DIM * sizeof(float) + 2 * SEQ_LEN * KV_DIM * sizeof(float);
    Benchmark::run_benchmark(
            round, totalBytes,
            attention_gpu_base,
            out, q, k_cache, v_cache, pos
    );
    return 0;
}