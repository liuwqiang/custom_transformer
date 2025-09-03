#include "cuda_runtime.h"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <time.h>

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
        if (std::fabs(out[i] - res[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

/**
 * int B int T int C 输入/输出的 shape (8,1024,768)
 * const float* inp 输入向量
 * float* mean float* rstd 均值和标准差的倒数  (8,1024)
 * const float* weight const float* bias 权重和偏置 (768)
 * float* out 输出向量
 */
void layernorm_cpu(const float* inp, float* mean, float* rstd, const float* weight, const float* bias, float* out
    , int B, int T, int C);

void layernorm_gpu(const float* inp, float* mean, float* rstd, const float* weight, const float* bias, float* out
    , int B, int T, int C, int block_size);

int main(int argc, char *argv[]) {
    int B = 64, T = 1024, C = 768, block_size = 128;
    float* inp = (float*) malloc(B * T * C * sizeof(float));
    rand(inp, B * T * C);

    float* mean = (float*) malloc(B * T * sizeof(float));
    float* rstd = (float*) malloc(B * T * sizeof(float));

    float* d_mean = (float*) malloc(B * T * sizeof(float));
    float* d_rstd = (float*) malloc(B * T * sizeof(float));

    float* weight = (float*) malloc(C * sizeof(float));
    rand(weight, C);
    float* bias = (float*) malloc(C * sizeof(float));
    rand(bias, C);

    float* out = (float*) malloc(B * T * C * sizeof(float));
    float* d_out = (float*) malloc(B * T * C * sizeof(float));

    clock_t start = clock();
    layernorm_cpu(inp, mean, rstd, weight, bias, out, B, T, C);
    clock_t end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU耗时: %.6f ms\n", cpu_time * 1000);

    layernorm_gpu(inp, d_mean, d_rstd, weight, bias, d_out, B, T, C, block_size);

    if (check(mean, d_mean, B * T) && check(rstd, d_rstd, B * T) && check(out, d_out, B * T * C)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    free(inp);
    free(out);
    free(mean);
    free(rstd);
    free(weight);
    free(bias);
}

__global__ void layernorm_kernel(const float* inp, float* mean, float* rstd, const float* weight, const float* bias, float* out
    , int B, int T, int C) {
    float eps = 1e-5f;
    //计算当前word的位置
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < B * T) {
        //计算均值
        float m = 0.0f;
        for (int c = 0; c < C; c++) {
            m += inp[tid * C + c];
        }
        m = m / C;
        //计算方差
        float v = 0.0f;
        for (int c = 0; c < C; c++) {
            float x_i = inp[tid * C + c] - m;
            v += x_i * x_i;
        }
        v = v / C;
        //计算标准差的倒数
        float s = 1.0f / sqrtf(v + eps);
        //计算结果输出
        for (int c = 0; c < C; c++) {
            float n = s * (inp[tid * C + c] - m);
            out[tid * C + c] = weight[c] * n + bias[c];
        }
        //记录均值和标准差倒数
        mean[tid] = m;
        rstd[tid] = s;
    }
}

void layernorm_gpu(const float *inp, float *mean, float *rstd, const float *weight, const float *bias, float *out, int B, int T, int C, int block_size) {
    //计算grid
    double N = B * T;
    int grid_size = ceil((N + block_size - 1)/ block_size);

    //分配显存
    float* d_inp;
    cudaMalloc(&d_inp, B * T * C * sizeof(float));

    float* d_mean;
    cudaMalloc(&d_mean, B * T * sizeof(float));

    float* d_rstd;
    cudaMalloc(&d_rstd, B * T * sizeof(float));

    float* d_weight;
    cudaMalloc(&d_weight, C * sizeof(float));

    float* d_bias;
    cudaMalloc(&d_bias, C * sizeof(float));

    float* d_out;
    cudaMalloc(&d_out, B * T * C * sizeof(float));

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start);

    //拷贝数据到显存
    cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice);

    layernorm_kernel<<<grid_size, block_size>>>(d_inp, d_mean, d_rstd, d_weight, d_bias, d_out, B, T, C);
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rstd, d_rstd, B * T * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mean, d_mean, B * T * sizeof(float), cudaMemcpyDeviceToHost);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算执行时间(毫秒)
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算带宽 (GB/s)
    float totalBytes = 2 * B * T * C * sizeof(float) + 2 * B * T * sizeof(float) + 2 * C * sizeof(float);
    float bandwidth = totalBytes / (milliseconds * 1e6);
    // 输出结果
    printf("GPU耗时: block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n",
           block_size, milliseconds, bandwidth);

    cudaFree(d_inp);
    cudaFree(d_mean);
    cudaFree(d_rstd);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_out);
}

void layernorm_cpu(const float* inp, float* mean, float* rstd, const float* weight, const float* bias, float* out
    , int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0;b < B; b++) {
        //按照word维度拆分，计算每个word的均值和方差
        for (int t = 0; t < T; t++) {
            const float* t_input = inp + b * T * C + t * C;
            //计算均值
            float m = 0.0f;
            for (int c = 0; c < C; c++) {
                m += t_input[c];
            }
            m = m / C;
            //计算方差
            float v = 0.0f;
            for (int c = 0; c < C; c++) {
                float x_i = t_input[c] - m;
                v += x_i * x_i;
            }
            v = v / C;
            //计算标准差的倒数
            float s = 1.0f / sqrtf(v + eps);
            //计算结果输出
            float* t_output = out + b * T * C + t * C;
            for (int c = 0; c < C; c++) {
                float n = s * (t_input[c] - m);
                t_output[c] = weight[c] * n + bias[c];
            }
            //记录均值和标准差的倒数
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}
