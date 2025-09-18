#include "cuda_runtime.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include "rmsnorm.cuh"

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
 * CPU rms_norm函数
 */
void rms_norm_cpu(float* inp, float* weight, float* out, int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        //对每个token维度的向量进行rms_norm归一化
        for (int t = 0; t < T; t++) {
            float* x = inp + b * T * C + t * C;
            //计算向量x的均方根
            float sq_sum = 0.0f;
            for (int c = 0; c < C; c++) {
                sq_sum += x[c] * x[c];
            }
            float rms = rsqrtf(sq_sum / C + eps);
            float* y = out + b * T * C + t * C;
            for (int c =0; c < C; c++ ) {
                y[c] = rms * x[c] * weight[c];
            }
        }
    }
}

/**
 * GPU rms_norm函数
 */
void rms_norm_base_gpu(float* inp, float* weight, float* out, int B, int T, int C) {
    //计算grid
    double N = B * T * C;
    int grid_size = ceil((N + block_size - 1)/ block_size);

    //分配显存
    float* d_inp;
    cudaMalloc(&d_inp, B * T * C * sizeof(float));

    float* d_weight;
    cudaMalloc(&d_weight, C * sizeof(float));

    float* d_out;
    cudaMalloc(&d_out, B * T * C * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice);

    rms_norm_kernel<<<grid_size, block_size>>>(d_inp, d_weight, d_out, B, T, C);
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inp);
    cudaFree(d_weight);
    cudaFree(d_out);
}

int main(int argc, char *argv[]) {
    int B = 1024, T = 768, C = 128;
    float* inp = (float*) malloc(B * T * C * sizeof(float));
    rand(inp, B * T * C);

    float* weight = (float*) malloc(C * sizeof(float));
    rand(weight, C);

    float* out = (float*) malloc(B * T * C * sizeof(float));
    float* d_out = (float*) malloc(B * T * C * sizeof(float));

    rms_norm_cpu(inp, weight, out, B, T, C);
    rms_norm_base_gpu(inp, weight, d_out, B, T, C);

    if (check(out, d_out, B * T * C)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    free(inp);
    free(weight);
    free(out);
    free(d_out);
}
