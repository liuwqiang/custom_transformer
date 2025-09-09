#include "cuda_runtime.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include "softmax.cuh"

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

void softmax_cpu(const float* inp, float* out, int B, int T, int C) {
    for (int b = 0;b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* t_input = inp + b * T * C + t * C;
            float* t_output = out + b * T * C + t * C;
            //计算最大值
            float maxValue = -INFINITY;
            for (int c = 0; c < C; c++) {
                if (t_input[c] > maxValue) {
                    maxValue = t_input[c];
                }
            }
            //求和
            float sum = 0.0f;
            for (int c = 0; c < C; c++) {
                t_output[c] = expf(t_input[c] - maxValue);
                sum += t_output[c];
            }
            float norm = 1.0f / sum;
            for (int c = 0; c < C; c++) {
                t_output[c] *= norm;
            }
        }
    }
}

void softmax_gpu_base(const float *inp,  float *out, int B, int T, int C, int block_size) {
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

    softmax_kernel<<<grid_size, block_size>>>(d_inp,  d_out, B, T, C);
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inp);
    cudaFree(d_out);
}

void softmax_gpu_v1(const float *inp,  float *out, int B, int T, int C, int block_size) {
    //计算grid
    double N = B * T;

    //分配显存
    float* d_inp;
    cudaMalloc(&d_inp, B * T * C * sizeof(float));
    float* d_out;
    cudaMalloc(&d_out, B * T * C * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice);

    softmax_kernel<<<N, block_size, (C + block_size + 1/ block_size) * sizeof(float)>>>(d_inp,  d_out, B, T, C);
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inp);
    cudaFree(d_out);
}

void softmax_gpu_v2(const float *inp,  float *out, int B, int T, int C, int block_size) {
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

    softmax_kernel<<<grid_size, block_size>>>(d_inp,  d_out, B, T, C);
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inp);
    cudaFree(d_out);
}


int main(int argc, char *argv[]) {
    int B = 64, T = 1024, C = 768, block_size = 128;
    float* inp = (float*) malloc(B * T * C * sizeof(float));
    rand(inp, B * T * C);

    float* out = (float*) malloc(B * T * C * sizeof(float));
    float* d_out = (float*) malloc(B * T * C * sizeof(float));

    softmax_cpu(inp, out, B, T, C);
    softmax_gpu_v2(inp, d_out, B, T, C, block_size);

    if (check(out, d_out, B * T * C)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    free(inp);
    free(out);
    free(d_out);
}
