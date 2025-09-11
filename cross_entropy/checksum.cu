#include "cuda_runtime.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <vector>

#include "cross_entropy.cuh"

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

void cross_entropy_cpu(float* loss, const float* probs, const int* targets, int B, int T, int V) {
    for (int b = 0;b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* t_probs = probs + b * T * V + t * V;
            const int idx = targets[b * T + t];
            loss[b * T + t] = -logf(t_probs[idx]);
        }
    }
}

void cross_entropy_gpu_base(float* loss, const float* probs, const int* targets, int B, int T, int V, int block_size) {
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

    cross_entropy_kernel<<<(N + block_size) / block_size, block_size>>>(d_loss, d_probs, d_targets, B, T, V);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(loss, d_loss, B * T * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_loss);
    cudaFree(d_probs);
    cudaFree(d_targets);
}

int main(int argc, char *argv[]) {
    int B = 64, T = 1024, V = 768, block_size = 128;
    float* probs = (float*) malloc(B * T * V * sizeof(float));
    rand(probs, B * T * V);

    float* losses = (float*) malloc(B * T * sizeof(float));
    float* d_losses = (float*) malloc(B * T * sizeof(float));

    int* targets = new int[B * T];
    for (int i = 0; i < B * T; ++i) {
        targets[i] = std::rand() % V;
    }

    cross_entropy_cpu(losses, probs, targets, B, T, V);
    cross_entropy_gpu_base(d_losses, probs, targets, B, T, V, block_size);

    if (check(losses, d_losses, B * T)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    free(probs);
    free(losses);
    free(d_losses);
}
