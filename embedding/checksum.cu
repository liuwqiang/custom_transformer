#include "cuda_runtime.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include "embedding.cuh"

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
 *  out:[B,T,C] 输出的token embedding
 *  inp:[B,T] 输入word在词表中的索引,T:[0,V]
 *  wte:[V,C] embedding矩阵
 *  wpe:[L,C] position矩阵，L:模型支持的最大输入，例如2048
 *  B: batch_size
 *  T: token
 *  C: embedding dimension
 */
void embedding_cpu(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* d_out = out + b * T * C + t * C;
            const int idx = inp[b * T + t];
            const float* d_wte = wte + idx * C;
            const float* d_wpe = wpe + t * C;
            for (int c = 0; c < C; c++) {
                d_out[c] = d_wte[c] + d_wpe[c];
            }
        }
    }
}

void embedding_gpu_base(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C) {
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

    dim3 dimBlock(32);
    embedding_kernel<<<(B * T * C + dimBlock.x - 1) / dimBlock.x, dimBlock>>>(d_out, d_inp, d_wte, d_wpe, B, T, C);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(out, d_out, B * T * C * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_inp);
    cudaFree(d_wte);
    cudaFree(d_wpe);
}

int main(int argc, char *argv[]) {
    int B = 64, T = 1024, C = 128;

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
    float* d_out = (float*) malloc(B * T * C * sizeof(float));

    embedding_cpu(out, inp, wte, wpe, B, T, C);
    embedding_gpu_base(d_out, inp , wte, wpe, B, T, C);

    if (check(out, d_out, B * T * C)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    free(inp);
    free(wte);
    free(wpe);
    free(out);
    free(d_out);
}
