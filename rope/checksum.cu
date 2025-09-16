#include "cuda_runtime.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include "rope.cuh"

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
 * CPU 版本旋转函数
 * q: 输入/输出数组，长度 = 2 * n (n个复数)
 * n: 复数个数
 * fcr, fci: 旋转用的 cosθ, sinθ
 */
void rotate_qk_cpu(float* q, float* k, int pos, size_t n) {
    for (int i = 0; i < n; i+=2) {
        //计算旋转矩阵
        int head_dim_idx = (i * 2) % HEAD_DIM;
        float freq = 1.0f / powf(ROPE_THETA, (float)head_dim_idx / (float)HEAD_DIM);
        float val = (float)pos * freq;
        float fcr, fci;
        sincosf(val, &fci, &fcr);

        // 旋转k向量
        float q0 = q[2 * i];
        float q1 = q[2 * i + 1];
        float new_q0 = q0 * fcr - q1 * fci;
        float new_q1 = q0 * fci + q1 * fcr;
        q[2 * i]     = new_q0;
        q[2 * i + 1] = new_q1;

        if (i < KV_DIM / 2) {
            // 旋转k向量
            float k0 = k[2 * i];     // 原来的实部
            float k1 = k[2 * i + 1]; // 原来的虚部
            //x′=k0cos−k1sin
            float new_k0 = k0 * fcr - k1 * fci; // 实部旋转
            //y′=k0sin+k1cos
            float new_k1 = k0 * fci + k1 * fcr; // 虚部旋转
            k[2 * i]     = new_k0;
            k[2 * i + 1] = new_k1;
        }
    }
}

void rotate_qk_kernel_gpu(float* q, float* k, int pos) {
    //分配显存
    float* d_q;
    cudaMalloc(&d_q, Q_DIM * sizeof(float));
    float* d_k;
    cudaMalloc(&d_k, KV_DIM * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_q, q, Q_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, KV_DIM * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(32);
    rotate_qk_kernel<<<(Q_DIM / 2 + dimBlock.x - 1) / dimBlock.x, dimBlock>>>(d_q, d_k, pos);
    CUDA_CHECK(cudaGetLastError());
    cudaMemcpy(q, d_q, Q_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(k, d_k, KV_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_k);
}

int main(int argc, char *argv[]) {
    float* q = (float*) malloc(Q_DIM * sizeof(float));
    rand(q, Q_DIM);

    float* k = (float*) malloc(KV_DIM * sizeof(float));
    rand(k, KV_DIM);

    float* d_q = (float*) malloc(Q_DIM * sizeof(float));
    memcpy(d_q, q, Q_DIM * sizeof(float));
    float* d_k = (float*) malloc(KV_DIM * sizeof(float));
    memcpy(d_k, k, KV_DIM * sizeof(float));

    rotate_qk_cpu(q, k, 0, Q_DIM);
    rotate_qk_kernel_gpu(d_q, d_k, 0);

    if (check(q, d_q, Q_DIM) && check(k, d_k, KV_DIM)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    free(q);
    free(k);
}
