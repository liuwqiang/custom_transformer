#include "cuda_runtime.h"
#include "cross_entropy.cuh"
#include "stdio.h"
//loss的维度是(B,T)，代表每个batch中每个token的损失值，为了后续的计算梯度
//probs的维度是(B,T,V)，代表每个batch中每个token在某个词上的概率分布。
//targets的维度是(B,T)，代表真是数据，每个值是(0,V]的范围，代表了真实的某个词
//V代表vocabulary的大小，也就是词典的大小
__global__ void cross_entropy_kernel(float* loss, const float* probs, const int* targets, int B, int T, int V) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < B * T) {
        int b = tid / T;
        int t = tid % T;
        const float* d_probs = probs + b * T * V + t * V;
        const int idx = targets[tid];
        loss[tid] = -logf(d_probs[idx]);
    }
}