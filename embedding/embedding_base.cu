#include "cuda_runtime.h"
#include "embedding.cuh"

__global__ void embedding_kernel(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    if (tid < N) {
        int bt = tid / C;
        int b = bt / T;
        int t = bt % T;
        int c = tid % C;

        float* d_out = out + b * T * C + t * C;
        int idx = inp[b * T + t];
        const float* d_wte = wte + idx * C;
        const float* d_wpe = wpe + t * C;
        d_out[c] = d_wte[c] + d_wpe[c];
    }
}