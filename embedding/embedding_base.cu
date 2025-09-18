#include "cuda_runtime.h"
#include "embedding.cuh"
/**
 * 整体实现逻辑很简单，就是查表，从模型参数中获取embedding的weight权重，然后查表就好了
 * wte: word term embedding matrix
 * wpe: word position embedding matrix
 * wpe这里用的最简单的基于位置的静态编码方式
 */
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