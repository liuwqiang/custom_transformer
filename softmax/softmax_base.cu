#include "cuda_runtime.h"
#include "softmax.cuh"

__global__ void softmax_kernel(const float *inp, float *out, int B, int T, int C) {
    int N = B * T;
    int tid = blockIdx.x * blockDim.x + threadIdx.x; //[0, N]
    if (tid < N) {
        const float* d_inp = inp + tid * C;
        //最大值
        float maxValue = -INFINITY;
        for (int c = 0; c < C; c++) {
            if (d_inp[c] > maxValue) {
                maxValue = d_inp[c];
            }
        }

        float* d_out = out + tid * C;
        //求和
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            float m = expf(d_inp[c] - maxValue);
            d_out[c] = m;
            sum += m;
        }

        float norm = 1.0f / sum;
        for (int c = 0; c < C; c++) {
            d_out[c] *= norm;
        }
    }
}