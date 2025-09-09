#include "cuda_runtime.h"
#include "softmax.cuh"

__global__ void softmax_kernel(const float *inp, float *out, int B, int T, int C) {
    const int N = B * T;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; //[0, N]
    if (tid < N) {
        const float* t_inp = inp + tid * C;
        //最大值
        float maxValue = -INFINITY;
        for (int c = 0; c < C; c++) {
            if (t_inp[c] > maxValue) {
                maxValue = t_inp[c];
            }
        }

        float* t_out = out + tid * C;
        //求和
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            const float m = expf(t_inp[c] - maxValue);
            t_out[c] = m;
            sum += m;
        }

        const float norm = 1.0f / sum;
        for (int c = 0; c < C; c++) {
            t_out[c] *= norm;
        }
    }
}