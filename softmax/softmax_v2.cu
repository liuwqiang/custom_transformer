#include "cuda_runtime.h"
#include "softmax.cuh"

/**
 *online softmax可以减少一次循环次数
 *具体的做法是在对C中的值求最大值的时候，可以迭代去求sum
 *1.记录当前最大值为maxPrevValue
 *2.如果当前值大于当前最大值maxValue,则更新当前最大值为当前值 计算sum = sum * expf(maxPrevValue - maxValue) + expf(t_inp[c] - maxValue)
 *3.如果小于当前最大值则计算sum += expf(t_inp[c] - maxValue)
 */
__global__ void softmax_kernel(const float *inp, float *out, int B, int T, int C) {
    const int N = B * T;
    unsigned const int tid = blockIdx.x * blockDim.x + threadIdx.x; //[0, N]
    if (tid < N) {
        const float* t_inp = inp + tid * C;
        //最大值
        float maxValue = -INFINITY;
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            const float maxPrevValue = maxValue;
            if (t_inp[c] > maxValue) {
                maxValue = t_inp[c];
                sum = sum * expf(maxPrevValue - maxValue) + expf(t_inp[c] - maxValue);
            } else {
                sum += expf(t_inp[c] - maxValue);
            }
        }

        float* t_out = out + tid * C;
        const float norm = 1.0f / sum;
        for (int c = 0; c < C; c++) {
            t_out[c] = expf(t_inp[c] - maxValue) * norm;
        }
    }
}