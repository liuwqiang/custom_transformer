#include "cuda_runtime.h"
#include "layernorm.cuh"

__global__ void layernorm_kernel(const float* inp, float* mean, float* rstd, const float* weight, const float* bias, float* out
    , int B, int T, int C, int block_size) {
    float eps = 1e-5f;
    //计算当前word的位置
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < B * T) {
        //计算均值
        float m = 0.0f;
        for (int c = 0; c < C; c++) {
            m += inp[tid * C + c];
        }
        m = m / C;
        //计算方差
        float v = 0.0f;
        for (int c = 0; c < C; c++) {
            float x_i = inp[tid * C + c] - m;
            v += x_i * x_i;
        }
        v = v / C;
        //计算标准差的倒数
        float s = 1.0f / sqrtf(v + eps);
        //计算结果输出
        for (int c = 0; c < C; c++) {
            float n = s * (inp[tid * C + c] - m);
            out[tid * C + c] = weight[c] * n + bias[c];
        }
        //记录均值和标准差倒数
        mean[tid] = m;
        rstd[tid] = s;
    }
}

__global__ void rstd_kernel(const float* inp, float* out, float* mean, float* rstd, int C,  int block_size) {

}

__global__ void mean_kernel(const float* inp, float* out, float* mean, int C, int block_size) {

}