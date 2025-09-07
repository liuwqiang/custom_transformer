#include "cuda_runtime.h"
#include "layernorm.cuh"

/**
 *v3版本简化方差计算
 */
__global__ void layernorm_kernel(const float* inp, float* mean, float* rstd, const float* weight, const float* bias, float* out
                                 , int B, int T, int C) {
    int idx = blockIdx.x; //[0, B*T]
    int tid = threadIdx.x; //[0, block_size]

    const float* x = inp + idx * C;
    float* y = out + idx * C;
    for (int i = tid; i < C; i += blockDim.x) {
        float n = rstd[idx] * (x[i] - mean[idx]);
        y[i] = weight[i] * n + bias[i];
    }
}

//简化方差计算，计算均值的过程中可以一起计算方差
__global__ void mean_rstd_kernel(const float* inp, float* mean, float* rstd, int C) {
    float eps = 1e-5f;
    //idx的维度是B*T的维度
    int idx = blockIdx.x; //[0, B*T]
    //tid的维度是C的维度,如果block_size小于C那么我们就要考虑无法整除的情况，一个线程可能要处理C维度的多个值
    int tid = threadIdx.x; //[0, block_size]

    const float* x = inp + idx * C;
    //使用共享内存缓存中间变量
    extern __shared__ float shared[];
    //共享内存的前半部分保存和,后半部分保存平方和
    float* sum_shard = shared;
    float* sumsq_shard = shared + blockDim.x;
    //单个线程要C中多个值的情况
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sum += x[i];
        sumsq += x[i] * x[i];
    }
    sum_shard[tid] = sum;
    sumsq_shard[tid] = sumsq;
    //同步所有结果
    __syncthreads();
    //对C维度进行规约求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        //每一次迭代代表的是对C中一半的数进行求和，需要同步等待所有线程完成
        __syncthreads();
        if (tid < stride) {
            sum_shard[tid] += sum_shard[tid + stride];
            sumsq_shard[tid] += sumsq_shard[tid + stride];
        }
    }
    //同步求和,最后一步的时候只会剩下0号线程，保存最后的求和结果
    __syncthreads();
    if (tid == 0) {
        float m = sum_shard[0] / C;
        mean[idx] = m;
        float var = sumsq_shard[0] / C - m * m;
        rstd[idx] = 1.0f / sqrtf(var + eps);
    }
}

//求标准差
__global__ void rstd_kernel(const float* inp, float* out, float* mean, float* rstd, int C) {

}

//求均值
__global__ void mean_kernel(const float* inp, float* out, float* mean, int C) {

}