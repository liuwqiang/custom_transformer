#include "cuda_runtime.h"
#include "layernorm.cuh"

/**
 *基础版本的实现实在是太简陋了，并没有发挥出gpu的整体的性能，最大的问题是在线程的分配上面，并没有一个线程仅计算一份数据，而是一个线程计算了C分数据。
 *v1版本优化了线程的分配方式，让更多的线程参与到计算中。
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

//求标准差
__global__ void rstd_kernel(const float* inp, float* out, float* mean, float* rstd, int C) {
    int idx = blockIdx.x; //[0, B*T]
    int tid = threadIdx.x; //[0, block_size]

    float eps = 1e-5f;
    const float* x = inp + idx * C;
    float* y = out + idx * C;
    y[tid] = 0.0f;
    //对每个值减去均值，然后求平方
    for (int i = tid; i < C; i += blockDim.x) {
        y[tid] += (x[i] - mean[idx]) * (x[i] - mean[idx]);
    }
    //同步所有结果
    __syncthreads();
    //规约求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            y[tid] += y[tid + stride];
        }
    }
    __syncthreads();
    if (tid == 0) {
       //计算标准差的倒数
       rstd[idx] = 1.0f / sqrtf(y[tid] / C + eps);
    }
}

//求均值
__global__ void mean_kernel(const float* inp, float* out, float* mean, int C) {
    //idx的维度是B*T的维度
    int idx = blockIdx.x; //[0, B*T]
    //tid的维度是C的维度,如果block_size小于C那么我们就要考虑无法整除的情况，一个线程可能要处理C维度的多个值
    int tid = threadIdx.x; //[0, block_size]

    const float* x = inp + idx * C;
    //使用out缓存中间变量
    float* y = out + idx * C;
    //单个线程要C中多个值的情况
    for (int i = tid; i < C; i += blockDim.x) {
        y[tid] += x[i];
    }
    //同步所有结果
    __syncthreads();
    //对C维度进行规约求和
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        //每一次迭代代表的是对C中一半的数进行求和，需要同步等待所有线程完成
        __syncthreads();
        if (tid < stride) {
            y[tid] += y[tid + stride];
        }
    }
    //同步求和,最后一步的时候只会剩下0号线程，保存最后的求和结果
    __syncthreads();
    if (tid == 0) {
        mean[idx] = y[0] / C;
    }
}

__global__ void mean_rstd_kernel(const float* inp, float* mean, float* rstd, int C) {

}