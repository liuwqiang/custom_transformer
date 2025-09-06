#include "cuda_runtime.h"
#include "layernorm.cuh"

/**
 *v2版本引入了共享内存，通过共享内存加速计算
 */
__global__ void layernorm_kernel(const float* inp, float* mean, float* rstd, const float* weight, const float* bias, float* out
                                 , int B, int T, int C, int block_size) {
    int idx = blockIdx.x; //[0, B*T]
    int tid = threadIdx.x; //[0, block_size]

    const float* x = inp + idx * C;
    float* y = out + idx * C;
    for (int i = tid; i < C; i += block_size) {
        float n = rstd[idx] * (x[i] - mean[idx]);
        y[i] = weight[i] * n + bias[i];
    }
}

//求标准差
__global__ void rstd_kernel(const float* inp, float* out, float* mean, float* rstd, int C,  int block_size) {
    int idx = blockIdx.x; //[0, B*T]
    int tid = threadIdx.x; //[0, block_size]

    float eps = 1e-5f;
    const float* x = inp + idx * C;
    //使用共享内存缓存中间变量
    extern __shared__ float shared[];
    //对每个值减去均值，然后求平方
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += (x[i] - mean[idx]) * (x[i] - mean[idx]);
    }
    shared[tid] = sum;
    //同步所有结果
    __syncthreads();
    //规约求和
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    __syncthreads();
    if (tid == 0) {
       //计算标准差的倒数
       rstd[idx] = 1.0f / sqrtf(shared[tid] / C + eps);
    }
}

//求均值
__global__ void mean_kernel(const float* inp, float* out, float* mean, int C, int block_size) {
    //idx的维度是B*T的维度
    int idx = blockIdx.x; //[0, B*T]
    //tid的维度是C的维度,如果block_size小于C那么我们就要考虑无法整除的情况，一个线程可能要处理C维度的多个值
    int tid = threadIdx.x; //[0, block_size]

    const float* x = inp + idx * C;
    //使用共享内存缓存中间变量
    extern __shared__ float shared[];
    //单个线程要C中多个值的情况
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    shared[tid] = sum;
    //同步所有结果
    __syncthreads();
    //对C维度进行规约求和
    for (int stride = block_size / 2; stride > 0; stride /= 2) {
        //每一次迭代代表的是对C中一半的数进行求和，需要同步等待所有线程完成
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    //同步求和,最后一步的时候只会剩下0号线程，保存最后的求和结果
    __syncthreads();
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}