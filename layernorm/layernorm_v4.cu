#include "cuda_runtime.h"
#include "layernorm.cuh"

/**
 *v4版本优化内存读取
 *展开 + warpReduce，减少一个同步操作，因为warp内的线程天生就是同步的
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

__device__ void warpReduce(volatile float* cache, unsigned int tid, int blockSize){
    if (blockSize >= 64)cache[tid]+=cache[tid+32];
    if (blockSize >= 32)cache[tid]+=cache[tid+16];
    if (blockSize >= 16)cache[tid]+=cache[tid+8];
    if (blockSize >= 8)cache[tid]+=cache[tid+4];
    if (blockSize >= 4)cache[tid]+=cache[tid+2];
    if (blockSize >= 2)cache[tid]+=cache[tid+1];
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

    if (blockDim.x >= 512) {
        if (tid < 256) {
            sum_shard[tid] += sum_shard[tid + 256];
            sumsq_shard[tid] += sumsq_shard[tid + 256];
        }
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) {
            sum_shard[tid] += sum_shard[tid + 128];
            sumsq_shard[tid] += sumsq_shard[tid + 128];
        }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) {
            sum_shard[tid] += sum_shard[tid + 64];
            sumsq_shard[tid] += sumsq_shard[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(sum_shard, tid, blockDim.x);
        warpReduce(sumsq_shard, tid, blockDim.x);
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