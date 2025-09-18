#include "cuda_runtime.h"
#include "rmsnorm.cuh"
#include "cub/cub.cuh"
/*
 * rms_norm解决了layer_norm计算复杂的问题，去掉了减去均值的这一个操作
 */
__global__ void rms_norm_kernel(float* inp, float* weight, float* out, int B, int T, int C) {
    float eps = 1e-5f;
    //idx的维度是B*T的维度
    int idx = blockIdx.x; //[0, B*T]
    //tid的维度是C的维度,如果block_size小于C那么我们就要考虑无法整除的情况，一个线程可能要处理C维度的多个值
    int tid = threadIdx.x; //[0, block_size]

    if (idx < B * T) {
        const float* x = inp + idx * C;
        float lsum = 0.0f;
        for (int c = tid; c < C; c += block_size) {
            lsum += x[c] * x[c];
        }

        //cub用来做block内的规约求和
        using BlockReduce = cub::BlockReduce<float, block_size>;
        __shared__ BlockReduce::TempStorage tmp;
        //block_sum就是C维度的和，因为一个block的32个线程会处理整个C维度的值
        float block_sum = BlockReduce(tmp).Sum(lsum);

        __shared__ float rms;
        //线程0保存了规约求和的结果
        if (tid == 0)
        {
            rms = rsqrtf(block_sum / C + eps);
        }
        __syncthreads();

        float* y = out + idx * C;
        for (int c = tid; c < C; c += block_size)
        {
            y[c] = rms * x[c] * weight[c];
        }
    }
}