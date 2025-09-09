#include "cuda_runtime.h"
#include "softmax.cuh"

/**
 总共的线程数为 N * blockSize ，即每一个 block 处理一行数据(长度为C)
 对于每个 block, 开辟一个长度为blockSize的共享内存，初始值为 C/blockSize个值的最大值
 对于共享内存，进行reduce操作，最终结果保存在索引为 0 的位置，这样可以求得最大值及和
*/
 __global__ void softmax_kernel(const float *inp, float *out, int B, int T, int C) {
     unsigned const int blockId = blockIdx.x;//[0, N]
     unsigned const int tid = threadIdx.x;//[0, blockSize]
     const float* t_inp = inp + blockId * C;
     extern __shared__ float shared[];
     shared[tid] = 0.0f;
     for (unsigned int i = tid; i < C; i += blockDim.x) {
         shared[tid] = fmaxf(shared[tid], t_inp[i]);
     }
     __syncthreads();
     for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
         __syncthreads();
         if (tid < stride) {
             shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
         }
     }
     __syncthreads();
     float* t_out = out + blockId * C;
     const float maxValue = shared[0];
     for (unsigned int i = tid; i < C; i += blockDim.x) {
         t_out[i] = expf(t_inp[i] - maxValue);
     }
     __syncthreads();
     shared[tid] = 0.0f;
     for (unsigned int i = tid; i < C; i += blockDim.x) {
         shared[tid] += t_out[i];
     }
     __syncthreads();
     for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
         __syncthreads();
         if (tid < stride) {
             shared[tid] += shared[tid + stride];
         }
     }
     __syncthreads();
     const float norm = 1.0f / shared[0];
     for (unsigned int i = tid; i < C; i += blockDim.x) {
         t_out[i] *= norm;
     }
}