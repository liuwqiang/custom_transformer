#include "cuda_runtime.h"
#include "rope.cuh"

__global__ void rope_kernel(float* q, float* k, int pos) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //每个线程处理两个向量
    if (tid < Q_DIM / 2) {
        //在多头注意力里，输入向量会被拆成多个 head，每个head处理一部分维度。
        //head_dim_idx: 因为Q向量的组成是：[num_heads * HEAD_DIM] 拼在一起的，我们需要位置编码要在每个head内重复，而不是跨head共享
        //例如 head_dim = 64，偶数维度索引2i,2i 会取模 64，保证频率公式用的是0–63之间的索引
        int head_dim_idx = (tid * 2) % HEAD_DIM;
        //θi=10000^-2i/d
        float freq = 1.0f / powf(ROPE_THETA, (float)head_dim_idx / (float)HEAD_DIM);
        //θi * P
        float val = (float)pos * freq;
        float fcr, fci;
        //同时得到sin(val)和cos(val)
        sincosf(val, &fci, &fcr);
        //旋转q向量
        float q0 = q[2 * tid];     // 原来的实部
        float q1 = q[2 * tid + 1]; // 原来的虚部
        //x′=q0cos−q1sin
        float new_q0 = q0 * fcr - q1 * fci; // 实部旋转
        //y′=q0sin+q1cos
        float new_q1 = q0 * fci + q1 * fcr; // 虚部旋转
        q[2 * tid]     = new_q0;
        q[2 * tid + 1] = new_q1;

        if (tid < KV_DIM / 2)
        {
            // 旋转k向量
            float k0 = k[2 * tid];     // 原来的实部
            float k1 = k[2 * tid + 1]; // 原来的虚部
            //x′=k0cos−k1sin
            float new_k0 = k0 * fcr - k1 * fci; // 实部旋转
            //y′=k0sin+k1cos
            float new_k1 = k0 * fci + k1 * fcr; // 虚部旋转
            k[2 * tid]     = new_k0;
            k[2 * tid + 1] = new_k1;
        }
    }
}