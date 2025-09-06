#pragma once

__global__ void layernorm_kernel(const float* inp, float* mean, float* rstd, const float* weight, const float* bias, float* out
    , int B, int T, int C, int block_size);

__global__ void rstd_kernel(const float* inp, float* out, float* mean, float* rstd, int C,  int block_size);

__global__ void mean_kernel(const float* inp, float* out, float* mean, int C, int block_size);