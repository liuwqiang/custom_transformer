#pragma once

__global__ void layernorm_kernel(const float* inp, float* mean, float* rstd, const float* weight, const float* bias, float* out
    , int B, int T, int C);