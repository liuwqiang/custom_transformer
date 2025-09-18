#pragma once
constexpr int block_size = 64;
__global__ void rms_norm_kernel(float* inp, float* weight, float* out, int B, int T, int C);