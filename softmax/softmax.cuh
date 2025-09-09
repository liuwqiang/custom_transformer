#pragma once

__global__ void softmax_kernel(const float* inp, float* out, int B, int T, int C);