#pragma once

__global__ void matmul_kernel(const float* a, const float* b, float* out, int M, int N, int K);