#pragma once
#define V 4096
#define L 2048
__global__ void embedding_kernel(float* out, const int* inp, const float* wte, const float* wpe, int B, int T, int C);