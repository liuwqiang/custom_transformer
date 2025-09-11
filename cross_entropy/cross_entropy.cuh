#pragma once

__global__ void cross_entropy_kernel(float* loss, const float* probs, const int* targets, int B, int T, int V);