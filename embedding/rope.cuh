#pragma once
#define ROPE_THETA 1000000
constexpr int HEAD_DIM = 128;
constexpr int N_HEADS = 16;
constexpr int N_KV_HEADS = 8;
constexpr int Q_DIM =     N_HEADS * HEAD_DIM; // 16 * 128 = 2048
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; //  8 * 128 = 1024
__global__ void rope_kernel(float* q, float* k, int pos);