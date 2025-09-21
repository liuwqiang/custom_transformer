#pragma once

//multi-query attention,q的head是key/value的2倍，也就是每两个query head共享一个key/value的head
constexpr int SEQ_LEN = 8192;
constexpr int N_HEADS = 16;
constexpr int N_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;
__global__ void attention_qk_kernel(float* att,const float* q,const float* k_cache,int pos);

__global__ void attention_softmax_kernel(float* att, int pos);

__global__ void attention_v_kernel(float* out, const float* att, const float* v_cache,int pos);