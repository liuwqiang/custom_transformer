#include "cuda_runtime.h"
#include "attention.cuh"
#include <stdio.h>

/**
 * 多头注意力
 * 注意力的计算分为三步
 * 1.计算 qk的值
 * 2.计算 softmax 将qk的分数转换为权重
 * 3.权重和v相乘得到最后的输出
 * 多头注意力：
 * 将原有的 [B,T,C]的维度映射到[B,T,H,HD]的维度
 */
//grid N_HEAD: query head的数量
//block pos + 1：最大值1024
//attr: [N_HEAD, SEQ_LEN]
//q:[N_HEAD,HEAD_DIM]
//k_cache:[SEQ_LEN,KV_DIM]
//pos: 当前正在输出的token的位置
__global__ void attention_qk_kernel(float* att,const float* q,const float* k_cache,int pos) {
    //h的维度是[0,N_HEAD]
    int h = blockIdx.x;
    //t的维度是[0,pos]
    int t = threadIdx.x;

    //计算k在cache中的索引，兼容MQA和GQA等变体
    int kv_mul = N_HEADS / N_KV_HEADS;

    //计算每一个token的key向量和当前正在输出的token的query向量注意力得分
    if (t <= pos) {
        //获取当前的query向量，因为是多头注意力，仅获取当前head的query向量
        const float* q_head = q + h * HEAD_DIM;
        //兼容MQA和GQA等变种，需要kv_cache的head索引，“多个query共享单个key”
        int kv_head_idx = h / kv_mul;
        //获取当前位置的key向量
        const float* k_vec = k_cache + t * KV_DIM + kv_head_idx * HEAD_DIM;

        //计算当前query和当前key的score分数
        float score = 0.0f;
        for (int i = 0; i < HEAD_DIM; i++) {
            score += q_head[i] * k_vec[i];
        }
        //执行缩放
        score /= sqrtf(HEAD_DIM);
        att[h * SEQ_LEN + t] = score;
    }
}

/**
 * 将注意力得分转换为权重，实际计算中由于引入了多头注意力，那么就需要将每个头的注意力得分转换为权重
 */
//grid N_HEAD: query head的数量
//block: 1
__global__ void attention_softmax_kernel(float* att, int pos) {
    int h = blockIdx.x;
    float* score = att + h * SEQ_LEN;

    //计算当前头中的最大分
    float max_val = -1e9f;
    for (int i = 0; i <= pos; i++) {
        if (score[i] > max_val) {
            max_val = score[i];
        }
    }

    //计算exp以及求和
    float sum = 0.0f;
    for (int i = 0; i <= pos; i++) {
        score[i] = expf(score[i] - max_val);
        sum += score[i];
    }

    //归一化
    float inv_sum = 1.0f / sum;
    for (int i = 0; i <= pos; i++) {
        score[i] *= inv_sum;
    }
}

/**
 * 将归一化后的注意力权重和value向量相乘，计算每个token和当前输出token的注意力做成当前输出token的向量
 */
//out:[N_HEAD,HEAD_DIM]
//att:[N_HEAD,SEQ_lEN]
//v_cache:[SEQ_LEN,KV_DIM]
//grid: N_HEAD
//block: HEAD_DIM
__global__ void attention_v_kernel(float* out, const float* att, const float* v_cache,int pos) {
    int h = blockIdx.x;
    int i = threadIdx.x; //第h个head的第i维向量

    //计算k在cache中的索引，兼容MQA和GQA等变体
    int kv_mul = N_HEADS / N_KV_HEADS;

    if (i < pos) {
        const float* score = att + h * SEQ_LEN;
        int kv_head_idx = h / kv_mul;
        float* out_head = out + h * HEAD_DIM;

        float weight_sum = 0.0f;
        //计算value和attention的权重
        for (int t = 0; t <= pos; t++) {
            //计算value的向量，这里是通过计算第t个token的第i维的向量和第t个token的注意力权重和得到的
            const float* v_vec = v_cache + t * KV_DIM + kv_head_idx * HEAD_DIM;
            weight_sum += v_vec[i] * score[t];
        }
        out_head[i] = weight_sum;
    }
}