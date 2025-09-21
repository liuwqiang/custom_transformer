#include "cuda_runtime.h"
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include "attention.cuh"

#define CUDA_CHECK(call) \
if ((call) != cudaSuccess) { \
fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(call), __FILE__, __LINE__); \
exit(1); \
}

void rand(float* elements, int size) {
    for(int i = 0; i < size; i++){
        elements[i] = (float)rand() / RAND_MAX;
    }
}

bool check(float *out, float *res, int n) {
    for (int i = 0; i < n; i++) {
        if (std::fabs(out[i] - res[i]) > 1e-4) {
            return false;
        }
    }
    return true;
}

/**
 * CPU attention函数
 */
//out: [N_HEAD,HEAD_DIM]
//attr: [N_HEAD, SEQ_LEN]
//q:[N_HEAD,HEAD_DIM]
//k_cache:[SEQ_LEN,KV_DIM]
//v_cache:[SEQ_LEN,KV_DIM]
//pos: [SEQ_LEN]
void attention_cpu(float* out, const float* q, const float* k_cache, const float* v_cache, int pos) {
    //计算qk的注意力得分
    float* att = (float*) malloc(N_HEADS * SEQ_LEN * sizeof(float));
    int kv_mul = N_HEADS / N_KV_HEADS;
    for (int h = 0; h < N_HEADS; h++) {
        const float* q_head = q + h * HEAD_DIM;
        int kv_head_idx = h / kv_mul;
        for (int t = 0; t <= pos; t++) {
            const float* k_vec = k_cache + t * KV_DIM + kv_head_idx * HEAD_DIM;
            float score = 0.0f;
            for (int i = 0; i < HEAD_DIM; i++) {
                score += q_head[i] * k_vec[i];
            }
            score /= sqrtf(HEAD_DIM);
            att[h * SEQ_LEN + t] = score;
        }
    }

    //计算softmax
    for (int h = 0;h < N_HEADS; h++) {
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
    //计算输出
    for (int h = 0;h < N_HEADS; h++) {
        const float* score = att + h * SEQ_LEN;
        int kv_head_idx = h / kv_mul;
        float* out_head = out + h * HEAD_DIM;
        for (int i = 0 ; i < HEAD_DIM; i++) {
            float weight_sum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                const float* v_vec = v_cache + t * KV_DIM + kv_head_idx * HEAD_DIM;
                weight_sum += v_vec[i] * score[t];
            }
            out_head[i] = weight_sum;
        }
    }
}

//out: [N_HEAD,HEAD_DIM]
//attr: [N_HEAD, SEQ_LEN]
//q:[N_HEAD,HEAD_DIM]
//k_cache:[SEQ_LEN,KV_DIM]
//v_cache:[SEQ_LEN,KV_DIM]
//pos: [SEQ_LEN]
void attention_gpu_base(float* out, const float* q, const float* k_cache, const float* v_cache, int pos) {
    float* d_out;
    cudaMalloc(&d_out, N_HEADS * HEAD_DIM * sizeof(float));

    float* d_att;
    cudaMalloc(&d_att, N_HEADS * SEQ_LEN * sizeof(float));

    float* d_q;
    cudaMalloc(&d_q, N_HEADS * HEAD_DIM * sizeof(float));

    float* d_k_cache;
    cudaMalloc(&d_k_cache, SEQ_LEN * KV_DIM * sizeof(float));

    float* d_v_cache;
    cudaMalloc(&d_v_cache, SEQ_LEN * KV_DIM * sizeof(float));

    //拷贝数据到显存
    cudaMemcpy(d_q, q, N_HEADS * HEAD_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_cache, k_cache, SEQ_LEN * KV_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_cache, v_cache, SEQ_LEN * KV_DIM * sizeof(float), cudaMemcpyHostToDevice);

    //计算kv注意力得分
    attention_qk_kernel<<<N_HEADS, pos + 1>>>(d_att, d_q, d_k_cache, pos);
    //计算softmax
    attention_softmax_kernel<<<N_HEADS, 1>>>(d_att, pos);
    //计算输出
    attention_v_kernel<<<N_HEADS, HEAD_DIM>>>(d_out, d_att, d_v_cache, pos);

    cudaMemcpy(out, d_out, N_HEADS * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_att);
    cudaFree(d_q);
    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
}

int main(int argc, char *argv[]) {
    int pos = 128;
    float* out = (float*) malloc(N_HEADS * HEAD_DIM * sizeof(float));
    float* d_out = (float*) malloc(N_HEADS * HEAD_DIM * sizeof(float));

    float* q = (float*) malloc(N_HEADS * HEAD_DIM * sizeof(float));
    rand(q, N_HEADS * HEAD_DIM);

    float* k_cache = (float*) malloc(SEQ_LEN * KV_DIM * sizeof(float));
    float* v_cache = (float*) malloc(SEQ_LEN * KV_DIM * sizeof(float));
    rand(k_cache, SEQ_LEN * KV_DIM);
    rand(v_cache, SEQ_LEN * KV_DIM);

    attention_cpu(out, q, k_cache, v_cache, pos);
    attention_gpu_base(d_out, q, k_cache, v_cache, pos);

    if (check(out, d_out, N_HEADS * HEAD_DIM)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
    }

    free(out);
    free(d_out);
    free(k_cache);
    free(v_cache);
}
