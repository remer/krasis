#pragma once

#include <cuda_bf16.h>
#include <math.h>

/*
 * GLM-5.3 Kimi Delta Attention (KDA).
 *
 * The released architecture uses equal Q/K/V head geometry and a recurrent
 * FP32 state [heads, head_dim, head_dim].  These kernels deliberately retain
 * the exact recurrent equations used by the official Transformers model.
 * They are shared by one-token decode and the correctness-first sequential
 * prefill path; a parallel chunked prefill implementation can replace the
 * latter without changing the persistent-state contract.
 */

__device__ __forceinline__ float glm5_next_kda_sigmoid(float value) {
    return 1.0f / (1.0f + __expf(-value));
}

/*
 * GLM-5.3's sparse indexer does not rank raw token keys.  It keeps a
 * four-token tail, applies a learned per-channel softmax across that tail,
 * and publishes one pooled BF16 key whenever the group is complete.  This
 * kernel also performs the indexer's LayerNorm so the rounding boundary
 * matches the official BF16 module before pooling.
 *
 * Serving inputs are unpadded contiguous token streams.  The absolute token
 * position therefore determines both the tail slot and the completed pool
 * row.  Padded batched prefill has a separate contract and must not call this
 * decode kernel.
 */
extern "C" __global__ void glm5_next_kpool_update_kernel(
    __nv_bfloat16* __restrict__ pooled_key_cache,
    __nv_bfloat16* __restrict__ key_tail,
    __nv_bfloat16* __restrict__ gate_tail,
    const __nv_bfloat16* __restrict__ raw_key,
    const __nv_bfloat16* __restrict__ raw_gate,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ norm_bias,
    const __nv_bfloat16* __restrict__ learned_ape,
    int position,
    int head_dim,
    int pool_size,
    float eps
) {
    if (position < 0 || head_dim <= 0 || pool_size <= 0 || pool_size > 32) {
        return;
    }
    int tid = (int)threadIdx.x;
    int threads = (int)blockDim.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps = (threads + 31) >> 5;
    int slot = position % pool_size;

    extern __shared__ float shared[];
    float* warp_sum = shared;
    float* warp_sq = shared + 32;

    float local_sum = 0.0f;
    float local_sq = 0.0f;
    for (int dim = tid; dim < head_dim; dim += threads) {
        float value = __bfloat162float(raw_key[dim]);
        local_sum += value;
        local_sq += value * value;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq += __shfl_down_sync(0xffffffff, local_sq, offset);
    }
    if (lane == 0) {
        warp_sum[warp] = local_sum;
        warp_sq[warp] = local_sq;
    }
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int index = 0; index < warps; ++index) {
            sum += warp_sum[index];
            sum_sq += warp_sq[index];
        }
        float mean = sum / (float)head_dim;
        float variance = fmaxf(sum_sq / (float)head_dim - mean * mean, 0.0f);
        warp_sum[0] = mean;
        warp_sq[0] = rsqrtf(variance + eps);
    }
    __syncthreads();

    float mean = warp_sum[0];
    float inv_std = warp_sq[0];
    for (int dim = tid; dim < head_dim; dim += threads) {
        float normalized = (__bfloat162float(raw_key[dim]) - mean) * inv_std;
        normalized = normalized * __bfloat162float(norm_weight[dim]) +
            __bfloat162float(norm_bias[dim]);
        key_tail[(long long)slot * head_dim + dim] =
            __float2bfloat16(normalized);
        gate_tail[(long long)slot * head_dim + dim] = raw_gate[dim];
    }
    __syncthreads();

    if (slot != pool_size - 1) {
        return;
    }
    int pool = position / pool_size;
    for (int dim = tid; dim < head_dim; dim += threads) {
        float maximum = -INFINITY;
        for (int item = 0; item < pool_size; ++item) {
            float logit = __bfloat162float(
                gate_tail[(long long)item * head_dim + dim]) +
                __bfloat162float(learned_ape[(long long)item * head_dim + dim]);
            maximum = fmaxf(maximum, logit);
        }
        float denominator = 0.0f;
        for (int item = 0; item < pool_size; ++item) {
            float logit = __bfloat162float(
                gate_tail[(long long)item * head_dim + dim]) +
                __bfloat162float(learned_ape[(long long)item * head_dim + dim]);
            denominator += __expf(logit - maximum);
        }
        float pooled = 0.0f;
        for (int item = 0; item < pool_size; ++item) {
            float logit = __bfloat162float(
                gate_tail[(long long)item * head_dim + dim]) +
                __bfloat162float(learned_ape[(long long)item * head_dim + dim]);
            __nv_bfloat16 probability =
                __float2bfloat16(__expf(logit - maximum) / denominator);
            // The released reference materializes the elementwise product in
            // BF16 before its reduction.  Preserve that rounding boundary;
            // accumulating the unrounded FP32 product measurably drifts from
            // the checkpoint's reference path over many pools.
            __nv_bfloat16 contribution = __float2bfloat16(
                __bfloat162float(probability) *
                __bfloat162float(key_tail[(long long)item * head_dim + dim]));
            pooled += __bfloat162float(contribution);
        }
        pooled_key_cache[(long long)pool * head_dim + dim] =
            __float2bfloat16(pooled);
    }
}

/* Expand selected pool rows back into raw token positions and append the
 * current incomplete tail.  Every unused output entry is explicitly -1 so
 * sparse MLA can use one fixed-capacity index buffer. */
extern "C" __global__ void glm5_next_kpool_expand_indices_kernel(
    int* __restrict__ output,
    const int* __restrict__ selected_pools,
    int selected_pool_count,
    int position,
    int pool_size,
    int output_capacity
) {
    int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (index >= output_capacity || selected_pool_count < 0 || position < 0 ||
        pool_size <= 0) {
        return;
    }
    int expanded_count = selected_pool_count * pool_size;
    int value = -1;
    if (index < expanded_count) {
        int pool = selected_pools[index / pool_size];
        if (pool >= 0) {
            value = pool * pool_size + index % pool_size;
        }
    } else {
        int tail_count = (position + 1) % pool_size;
        int tail_offset = index - expanded_count;
        if (tail_offset < tail_count) {
            value = (position + 1 - tail_count) + tail_offset;
        }
    }
    output[index] = value;
}

/* GLM's final mHC head is the unweighted mean of the residual streams. */
extern "C" __global__ void glm5_next_hc_mean_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ state,
    int hidden_size,
    int hc_mult
) {
    int hidden = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (hidden >= hidden_size || hc_mult <= 0) {
        return;
    }
    float sum = 0.0f;
    for (int stream = 0; stream < hc_mult; ++stream) {
        sum += __bfloat162float(state[(long long)stream * hidden_size + hidden]);
    }
    output[hidden] = __float2bfloat16(sum / (float)hc_mult);
}

/* Shift one depthwise-convolution row, append the new projected value, and
 * emit SiLU(conv(row)).  State and checkpoint convolution weights are BF16,
 * while accumulation is FP32. */
extern "C" __global__ void glm5_next_kda_conv_decode_kernel(
    __nv_bfloat16* __restrict__ convolved_qkv,
    const __nv_bfloat16* __restrict__ projected_qkv,
    const __nv_bfloat16* __restrict__ conv_weight,
    __nv_bfloat16* __restrict__ conv_state,
    int conv_dim,
    int kernel_dim
) {
    int channel = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (channel >= conv_dim || kernel_dim <= 0) {
        return;
    }
    long long row = (long long)channel * kernel_dim;
    float sum = 0.0f;
    for (int tap = 0; tap < kernel_dim - 1; ++tap) {
        __nv_bfloat16 value = conv_state[row + tap + 1];
        conv_state[row + tap] = value;
        sum = fmaf(
            __bfloat162float(value),
            __bfloat162float(conv_weight[row + tap]),
            sum
        );
    }
    __nv_bfloat16 newest = projected_qkv[channel];
    conv_state[row + kernel_dim - 1] = newest;
    sum = fmaf(
        __bfloat162float(newest),
        __bfloat162float(conv_weight[row + kernel_dim - 1]),
        sum
    );
    float activated = sum / (1.0f + __expf(-sum));
    convolved_qkv[channel] = __float2bfloat16(activated);
}

/* One CUDA block owns one KDA head and one thread owns one value column of
 * the recurrent state.  This makes the decay, delta update, and output read
 * race-free while preserving the official operation order:
 *
 *   S <- exp(forget) * S
 *   delta <- beta * (v - k^T S)
 *   S <- S + k * delta^T
 *   o <- q^T S
 *
 * The output is then RMS-normalized and sigmoid-gated per head. */
extern "C" __global__ void glm5_next_kda_recurrent_decode_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ convolved_qkv,
    const __nv_bfloat16* __restrict__ forget_projection,
    const __nv_bfloat16* __restrict__ beta_logits,
    const __nv_bfloat16* __restrict__ output_gate,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ output_norm,
    float* __restrict__ recurrent_state,
    int num_heads,
    int head_dim,
    float query_scale,
    float gate_lower_bound,
    float norm_eps
) {
    int head = (int)blockIdx.x;
    int value_col = (int)threadIdx.x;
    if (head >= num_heads || value_col >= head_dim || head_dim <= 0) {
        return;
    }

    extern __shared__ float shared[];
    float* query = shared;
    float* key = query + head_dim;
    float* attention = key + head_dim;
    float* reduction = attention + head_dim;

    long long head_offset = (long long)head * head_dim;
    long long q_offset = head_offset;
    long long k_offset = (long long)num_heads * head_dim + head_offset;
    long long v_offset = 2LL * num_heads * head_dim + head_offset;
    float q_value = __bfloat162float(convolved_qkv[q_offset + value_col]);
    float k_value = __bfloat162float(convolved_qkv[k_offset + value_col]);
    query[value_col] = q_value;
    key[value_col] = k_value;
    reduction[value_col] = q_value * q_value;
    __syncthreads();

    for (int width = head_dim >> 1; width > 0; width >>= 1) {
        if (value_col < width) {
            reduction[value_col] += reduction[value_col + width];
        }
        __syncthreads();
    }
    float q_inverse = rsqrtf(reduction[0] + 1.0e-6f) * query_scale;
    reduction[value_col] = k_value * k_value;
    __syncthreads();
    for (int width = head_dim >> 1; width > 0; width >>= 1) {
        if (value_col < width) {
            reduction[value_col] += reduction[value_col + width];
        }
        __syncthreads();
    }
    float k_inverse = rsqrtf(reduction[0] + 1.0e-6f);
    query[value_col] *= q_inverse;
    key[value_col] *= k_inverse;
    __syncthreads();

    float decay_rate = __expf(a_log[head]);
    float memory = 0.0f;
    long long state_base = (long long)head * head_dim * head_dim;
    for (int key_row = 0; key_row < head_dim; ++key_row) {
        long long state_index = state_base + (long long)key_row * head_dim + value_col;
        float raw_forget = __bfloat162float(
            forget_projection[head_offset + key_row]
        ) + dt_bias[head_offset + key_row];
        float forget = gate_lower_bound *
            glm5_next_kda_sigmoid(decay_rate * raw_forget);
        float state_value = recurrent_state[state_index] * __expf(forget);
        recurrent_state[state_index] = state_value;
        memory = fmaf(state_value, key[key_row], memory);
    }
    float beta = glm5_next_kda_sigmoid(__bfloat162float(beta_logits[head]));
    float value = __bfloat162float(convolved_qkv[v_offset + value_col]);
    float delta = (value - memory) * beta;
    float result = 0.0f;
    for (int key_row = 0; key_row < head_dim; ++key_row) {
        long long state_index = state_base + (long long)key_row * head_dim + value_col;
        float state_value = recurrent_state[state_index] + key[key_row] * delta;
        recurrent_state[state_index] = state_value;
        result = fmaf(state_value, query[key_row], result);
    }
    // The released recurrent helper casts core_attn_out back to the incoming
    // BF16 dtype before Glm5NextTextRMSNormGated converts it to FP32. Preserve
    // that architecture-visible rounding boundary.
    __nv_bfloat16 attention_bf16 = __float2bfloat16(result);
    float rounded_attention = __bfloat162float(attention_bf16);
    attention[value_col] = rounded_attention;
    reduction[value_col] = rounded_attention * rounded_attention;
    __syncthreads();

    for (int width = head_dim >> 1; width > 0; width >>= 1) {
        if (value_col < width) {
            reduction[value_col] += reduction[value_col + width];
        }
        __syncthreads();
    }
    float inv_rms = rsqrtf(reduction[0] / (float)head_dim + norm_eps);
    float gated = rounded_attention
        * inv_rms
        * __bfloat162float(output_norm[value_col])
        * glm5_next_kda_sigmoid(
            __bfloat162float(output_gate[head_offset + value_col])
        );
    output[head_offset + value_col] = __float2bfloat16(gated);
}
