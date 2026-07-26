/* model.h -- config -> IR -> bound weights.
 *
 * The engine reads the architecture from the container (`meta.config`), never
 * from a `config.json` sitting next to it that may have drifted (docs/14).  Each
 * weight is located once at open time, with its shape checked against what the
 * config promises: a swapped tensor has to error out here, with its name and
 * shape, and not become numerical garbage 47 layers later.
 *
 * The architectural variation points (attention, routing) exist as IR fields
 * from M1 on, but only GLM-4.7-Flash's combination is implemented: MLA with
 * q-LoRA and a top-k router with n_group = 1 (docs/05, "Out of scope").
 */
#ifndef TOOTEGA_MODEL_H
#define TOOTEGA_MODEL_H

#include <stdint.h>

#include "container.h"

typedef struct {
    int64_t hidden_size;
    int64_t vocab_size;
    int64_t num_hidden_layers;
    int64_t num_attention_heads;
    int64_t intermediate_size;         /* the dense layer's MLP */
    int64_t moe_intermediate_size;     /* one expert's MLP */
    int64_t n_routed_experts;
    int64_t n_shared_experts;
    int64_t num_experts_per_tok;       /* top-k */
    int64_t first_k_dense_replace;     /* dense layers at the start */
    int64_t n_group, topk_group;
    int64_t max_position_embeddings;
    int64_t mtp_layer;                 /* -1 if there is none */

    /* MLA */
    int64_t q_lora_rank;
    int64_t kv_lora_rank;
    int64_t qk_nope_head_dim;
    int64_t qk_rope_head_dim;
    int64_t qk_head_dim;               /* nope + rope */
    int64_t v_head_dim;

    float   routed_scaling_factor;
    float   rms_norm_eps;
    double  rope_theta;
    int     norm_topk_prob;

    float   attn_scale;                /* qk_head_dim ^ -0.5 */
} tt_config;

typedef struct {
    int is_moe;

    const tt_tensor *input_layernorm;
    const tt_tensor *post_attention_layernorm;

    const tt_tensor *q_a_proj, *q_a_layernorm, *q_b_proj;
    const tt_tensor *kv_a_proj, *kv_a_layernorm, *kv_b_proj;
    const tt_tensor *o_proj;

    /* dense (is_moe == 0) */
    const tt_tensor *gate_proj, *up_proj, *down_proj;

    /* MoE (is_moe == 1) */
    const tt_tensor  *router, *router_bias;
    const tt_tensor  *sh_gate, *sh_up, *sh_down;
    const tt_tensor **e_gate, **e_up, **e_down;   /* [n_routed_experts] */
} tt_layer;

/* The MLA attention formulation.  The two are algebraically equivalent and do
 * NOT produce the same bits (docs/15); the choice is explicit precisely because
 * of that.
 *
 *   EXPANDED  unabsorbed, the oracle's.  Caches full K and V: 10,240 values per
 *             token per layer.
 *   ABSORBED  folds `kv_b` into the query and operates in the compressed space.
 *             Caches 576 values per token per layer -- 18x less traffic, which
 *             is what decides performance at long context (docs/13, D11).
 */
typedef enum {
    TT_ATTN_ABSORBED = 0,
    TT_ATTN_EXPANDED
} tt_attn_mode;

/* KV cache precision.  bf16 keeps fp32's 8 exponent bits and halves both the
 * traffic and the L3 pressure; it is an output change, authorised by the same
 * 32/32 test that authorised absorption. */
typedef enum {
    TT_KV_BF16 = 0,
    TT_KV_F32,
    TT_KV_INT8            /* g64 with a scale, like the container */
} tt_kv_prec;

#define TT_KV_GROUP 64

int tt_kv_prec_parse(const char *s, tt_kv_prec *out);

/* "f32" | "bf16" | "int8".  It exists for the measurement header: a number
 * without its configuration next to it is not comparable later (docs/17). */
const char *tt_kv_prec_name(tt_kv_prec p);

typedef struct {
    tt_container c;
    tt_config    cfg;
    tt_arena     ir;               /* IR pointers + RoPE tables */

    tt_attn_mode attn;
    tt_kv_prec   kv_prec;

    tt_layer    *layers;
    const tt_tensor *embed;        /* [vocab, hidden], bf16, file-backed */
    const tt_tensor *norm;         /* final norm */
    const tt_tensor *lm_head;

    /* Precomputed RoPE: cos/sin[pos][qk_rope_head_dim/2].  A table rather than
     * sinf/cosf in the loop -- docs/15, D4. */
    float   *rope_cos, *rope_sin;
    int64_t  rope_positions;
} tt_model;

/* `max_pos` sizes the RoPE table; pass the context you are going to use. */
int  tt_model_open(tt_model *m, const char *path, int64_t max_pos,
                   tt_attn_mode attn, tt_kv_prec kv_prec);
void tt_model_close(tt_model *m);

int tt_attn_parse(const char *s, tt_attn_mode *out);

/* First row of `kv_b` that belongs to this head.  W_k is the next
 * `qk_nope_head_dim` rows; W_v, the `v_head_dim` rows after those. */
static inline int64_t tt_kvb_head_row(const tt_config *c, int64_t head)
{
    return head * (c->qk_nope_head_dim + c->v_head_dim);
}

/* Weight bytes that stay resident, for the docs/09 budget. */
uint64_t tt_model_weight_bytes(const tt_model *m);

/* Brings the resident weights into RAM before the first token.
 *
 * Without this, every weight arrives through a page fault in the middle of the
 * decode loop, and the first pass costs 5x more per token -- measured: 480 ms
 * against 95 ms.  On a server the whole bill falls on the first user, who has no
 * way of knowing why they waited.
 *
 * `embed_tokens` is left out: it is read one row per token and needs no
 * residency (docs/04).  Returns the bytes touched. */
uint64_t tt_model_warm(const tt_model *m);

#endif /* TOOTEGA_MODEL_H */
