/* forward.h -- one sequence's state and one token's pass.
 *
 * A `tt_state` is a slot: its own KV cache, its own scratch buffers, the list of
 * ids already processed.  All memory comes from two arenas sized in
 * `tt_state_init` -- after that the token loop calls no malloc and reads no
 * environment variable (M1 exit criterion, docs/05).
 *
 * M1's KV is contiguous per slot and stores K and V ALREADY EXPANDED.  That is a
 * direct consequence of docs/15: the unabsorbed MLA formulation is the oracle's,
 * and it needs complete K and V.  It costs 1.9 MB per token (47 layers x 20
 * heads x 512 values x 4 B) -- 7.9 GB at 4k of context, which is M1's
 * measurement configuration.  Agent-sized context requires the absorbed
 * formulation or half-precision KV, and that is M5.
 */
#ifndef TOOTEGA_FORWARD_H
#define TOOTEGA_FORWARD_H

#include "model.h"

typedef struct {
    const tt_model *m;

    tt_arena kv;               /* the cache, lives between tokens */
    tt_arena scratch;          /* pass buffers, reset every token */

    int64_t  ctx;              /* capacity in tokens */
    int64_t  len;              /* tokens already in the cache */
    int64_t  logits_len;       /* value of `len` when `logits` was produced; 0 = stale */

    /* EXPANDED mode: full K and V.  ABSORBED mode: `kvc` in place of both,
     * [layer][pos][kv_lora_rank + qk_rope_head_dim]. */
    float   *k;                /* [layer][pos][H * qk_head_dim] */
    float   *v;                /* [layer][pos][H * v_head_dim]  */
    float    *kvc;             /* [layer][pos][576], f32 mode */
    uint16_t *kvc16;           /* the same in bf16 -- half the traffic */
    int8_t   *kvq;             /* the same in int8 g64 -- 3.8x less */
    float    *kvq_scales;      /* [layer][pos][576/64] */
    int32_t *tokens;           /* ids in the cache, for prefix matching */

    /* scratch, everything pointing inside `scratch` */
    float *h, *hn, *resid;
    float *q_lora, *q, *kv_a, *ckv, *kvb, *krot, *attn, *scores;
    /* [H][kv_lora_rank] each, absorbed mode only.  `wrow` is the dequantised
     * row of `kv_b` -- one per head, because the heads run in parallel and each
     * walks its own 448 rows. */
    float *q_abs, *o_c, *wrow;
    float *gate_buf, *up_buf, *mlp_out, *expert_out;
    /* Fused MoE: gate+up of the K experts plus the shared one, and the down
     * outputs before combination. */
    float *moe_gu, *moe_down;
    float *router_logits, *router_scores, *router_choice;
    float *logits;
    int32_t *topk_idx;
    float   *topk_w;

    /* Debug trace: if not NULL, receives the hidden state at the end of each
     * layer -- [num_hidden_layers + 1][hidden_size], with the first entry being
     * the embedding.  It is what allows bisecting against transformers and
     * finding the FIRST layer that diverges, instead of rereading the code. */
    float   *trace;

    /* Routing trace: if not NULL, receives which experts the router picked at
     * each layer of the last pass and with what weight --
     * [num_hidden_layers][num_experts_per_tok], ascending expert index, the
     * order they are accumulated in.  A dense layer writes -1.  The caller owns
     * both buffers; `Tests/route` is what reads them. */
    int32_t *route;
    float   *route_w;

    /* report counters */
    uint64_t tokens_processed;
    double   seconds_in_forward;
} tt_state;

int  tt_state_init(tt_state *s, const tt_model *m, int64_t ctx);
void tt_state_free(tt_state *s);
void tt_state_reset(tt_state *s);       /* forgets the sequence, keeps the memory */

/* KV bytes for `ctx` tokens -- the docs/09 budget needs this before anything is
 * allocated. */
uint64_t tt_kv_bytes(const tt_model *m, int64_t ctx);

/* The pass's scratch.  It exists separately from `tt_state_init` because
 * `--plan` has to answer "does it fit?" without allocating (docs/09). */
uint64_t tt_scratch_bytes(const tt_model *m, int64_t ctx);

/* Processes `token` at position s->len, advances the cache and returns the
 * logits (a pointer inside the scratch, valid until the next pass). */
int tt_forward(tt_state *s, int32_t token, float **logits_out);

/* The same pass WITHOUT the vocabulary projection: the cache advances exactly
 * as in `tt_forward`, and `s->logits` is left stale.  For prompt positions whose
 * next token is already known -- every one but the last.  `lm_head` is 337 MB of
 * the 2.38 GB a token reads, and a prompt position pays it to predict an id the
 * caller is holding in its hand. */
int tt_forward_prefill(tt_state *s, int32_t token);

/* Runs ids[from..n) and leaves the logits of position n-1 ready to sample: only
 * the last position pays for `lm_head`.  This is the one entry point a prefill
 * should use -- it is also where a batched prefill (docs/13, D16) will land,
 * without any caller changing. */
int tt_prefill(tt_state *s, const int32_t *ids, int64_t from, int64_t n);

/* The logits of the LAST processed position, or NULL when that position ran
 * without the projection.  Sampling code should read through this instead of
 * `s->logits`, so that a stale distribution is an error and not an answer. */
float *tt_state_logits(const tt_state *s);

/* Matches the longest common prefix with the cache and truncates at the
 * divergence.  Returns how many tokens remain valid -- the prefill only has to
 * run over the rest.
 *
 * At most `n - 1`, never `n`: the caller samples from the logits of the last
 * position, and those are produced by the forward pass, not stored in the
 * cache.  See the comment on the implementation. */
int64_t tt_state_match_prefix(tt_state *s, const int32_t *ids, int64_t n);

#endif /* TOOTEGA_FORWARD_H */
