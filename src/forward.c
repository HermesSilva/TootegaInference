/* forward.c -- o passe de um token.
 *
 * A referencia e `Glm4MoeLite*` do transformers, e a ordem das operacoes segue
 * a dela deliberadamente: onde ha duas formas algebricamente equivalentes,
 * usamos a que o oraculo usa (docs/15, "Onde o MLA pode divergir").
 */
#include "forward.h"
#include "kernels.h"
#include "pool.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------- estado */

uint64_t tt_kv_bytes(const tt_model *m, int64_t ctx)
{
    const tt_config *c = &m->cfg;

    /* A diferenca entre os dois modos e o numero inteiro do D11: 10.240
     * valores por token por camada contra 576. */
    const uint64_t per_token = (m->attn == TT_ATTN_ABSORBED)
        ? (uint64_t)(c->kv_lora_rank + c->qk_rope_head_dim)
        : (uint64_t)c->num_attention_heads * (uint64_t)(c->qk_head_dim + c->v_head_dim);

    uint64_t bytes_per_value;
    switch (m->kv_prec) {
    case TT_KV_BF16: bytes_per_value = sizeof(uint16_t); break;
    case TT_KV_INT8: bytes_per_value = 1; break;
    default:         bytes_per_value = sizeof(float); break;
    }

    uint64_t total = (uint64_t)c->num_hidden_layers * (uint64_t)ctx
                   * per_token * bytes_per_value;

    /* int8 carrega uma escala f32 por grupo de 64 -- 4/64 de byte por valor. */
    if (m->kv_prec == TT_KV_INT8)
        total += (uint64_t)c->num_hidden_layers * (uint64_t)ctx
               * (per_token / TT_KV_GROUP) * sizeof(float);

    return total;
}

uint64_t tt_scratch_bytes(const tt_model *m, int64_t ctx)
{
    const tt_config *c = &m->cfg;
    const int64_t H  = c->num_attention_heads;
    const int64_t qk = c->qk_head_dim, vd = c->v_head_dim;
    const int64_t inter = c->intermediate_size > c->moe_intermediate_size * c->n_shared_experts
                        ? c->intermediate_size
                        : c->moe_intermediate_size * c->n_shared_experts;
    if (ctx < 1) ctx = 1;

    return (uint64_t)(c->hidden_size * 3
                    + c->q_lora_rank
                    + H * qk
                    + (c->kv_lora_rank + c->qk_rope_head_dim)
                    + c->kv_lora_rank
                    + H * (c->qk_nope_head_dim + vd)
                    + c->qk_rope_head_dim
                    + H * vd
                    + H * ctx
                    + H * c->kv_lora_rank * 3
                    + (c->num_experts_per_tok + 1) * (2 * c->moe_intermediate_size + c->hidden_size)
                    + inter * 2
                    + c->hidden_size * 2
                    + c->n_routed_experts * 3
                    + c->vocab_size
                    + c->num_experts_per_tok * 2) * sizeof(float)
         + (uint64_t)ctx * sizeof(int32_t)
         + 40 * TT_ALIGN;
}

int tt_state_init(tt_state *s, const tt_model *m, int64_t ctx)
{
    memset(s, 0, sizeof(*s));
    s->m = m;
    s->ctx = ctx > 0 ? ctx : 1;

    const tt_config *c = &m->cfg;
    const int64_t H  = c->num_attention_heads;
    const int64_t qk = c->qk_head_dim, vd = c->v_head_dim;

    /* KV: uma alocacao so, para os 47 x ctx blocos.  Se nao couber, e melhor
     * saber agora e com o numero na mao (docs/09: recusar, nao degradar). */
    uint64_t kv_bytes = tt_kv_bytes(m, s->ctx);
    if (tt_arena_init(&s->kv, (size_t)kv_bytes + 2 * TT_ALIGN, "kv") != 0)
        return -1;

    if (m->attn == TT_ATTN_ABSORBED) {
        const int64_t w = c->kv_lora_rank + c->qk_rope_head_dim;
        const int64_t n = c->num_hidden_layers * s->ctx * w;
        if (m->kv_prec == TT_KV_INT8) {
            if (w % TT_KV_GROUP) { tt_state_free(s);
                return tt_fail("KV int8: %lld valores nao divide por %d",
                               (long long)w, TT_KV_GROUP); }
            s->kvq = (int8_t *)tt_arena_alloc(&s->kv, (size_t)n, TT_ALIGN);
            s->kvq_scales = tt_arena_new(&s->kv, float, n / TT_KV_GROUP);
            if (!s->kvq || !s->kvq_scales) { tt_state_free(s); return -1; }
        } else if (m->kv_prec == TT_KV_BF16) {
            s->kvc16 = (uint16_t *)tt_arena_alloc(&s->kv,
                          (size_t)n * sizeof(uint16_t), TT_ALIGN);
            if (!s->kvc16) { tt_state_free(s); return -1; }
        } else {
            s->kvc = tt_arena_new(&s->kv, float, n);
            if (!s->kvc) { tt_state_free(s); return -1; }
        }
    } else {
        s->k = tt_arena_new(&s->kv, float, c->num_hidden_layers * s->ctx * H * qk);
        s->v = tt_arena_new(&s->kv, float, c->num_hidden_layers * s->ctx * H * vd);
        if (!s->k || !s->v) { tt_state_free(s); return -1; }
    }

    size_t need = (size_t)tt_scratch_bytes(m, s->ctx);
    if (tt_arena_init(&s->scratch, need, "scratch") != 0) { tt_state_free(s); return -1; }

    const int64_t inter = c->intermediate_size > c->moe_intermediate_size * c->n_shared_experts
                        ? c->intermediate_size
                        : c->moe_intermediate_size * c->n_shared_experts;

    #define TAKE(field, n) do {                                          \
            (field) = tt_arena_new(&s->scratch, float, (n));             \
            if (!(field)) { tt_state_free(s); return -1; }               \
        } while (0)

    TAKE(s->h,      c->hidden_size);
    TAKE(s->hn,     c->hidden_size);
    TAKE(s->resid,  c->hidden_size);
    TAKE(s->q_lora, c->q_lora_rank);
    TAKE(s->q,      H * qk);
    TAKE(s->kv_a,   c->kv_lora_rank + c->qk_rope_head_dim);
    TAKE(s->ckv,    c->kv_lora_rank);
    TAKE(s->kvb,    H * (c->qk_nope_head_dim + vd));
    TAKE(s->krot,   c->qk_rope_head_dim);
    TAKE(s->attn,   H * vd);
    TAKE(s->scores, H * s->ctx);
    TAKE(s->gate_buf, inter);
    TAKE(s->up_buf,   inter);
    TAKE(s->mlp_out,    c->hidden_size);
    TAKE(s->moe_gu,   (c->num_experts_per_tok + 1) * 2 * c->moe_intermediate_size);
    TAKE(s->moe_down, (c->num_experts_per_tok + 1) * c->hidden_size);
    TAKE(s->expert_out, c->hidden_size);
    TAKE(s->router_logits, c->n_routed_experts);
    TAKE(s->router_scores, c->n_routed_experts);
    TAKE(s->router_choice, c->n_routed_experts);
    TAKE(s->topk_w,  c->num_experts_per_tok);
    TAKE(s->logits,  c->vocab_size);
    if (m->attn == TT_ATTN_ABSORBED) {
        /* Por cabeca, porque as cabecas rodam em paralelo. */
        TAKE(s->q_abs, H * c->kv_lora_rank);
        TAKE(s->o_c,   H * c->kv_lora_rank);
        TAKE(s->wrow,  H * c->kv_lora_rank);
    }
    #undef TAKE

    s->topk_idx = (int32_t *)tt_arena_alloc(&s->scratch,
                    (size_t)c->num_experts_per_tok * sizeof(int32_t), TT_ALIGN);
    s->tokens = (int32_t *)tt_arena_alloc(&s->scratch,
                    (size_t)s->ctx * sizeof(int32_t), TT_ALIGN);
    if (!s->topk_idx || !s->tokens) { tt_state_free(s); return -1; }

    tt_dbg("estado: ctx %lld, kv %.2f GB, scratch %.1f MB",
           (long long)s->ctx, (double)kv_bytes / 1e9,
           (double)s->scratch.used / 1e6);
    return 0;
}

void tt_state_free(tt_state *s)
{
    tt_arena_free(&s->kv);
    tt_arena_free(&s->scratch);
    memset(s, 0, sizeof(*s));
}

void tt_state_reset(tt_state *s)
{
    s->len = 0;
}

int64_t tt_state_match_prefix(tt_state *s, const int32_t *ids, int64_t n)
{
    int64_t i = 0;
    while (i < n && i < s->len && s->tokens[i] == ids[i]) i++;
    s->len = i;                      /* trunca no divergente */
    return i;
}

/* --------------------------------------------------------------- RoPE */

/* RoPE na variante INTERCALADA -- `rope_interleave` e `True` por default no
 * `Glm4MoeLiteConfig`, e o `config.json` deste modelo nao sobrescreve.
 *
 * A diferenca entre as duas variantes e QUEM roda com QUEM:
 *
 *   rotate_half   pareia x[i] com x[i + rot/2]
 *   interleave    pareia x[2i] com x[2i+1]      <- esta
 *
 * O transformers chega nela de-intercalando antes do rotate_half:
 *
 *   q.view(b,h,s,d//2,2).transpose(4,3).reshape(b,h,s,d)
 *
 * que leva o elemento 2i para a posicao i e o 2i+1 para i + d/2. Feita a
 * conta, sobra o par (2i, 2i+1) girando junto, com a saida no layout
 * de-intercalado -- que e o que este laco escreve.
 *
 * Errar isto nao estoura nada: na posicao 0 as duas variantes coincidem
 * (cos=1, sin=0) e o erro cresce com a posicao. O sintoma e texto que comeca
 * bem e desanda -- e perplexidade alta que se confunde com "corpus dificil". */
static void rope_apply(float *x, const float *cosv, const float *sinv, int64_t rot)
{
    const int64_t half = rot / 2;
    float tmp[128];                       /* qk_rope_head_dim = 64 aqui */

    for (int64_t i = 0; i < half; i++) {
        const float a = x[2 * i], b = x[2 * i + 1];
        const float c = cosv[i], sn = sinv[i];
        tmp[i]        = a * c - b * sn;
        tmp[i + half] = b * c + a * sn;
    }
    memcpy(x, tmp, (size_t)rot * sizeof(float));
}

/* ----------------------------------------------------------- atencao */

/* Orcamento de L3 abaixo do qual o cache de uma camada e considerado
 * residente.  Conservador de proposito: o L3 e compartilhado com os pesos que
 * estao passando e com o resto da maquina.  36 MB nesta CPU; usamos 12. */
#define TT_L3_BUDGET (12 * 1024 * 1024)

typedef struct {
    tt_state       *s;
    const tt_layer *L;
    int64_t         li;
    int64_t         T;    /* posicoes visiveis, incluindo a atual */
} attn_job;

/* As tres precisoes de cache atras de uma assinatura so.  Exatamente um dos
 * tres ponteiros e nao-nulo; o desvio e por posicao de contexto, nao por
 * elemento, e some no custo do produto interno de 576 dimensoes. */
static float kv_score(const float *q_abs, const float *q_rot,
                      const float *kvc, const uint16_t *kvc16,
                      const int8_t *kvq, const float *kvqs,
                      int64_t t, int64_t W, int64_t R, int64_t rot,
                      int64_t ng, int64_t ngR)
{
    if (kvc) {
        const float *ct = kvc + t * W;
        return tt_dot(q_abs, ct, R) + tt_dot(q_rot, ct + R, rot);
    }
    if (kvq) {
        const int8_t *ct = kvq + t * W;
        const float  *sc = kvqs + t * ng;
        return tt_dot_int8g(q_abs, ct, sc, R, TT_KV_GROUP)
             + tt_dot_int8g(q_rot, ct + R, sc + ngR, rot, TT_KV_GROUP);
    }
    const uint16_t *ct = kvc16 + t * W;
    return tt_dot_bf16(q_abs, ct, R) + tt_dot_bf16(q_rot, ct + R, rot);
}

static void kv_accum(float *o_c, float w,
                     const float *kvc, const uint16_t *kvc16,
                     const int8_t *kvq, const float *kvqs,
                     int64_t t, int64_t W, int64_t R, int64_t ng)
{
    if (kvc)      tt_axpy(o_c, w, kvc + t * W, R);
    else if (kvq) tt_axpy_int8g(o_c, w, kvq + t * W, kvqs + t * ng,
                                R, TT_KV_GROUP);
    else          tt_axpy_bf16(o_c, w, kvc16 + t * W, R);
}

static void attn_head(void *raw, int64_t head0, int64_t head1)
{
    const attn_job *j = (const attn_job *)raw;
    tt_state *s = j->s;
    const tt_config *c = &s->m->cfg;
    const int64_t H = c->num_attention_heads;
    const int64_t nope = c->qk_nope_head_dim, rot = c->qk_rope_head_dim;
    const int64_t qk = c->qk_head_dim, vd = c->v_head_dim;
    const int64_t T = j->T;
    const float *cosv = s->m->rope_cos + (T - 1) * (rot / 2);
    const float *sinv = s->m->rope_sin + (T - 1) * (rot / 2);

    for (int64_t hh = head0; hh < head1; hh++) {
        float *qh = s->q + hh * qk;
        rope_apply(qh + nope, cosv, sinv, rot);

        /* Uma faixa de scores por cabeca: sem isso as cabecas se
         * atropelariam ao rodar em paralelo. */
        float *sc = s->scores + hh * s->ctx;

        for (int64_t t = 0; t < T; t++) {
            const float *kt = s->k + (((j->li * s->ctx) + t) * H + hh) * qk;
            sc[t] = tt_dot(qh, kt, qk) * c->attn_scale;
        }
        tt_softmax(sc, T);

        float *out = s->attn + hh * vd;
        tt_zero(out, vd);
        for (int64_t t = 0; t < T; t++) {
            const float *vt = s->v + (((j->li * s->ctx) + t) * H + hh) * vd;
            tt_axpy(out, sc[t], vt, vd);
        }
    }
}

/* Atencao no espaco comprimido (docs/13, D11).
 *
 * A identidade que faz isso funcionar:
 *
 *   q_nope . k_nope[t]  =  q_nope^T (W_k c[t])  =  (W_k^T q_nope)^T c[t]
 *   sum_t w[t] v[t]     =  sum_t w[t] (W_v c[t]) =  W_v (sum_t w[t] c[t])
 *
 * Ou seja: projeta-se a QUERY uma vez (custo fixo, independe do contexto) em
 * vez de expandir CADA posicao do cache.  O laco sobre o contexto passa a ler
 * 576 floats por posicao em vez de 10.240.
 *
 * Nao e a formulacao do oraculo e nao da os mesmos bits -- ver docs/15.  O que
 * a autoriza e o teste de 32/32 contra `ref.json`, nao a algebra. */
static void attn_head_absorbed(void *raw, int64_t head0, int64_t head1)
{
    const attn_job *j = (const attn_job *)raw;
    tt_state *s = j->s;
    const tt_model  *m = s->m;
    const tt_config *c = &m->cfg;

    const int64_t nope = c->qk_nope_head_dim, rot = c->qk_rope_head_dim;
    const int64_t qk = c->qk_head_dim, vd = c->v_head_dim;
    const int64_t R = c->kv_lora_rank, W = R + rot;
    const int64_t T = j->T;

    const float *cosv = m->rope_cos + (T - 1) * (rot / 2);
    const float *sinv = m->rope_sin + (T - 1) * (rot / 2);
    const float    *kvc   = s->kvc   ? s->kvc   + (j->li * s->ctx) * W : NULL;
    const uint16_t *kvc16 = s->kvc16 ? s->kvc16 + (j->li * s->ctx) * W : NULL;
    const int8_t   *kvq   = s->kvq   ? s->kvq   + (j->li * s->ctx) * W : NULL;
    const float    *kvqs  = s->kvq
        ? s->kvq_scales + (j->li * s->ctx) * (W / TT_KV_GROUP) : NULL;
    const int64_t   ng    = W / TT_KV_GROUP;      /* escalas por posicao */
    const int64_t   ngR   = R / TT_KV_GROUP;      /* escalas da parte comprimida */

    for (int64_t hh = head0; hh < head1; hh++) {
        float *qh = s->q + hh * qk;
        rope_apply(qh + nope, cosv, sinv, rot);

        /* q_abs = W_k^T q_nope.
         *
         * A transposta NAO obriga a materializar `kv_b` em fp32: como W_k e
         * [nope][R] linha-maior, o produto transposto e uma acumulacao de
         * LINHAS, e linha e exatamente o que o container entrega barato.
         * Materializar custaria 863 MB lidos por token contra os 108 MB do
         * int4 -- o oposto do que a absorcao veio comprar. */
        const tt_tensor *kvb = j->L->kv_b_proj;
        const int64_t    r0  = tt_kvb_head_row(c, hh);
        float *wrow  = s->wrow + hh * R;
        float *q_abs = s->q_abs + hh * R;

        tt_zero(q_abs, R);
        for (int64_t r = 0; r < nope; r++) {
            tt_dequant_row(wrow, kvb, r0 + r);
            tt_axpy(q_abs, qh[r], wrow, R);
        }

        /* Softmax online: UM passe sobre o cache em vez de dois.
         *
         * A versao de dois passes calcula todos os scores, normaliza, e volta
         * para acumular -- lendo o cache inteiro duas vezes.  Aqui o maximo e
         * a soma sao mantidos correntes e o acumulador e reescalado quando
         * aparece um maximo novo:
         *
         *   m' = max(m, s)      acc' = acc * exp(m - m') + exp(s - m') * c[t]
         *
         * Algebricamente identico ao softmax com subtracao do maximo, e pela
         * mesma razao: o fator comum sai da soma.  O reescalonamento acontece
         * O(log T) vezes em media, contra T leituras economizadas. */
        float *o_c = s->o_c + hh * R;
        tt_zero(o_c, R);

        /* Dois passes ou um?  Medido: a 4k o cache de uma camada sao 9,4 MB em
         * f32 e cabem no L3 de 36 MB, entao o segundo passe custa quase nada e
         * o softmax online -- que paga um desvio por posicao e reescalona o
         * acumulador a cada maximo novo -- sai 9% MAIS LENTO (9,19 contra
         * 10,09 tok/s).  Acima do L3 a conta inverte: o segundo passe passa a
         * vir da DRAM.  A escolha e por tamanho, nao por gosto. */
        const int one_pass = (T * W * (kvc ? 4 : (kvq ? 1 : 2))) > TT_L3_BUDGET;

        if (one_pass) {
            float mx = -INFINITY, l = 0.0f;
            for (int64_t t = 0; t < T; t++) {
                const float sc = kv_score(q_abs, qh + nope, kvc, kvc16, kvq,
                                          kvqs, t, W, R, rot, ng, ngR)
                                 * c->attn_scale;

                /* m' = max(m, s);  acc' = acc*exp(m-m') + exp(s-m')*c[t].
                 * E o mesmo softmax com subtracao do maximo, so que o maximo
                 * e descoberto durante a soma em vez de antes dela. */
                if (sc > mx) {
                    const float corr = (mx == -INFINITY) ? 0.0f : expf(mx - sc);
                    tt_scale(o_c, corr, R);
                    l *= corr;
                    mx = sc;
                }
                const float e = expf(sc - mx);
                l += e;
                kv_accum(o_c, e, kvc, kvc16, kvq, kvqs, t, W, R, ng);
            }
            tt_scale(o_c, 1.0f / l, R);
        } else {
            float *sc = s->scores + hh * s->ctx;
            for (int64_t t = 0; t < T; t++)
                sc[t] = kv_score(q_abs, qh + nope, kvc, kvc16, kvq, kvqs,
                                 t, W, R, rot, ng, ngR) * c->attn_scale;
            tt_softmax(sc, T);

            for (int64_t t = 0; t < T; t++)
                kv_accum(o_c, sc[t], kvc, kvc16, kvq, kvqs, t, W, R, ng);
        }

        /* out = W_v o_c: uma projecao normal, [vd][R] . [R], linha a linha. */
        float *out = s->attn + hh * vd;
        for (int64_t d = 0; d < vd; d++) {
            tt_dequant_row(wrow, kvb, r0 + nope + d);
            out[d] = tt_dot(wrow, o_c, R);
        }
    }
}

static int attention_absorbed(tt_state *s, const tt_layer *L, int64_t li, int64_t pos)
{
    const tt_config *c = &s->m->cfg;
    const int64_t rot = c->qk_rope_head_dim;
    const int64_t R = c->kv_lora_rank, W = R + rot;

    if (tt_matmul(s->q_lora, L->q_a_proj, s->hn) != 0) return -1;
    tt_rmsnorm(s->q_lora, s->q_lora, (const float *)L->q_a_layernorm->data,
               c->q_lora_rank, c->rms_norm_eps);
    if (tt_matmul(s->q, L->q_b_proj, s->q_lora) != 0) return -1;

    if (tt_matmul(s->kv_a, L->kv_a_proj, s->hn) != 0) return -1;

    /* O cache guarda o KV comprimido JA NORMALIZADO -- e sobre ele que `kv_b`
     * opera, e normalizar uma vez por token e mais barato que a cada leitura. */
    tt_rmsnorm(s->ckv, s->kv_a, (const float *)L->kv_a_layernorm->data,
               R, c->rms_norm_eps);
    memcpy(s->krot, s->kv_a + R, (size_t)rot * sizeof(float));
    rope_apply(s->krot, s->m->rope_cos + pos * (rot / 2),
               s->m->rope_sin + pos * (rot / 2), rot);

    const int64_t off = ((li * s->ctx) + pos) * W;
    if (s->kvc) {
        memcpy(s->kvc + off,     s->ckv,  (size_t)R * sizeof(float));
        memcpy(s->kvc + off + R, s->krot, (size_t)rot * sizeof(float));
    } else if (s->kvq) {
        /* As duas partes sao quantizadas separadamente porque tem escalas
         * muito diferentes: a rotativa sai de seno e cosseno e vive em
         * [-1, 1], a comprimida nao. */
        const int64_t soff = off / TT_KV_GROUP;
        tt_store_int8g(s->kvq + off, s->kvq_scales + soff, s->ckv,
                       R, TT_KV_GROUP);
        tt_store_int8g(s->kvq + off + R, s->kvq_scales + soff + R / TT_KV_GROUP,
                       s->krot, rot, TT_KV_GROUP);
    } else {
        tt_store_bf16(s->kvc16 + off,     s->ckv,  R);
        tt_store_bf16(s->kvc16 + off + R, s->krot, rot);
    }

    attn_job job = { s, L, li, pos + 1 };
    tt_pool_run(tt_kernels_pool(), attn_head_absorbed, &job,
                c->num_attention_heads, 1);

    return tt_matmul(s->mlp_out, L->o_proj, s->attn);
}

static int attention(tt_state *s, const tt_layer *L, int64_t li, int64_t pos)
{
    const tt_config *c = &s->m->cfg;
    const int64_t H = c->num_attention_heads;
    const int64_t nope = c->qk_nope_head_dim, rot = c->qk_rope_head_dim;
    const int64_t qk = c->qk_head_dim, vd = c->v_head_dim;
    const int64_t kvb_stride = nope + vd;

    /* q = q_b_proj(rmsnorm(q_a_proj(h))) */
    if (tt_matmul(s->q_lora, L->q_a_proj, s->hn) != 0) return -1;
    tt_rmsnorm(s->q_lora, s->q_lora, (const float *)L->q_a_layernorm->data,
               c->q_lora_rank, c->rms_norm_eps);
    if (tt_matmul(s->q, L->q_b_proj, s->q_lora) != 0) return -1;

    /* compressed_kv = kv_a_proj(h) -> [kv_lora_rank | qk_rope_head_dim] */
    if (tt_matmul(s->kv_a, L->kv_a_proj, s->hn) != 0) return -1;
    tt_rmsnorm(s->ckv, s->kv_a, (const float *)L->kv_a_layernorm->data,
               c->kv_lora_rank, c->rms_norm_eps);
    if (tt_matmul(s->kvb, L->kv_b_proj, s->ckv) != 0) return -1;

    /* A parte rotativa da chave e UMA so, compartilhada por todas as cabecas
     * (MQA): rotaciona-se uma vez e replica-se. */
    memcpy(s->krot, s->kv_a + c->kv_lora_rank, (size_t)rot * sizeof(float));
    const float *cosv = s->m->rope_cos + pos * (rot / 2);
    const float *sinv = s->m->rope_sin + pos * (rot / 2);
    rope_apply(s->krot, cosv, sinv, rot);

    /* Escreve K e V desta posicao no cache, ja expandidos. */
    float *kbase = s->k + ((li * s->ctx) + pos) * H * qk;
    float *vbase = s->v + ((li * s->ctx) + pos) * H * vd;
    for (int64_t hh = 0; hh < H; hh++) {
        const float *src = s->kvb + hh * kvb_stride;
        float *kh = kbase + hh * qk;
        memcpy(kh, src, (size_t)nope * sizeof(float));
        memcpy(kh + nope, s->krot, (size_t)rot * sizeof(float));
        memcpy(vbase + hh * vd, src + nope, (size_t)vd * sizeof(float));
    }

    /* Query: rotaciona a cauda de cada cabeca e faz a atencao causal.  As
     * cabecas sao independentes -- cada uma le todo o KV da camada e escreve
     * so a sua fatia de `attn`, entao a particao por cabeca tem a mesma
     * propriedade da particao por linha: nao muda um bit. */
    attn_job job = { s, L, li, pos + 1 };
    tt_pool_run(tt_kernels_pool(), attn_head, &job, H, 1);

    return tt_matmul(s->mlp_out, L->o_proj, s->attn);   /* reusa mlp_out */
}

/* --------------------------------------------------------------- MLP */

static int mlp_dense(tt_state *s, const tt_tensor *gate, const tt_tensor *up,
                     const tt_tensor *down, const float *x, float *out)
{
    const int64_t inter = gate->rows;
    if (tt_matmul(s->gate_buf, gate, x) != 0) return -1;
    if (tt_matmul(s->up_buf,   up,   x) != 0) return -1;
    tt_silu_mul(s->gate_buf, s->gate_buf, s->up_buf, inter);
    (void)inter;
    return tt_matmul(out, down, s->gate_buf);
}

static float sigmoidf_(float v) { return 1.0f / (1.0f + expf(-v)); }

static int mlp_moe(tt_state *s, const tt_layer *L, float *out)
{
    const tt_config *c = &s->m->cfg;
    const int64_t E = c->n_routed_experts;
    const int     K = (int)c->num_experts_per_tok;

    /* Roteador em f32, sobre a hidden normalizada.  `sigmoid` e nao softmax:
     * e o `noaux_tc` do GLM/DeepSeek. */
    if (tt_matmul(s->router_logits, L->router, s->hn) != 0) return -1;

    const float *bias = (const float *)L->router_bias->data;
    for (int64_t e = 0; e < E; e++) {
        s->router_scores[e] = sigmoidf_(s->router_logits[e]);
        /* O vies de correcao entra APENAS na escolha; o peso usado depois e o
         * score puro.  Trocar isso muda quais experts rodam E o peso deles. */
        s->router_choice[e] = s->router_scores[e] + bias[e];
    }

    tt_topk(s->router_choice, E, K, s->topk_idx, s->topk_w);

    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        s->topk_w[i] = s->router_scores[s->topk_idx[i]];
        sum += s->topk_w[i];
    }
    if (c->norm_topk_prob) {
        const float denom = sum + 1e-20f;
        for (int i = 0; i < K; i++) s->topk_w[i] /= denom;
    }
    for (int i = 0; i < K; i++) s->topk_w[i] *= c->routed_scaling_factor;

    /* Ordem de acumulacao fixa: por indice de expert crescente, que e a ordem
     * em que o `index_add_` da referencia soma as contribuicoes.  Sem isto o
     * resultado dependeria da ordem do top-k, que nem e definida. */
    for (int i = 1; i < K; i++) {
        const int32_t ki = s->topk_idx[i];
        const float   kw = s->topk_w[i];
        int j = i - 1;
        while (j >= 0 && s->topk_idx[j] > ki) {
            s->topk_idx[j + 1] = s->topk_idx[j];
            s->topk_w[j + 1]   = s->topk_w[j];
            j--;
        }
        s->topk_idx[j + 1] = ki;
        s->topk_w[j + 1]   = kw;
    }

    /* Os K experts escolhidos mais o compartilhado sao independentes entre si:
     * mesma entrada, matrizes diferentes, saidas diferentes. Fundir os seus
     * matmuls em dois dispatches -- um para gate+up, outro para down -- troca
     * 15 barreiras por 2 sem mudar uma soma sequer. */
    const int64_t inter = L->e_gate[0]->rows;
    const int64_t hid = c->hidden_size;
    const int n_mlp = K + 1;                 /* +1 = expert compartilhado */

    const tt_tensor *mats[TT_BATCH_MAX];
    float           *outs[TT_BATCH_MAX];

    for (int i = 0; i < K; i++) {
        const int32_t e = s->topk_idx[i];
        if (e < 0 || e >= E) return tt_fail("roteador escolheu expert %d", e);
        mats[2 * i]     = L->e_gate[e];
        mats[2 * i + 1] = L->e_up[e];
        outs[2 * i]     = s->moe_gu + (2 * i)     * inter;
        outs[2 * i + 1] = s->moe_gu + (2 * i + 1) * inter;
    }
    mats[2 * K]     = L->sh_gate;
    mats[2 * K + 1] = L->sh_up;
    outs[2 * K]     = s->moe_gu + (2 * K)     * inter;
    outs[2 * K + 1] = s->moe_gu + (2 * K + 1) * inter;

    if (tt_matmul_batch(outs, mats, 2 * n_mlp, s->hn) != 0) return -1;

    for (int i = 0; i < n_mlp; i++) {
        float *g = s->moe_gu + (2 * i) * inter;
        tt_silu_mul(g, g, s->moe_gu + (2 * i + 1) * inter, inter);
        mats[i] = (i < K) ? L->e_down[s->topk_idx[i]] : L->sh_down;
        outs[i] = s->moe_down + i * hid;
    }

    /* `down` de cada expert le uma entrada diferente, entao nao cabe no mesmo
     * `tt_matmul_batch` -- que assume entrada comum. Sao K+1 dispatches, e
     * ainda assim 15 viraram 6. */
    for (int i = 0; i < n_mlp; i++)
        if (tt_matmul(outs[i], mats[i], s->moe_gu + (2 * i) * inter) != 0)
            return -1;

    /* Combinacao em ordem de expert crescente, como o `index_add_` da
     * referencia; o compartilhado entra por ultimo, sem peso (docs/04). */
    tt_zero(out, hid);
    for (int i = 0; i < K; i++) {
        const float w = s->topk_w[i];
        const float *e_out = s->moe_down + i * hid;
        for (int64_t d = 0; d < hid; d++) out[d] += e_out[d] * w;
    }
    tt_add(out, s->moe_down + K * hid, hid);
    return 0;
}

/* ------------------------------------------------------------ forward */

int tt_forward(tt_state *s, int32_t token, float **logits_out)
{
    const tt_model  *m = s->m;
    const tt_config *c = &m->cfg;

    if (token < 0 || token >= c->vocab_size)
        return tt_fail("token %d fora de [0, %lld)", token, (long long)c->vocab_size);
    if (s->len >= s->ctx)
        return tt_fail("contexto cheio (%lld tokens). Aumente --ctx ou "
                       "descarte o inicio", (long long)s->ctx);
    if (s->len >= m->rope_positions)
        return tt_fail("tabela de RoPE cobre %lld posicoes",
                       (long long)m->rope_positions);

    const double t0 = tt_now();
    const int64_t pos = s->len;

    /* Embedding: uma linha, lida do arquivo mapeado.  E o unico peso que nao
     * precisa de residencia -- 605 MB que ficam file-backed (docs/04). */
    if (tt_dequant_row(s->h, m->embed, token) != 0) return -1;
    if (s->trace) tt_copy(s->trace, s->h, c->hidden_size);

    for (int64_t li = 0; li < c->num_hidden_layers; li++) {
        const tt_layer *L = &m->layers[li];

        tt_rmsnorm(s->hn, s->h, (const float *)L->input_layernorm->data,
                   c->hidden_size, c->rms_norm_eps);
        if (m->attn == TT_ATTN_ABSORBED) {
            if (attention_absorbed(s, L, li, pos) != 0) return -1;
        } else {
            if (attention(s, L, li, pos) != 0) return -1;
        }
        tt_add(s->h, s->mlp_out, c->hidden_size);

        tt_rmsnorm(s->hn, s->h, (const float *)L->post_attention_layernorm->data,
                   c->hidden_size, c->rms_norm_eps);
        if (L->is_moe) {
            if (mlp_moe(s, L, s->mlp_out) != 0) return -1;
        } else {
            if (mlp_dense(s, L->gate_proj, L->up_proj, L->down_proj,
                          s->hn, s->mlp_out) != 0) return -1;
        }
        tt_add(s->h, s->mlp_out, c->hidden_size);

        if (s->trace)
            tt_copy(s->trace + (li + 1) * c->hidden_size, s->h, c->hidden_size);
    }

    tt_rmsnorm(s->hn, s->h, (const float *)m->norm->data,
               c->hidden_size, c->rms_norm_eps);
    if (tt_matmul(s->logits, m->lm_head, s->hn) != 0) return -1;

    s->tokens[pos] = token;
    s->len = pos + 1;
    s->tokens_processed++;
    s->seconds_in_forward += tt_now() - t0;

    if (logits_out) *logits_out = s->logits;
    return 0;
}
