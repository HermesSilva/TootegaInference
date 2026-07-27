/* oracle -- the M1 reference: a forward that is INDEPENDENT of the engine's,
 * over the SAME container.
 *
 * Why not run the original checkpoint instead: the model in fp32 is ~125 GB and
 * in bf16 ~62 GB on a 64 GB machine.  And, more importantly, comparing the
 * engine against the BF16 checkpoint would measure QUANTISATION, not the engine
 * -- the 11.7% output error of docs/04 would show up as "the engine is wrong".
 *
 * So the oracle reads the same int4 weights the engine reads, dequantises on
 * demand (one row at a time, never the whole model in memory) and redoes the
 * arithmetic following `modeling_glm4_moe_lite.py` operation by operation.  What
 * it tests is exactly what M1 promises: that the forward in C reproduces the
 * mathematics, given the same weights.
 *
 * INDEPENDENCE IS THE WHOLE POINT, and it is what every line here is arranged
 * around.  This file shares NOTHING with `src/forward.c`, `src/kernels.c`,
 * `src/model.c` or `src/container.c`: it has its own container reader, its own
 * dequantiser, its own matmul, its own attention and its own routing.  Two
 * implementations agreeing is evidence; one implementation agreeing with itself
 * is a tautology.  What it does borrow is infrastructure that is not the thing
 * under test and that fails loudly rather than silently: the memory mapping
 * (`compat`), the JSON parser (`json`, which has its own selftest), the thread
 * pool (`pool`) and, only for `--prompt`, the tokeniser -- whose own reference
 * is `tokfixture.json`.
 *
 *   Tests\oracle.exe models\m.tootega --ids-file ids32.txt --out ref32.json
 *   Tests\oracle.exe models\m.tootega --ids-file ids32.txt --compare logits.bin
 *   Tests\oracle.exe models\m.tootega --prompt "..." --tokenizer tok.json --generate 40
 *   Tests\oracle.exe --check-ref ref32.json --logits logits.bin
 *
 * The last mode has no model and no forward: it compares a logits dump the
 * engine produced against a reference that already exists.  It exists because it
 * runs in milliseconds -- once the reference exists, every kernel change (SIMD,
 * threads) has to be verified, and recomputing a 47-layer oracle each time is
 * exactly what stops one from verifying.
 */
#include "common.h"
#include "compat.h"
#include "json.h"
#include "pool.h"
#include "tokenizer.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------- container */

/* Own reader.  `src/container.c` reads the same file, and that is the reason
 * not to call it: if the index were being misread, both would be misread the
 * same way and the comparison would agree on the wrong thing. */

#define OC_HEADER 128
#define OC_MAXDIM 4

typedef enum { OK_INT4 = 0, OK_INT8, OK_F32, OK_BF16 } okind;

typedef struct {
    const char *name;
    size_t      name_len;
    okind       kind;
    int64_t     rows, cols;
    const void *data;
    const float *scales;
    int         group;
} otensor;

typedef struct {
    tt_map    map;
    tt_arena  arena;
    otensor  *t;
    size_t    n;
    const tt_json *config;
} ocontainer;

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const unsigned char *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static int oc_open(ocontainer *c, const char *path)
{
    memset(c, 0, sizeof(*c));
    if (tt_map_open(&c->map, path) != 0) return -1;
    if (c->map.size < OC_HEADER) return tt_fail("%s: arquivo curto demais", path);

    const unsigned char *h = (const unsigned char *)c->map.base;
    if (memcmp(h, "TOOTEGA\x01", 8) != 0) return tt_fail("%s: magic invalido", path);
    if (rd32(h + 8) != 1) return tt_fail("%s: formato v%u desconhecido", path, rd32(h + 8));

    const uint64_t index_off = rd64(h + 24), index_len = rd64(h + 32);
    if (index_off + index_len > c->map.size)
        return tt_fail("%s: indice fora do arquivo", path);

    const char *idx = (const char *)c->map.base + index_off;
    if (tt_arena_init(&c->arena, tt_json_bound(idx, (size_t)index_len) + (64u << 20),
                      "oracle index") != 0)
        return -1;

    const tt_json *root = tt_json_parse(idx, (size_t)index_len, &c->arena);
    if (!root) return -1;

    const tt_json *meta = tt_json_get(root, "meta");
    c->config = tt_json_get(meta, "config");
    if (!c->config) return tt_fail("%s: meta.config ausente", path);

    const tt_json *tensors = tt_json_get(root, "tensors");
    if (!tensors) return tt_fail("%s: tensors ausente", path);

    c->n = tensors->count;
    c->t = (otensor *)tt_arena_alloc(&c->arena, c->n * sizeof(otensor), 16);
    if (!c->t) return -1;

    size_t i = 0;
    for (const tt_json *e = tensors->child; e; e = e->next, i++) {
        otensor *t = &c->t[i];
        memset(t, 0, sizeof(*t));
        t->name = e->key;
        t->name_len = e->key_len;

        size_t kl = 0;
        const char *k = tt_json_str(tt_json_get(e, "kind"), &kl);
        if      (kl == 4 && !memcmp(k, "int4", 4)) t->kind = OK_INT4;
        else if (kl == 4 && !memcmp(k, "int8", 4)) t->kind = OK_INT8;
        else if (kl == 3 && !memcmp(k, "f32", 3))  t->kind = OK_F32;
        else if (kl == 4 && !memcmp(k, "bf16", 4)) t->kind = OK_BF16;
        else return tt_fail("%.*s: kind desconhecido", (int)t->name_len, t->name);

        const tt_json *shape = tt_json_get(e, "shape");
        int64_t dims[OC_MAXDIM] = {0};
        int nd = 0;
        for (const tt_json *d = shape ? shape->child : NULL; d && nd < OC_MAXDIM; d = d->next)
            dims[nd++] = tt_json_int(d, 0);
        if (nd < 1) return tt_fail("%.*s: shape vazio", (int)t->name_len, t->name);

        t->cols = dims[nd - 1];
        t->rows = 1;
        for (int d = 0; d < nd - 1; d++) t->rows *= dims[d];

        const tt_json *span = tt_json_get(e, "data");
        const uint64_t off = (uint64_t)tt_json_int(tt_json_at(span, 0), -1);
        const uint64_t len = (uint64_t)tt_json_int(tt_json_at(span, 1), 0);
        if (off + len > c->map.size)
            return tt_fail("%.*s: blob fora do arquivo", (int)t->name_len, t->name);
        t->data = (const unsigned char *)c->map.base + off;

        const tt_json *sp = tt_json_get(e, "scales");
        if (sp && sp->type == TT_JSON_ARRAY) {
            const uint64_t so = (uint64_t)tt_json_int(tt_json_at(sp, 0), -1);
            t->scales = (const float *)((const unsigned char *)c->map.base + so);
            t->group = (int)tt_json_int(tt_json_get(e, "group"), 0);
            if (t->group <= 0 || t->cols % t->group)
                return tt_fail("%.*s: group %d invalido para %lld colunas",
                               (int)t->name_len, t->name, t->group, (long long)t->cols);
        }
    }
    return 0;
}

static const otensor *oc_find(const ocontainer *c, const char *name)
{
    const size_t n = strlen(name);
    for (size_t i = 0; i < c->n; i++)
        if (c->t[i].name_len == n && memcmp(c->t[i].name, name, n) == 0)
            return &c->t[i];
    return NULL;
}

static const otensor *oc_findf(const ocontainer *c, const char *fmt, ...)
{
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);
    const otensor *t = oc_find(c, name);
    if (!t) { fprintf(stderr, "tensor ausente: %s\n", name); exit(1); }
    return t;
}

/* Uma linha em float32.  A ordem dentro do grupo -- metade baixa nos nibbles
 * baixos -- e a de docs/14; escrita aqui de novo, de proposito. */
static void oc_row(const otensor *t, int64_t r, float *dst)
{
    const int64_t I = t->cols;

    switch (t->kind) {
    case OK_F32:
        memcpy(dst, (const float *)t->data + r * I, (size_t)I * sizeof(float));
        return;

    case OK_BF16: {
        const uint16_t *p = (const uint16_t *)t->data + r * I;
        for (int64_t i = 0; i < I; i++) {
            union { uint32_t u; float f; } c;
            c.u = (uint32_t)p[i] << 16;
            dst[i] = c.f;
        }
        return;
    }

    case OK_INT8: {
        const int8_t *q = (const int8_t *)t->data + r * I;
        const float *s = t->scales + r * (I / t->group);
        for (int64_t i = 0; i < I; i++)
            dst[i] = (float)q[i] * s[i / t->group];
        return;
    }

    default: {
        const int half = t->group / 2;
        const uint8_t *p = (const uint8_t *)t->data + r * (I / 2);
        const float *s = t->scales + r * (I / t->group);
        const int64_t ng = I / t->group;
        for (int64_t g = 0; g < ng; g++) {
            const uint8_t *b = p + g * half;
            for (int j = 0; j < half; j++) {
                dst[g * t->group + j]        = (float)((int)(b[j] & 0x0F) - 8) * s[g];
                dst[g * t->group + half + j] = (float)((int)(b[j] >> 4)   - 8) * s[g];
            }
        }
        return;
    }
    }
}

/* --------------------------------------------------------------- pecas */

/* y[t][O] = x[t][I] @ w[O][I]^T -- a convencao [O, I] do container e a de
 * nn.Linear.
 *
 * Uma linha de peso e desquantizada uma vez e consumida pelas T posicoes: e o
 * que mantem a conta viavel sem materializar o tensor inteiro em fp32.
 * Particionado por linha de SAIDA, entao cada thread faz a sua reducao inteira,
 * na mesma ordem, com o proprio acumulador -- o resultado nao muda com
 * `--threads` (docs/15). */
typedef struct {
    float         *y;
    const otensor *w;
    const float   *x;
    int64_t        T, I, O;
} linjob;

static void lin_rows(void *arg, int64_t r0, int64_t r1)
{
    linjob *j = (linjob *)arg;

    /* Um buffer por bloco: a fatia [r0,r1) e exclusiva desta thread, e o buffer
     * so vive dentro dela. */
    float *buf = (float *)malloc((size_t)j->I * sizeof(float));
    if (!buf) { fprintf(stderr, "sem memoria no oraculo\n"); exit(1); }

    for (int64_t r = r0; r < r1; r++) {
        oc_row(j->w, r, buf);
        for (int64_t t = 0; t < j->T; t++) {
            const float *x = j->x + t * j->I;
            float acc = 0.0f;
            for (int64_t i = 0; i < j->I; i++) acc = acc + buf[i] * x[i];
            j->y[t * j->O + r] = acc;
        }
    }
    free(buf);
}

static tt_pool *g_pool = NULL;

static void olinear(float *y, const otensor *w, const float *x, int64_t T)
{
    linjob j = { y, w, x, T, w->cols, w->rows };
    tt_pool_run(g_pool, lin_rows, &j, w->rows, 8);
}

/* Glm4MoeLiteRMSNorm: variancia em float32, depois multiplica pelo peso. */
static void ormsnorm(float *y, const float *x, const float *w, int64_t n, float eps)
{
    float ss = 0.0f;
    for (int64_t i = 0; i < n; i++) ss = ss + x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (int64_t i = 0; i < n; i++) y[i] = (x[i] * inv) * w[i];
}

static float osilu(float x) { return x / (1.0f + expf(-x)); }

/* ----------------------------------------------------------- oraculo */

typedef struct {
    int64_t hidden, heads, nope, rot, qk, vdim;
    int64_t kv_rank, q_rank, layers, first_dense;
    int64_t top_k, n_exp, vocab;
    float   eps, scaling, theta;
    int     norm_topk;
} ocfg;

static void read_cfg(const ocontainer *c, ocfg *g)
{
    const tt_json *j = c->config;
    g->hidden      = tt_json_int(tt_json_get(j, "hidden_size"), 0);
    g->heads       = tt_json_int(tt_json_get(j, "num_attention_heads"), 0);
    g->nope        = tt_json_int(tt_json_get(j, "qk_nope_head_dim"), 0);
    g->rot         = tt_json_int(tt_json_get(j, "qk_rope_head_dim"), 0);
    g->vdim        = tt_json_int(tt_json_get(j, "v_head_dim"), 0);
    g->kv_rank     = tt_json_int(tt_json_get(j, "kv_lora_rank"), 0);
    g->q_rank      = tt_json_int(tt_json_get(j, "q_lora_rank"), 0);
    g->layers      = tt_json_int(tt_json_get(j, "num_hidden_layers"), 0);
    g->first_dense = tt_json_int(tt_json_get(j, "first_k_dense_replace"), 0);
    g->top_k       = tt_json_int(tt_json_get(j, "num_experts_per_tok"), 0);
    g->n_exp       = tt_json_int(tt_json_get(j, "n_routed_experts"), 0);
    g->vocab       = tt_json_int(tt_json_get(j, "vocab_size"), 0);
    g->eps         = (float)tt_json_real(tt_json_get(j, "rms_norm_eps"), 1e-5);
    g->scaling     = (float)tt_json_real(tt_json_get(j, "routed_scaling_factor"), 1.0);
    g->theta       = (float)tt_json_real(tt_json_get(j, "rope_theta"), 10000.0);
    g->norm_topk   = tt_json_bool(tt_json_get(j, "norm_topk_prob"), 1);
    g->qk          = g->nope + g->rot;
}

/* `Glm4MoeLiteConfig.rope_interleave` e True por default e o `config.json`
 * deste modelo nao sobrescreve, entao a atencao usa
 * `apply_rotary_pos_emb_interleave` -- que leva o elemento 2i para a posicao i
 * e o 2i+1 para i + d/2 ANTES do rotate_half.
 *
 * Omitir este passo nao estoura nada: na posicao 0 as duas variantes coincidem
 * (cos=1, sin=0) e o erro cresce com a posicao.  Foi assim que a variante
 * errada sobreviveu a uma comparacao inteira. */
static void deinterleave(float *dst, const float *src, int64_t d)
{
    const int64_t half = d / 2;
    for (int64_t i = 0; i < half; i++) {
        dst[i]        = src[2 * i];
        dst[half + i] = src[2 * i + 1];
    }
}

static void rope_apply(float *v, const float *cos_t, const float *sin_t, int64_t d)
{
    const int64_t half = d / 2;
    float tmp[512];
    memcpy(tmp, v, (size_t)d * sizeof(float));
    for (int64_t i = 0; i < half; i++) {
        v[i]        = tmp[i] * cos_t[i] - tmp[half + i] * sin_t[i];
        v[half + i] = tmp[half + i] * cos_t[half + i] + tmp[i] * sin_t[half + i];
    }
}

typedef struct {
    float  *logits;      /* [T][vocab] */
    float  *hidden;      /* [T][hidden], depois da norma final */
    double *layer_norm;  /* [layers] -- |h| da ultima posicao */
} oresult;

static void forward(const ocontainer *c, const ocfg *g, const int32_t *ids,
                    int64_t T, oresult *out, int verbose)
{
    const int64_t H = g->heads, hid = g->hidden, qk = g->qk, vd = g->vdim;
    const float scale = 1.0f / sqrtf((float)qk);

    float *h   = (float *)calloc((size_t)T * hid, sizeof(float));
    float *hn  = (float *)calloc((size_t)T * hid, sizeof(float));
    float *qa  = (float *)calloc((size_t)T * g->q_rank, sizeof(float));
    float *q   = (float *)calloc((size_t)T * H * qk, sizeof(float));
    float *kva = (float *)calloc((size_t)T * (g->kv_rank + g->rot), sizeof(float));
    float *ckv = (float *)calloc((size_t)T * g->kv_rank, sizeof(float));
    float *kvb = (float *)calloc((size_t)T * H * (g->nope + vd), sizeof(float));
    float *krot= (float *)calloc((size_t)T * g->rot, sizeof(float));
    float *att = (float *)calloc((size_t)T * H * vd, sizeof(float));
    float *acc = (float *)calloc((size_t)T * hid, sizeof(float));
    float *tmp = (float *)calloc((size_t)T * hid, sizeof(float));
    float *cosb= (float *)calloc((size_t)T * g->rot, sizeof(float));
    float *sinb= (float *)calloc((size_t)T * g->rot, sizeof(float));
    float *scores = (float *)calloc((size_t)T, sizeof(float));
    float *vecw = (float *)calloc((size_t)(hid > 32768 ? hid : 32768), sizeof(float));

    if (!h || !hn || !qa || !q || !kva || !ckv || !kvb || !krot || !att ||
        !acc || !tmp || !cosb || !sinb || !scores || !vecw) {
        fprintf(stderr, "sem memoria para as ativacoes\n"); exit(1);
    }

    /* embeddings: uma linha por token, sem materializar a tabela. */
    const otensor *emb = oc_findf(c, "model.embed_tokens.weight");
    for (int64_t t = 0; t < T; t++) oc_row(emb, ids[t], h + t * hid);

    /* tabelas de RoPE.  `inv` em double para que o expoente nao carregue erro
     * de arredondamento para dentro do cosseno. */
    for (int64_t t = 0; t < T; t++) {
        for (int64_t i = 0; i < g->rot / 2; i++) {
            const double inv = 1.0 / pow((double)g->theta,
                                         (double)(2 * i) / (double)g->rot);
            const float f = (float)((double)t * inv);
            cosb[t * g->rot + i]              = cosf(f);
            cosb[t * g->rot + g->rot / 2 + i] = cosf(f);
            sinb[t * g->rot + i]              = sinf(f);
            sinb[t * g->rot + g->rot / 2 + i] = sinf(f);
        }
    }

    for (int64_t li = 0; li < g->layers; li++) {
        const double t_layer = tt_now();

        /* ------------------------------------------------ atencao MLA */

        oc_row(oc_findf(c, "model.layers.%lld.input_layernorm.weight", (long long)li),
               0, vecw);
        for (int64_t t = 0; t < T; t++)
            ormsnorm(hn + t * hid, h + t * hid, vecw, hid, g->eps);

        olinear(qa, oc_findf(c, "model.layers.%lld.self_attn.q_a_proj.weight",
                             (long long)li), hn, T);
        oc_row(oc_findf(c, "model.layers.%lld.self_attn.q_a_layernorm.weight",
                        (long long)li), 0, vecw);
        for (int64_t t = 0; t < T; t++)
            ormsnorm(qa + t * g->q_rank, qa + t * g->q_rank, vecw, g->q_rank, g->eps);

        olinear(q, oc_findf(c, "model.layers.%lld.self_attn.q_b_proj.weight",
                            (long long)li), qa, T);

        olinear(kva, oc_findf(c, "model.layers.%lld.self_attn.kv_a_proj_with_mqa.weight",
                              (long long)li), hn, T);

        const int64_t kvw = g->kv_rank + g->rot;
        for (int64_t t = 0; t < T; t++) {
            memcpy(ckv + t * g->kv_rank, kva + t * kvw,
                   (size_t)g->kv_rank * sizeof(float));
            /* deinterleave ja aqui: o k rotativo e compartilhado por todas as
             * cabecas, entao a rotacao acontece uma vez por posicao. */
            deinterleave(krot + t * g->rot, kva + t * kvw + g->kv_rank, g->rot);
            rope_apply(krot + t * g->rot, cosb + t * g->rot, sinb + t * g->rot, g->rot);
        }

        oc_row(oc_findf(c, "model.layers.%lld.self_attn.kv_a_layernorm.weight",
                        (long long)li), 0, vecw);
        for (int64_t t = 0; t < T; t++)
            ormsnorm(ckv + t * g->kv_rank, ckv + t * g->kv_rank, vecw,
                     g->kv_rank, g->eps);

        olinear(kvb, oc_findf(c, "model.layers.%lld.self_attn.kv_b_proj.weight",
                              (long long)li), ckv, T);

        /* q rotativo, por cabeca. */
        for (int64_t t = 0; t < T; t++)
            for (int64_t hh = 0; hh < H; hh++) {
                float *qh = q + (t * H + hh) * qk;
                float rot[512];
                deinterleave(rot, qh + g->nope, g->rot);
                rope_apply(rot, cosb + t * g->rot, sinb + t * g->rot, g->rot);
                memcpy(qh + g->nope, rot, (size_t)g->rot * sizeof(float));
            }

        /* scores + softmax + soma ponderada de v, causal, uma cabeca por vez. */
        const int64_t kvstride = g->nope + vd;
        for (int64_t hh = 0; hh < H; hh++) {
            for (int64_t t = 0; t < T; t++) {
                const float *qh = q + (t * H + hh) * qk;
                float max = -INFINITY;

                for (int64_t s = 0; s <= t; s++) {
                    const float *knope = kvb + (s * H + hh) * kvstride;
                    float acc2 = 0.0f;
                    for (int64_t i = 0; i < g->nope; i++)
                        acc2 = acc2 + qh[i] * knope[i];
                    const float *kr = krot + s * g->rot;
                    for (int64_t i = 0; i < g->rot; i++)
                        acc2 = acc2 + qh[g->nope + i] * kr[i];
                    acc2 *= scale;
                    scores[s] = acc2;
                    if (acc2 > max) max = acc2;
                }

                float sum = 0.0f;
                for (int64_t s = 0; s <= t; s++) {
                    scores[s] = expf(scores[s] - max);
                    sum += scores[s];
                }

                float *o = att + (t * H + hh) * vd;
                memset(o, 0, (size_t)vd * sizeof(float));
                for (int64_t s = 0; s <= t; s++) {
                    /* Divisao, e nao multiplicacao pelo reciproco: e a conta que
                     * a referencia faz, e aqui o papel e ser a referencia. */
                    const float wgt = scores[s] / sum;
                    const float *vv = kvb + (s * H + hh) * kvstride + g->nope;
                    for (int64_t i = 0; i < vd; i++) o[i] += wgt * vv[i];
                }
            }
        }

        olinear(acc, oc_findf(c, "model.layers.%lld.self_attn.o_proj.weight",
                              (long long)li), att, T);
        for (int64_t i = 0; i < T * hid; i++) h[i] += acc[i];

        /* ------------------------------------------------------- MLP */

        oc_row(oc_findf(c, "model.layers.%lld.post_attention_layernorm.weight",
                        (long long)li), 0, vecw);
        for (int64_t t = 0; t < T; t++)
            ormsnorm(hn + t * hid, h + t * hid, vecw, hid, g->eps);

        if (li < g->first_dense) {
            const otensor *gate = oc_findf(c, "model.layers.%lld.mlp.gate_proj.weight",
                                           (long long)li);
            const otensor *up   = oc_findf(c, "model.layers.%lld.mlp.up_proj.weight",
                                           (long long)li);
            const otensor *down = oc_findf(c, "model.layers.%lld.mlp.down_proj.weight",
                                           (long long)li);
            const int64_t inter = gate->rows;
            float *a = (float *)malloc((size_t)T * inter * sizeof(float));
            float *b = (float *)malloc((size_t)T * inter * sizeof(float));
            olinear(a, gate, hn, T);
            olinear(b, up, hn, T);
            for (int64_t i = 0; i < T * inter; i++) a[i] = osilu(a[i]) * b[i];
            olinear(acc, down, a, T);
            for (int64_t i = 0; i < T * hid; i++) h[i] += acc[i];
            free(a); free(b);
        } else {
            const otensor *router = oc_findf(c, "model.layers.%lld.mlp.gate.weight",
                                             (long long)li);
            const otensor *bias = oc_findf(c,
                "model.layers.%lld.mlp.gate.e_score_correction_bias", (long long)li);
            const int64_t E = router->rows;

            float *rl = (float *)malloc((size_t)T * E * sizeof(float));
            olinear(rl, router, hn, T);
            oc_row(bias, 0, vecw);

            int32_t *pick = (int32_t *)malloc((size_t)T * g->top_k * sizeof(int32_t));
            float   *pw   = (float *)malloc((size_t)T * g->top_k * sizeof(float));

            for (int64_t t = 0; t < T; t++) {
                float *probs = rl + t * E;
                for (int64_t e = 0; e < E; e++)
                    probs[e] = 1.0f / (1.0f + expf(-probs[e]));

                /* top-k por `probs + bias`, mas o PESO e `probs` puro -- e a
                 * regra do noaux_tc.  Empate fica com o indice menor, como o
                 * argsort estavel da referencia. */
                char taken[512] = {0};
                float sum = 0.0f;
                for (int64_t k = 0; k < g->top_k; k++) {
                    int64_t best = -1;
                    float bestv = 0.0f;
                    for (int64_t e = 0; e < E; e++) {
                        if (taken[e]) continue;
                        const float v = probs[e] + vecw[e];
                        if (best < 0 || v > bestv) { best = e; bestv = v; }
                    }
                    taken[best] = 1;
                    pick[t * g->top_k + k] = (int32_t)best;
                    pw[t * g->top_k + k] = probs[best];
                    sum += probs[best];
                }
                for (int64_t k = 0; k < g->top_k; k++) {
                    if (g->norm_topk) pw[t * g->top_k + k] /= (sum + 1e-20f);
                    pw[t * g->top_k + k] *= g->scaling;
                }
            }

            memset(acc, 0, (size_t)T * hid * sizeof(float));

            /* Por expert crescente, como o `index_add_` da referencia: a ordem
             * da soma muda o ultimo bit e a comparacao aqui e de bits. */
            int64_t *rows_of = (int64_t *)malloc((size_t)T * sizeof(int64_t));
            float   *w_of    = (float *)malloc((size_t)T * sizeof(float));
            float   *xin = NULL, *a = NULL, *b = NULL, *yo = NULL;
            int64_t inter = 0;

            for (int64_t e = 0; e < E; e++) {
                int64_t n = 0;
                for (int64_t t = 0; t < T; t++)
                    for (int64_t k = 0; k < g->top_k; k++)
                        if (pick[t * g->top_k + k] == (int32_t)e) {
                            rows_of[n] = t; w_of[n] = pw[t * g->top_k + k]; n++;
                        }
                if (!n) continue;

                const otensor *gp = oc_findf(c,
                    "model.layers.%lld.mlp.experts.%lld.gate_proj.weight",
                    (long long)li, (long long)e);
                const otensor *up = oc_findf(c,
                    "model.layers.%lld.mlp.experts.%lld.up_proj.weight",
                    (long long)li, (long long)e);
                const otensor *dn = oc_findf(c,
                    "model.layers.%lld.mlp.experts.%lld.down_proj.weight",
                    (long long)li, (long long)e);

                if (!inter) {
                    inter = gp->rows;
                    xin = (float *)malloc((size_t)T * hid * sizeof(float));
                    a   = (float *)malloc((size_t)T * inter * sizeof(float));
                    b   = (float *)malloc((size_t)T * inter * sizeof(float));
                    yo  = (float *)malloc((size_t)T * hid * sizeof(float));
                }

                for (int64_t i = 0; i < n; i++)
                    memcpy(xin + i * hid, hn + rows_of[i] * hid,
                           (size_t)hid * sizeof(float));

                olinear(a, gp, xin, n);
                olinear(b, up, xin, n);
                for (int64_t i = 0; i < n * inter; i++) a[i] = osilu(a[i]) * b[i];
                olinear(yo, dn, a, n);

                for (int64_t i = 0; i < n; i++)
                    for (int64_t j = 0; j < hid; j++)
                        acc[rows_of[i] * hid + j] += yo[i * hid + j] * w_of[i];
            }

            /* shared expert: roda para todas as posicoes, sem peso. */
            {
                const otensor *gp = oc_findf(c,
                    "model.layers.%lld.mlp.shared_experts.gate_proj.weight", (long long)li);
                const otensor *up = oc_findf(c,
                    "model.layers.%lld.mlp.shared_experts.up_proj.weight", (long long)li);
                const otensor *dn = oc_findf(c,
                    "model.layers.%lld.mlp.shared_experts.down_proj.weight", (long long)li);
                const int64_t si = gp->rows;
                float *sa = (float *)malloc((size_t)T * si * sizeof(float));
                float *sb = (float *)malloc((size_t)T * si * sizeof(float));
                olinear(sa, gp, hn, T);
                olinear(sb, up, hn, T);
                for (int64_t i = 0; i < T * si; i++) sa[i] = osilu(sa[i]) * sb[i];
                olinear(tmp, dn, sa, T);
                for (int64_t i = 0; i < T * hid; i++) h[i] += acc[i] + tmp[i];
                free(sa); free(sb);
            }

            free(rl); free(pick); free(pw);
            free(rows_of); free(w_of);
            free(xin); free(a); free(b); free(yo);
        }

        double nrm = 0.0;
        for (int64_t j = 0; j < hid; j++) {
            const double v = h[(T - 1) * hid + j];
            nrm += v * v;
        }
        out->layer_norm[li] = sqrt(nrm);
        if (verbose)
            printf("  camada %2lld  |h| = %.4f   %.1f s\n", (long long)li,
                   out->layer_norm[li], tt_now() - t_layer);
        fflush(stdout);
    }

    oc_row(oc_findf(c, "model.norm.weight"), 0, vecw);
    for (int64_t t = 0; t < T; t++)
        ormsnorm(out->hidden + t * hid, h + t * hid, vecw, hid, g->eps);

    olinear(out->logits, oc_findf(c, "lm_head.weight"), out->hidden, T);

    free(h); free(hn); free(qa); free(q); free(kva); free(ckv); free(kvb);
    free(krot); free(att); free(acc); free(tmp); free(cosb); free(sinb);
    free(scores); free(vecw);
}

/* ------------------------------------------------------------- ref.json */

static void top3_of(const float *row, int64_t n, int32_t *idx)
{
    for (int k = 0; k < 3; k++) {
        int64_t best = -1;
        for (int64_t i = 0; i < n; i++) {
            int seen = 0;
            for (int p = 0; p < k; p++) if (idx[p] == (int32_t)i) seen = 1;
            if (seen) continue;
            if (best < 0 || row[i] > row[best]) best = i;
        }
        idx[k] = (int32_t)best;
    }
}

static int write_ref(const char *path, const char *model, const int32_t *ids,
                     int64_t T, const ocfg *g, const oresult *r)
{
    FILE *f = fopen(path, "wb");
    if (!f) return tt_fail("nao consegui escrever '%s'", path);

    fprintf(f, "{\n \"model\": \"");
    for (const char *p = model; *p; p++) {
        if (*p == '\\' || *p == '"') fputc('\\', f);
        fputc(*p, f);
    }
    fprintf(f, "\",\n \"input_ids\": [\n");
    for (int64_t t = 0; t < T; t++)
        fprintf(f, "  %d%s\n", ids[t], t + 1 < T ? "," : "");
    fprintf(f, " ],\n \"argmax\": [\n");

    int32_t *top = (int32_t *)malloc((size_t)T * 3 * sizeof(int32_t));
    for (int64_t t = 0; t < T; t++)
        top3_of(r->logits + t * g->vocab, g->vocab, top + t * 3);

    for (int64_t t = 0; t < T; t++)
        fprintf(f, "  %d%s\n", top[t * 3], t + 1 < T ? "," : "");

    fprintf(f, " ],\n \"top3\": [\n");
    for (int64_t t = 0; t < T; t++)
        fprintf(f, "  [%d, %d, %d]%s\n", top[t * 3], top[t * 3 + 1], top[t * 3 + 2],
                t + 1 < T ? "," : "");

    fprintf(f, " ],\n \"top3_logits\": [\n");
    for (int64_t t = 0; t < T; t++) {
        const float *row = r->logits + t * g->vocab;
        fprintf(f, "  [%.7g, %.7g, %.7g]%s\n", (double)row[top[t * 3]],
                (double)row[top[t * 3 + 1]], (double)row[top[t * 3 + 2]],
                t + 1 < T ? "," : "");
    }

    fprintf(f, " ],\n \"hidden_norm\": [\n");
    for (int64_t t = 0; t < T; t++) {
        double n = 0.0;
        for (int64_t j = 0; j < g->hidden; j++) {
            const double v = r->hidden[t * g->hidden + j];
            n += v * v;
        }
        fprintf(f, "  %.7g%s\n", sqrt(n), t + 1 < T ? "," : "");
    }

    fprintf(f, " ],\n \"layer_norm_last\": [\n");
    for (int64_t l = 0; l < g->layers; l++)
        fprintf(f, "  %.7g%s\n", r->layer_norm[l], l + 1 < g->layers ? "," : "");
    fprintf(f, " ]\n}\n");

    fclose(f);
    free(top);
    return 0;
}

/* Compara um dump de logits do engine com um `ref.json` que ja existe.  Sem
 * modelo e sem forward: roda em milissegundos, que e o motivo de existir. */
static int check_ref(const char *ref_path, const char *logits_path)
{
    tt_map m;
    if (tt_map_open(&m, ref_path) != 0) return 1;

    tt_arena arena;
    if (tt_arena_init(&arena, tt_json_bound((const char *)m.base, m.size) + (1u << 20),
                      "ref") != 0) return 1;
    const tt_json *root = tt_json_parse((const char *)m.base, m.size, &arena);
    if (!root) { fprintf(stderr, "%s: %s\n", ref_path, tt_error()); return 1; }

    const tt_json *argmax = tt_json_get(root, "argmax");
    if (!argmax) { fprintf(stderr, "%s: sem 'argmax'\n", ref_path); return 1; }
    const int64_t T = (int64_t)argmax->count;

    FILE *f = fopen(logits_path, "rb");
    if (!f) { fprintf(stderr, "nao consegui abrir '%s'\n", logits_path); return 1; }
    fseek(f, 0, SEEK_END);
    const long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);

    const int64_t total = bytes / (long)sizeof(float);
    if (total % T) {
        fprintf(stderr, "%s tem %lld floats, indivisivel por %lld posicoes\n",
                logits_path, (long long)total, (long long)T);
        fclose(f);
        return 1;
    }
    const int64_t vocab = total / T;

    float *row = (float *)malloc((size_t)vocab * sizeof(float));
    int same = 0, shown = 0;

    printf("%s x %s\n", logits_path, ref_path);
    printf("  %lld posicoes, vocab %lld\n", (long long)T, (long long)vocab);

    const tt_json *node = argmax->child;
    for (int64_t t = 0; t < T; t++, node = node->next) {
        if (fread(row, sizeof(float), (size_t)vocab, f) != (size_t)vocab) {
            fprintf(stderr, "leitura curta na posicao %lld\n", (long long)t);
            fclose(f); free(row); return 1;
        }
        int64_t best = 0;
        for (int64_t i = 1; i < vocab; i++) if (row[i] > row[best]) best = i;

        const int64_t want = tt_json_int(node, -1);
        if (best == want) same++;
        else if (shown++ < 10)
            printf("  pos %lld: engine %lld (logit %.4f), referencia %lld (logit %.4f)\n",
                   (long long)t, (long long)best, (double)row[best],
                   (long long)want, (double)row[want]);
    }
    fclose(f);
    free(row);

    printf("  tokens iguais      %d/%lld\n", same, (long long)T);
    printf("  %s\n", same == T ? "OK" : "FALHOU");
    tt_map_close(&m);
    return same == (int)T ? 0 : 1;
}

/* ------------------------------------------------------------------ CLI */

static int64_t parse_ids(const char *s, int32_t *out, int64_t max)
{
    int64_t n = 0;
    while (*s && n < max) {
        while (*s && (*s == ',' || *s == ' ' || *s == '\n' || *s == '\r' || *s == '\t')) s++;
        if (!*s) break;
        char *end = NULL;
        const long v = strtol(s, &end, 10);
        if (end == s) break;
        out[n++] = (int32_t)v;
        s = end;
    }
    return n;
}

int main(int argc, char **argv)
{
    const char *model = NULL, *ids_str = NULL, *ids_file = NULL, *prompt = NULL;
    const char *tokpath = NULL, *out_ref = NULL, *save = NULL, *compare = NULL;
    const char *ref_only = NULL, *logits_only = NULL;
    int64_t n_pos = 32, generate = 0;
    int threads = 0, verbose = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = (i + 1 < argc);
        if      (!strcmp(a, "--ids") && has)        ids_str = argv[++i];
        else if (!strcmp(a, "--ids-file") && has)   ids_file = argv[++i];
        else if (!strcmp(a, "--prompt") && has)     prompt = argv[++i];
        else if (!strcmp(a, "--tokenizer") && has)  tokpath = argv[++i];
        else if (!strcmp(a, "--out") && has)        out_ref = argv[++i];
        else if (!strcmp(a, "--save-logits") && has) save = argv[++i];
        else if (!strcmp(a, "--compare") && has)    compare = argv[++i];
        else if (!strcmp(a, "--check-ref") && has)  ref_only = argv[++i];
        else if (!strcmp(a, "--logits") && has)     logits_only = argv[++i];
        else if (!strcmp(a, "--n") && has)          n_pos = atoll(argv[++i]);
        else if (!strcmp(a, "--generate") && has)   generate = atoll(argv[++i]);
        else if (!strcmp(a, "--threads") && has)    threads = atoi(argv[++i]);
        else if (!strcmp(a, "-v"))                  verbose = 1;
        else if (a[0] == '-') { fprintf(stderr, "opcao desconhecida: %s\n", a); return 2; }
        else if (!model) model = a;
    }

    if (ref_only || logits_only) {
        if (!ref_only || !logits_only) {
            fprintf(stderr, "--check-ref e --logits andam juntos\n");
            return 2;
        }
        return check_ref(ref_only, logits_only);
    }

    if (!model || (!ids_str && !ids_file && !prompt)) {
        fprintf(stderr,
            "uso: oracle <model.tootega> (--ids 1,2,3 | --ids-file <f> | --prompt <txt>)\n"
            "     [--tokenizer <tokenizer.json>] [--n 32] [--generate N]\n"
            "     [--out ref.json] [--save-logits <f>] [--compare <f>]\n"
            "     [--threads N] [-v]\n"
            "     oracle --check-ref <ref.json> --logits <f>\n\n"
            "  --compare    logits crus do engine (f32, [T][vocab]); sai 1 se\n"
            "               algum argmax divergir\n"
            "  --generate   geracao gulosa a partir do prompt, N tokens\n"
            "  --check-ref  compara um dump com uma referencia pronta, sem\n"
            "               recalcular nada\n");
        return 2;
    }

    ocontainer c;
    if (oc_open(&c, model) != 0) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }

    ocfg g;
    read_cfg(&c, &g);

    g_pool = tt_pool_create(threads, TT_AFFINITY_PCORES);
    if (!g_pool) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }

    int32_t *ids = (int32_t *)malloc((size_t)(n_pos + generate + 8) * sizeof(int32_t));
    int64_t T = 0;

    tt_tokenizer tok;
    int have_tok = 0;
    if (tokpath) {
        if (tt_tokenizer_load(&tok, tokpath, (int32_t)g.vocab) != 0) {
            fprintf(stderr, "erro: %s\n", tt_error());
            return 1;
        }
        have_tok = 1;
    }

    if (ids_file) {
        tt_map m;
        if (tt_map_open(&m, ids_file) != 0) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }
        char *txt = (char *)malloc(m.size + 1);
        memcpy(txt, m.base, m.size);
        txt[m.size] = '\0';
        T = parse_ids(txt, ids, n_pos);
        free(txt);
        tt_map_close(&m);
    } else if (ids_str) {
        T = parse_ids(ids_str, ids, n_pos);
    } else {
        if (!have_tok) {
            fprintf(stderr, "--prompt precisa de --tokenizer\n");
            return 2;
        }
        T = tt_encode(&tok, prompt, strlen(prompt), ids, n_pos);
        if (T < 0) { fprintf(stderr, "erro: %s\n", tt_error()); return 1; }
    }
    if (T <= 0) { fprintf(stderr, "nenhum id de entrada\n"); return 2; }

    oresult r;
    const int64_t cap = T + generate;
    r.logits     = (float *)malloc((size_t)cap * g.vocab * sizeof(float));
    r.hidden     = (float *)malloc((size_t)cap * g.hidden * sizeof(float));
    r.layer_norm = (double *)malloc((size_t)g.layers * sizeof(double));
    if (!r.logits || !r.hidden || !r.layer_norm) {
        fprintf(stderr, "sem memoria para %lld posicoes de vocabulario %lld\n",
                (long long)cap, (long long)g.vocab);
        return 1;
    }

    printf("oraculo: %lld posicoes, teacher-forcing, %d threads\n",
           (long long)T, tt_pool_threads(g_pool));

    const double t0 = tt_now();
    forward(&c, &g, ids, T, &r, verbose);
    printf("  %.1f s\n", tt_now() - t0);

    /* ---------------------------------------------------- geracao gulosa */

    if (generate > 0) {
        /* Existe para separar bug de forward de bug de template ou de
         * amostragem: se a geracao pelo caminho independente for coerente e a do
         * engine nao, o problema nao esta na quantizacao nem na matematica.
         * Recalcula o prefixo inteiro a cada token -- e lento de proposito, sao
         * dezenas de tokens para responder uma pergunta, nao um benchmark. */
        for (int64_t k = 0; k < generate; k++) {
            const float *row = r.logits + (T - 1) * g.vocab;
            int64_t best = 0;
            for (int64_t i = 1; i < g.vocab; i++) if (row[i] > row[best]) best = i;
            ids[T++] = (int32_t)best;

            if (have_tok) {
                char piece[512];
                const int64_t n = tt_decode(&tok, ids + T - 1, 1, piece, sizeof(piece));
                if (n > 0) { fwrite(piece, 1, (size_t)n, stdout); fflush(stdout); }
            } else {
                printf(" %lld", (long long)best);
                fflush(stdout);
            }
            if (k + 1 < generate) forward(&c, &g, ids, T, &r, 0);
        }
        printf("\n");
    }

    for (int64_t t = 0; t < T && !generate; t++) {
        const float *row = r.logits + t * g.vocab;
        int64_t best = 0, second = -1;
        for (int64_t i = 1; i < g.vocab; i++) if (row[i] > row[best]) best = i;
        for (int64_t i = 0; i < g.vocab; i++)
            if (i != best && (second < 0 || row[i] > row[second])) second = i;
        printf("  %3lld %8d -> %8lld  logit %8.3f  margem %6.3f\n",
               (long long)t, ids[t], (long long)best,
               (double)row[best], (double)(row[best] - row[second]));
    }

    if (out_ref && write_ref(out_ref, model, ids, T, &g, &r) == 0)
        printf("  %s escrito\n", out_ref);

    if (save) {
        FILE *f = fopen(save, "wb");
        if (f) {
            fwrite(r.logits, sizeof(float), (size_t)T * g.vocab, f);
            fclose(f);
            printf("  %s escrito\n", save);
        }
    }

    int rc = 0;

    if (compare) {
        FILE *f = fopen(compare, "rb");
        if (!f) { fprintf(stderr, "nao consegui abrir '%s'\n", compare); return 1; }

        float *row = (float *)malloc((size_t)g.vocab * sizeof(float));
        int same = 0;
        double num = 0.0, den = 0.0, worst = 0.0;

        printf("\ncomparacao com o engine:\n");
        for (int64_t t = 0; t < T; t++) {
            if (fread(row, sizeof(float), (size_t)g.vocab, f) != (size_t)g.vocab) {
                fprintf(stderr, "  %s acabou na posicao %lld\n", compare, (long long)t);
                fclose(f); free(row); return 1;
            }
            const float *ref = r.logits + t * g.vocab;
            int64_t gb = 0, rb = 0;
            for (int64_t i = 0; i < g.vocab; i++) {
                if (row[i] > row[gb]) gb = i;
                if (ref[i] > ref[rb]) rb = i;
                const double d = (double)row[i] - (double)ref[i];
                num += d * d;
                den += (double)ref[i] * (double)ref[i];
                if (fabs(d) > worst) worst = fabs(d);
            }
            if (gb == rb) same++;
            else printf("  pos %lld: engine %lld (logit %.4f), oraculo %lld (logit %.4f)\n",
                        (long long)t, (long long)gb, (double)row[gb],
                        (long long)rb, (double)ref[rb]);
        }
        fclose(f);
        free(row);

        printf("  tokens iguais      %d/%lld\n", same, (long long)T);
        printf("  erro relativo      %.4f%%\n", 100.0 * sqrt(num / den));
        printf("  maior diferenca    %.6f\n", worst);
        rc = (same == (int)T) ? 0 : 1;
    }

    tt_pool_destroy(g_pool);
    if (have_tok) tt_tokenizer_free(&tok);
    free(ids); free(r.logits); free(r.hidden); free(r.layer_norm);
    tt_map_close(&c.map);
    tt_arena_free(&c.arena);
    return rc;
}
