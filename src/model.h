/* model.h -- config -> IR -> pesos ligados.
 *
 * O engine le a arquitetura do container (`meta.config`), nunca de um
 * `config.json` ao lado que pode divergir (docs/14).  Cada peso e localizado
 * uma vez na abertura, com a forma conferida contra o que a config promete: um
 * tensor trocado tem de dar erro aqui, com nome e forma, e nao virar lixo
 * numerico 47 camadas adiante.
 *
 * Os pontos de variacao de arquitetura (atencao, roteamento) existem como
 * campos do IR desde o M1, mas so a combinacao do GLM-4.7-Flash e implementada:
 * MLA com q-LoRA e roteador top-k com n_group = 1 (docs/05, "Fora de escopo").
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
    int64_t intermediate_size;         /* MLP da camada densa */
    int64_t moe_intermediate_size;     /* MLP de um expert */
    int64_t n_routed_experts;
    int64_t n_shared_experts;
    int64_t num_experts_per_tok;       /* top-k */
    int64_t first_k_dense_replace;     /* camadas densas no inicio */
    int64_t n_group, topk_group;
    int64_t max_position_embeddings;
    int64_t mtp_layer;                 /* -1 se nao ha */

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

    /* densa (is_moe == 0) */
    const tt_tensor *gate_proj, *up_proj, *down_proj;

    /* MoE (is_moe == 1) */
    const tt_tensor  *router, *router_bias;
    const tt_tensor  *sh_gate, *sh_up, *sh_down;
    const tt_tensor **e_gate, **e_up, **e_down;   /* [n_routed_experts] */
} tt_layer;

/* Formulacao da atencao MLA.  As duas sao algebricamente equivalentes e NAO
 * dao os mesmos bits (docs/15); a escolha e explicita justamente por isso.
 *
 *   EXPANDED  desabsorvida, a do oraculo.  Cacheia K e V completos:
 *             10.240 valores por token por camada.
 *   ABSORBED  dobra `kv_b` na query e opera no espaco comprimido.  Cacheia
 *             576 valores por token por camada -- 18x menos trafego, que e o
 *             que decide o desempenho a contexto longo (docs/13, D11).
 */
typedef enum {
    TT_ATTN_ABSORBED = 0,
    TT_ATTN_EXPANDED
} tt_attn_mode;

/* Precisao do cache de KV.  bf16 mantem os 8 bits de expoente do fp32 e corta
 * o trafego e a pressao de L3 pela metade; e mudanca de saida, autorizada pelo
 * mesmo teste de 32/32 que autorizou a absorcao. */
typedef enum {
    TT_KV_BF16 = 0,
    TT_KV_F32,
    TT_KV_INT8            /* g64 com escala, como o container */
} tt_kv_prec;

#define TT_KV_GROUP 64

int tt_kv_prec_parse(const char *s, tt_kv_prec *out);

/* "f32" | "bf16" | "int8".  Existe para o cabecalho das medicoes: numero sem a
 * configuracao ao lado nao e comparavel depois (docs/17). */
const char *tt_kv_prec_name(tt_kv_prec p);

typedef struct {
    tt_container c;
    tt_config    cfg;
    tt_arena     ir;               /* ponteiros do IR + tabelas de RoPE */

    tt_attn_mode attn;
    tt_kv_prec   kv_prec;

    tt_layer    *layers;
    const tt_tensor *embed;        /* [vocab, hidden], bf16, file-backed */
    const tt_tensor *norm;         /* norma final */
    const tt_tensor *lm_head;

    /* RoPE pre-computado: cos/sin[pos][qk_rope_head_dim/2].  Tabela e nao
     * sinf/cosf no laco -- docs/15, D4. */
    float   *rope_cos, *rope_sin;
    int64_t  rope_positions;
} tt_model;

/* `max_pos` dimensiona a tabela de RoPE; passe o contexto que vai usar. */
int  tt_model_open(tt_model *m, const char *path, int64_t max_pos,
                   tt_attn_mode attn, tt_kv_prec kv_prec);
void tt_model_close(tt_model *m);

int tt_attn_parse(const char *s, tt_attn_mode *out);

/* Primeira linha de `kv_b` que pertence a esta cabeca.  W_k sao as
 * `qk_nope_head_dim` linhas seguintes; W_v, as `v_head_dim` depois delas. */
static inline int64_t tt_kvb_head_row(const tt_config *c, int64_t head)
{
    return head * (c->qk_nope_head_dim + c->v_head_dim);
}

/* Bytes de peso que ficam residentes, para o orcamento de docs/09. */
uint64_t tt_model_weight_bytes(const tt_model *m);

/* Traz os pesos residentes para a RAM antes do primeiro token.
 *
 * Sem isto cada peso chega por falta de pagina no meio do laco de decode, e o
 * primeiro passe custa 5x mais por token -- medido: 480 ms contra 95 ms. Num
 * servidor a conta inteira recai sobre o primeiro usuario, que nao tem como
 * saber por que esperou.
 *
 * `embed_tokens` fica de fora: e lido uma linha por token e nao precisa de
 * residencia (docs/04).  Devolve os bytes tocados. */
uint64_t tt_model_warm(const tt_model *m);

#endif /* TOOTEGA_MODEL_H */
