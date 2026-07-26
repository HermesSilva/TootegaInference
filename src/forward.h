/* forward.h -- estado de uma sequencia e o passe de um token.
 *
 * Um `tt_state` e um slot: cache de KV proprio, buffers de scratch proprios,
 * lista dos ids ja processados.  Toda a memoria sai de duas arenas
 * dimensionadas em `tt_state_init` -- depois disso o laco de token nao chama
 * malloc nem le variavel de ambiente (criterio de saida do M1, docs/05).
 *
 * O KV do M1 e contiguo por slot e guarda K e V JA EXPANDIDOS.  E consequencia
 * direta de docs/15: a formulacao desabsorvida do MLA e a do oraculo, e ela
 * precisa de K e V completos.  Custa 1,9 MB por token (47 camadas x 20 cabecas
 * x 512 valores x 4 B) -- 7,9 GB a 4k de contexto, que e a configuracao de
 * medida do M1.  Contexto de agente exige a formulacao absorvida ou KV em
 * meia precisao, e isso e M5.
 */
#ifndef TOOTEGA_FORWARD_H
#define TOOTEGA_FORWARD_H

#include "model.h"

typedef struct {
    const tt_model *m;

    tt_arena kv;               /* cache, vive entre tokens */
    tt_arena scratch;          /* buffers do passe, resetados por token */

    int64_t  ctx;              /* capacidade em tokens */
    int64_t  len;              /* tokens ja no cache */

    /* Modo EXPANDED: K e V completos.  Modo ABSORBED: `kvc` no lugar dos dois,
     * [camada][pos][kv_lora_rank + qk_rope_head_dim]. */
    float   *k;                /* [camada][pos][H * qk_head_dim] */
    float   *v;                /* [camada][pos][H * v_head_dim]  */
    float    *kvc;             /* [camada][pos][576], modo f32 */
    uint16_t *kvc16;           /* idem em bf16 -- metade do trafego */
    int8_t   *kvq;             /* idem em int8 g64 -- 3,8x menos */
    float    *kvq_scales;      /* [camada][pos][576/64] */
    int32_t *tokens;           /* ids no cache, para casamento de prefixo */

    /* scratch, tudo apontando para dentro de `scratch` */
    float *h, *hn, *resid;
    float *q_lora, *q, *kv_a, *ckv, *kvb, *krot, *attn, *scores;
    /* [H][kv_lora_rank] cada, so no modo absorvido.  `wrow` e a linha de
     * `kv_b` desquantizada -- uma por cabeca, porque as cabecas rodam em
     * paralelo e cada uma percorre as suas 448 linhas. */
    float *q_abs, *o_c, *wrow;
    float *gate_buf, *up_buf, *mlp_out, *expert_out;
    /* MoE fundido: gate+up dos K experts mais o compartilhado, e as saidas
     * de down antes da combinacao. */
    float *moe_gu, *moe_down;
    float *router_logits, *router_scores, *router_choice;
    float *logits;
    int32_t *topk_idx;
    float   *topk_w;

    /* Traco de depuracao: se nao for NULL, recebe o estado oculto ao fim de
     * cada camada -- [num_hidden_layers + 1][hidden_size], sendo a primeira
     * entrada o embedding.  E o que permite bissecar contra o transformers e
     * achar a PRIMEIRA camada que diverge, em vez de reler o codigo. */
    float   *trace;

    /* contadores do relatorio */
    uint64_t tokens_processed;
    double   seconds_in_forward;
} tt_state;

int  tt_state_init(tt_state *s, const tt_model *m, int64_t ctx);
void tt_state_free(tt_state *s);
void tt_state_reset(tt_state *s);       /* esquece a sequencia, mantem memoria */

/* Bytes de KV para `ctx` tokens -- o orcamento de docs/09 precisa disto antes
 * de alocar qualquer coisa. */
uint64_t tt_kv_bytes(const tt_model *m, int64_t ctx);

/* Scratch do passe.  Existe separado de `tt_state_init` porque o `--plan` tem
 * de responder "cabe?" sem alocar nada (docs/09). */
uint64_t tt_scratch_bytes(const tt_model *m, int64_t ctx);

/* Processa `token` na posicao s->len, avanca o cache e devolve os logits
 * (ponteiro para dentro do scratch, valido ate o proximo passe). */
int tt_forward(tt_state *s, int32_t token, float **logits_out);

/* Casa o maior prefixo comum com o cache e trunca no divergente.  Devolve
 * quantos tokens continuam validos -- o prefill so precisa rodar do resto. */
int64_t tt_state_match_prefix(tt_state *s, const int32_t *ids, int64_t n);

#endif /* TOOTEGA_FORWARD_H */
