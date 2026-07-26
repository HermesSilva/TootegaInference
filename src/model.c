#include "model.h"
#include "kernels.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------- config */

static int64_t cfg_int(const tt_container *c, const char *key, int64_t def, int *missing)
{
    const tt_json *n = tt_json_get(c->config, key);
    if (!n) { if (missing) (*missing)++; return def; }
    return tt_json_int(n, def);
}

static double cfg_real(const tt_container *c, const char *key, double def)
{
    return tt_json_real(tt_json_get(c->config, key), def);
}

static int read_config(tt_model *m)
{
    tt_config *k = &m->cfg;
    const tt_container *c = &m->c;

    if (!c->config)
        return tt_fail("container sem meta.config -- o engine nao tem de onde "
                       "ler a arquitetura");

    int missing = 0;
    k->hidden_size           = cfg_int(c, "hidden_size", 0, &missing);
    k->vocab_size            = cfg_int(c, "vocab_size", 0, &missing);
    k->num_hidden_layers     = cfg_int(c, "num_hidden_layers", 0, &missing);
    k->num_attention_heads   = cfg_int(c, "num_attention_heads", 0, &missing);
    k->intermediate_size     = cfg_int(c, "intermediate_size", 0, &missing);
    k->moe_intermediate_size = cfg_int(c, "moe_intermediate_size", 0, &missing);
    k->n_routed_experts      = cfg_int(c, "n_routed_experts", 0, &missing);
    k->n_shared_experts      = cfg_int(c, "n_shared_experts", 1, NULL);
    k->num_experts_per_tok   = cfg_int(c, "num_experts_per_tok", 0, &missing);
    k->first_k_dense_replace = cfg_int(c, "first_k_dense_replace", 0, NULL);
    k->n_group               = cfg_int(c, "n_group", 1, NULL);
    k->topk_group            = cfg_int(c, "topk_group", 1, NULL);
    k->max_position_embeddings = cfg_int(c, "max_position_embeddings", 4096, NULL);

    k->q_lora_rank      = cfg_int(c, "q_lora_rank", 0, &missing);
    k->kv_lora_rank     = cfg_int(c, "kv_lora_rank", 0, &missing);
    k->qk_nope_head_dim = cfg_int(c, "qk_nope_head_dim", 0, &missing);
    k->qk_rope_head_dim = cfg_int(c, "qk_rope_head_dim", 0, &missing);
    k->v_head_dim       = cfg_int(c, "v_head_dim", 0, &missing);
    k->qk_head_dim      = k->qk_nope_head_dim + k->qk_rope_head_dim;

    k->routed_scaling_factor = (float)cfg_real(c, "routed_scaling_factor", 1.0);
    k->rms_norm_eps          = (float)cfg_real(c, "rms_norm_eps", 1e-6);
    k->rope_theta            = cfg_real(c, "rope_theta", 10000.0);
    k->norm_topk_prob        = tt_json_bool(tt_json_get(c->config, "norm_topk_prob"), 1);

    k->mtp_layer = tt_json_int(tt_json_get(c->meta, "mtp_layer"), -1);

    if (missing)
        return tt_fail("config incompleta: %d campo(s) obrigatorio(s) ausente(s)",
                       missing);

    /* Restricoes que o M1 assume.  Recusar e melhor que rodar errado: cada uma
     * destas mudaria a matematica, e docs/05 diz que outras arquiteturas so
     * entram quando houver um modelo concreto que valha. */
    if (k->q_lora_rank <= 0)
        return tt_fail("q_lora_rank = %lld: o M1 so implementa MLA com q-LoRA",
                       (long long)k->q_lora_rank);
    if (k->n_group != 1 || k->topk_group != 1)
        return tt_fail("n_group = %lld, topk_group = %lld: o M1 so implementa "
                       "roteamento sem grupos (docs/05)",
                       (long long)k->n_group, (long long)k->topk_group);
    if (k->num_experts_per_tok > k->n_routed_experts)
        return tt_fail("top-k = %lld com %lld experts",
                       (long long)k->num_experts_per_tok,
                       (long long)k->n_routed_experts);

    const char *act = NULL;
    size_t act_len = 0;
    act = tt_json_str(tt_json_get(c->config, "hidden_act"), &act_len);
    if (act && !(act_len == 4 && memcmp(act, "silu", 4) == 0))
        return tt_fail("hidden_act = '%.*s': o M1 so implementa silu",
                       (int)act_len, act);

    k->attn_scale = 1.0f / sqrtf((float)k->qk_head_dim);
    return 0;
}

/* ----------------------------------------------------- ligacao de pesos */

/* Localiza e confere a forma.  A checagem existe porque o erro que ela pega --
 * tensor certo, forma errada -- nao produz crash, produz numero errado. */
static const tt_tensor *bind(const tt_model *m, int64_t rows, int64_t cols,
                             const char *fmt, ...)
{
    char name[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof(name), fmt, ap);
    va_end(ap);

    const tt_tensor *t = tt_container_find(&m->c, name);
    if (!t) {
        tt_fail("peso ausente: %s", name);
        return NULL;
    }

    if (cols > 0) {
        if (t->rows != rows || t->cols != cols) {
            tt_fail("%s tem forma %lldx%lld, a config exige %lldx%lld",
                    name, (long long)t->rows, (long long)t->cols,
                    (long long)rows, (long long)cols);
            return NULL;
        }
    } else {
        /* vetor: uma dimensao, `rows` elementos */
        if (t->ndim != 1 || t->shape[0] != rows) {
            tt_fail("%s deveria ser vetor de %lld", name, (long long)rows);
            return NULL;
        }
    }
    return t;
}

#define BIND(dst, rows, cols, ...) do {                       \
        (dst) = bind(m, (rows), (cols), __VA_ARGS__);         \
        if (!(dst)) return -1;                                \
    } while (0)

static int bind_attention(tt_model *m, tt_layer *L, int64_t li)
{
    const tt_config *k = &m->cfg;
    const int64_t H = k->num_attention_heads;

    BIND(L->q_a_proj,      k->q_lora_rank, k->hidden_size,  "model.layers.%lld.self_attn.q_a_proj.weight", (long long)li);
    BIND(L->q_a_layernorm, k->q_lora_rank, 0,               "model.layers.%lld.self_attn.q_a_layernorm.weight", (long long)li);
    BIND(L->q_b_proj,      H * k->qk_head_dim, k->q_lora_rank, "model.layers.%lld.self_attn.q_b_proj.weight", (long long)li);

    BIND(L->kv_a_proj,     k->kv_lora_rank + k->qk_rope_head_dim, k->hidden_size,
         "model.layers.%lld.self_attn.kv_a_proj_with_mqa.weight", (long long)li);
    BIND(L->kv_a_layernorm, k->kv_lora_rank, 0,             "model.layers.%lld.self_attn.kv_a_layernorm.weight", (long long)li);
    BIND(L->kv_b_proj,     H * (k->qk_nope_head_dim + k->v_head_dim), k->kv_lora_rank,
         "model.layers.%lld.self_attn.kv_b_proj.weight", (long long)li);

    BIND(L->o_proj,        k->hidden_size, H * k->v_head_dim, "model.layers.%lld.self_attn.o_proj.weight", (long long)li);
    return 0;
}

static int bind_layer(tt_model *m, int64_t li)
{
    const tt_config *k = &m->cfg;
    tt_layer *L = &m->layers[li];

    BIND(L->input_layernorm,          k->hidden_size, 0, "model.layers.%lld.input_layernorm.weight", (long long)li);
    BIND(L->post_attention_layernorm, k->hidden_size, 0, "model.layers.%lld.post_attention_layernorm.weight", (long long)li);

    if (bind_attention(m, L, li) != 0) return -1;

    L->is_moe = (li >= k->first_k_dense_replace);

    if (!L->is_moe) {
        BIND(L->gate_proj, k->intermediate_size, k->hidden_size,       "model.layers.%lld.mlp.gate_proj.weight", (long long)li);
        BIND(L->up_proj,   k->intermediate_size, k->hidden_size,       "model.layers.%lld.mlp.up_proj.weight", (long long)li);
        BIND(L->down_proj, k->hidden_size,       k->intermediate_size, "model.layers.%lld.mlp.down_proj.weight", (long long)li);
        return 0;
    }

    BIND(L->router,      k->n_routed_experts, k->hidden_size, "model.layers.%lld.mlp.gate.weight", (long long)li);
    BIND(L->router_bias, k->n_routed_experts, 0,              "model.layers.%lld.mlp.gate.e_score_correction_bias", (long long)li);

    /* O roteador decide QUAIS experts rodam; um erro de precisao aqui troca o
     * expert, nao o ultimo bit.  Por isso o conversor o manteve em f32 e o
     * engine recusa qualquer outra coisa. */
    if (L->router->kind != TT_KIND_F32 || L->router_bias->kind != TT_KIND_F32)
        return tt_fail("camada %lld: roteador em %s -- tem de ser f32",
                       (long long)li, tt_kind_name(L->router->kind));

    const int64_t shared = k->moe_intermediate_size * k->n_shared_experts;
    BIND(L->sh_gate, shared,          k->hidden_size, "model.layers.%lld.mlp.shared_experts.gate_proj.weight", (long long)li);
    BIND(L->sh_up,   shared,          k->hidden_size, "model.layers.%lld.mlp.shared_experts.up_proj.weight", (long long)li);
    BIND(L->sh_down, k->hidden_size,  shared,         "model.layers.%lld.mlp.shared_experts.down_proj.weight", (long long)li);

    const size_t ne = (size_t)k->n_routed_experts;
    L->e_gate = (const tt_tensor **)tt_arena_alloc(&m->ir, ne * sizeof(void *), 8);
    L->e_up   = (const tt_tensor **)tt_arena_alloc(&m->ir, ne * sizeof(void *), 8);
    L->e_down = (const tt_tensor **)tt_arena_alloc(&m->ir, ne * sizeof(void *), 8);
    if (!L->e_gate || !L->e_up || !L->e_down) return -1;

    for (size_t e = 0; e < ne; e++) {
        BIND(L->e_gate[e], k->moe_intermediate_size, k->hidden_size,
             "model.layers.%lld.mlp.experts.%zu.gate_proj.weight", (long long)li, e);
        BIND(L->e_up[e],   k->moe_intermediate_size, k->hidden_size,
             "model.layers.%lld.mlp.experts.%zu.up_proj.weight", (long long)li, e);
        BIND(L->e_down[e], k->hidden_size, k->moe_intermediate_size,
             "model.layers.%lld.mlp.experts.%zu.down_proj.weight", (long long)li, e);
    }
    return 0;
}

/* ---------------------------------------------------------------- RoPE */

static int build_rope(tt_model *m, int64_t max_pos)
{
    const int64_t rot = m->cfg.qk_rope_head_dim;
    const int64_t half = rot / 2;

    if (rot % 2) return tt_fail("qk_rope_head_dim impar (%lld)", (long long)rot);
    if (max_pos <= 0) max_pos = 1;

    m->rope_cos = tt_arena_new(&m->ir, float, max_pos * half);
    m->rope_sin = tt_arena_new(&m->ir, float, max_pos * half);
    if (!m->rope_cos || !m->rope_sin) return -1;
    m->rope_positions = max_pos;

    for (int64_t i = 0; i < half; i++) {
        /* inv_freq guardado como float32, igual ao buffer do oraculo; o
         * produto pos*inv_freq e uma multiplicacao float32 unica, entao
         * calcular em double e arredondar da o MESMO resultado. */
        const float inv = (float)(1.0 / pow(m->cfg.rope_theta,
                                            (double)(2 * i) / (double)rot));
        for (int64_t p = 0; p < max_pos; p++) {
            const float f = (float)((double)p * (double)inv);
            m->rope_cos[p * half + i] = (float)cos((double)f);
            m->rope_sin[p * half + i] = (float)sin((double)f);
        }
    }
    return 0;
}

/* ------------------------------------------------------------ absorcao */

int tt_kv_prec_parse(const char *s, tt_kv_prec *out)
{
    if (!s || !*s || !strcmp(s, "bf16")) { *out = TT_KV_BF16; return 0; }
    if (!strcmp(s, "f32"))               { *out = TT_KV_F32;  return 0; }
    if (!strcmp(s, "int8"))              { *out = TT_KV_INT8; return 0; }
    return tt_fail("--kv '%s' desconhecido (f32 | int8 | bf16)", s);
}

const char *tt_kv_prec_name(tt_kv_prec p)
{
    switch (p) {
    case TT_KV_BF16: return "bf16";
    case TT_KV_INT8: return "int8";
    default:         return "f32";
    }
}

int tt_attn_parse(const char *s, tt_attn_mode *out)
{
    if (!s || !*s || !strcmp(s, "absorbed")) { *out = TT_ATTN_ABSORBED; return 0; }
    if (!strcmp(s, "expanded"))              { *out = TT_ATTN_EXPANDED; return 0; }
    return tt_fail("--attn '%s' desconhecido (absorbed | expanded)", s);
}

/* -------------------------------------------------------------- abrir */

int tt_model_open(tt_model *m, const char *path, int64_t max_pos,
                  tt_attn_mode attn, tt_kv_prec kv_prec)
{
    memset(m, 0, sizeof(*m));
    m->attn = attn;
    /* A formulacao expandida e a referencia do oraculo; ela nao ganha modo de
     * precisao para nao virar duas referencias (docs/15). */
    m->kv_prec = (attn == TT_ATTN_ABSORBED) ? kv_prec : TT_KV_F32;

    if (tt_container_open(&m->c, path) != 0) return -1;
    if (read_config(m) != 0) { tt_model_close(m); return -1; }

    const tt_config *k = &m->cfg;
    if (max_pos <= 0) max_pos = k->max_position_embeddings;

    /* IR: ponteiros por camada + tabelas de RoPE.  Tudo dimensionado aqui,
     * porque depois disto nao ha alocacao nenhuma (docs/11). */
    size_t need = (size_t)k->num_hidden_layers * sizeof(tt_layer)
                + (size_t)k->num_hidden_layers * 3 * (size_t)k->n_routed_experts * sizeof(void *)
                + (size_t)max_pos * (size_t)k->qk_rope_head_dim * sizeof(float)
                + 64 * 1024;

    if (tt_arena_init(&m->ir, need, "ir") != 0) { tt_model_close(m); return -1; }

    m->layers = tt_arena_new(&m->ir, tt_layer, k->num_hidden_layers);
    if (!m->layers) { tt_model_close(m); return -1; }
    memset(m->layers, 0, (size_t)k->num_hidden_layers * sizeof(tt_layer));

    for (int64_t li = 0; li < k->num_hidden_layers; li++)
        if (bind_layer(m, li) != 0) { tt_model_close(m); return -1; }

    m->embed = bind(m, k->vocab_size, k->hidden_size, "model.embed_tokens.weight");
    if (!m->embed) { tt_model_close(m); return -1; }
    m->norm = bind(m, k->hidden_size, 0, "model.norm.weight");
    if (!m->norm) { tt_model_close(m); return -1; }
    m->lm_head = bind(m, k->vocab_size, k->hidden_size, "lm_head.weight");
    if (!m->lm_head) { tt_model_close(m); return -1; }

    if (build_rope(m, max_pos) != 0) { tt_model_close(m); return -1; }

    tt_dbg("modelo: %lld camadas (%lld densa[s]), %lld experts top-%lld, "
           "hidden %lld, vocab %lld",
           (long long)k->num_hidden_layers, (long long)k->first_k_dense_replace,
           (long long)k->n_routed_experts, (long long)k->num_experts_per_tok,
           (long long)k->hidden_size, (long long)k->vocab_size);
    return 0;
}

void tt_model_close(tt_model *m)
{
    tt_arena_free(&m->ir);
    tt_container_close(&m->c);
    memset(m, 0, sizeof(*m));
}

uint64_t tt_model_weight_bytes(const tt_model *m)
{
    return tt_container_resident_bytes(&m->c);
}

uint64_t tt_model_warm(const tt_model *m)
{
    static const char suffix[] = "embed_tokens.weight";
    const size_t slen = sizeof(suffix) - 1;
    const tt_container *c = &m->c;

    /* Duas fases, e as duas importam: `willneed` e uma DICA assincrona -- diz
     * ao SO para comecar a ler --, e `touch` e sincrono, forcando a falta de
     * pagina agora. Sozinho, o hint nao garante residencia; sozinho, o toque
     * serializa a leitura. Juntos, o disco trabalha adiantado enquanto o laco
     * caminha (docs/11). */
    for (size_t i = 0; i < c->n_tensors; i++) {
        const tt_tensor *t = &c->tensors[i];
        if (t->name_len >= slen &&
            memcmp(t->name + t->name_len - slen, suffix, slen) == 0)
            continue;
        tt_map_willneed(&c->map, t->data_off, t->data_bytes);
    }

    uint64_t touched = 0;
    for (size_t i = 0; i < c->n_tensors; i++) {
        const tt_tensor *t = &c->tensors[i];
        if (t->name_len >= slen &&
            memcmp(t->name + t->name_len - slen, suffix, slen) == 0)
            continue;

        tt_map_touch(&c->map, t->data_off, t->data_bytes);
        if (t->scales_bytes) tt_map_touch(&c->map, t->scales_off, t->scales_bytes);
        touched += t->data_bytes + t->scales_bytes;
    }
    return touched;
}
