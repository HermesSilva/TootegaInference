#include "offload.h"

#include "backend.h"
#include "common.h"

#include <stdio.h>
#include <string.h>

/* Nome do tensor: comparacoes sobre `name`/`name_len`, que nao tem terminador
 * (aponta para dentro do indice mapeado). */
static int has(const tt_tensor *t, const char *needle)
{
    const size_t n = strlen(needle);
    if (t->name_len < n) return 0;
    for (size_t i = 0; i + n <= t->name_len; i++)
        if (!memcmp(t->name + i, needle, n)) return 1;
    return 0;
}

/* Numero da camada em "model.layers.<N>.", ou -1. */
static int layer_of(const tt_tensor *t)
{
    static const char pre[] = "model.layers.";
    const size_t n = sizeof(pre) - 1;
    if (t->name_len < n + 1 || memcmp(t->name, pre, n)) return -1;

    int v = 0;
    size_t i = n;
    if (t->name[i] < '0' || t->name[i] > '9') return -1;
    while (i < t->name_len && t->name[i] >= '0' && t->name[i] <= '9')
        v = v * 10 + (t->name[i++] - '0');
    return v;
}

/* Pode subir?  Fora ficam os 1-D (normas, bias -- somados na CPU, e o trafego
 * deles e ruido) e o bf16, que o backend nao desempacota. */
static int uploadable(const tt_tensor *t)
{
    return t->ndim >= 2 && t->rows > 0 && t->cols > 0 &&
           (t->kind == TT_KIND_INT4 || t->kind == TT_KIND_INT8 ||
            t->kind == TT_KIND_F32) &&
           !has(t, "embed_tokens");
}

static uint64_t tensor_bytes(const tt_tensor *t)
{
    return t->data_bytes + t->scales_bytes;
}

/* Piso de tamanho para tensor que sera chamado SOZINHO.
 *
 * Cada `matmul` isolado paga ~17 us de protocolo (H2D, lancamento, D2H,
 * sincronizacao). Subir um tensor so compensa se o que a GPU economiza no
 * calculo for maior que isso:
 *
 *   1,5 MB (expert)   CPU 22 us   GPU  9 us   economia 13 us  ->  PREJUIZO
 *   2,0 MB (q_b)      CPU 29 us   GPU 11 us   economia 18 us  ->  empate
 *   5,2 MB (o_proj)   CPU 76 us   GPU 30 us   economia 46 us  ->  ganho
 *
 * Os experts escapam da regra porque nunca sao chamados sozinhos: os 15 de uma
 * camada MoE vao num `matmul_many` e pagam o protocolo UMA vez.  Foi medido:
 * subir tudo deu 13,7 tok/s contra 14,4 da CPU pura -- o offload indiscriminado
 * e mais lento que nao usar GPU nenhuma. */
#define SOLO_MIN_BYTES (3u << 20)

static int called_solo(const tt_tensor *t)
{
    return !has(t, "experts.");
}

/* Sobe um tensor se couber no que resta.  Devolve os bytes gastos. */
static uint64_t put(const tt_backend *be, tt_tensor *t, uint64_t *left)
{
    if (t->dev || !uploadable(t)) return 0;
    if (called_solo(t) && tensor_bytes(t) < SOLO_MIN_BYTES) return 0;

    const uint64_t need = tensor_bytes(t);
    if (need > *left) return 0;

    const tt_dev_tensor id = be->upload(t->data, t->data_bytes,
                                        t->scales, t->scales_bytes,
                                        (int)t->kind, t->rows, t->cols,
                                        t->group);
    if (id == TT_DEV_NONE) return 0;

    t->dev = id;
    *left -= need;
    return need;
}

int tt_gpu_place(tt_container *c, const char *dll, double vram_gb,
                 tt_offload_mode mode, tt_gpu_plan *out)
{
    memset(out, 0, sizeof(*out));

    const tt_backend *be = tt_backend_load(dll, 0);
    if (!be) return 0;                      /* razao ja esta em tt_error() */
    snprintf(out->device, sizeof(out->device), "%s", be->name);

    if (!be->matmul_many) {
        tt_log("backend sem matmul_many (ABI velha) -- seguindo em CPU");
        tt_backend_unload();
        return 0;
    }

    /* Sem numero pedido, deixa 1 GB para o driver e o desktop.  Encher a VRAM
     * ate a borda faz o compositor do Windows comecar a paginar, e ai a
     * maquina inteira engasga -- inclusive o que nao e nosso. */
    const uint64_t headroom = 1600ull << 20;    /* 1,6 GB */
    uint64_t left = vram_gb > 0 ? (uint64_t)(vram_gb * 1e9)
                                : (be->vram_free > headroom
                                   ? be->vram_free - headroom : 0);

    if (mode == TT_OFFLOAD_ALL) {
        /* --- 1. atencao de todas as camadas ---------------------------- */
        int layers = 0;
        for (size_t i = 0; i < c->n_tensors; i++) {
            tt_tensor *t = &c->tensors[i];
            if (!has(t, "self_attn")) continue;
            if (put(be, t, &left)) {
                out->bytes += tensor_bytes(t);
                out->tensors++;
            }
        }
        for (size_t i = 0; i < c->n_tensors; i++) {
            const tt_tensor *t = &c->tensors[i];
            if (has(t, "o_proj") && t->dev) layers++;
        }
        out->attn_layers = layers;

        /* --- 2. lm_head ------------------------------------------------ */
        for (size_t i = 0; i < c->n_tensors; i++) {
            tt_tensor *t = &c->tensors[i];
            if (!has(t, "lm_head")) continue;
            if (put(be, t, &left)) {
                out->bytes += tensor_bytes(t);
                out->tensors++;
                out->lm_head = 1;
            }
        }
    }

    /* --- 3. camada a camada, TUDO ou NADA ------------------------------ */
    for (int L = 0; L < 64; L++) {
        /* Custo da camada inteira antes de subir qualquer coisa: subir metade
         * gastaria VRAM sem tirar trafego da CPU -- o token que rotear um
         * expert que ficou de fora paga o caminho lento do mesmo jeito. */
        uint64_t need = 0;
        int found = 0;
        for (size_t i = 0; i < c->n_tensors; i++) {
            const tt_tensor *t = &c->tensors[i];
            if (layer_of(t) != L || t->dev || !uploadable(t)) continue;
            if (mode == TT_OFFLOAD_MOE && !has(t, "experts.")) continue;
            need += tensor_bytes(t);
            found = 1;
        }
        if (!found) continue;
        if (need > left) break;             /* a proxima tambem nao cabe */

        int up = 0;
        for (size_t i = 0; i < c->n_tensors; i++) {
            tt_tensor *t = &c->tensors[i];
            if (layer_of(t) != L) continue;
            if (mode == TT_OFFLOAD_MOE && !has(t, "experts.")) continue;
            if (put(be, t, &left)) {
                out->bytes += tensor_bytes(t);
                out->tensors++;
                up++;
            }
        }
        if (up) out->moe_layers++;
    }

    out->enabled = out->tensors > 0;
    out->vram_free_gb = (double)be->vram_free / 1e9;
    if (!out->enabled) tt_backend_unload();
    return out->enabled;
}

void tt_gpu_release(tt_container *c)
{
    const tt_backend *be = tt_backend_get();
    if (be) {
        for (size_t i = 0; i < c->n_tensors; i++)
            if (c->tensors[i].dev) {
                be->release(c->tensors[i].dev);
                c->tensors[i].dev = 0;
            }
    }
    tt_backend_unload();
}

void tt_gpu_print(const tt_gpu_plan *p)
{
    char buf[32];
    if (!p->enabled) {
        tt_log("gpu       nao usada -- %s", tt_error());
        return;
    }
    tt_log("gpu       %s", p->device);
    tt_log("  residente  %s em %d tensores", tt_bytes(buf, sizeof(buf), p->bytes),
           p->tensors);
    tt_log("  atencao    %d camadas    experts  %d camadas%s",
           p->attn_layers, p->moe_layers, p->lm_head ? "    lm_head" : "");
    tt_log("  vram livre %.2f GB", p->vram_free_gb);
}
