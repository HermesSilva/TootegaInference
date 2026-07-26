/* offload.h -- colocacao de tensores na VRAM.
 *
 * O engine nao "roda na GPU": ele decide QUAIS pesos moram la, e o despacho do
 * matmul (kernels.c) segue o tensor.  Quem nao subiu continua na CPU, no mesmo
 * passe, sem ramo especial no forward.
 *
 * A ordem de subida vem da tabela de trafego de docs/18 e nao do tamanho:
 *
 *   1. atencao de TODAS as camadas   0,51 GB   28% do trafego
 *   2. lm_head                       0,16 GB    9%
 *   3. camada a camada, os experts   302 MB/camada, 61% no total
 *
 * A atencao vem primeiro porque roda inteira, em toda camada, para todo token
 * -- nunca ha parte ociosa.  Um expert so ajuda se os 64 da camada estiverem
 * residentes, porque qualquer um pode ser roteado no proximo token; por isso a
 * unidade da terceira fase e a camada, e ela e tudo ou nada.
 */
#ifndef TOOTEGA_OFFLOAD_H
#define TOOTEGA_OFFLOAD_H

#include "container.h"

typedef struct {
    int      enabled;
    uint64_t bytes;            /* pesos residentes na VRAM */
    int      tensors;          /* quantos subiram */
    int      attn_layers;      /* camadas com a atencao residente */
    int      moe_layers;       /* camadas com os experts residentes */
    int      lm_head;
    double   vram_free_gb;     /* depois da colocacao */
    char     device[64];
} tt_gpu_plan;

/* Carrega o backend e sobe o que couber em `vram_gb` (0 = livre menos 1 GB).
 *
 * Nunca e erro fatal: sem DLL, sem placa ou sem VRAM, devolve `enabled = 0` e
 * o engine roda em CPU. A razao vai para o log. */
int  tt_gpu_place(tt_container *c, const char *dll, double vram_gb,
                  tt_gpu_plan *out);

/* Libera os tensores e zera os handles.  Seguro sem colocacao. */
void tt_gpu_release(tt_container *c);

void tt_gpu_print(const tt_gpu_plan *p);

#endif /* TOOTEGA_OFFLOAD_H */
