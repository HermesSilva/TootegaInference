/* pool.h -- pool de threads proprio.
 *
 * Nao e OpenMP.  OpenMP resolveria, mas arrasta `libgomp` e a promessa de
 * docs/10 e linkagem estatica com o minimo de runtime (docs/15, D6).
 *
 * O contrato que importa vem de docs/15: **particionamento por linha de
 * saida**.  Cada thread possui linhas inteiras e faz a reducao ao longo de I
 * sozinha, na mesma ordem, com acumulador proprio.  Threads nunca somam
 * parcelas umas das outras -- por isso o resultado nao muda com --threads, e o
 * teste de determinismo pode existir.
 *
 * Granularidade dinamica (contador atomico) porque a maquina e hibrida: com
 * faixas fixas, um E-core segura a barreira dos oito P-cores.
 */
#ifndef TOOTEGA_POOL_H
#define TOOTEGA_POOL_H

#include <stdint.h>

typedef void (*tt_task_fn)(void *arg, int64_t row0, int64_t row1);

typedef struct tt_pool tt_pool;

typedef enum {
    TT_AFFINITY_PCORES = 0,   /* default de docs/09 */
    TT_AFFINITY_ALL,          /* todos os cores fisicos, para medir */
    TT_AFFINITY_NONE          /* deixa o escalonador do SO decidir */
} tt_affinity;

int tt_affinity_parse(const char *s, tt_affinity *out);

/* `threads <= 0` usa o default: um por P-core fisico. */
tt_pool *tt_pool_create(int threads, tt_affinity affinity);
void     tt_pool_destroy(tt_pool *p);
int      tt_pool_threads(const tt_pool *p);

/* Roda fn sobre [0, rows) particionado em blocos de `grain` linhas.  A thread
 * chamadora participa do trabalho -- nao faz sentido ela dormir enquanto ha
 * fila.  Retorna quando tudo terminou (a barreira e aqui). */
void tt_pool_run(tt_pool *p, tt_task_fn fn, void *arg, int64_t rows, int64_t grain);

int tt_pool_selftest(void);

#endif /* TOOTEGA_POOL_H */
