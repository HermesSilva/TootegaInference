/* budget.h -- orcamento de recursos.  Implementa docs/09-limites-de-recursos.md.
 *
 * A regra que este arquivo existe para cumprir: **recusar em vez de degradar**.
 * Um engine que aceita o pedido e depois faz o SO paginar entrega 0,05 tok/s e
 * a pessoa nao sabe por que; um que recusa dizendo o que reduzir custa cinco
 * segundos e nenhuma duvida.
 *
 * Por isso toda conta aqui e feita ANTES de alocar qualquer coisa, e a
 * verificacao continua durante a execucao -- estimar e prometer, medir e
 * cumprir.
 */
#ifndef TOOTEGA_BUDGET_H
#define TOOTEGA_BUDGET_H

#include <stdint.h>

#include "forward.h"
#include "model.h"

typedef struct {
    uint64_t weights;          /* pesos residentes */
    uint64_t kv;               /* cache para `ctx` posicoes */
    uint64_t scratch;
    uint64_t total;

    double   budget_gb;        /* teto pedido, ou o default de docs/09 */
    double   ram_total_gb;
    double   ram_avail_gb;

    int      fits;
    int64_t  ctx_max;          /* maior ctx que caberia no orcamento */
} tt_plan;

/* `ram_gb <= 0` usa o default `auto`: total menos 8 GB de folga para o SO e
 * para o resto do que a pessoa esta fazendo. */
void tt_plan_compute(tt_plan *p, const tt_model *m, int64_t ctx, double ram_gb);
void tt_plan_print(const tt_plan *p, const tt_model *m, int64_t ctx, int threads);

/* -1 com mensagem acionavel se nao couber -- a mensagem diz o novo `--ctx`. */
int  tt_plan_check(const tt_plan *p, int64_t ctx);

/* Verificacao em execucao: o RSS medido contra o teto prometido. */
int  tt_budget_verify(const tt_plan *p);

int tt_cmd_plan(int argc, char **argv);

#endif /* TOOTEGA_BUDGET_H */
