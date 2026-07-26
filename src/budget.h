/* budget.h -- resource budget.  Implements docs/09-resource-limits.md.
 *
 * The rule this file exists to enforce: **refuse instead of degrading**.  An
 * engine that accepts the request and then makes the OS page delivers
 * 0.05 tok/s and the person does not know why; one that refuses and says what
 * to reduce costs five seconds and no doubt at all.
 *
 * That is why every calculation here happens BEFORE anything is allocated, and
 * why verification continues at runtime -- estimating is promising, measuring is
 * keeping the promise.
 */
#ifndef TOOTEGA_BUDGET_H
#define TOOTEGA_BUDGET_H

#include <stdint.h>

#include "forward.h"
#include "model.h"

typedef struct {
    uint64_t weights;          /* resident weights */
    uint64_t kv;               /* cache for `ctx` positions */
    uint64_t scratch;
    uint64_t total;

    double   budget_gb;        /* ceiling requested, or the docs/09 default */
    double   ram_total_gb;
    double   ram_avail_gb;

    int      fits;
    int64_t  ctx_max;          /* largest ctx that would fit in the budget */
} tt_plan;

/* `ram_gb <= 0` uses the `auto` default: total minus 8 GB of headroom for the
 * OS and for whatever else the person is doing. */
void tt_plan_compute(tt_plan *p, const tt_model *m, int64_t ctx, double ram_gb);
void tt_plan_print(const tt_plan *p, const tt_model *m, int64_t ctx, int threads);

/* -1 with an actionable message if it does not fit -- the message states the
 * new `--ctx`. */
int  tt_plan_check(const tt_plan *p, int64_t ctx);

/* Runtime verification: measured RSS against the promised ceiling. */
int  tt_budget_verify(const tt_plan *p);

int tt_cmd_plan(int argc, char **argv);

#endif /* TOOTEGA_BUDGET_H */
