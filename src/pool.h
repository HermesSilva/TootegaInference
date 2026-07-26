/* pool.h -- our own thread pool.
 *
 * It is not OpenMP.  OpenMP would work, but it drags in `libgomp`, and docs/10
 * promises static linking with minimal runtime (docs/15, D6).
 *
 * The contract that matters comes from docs/15: **partitioning by output row**.
 * Each thread owns whole rows and performs the reduction along I on its own, in
 * the same order, with its own accumulator.  Threads never sum each other's
 * partials -- which is why the result does not change with --threads, and why
 * the determinism test can exist at all.
 *
 * Dynamic granularity (an atomic counter) because the machine is hybrid: with
 * fixed ranges, one E-core holds the barrier for all eight P-cores.
 */
#ifndef TOOTEGA_POOL_H
#define TOOTEGA_POOL_H

#include <stdint.h>

typedef void (*tt_task_fn)(void *arg, int64_t row0, int64_t row1);

typedef struct tt_pool tt_pool;

typedef enum {
    TT_AFFINITY_PCORES = 0,   /* docs/09 default */
    TT_AFFINITY_ALL,          /* every physical core, for measurement */
    TT_AFFINITY_NONE          /* let the OS scheduler decide */
} tt_affinity;

int tt_affinity_parse(const char *s, tt_affinity *out);

/* `threads <= 0` uses the default: one per physical P-core. */
tt_pool *tt_pool_create(int threads, tt_affinity affinity);
void     tt_pool_destroy(tt_pool *p);
int      tt_pool_threads(const tt_pool *p);

/* Runs fn over [0, rows) partitioned into blocks of `grain` rows.  The calling
 * thread joins the work -- there is no sense in it sleeping while the queue has
 * items.  Returns when everything has finished (the barrier is here). */
void tt_pool_run(tt_pool *p, tt_task_fn fn, void *arg, int64_t rows, int64_t grain);

int tt_pool_selftest(void);

#endif /* TOOTEGA_POOL_H */
