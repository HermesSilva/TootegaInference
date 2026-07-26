#include "pool.h"
#include "common.h"
#include "compat.h"

#include <stdlib.h>
#include <string.h>

#define TT_MAX_THREADS 128

struct tt_pool {
    int        n;                     /* total, contando a thread chamadora */
    tt_thread  th[TT_MAX_THREADS];    /* n-1 usadas */

    tt_atomic32 stop;
    tt_atomic32 generation;           /* muda a cada dispatch */
    tt_atomic64 next;                 /* proximo bloco da fila */
    tt_atomic32 finished;             /* workers que acabaram esta rodada */
    tt_atomic32 sleepers;             /* quantos estao bloqueados agora */

    tt_task_fn fn;
    void      *arg;
    int64_t    rows;
    int64_t    grain;
};

int tt_affinity_parse(const char *s, tt_affinity *out)
{
    if (!s || !*s || !strcmp(s, "p-cores") || !strcmp(s, "pcores")) {
        *out = TT_AFFINITY_PCORES;
        return 0;
    }
    if (!strcmp(s, "all"))  { *out = TT_AFFINITY_ALL;  return 0; }
    if (!strcmp(s, "none")) { *out = TT_AFFINITY_NONE; return 0; }
    return tt_fail("--affinity '%s' desconhecido (p-cores | all | none)", s);
}

/* Espera em tres degraus: gira, cede, e por fim BLOQUEIA (docs/15).
 *
 * O terceiro degrau ja foi `tt_sleep_ms(1)`, e isso era polling disfarcado. O
 * worker acordava por tempo, nao por evento: depois de uma pausa longa -- a
 * inicializacao do contexto CUDA, por exemplo -- todos estavam dormindo, e
 * cada dispatch seguinte esperava o tique do relogio. Um matmul de 1,6 MB
 * passava de 0,25 ms para 26 ms, e a CPU com 20 threads ficava 26x MAIS LENTA
 * que a mesma CPU com uma.
 *
 * Agora quem despacha acorda quem espera. O timeout de 50 ms fica so como rede
 * contra acordar perdido -- nao e o mecanismo, e o seguro dele. */
static void wait_for_work(struct tt_pool *p, int32_t last, int *spins)
{
    const int n = ++(*spins);
    if (n < 512)   { tt_cpu_relax(); return; }
    if (n < 20000) { tt_yield();     return; }

    /* Anunciar-se ANTES de conferir de novo: se o dispatch acontecer entre a
     * conferencia e o bloqueio, ele ja vera o contador e acordara. */
    tt_at32_add(&p->sleepers, 1);
    if (tt_at32_load(&p->generation) == last)
        tt_futex_wait(&p->generation.v, last, 50);
    tt_at32_add(&p->sleepers, -1);
}

/* Consome blocos ate a fila secar.  O mesmo laco roda na chamadora e nos
 * workers -- a chamadora nao dorme enquanto ha trabalho. */
static void drain(struct tt_pool *p)
{
    for (;;) {
        long long i = tt_at64_add(&p->next, p->grain);
        if (i >= p->rows) break;
        long long end = i + p->grain;
        if (end > p->rows) end = p->rows;
        p->fn(p->arg, (int64_t)i, (int64_t)end);
    }
}

typedef struct {
    struct tt_pool *pool;
    int             cpu;
} worker_arg;

static void worker(void *raw)
{
    worker_arg *wa = (worker_arg *)raw;
    struct tt_pool *p = wa->pool;
    const int cpu = wa->cpu;
    free(wa);

    /* O pinning acontece na criacao da thread, nao no meio do trabalho
     * (docs/09). */
    tt_thread_pin(cpu);

    int32_t last = 0;
    int spins = 0;

    for (;;) {
        int32_t gen = tt_at32_load(&p->generation);
        if (gen == last) { wait_for_work(p, last, &spins); continue; }
        if (tt_at32_load(&p->stop)) break;

        last = gen;
        spins = 0;
        drain(p);
        tt_at32_add(&p->finished, 1);
    }
}

tt_pool *tt_pool_create(int threads, tt_affinity affinity)
{
    int cpus[TT_MAX_THREADS];
    int ncpu;

    switch (affinity) {
    case TT_AFFINITY_ALL:  ncpu = tt_cpu_core_list(cpus, TT_MAX_THREADS);  break;
    case TT_AFFINITY_NONE: ncpu = tt_cpu_threads(); if (ncpu > TT_MAX_THREADS) ncpu = TT_MAX_THREADS;
                           for (int i = 0; i < ncpu; i++) cpus[i] = -1;    break;
    default:               ncpu = tt_cpu_pcore_list(cpus, TT_MAX_THREADS); break;
    }
    if (ncpu < 1) { cpus[0] = -1; ncpu = 1; }

    if (threads <= 0) threads = ncpu;
    if (threads > TT_MAX_THREADS) threads = TT_MAX_THREADS;

    struct tt_pool *p = (struct tt_pool *)calloc(1, sizeof(*p));
    if (!p) { tt_fail("sem memoria para o pool"); return NULL; }

    p->n = threads;

    for (int i = 0; i + 1 < threads; i++) {
        worker_arg *wa = (worker_arg *)malloc(sizeof(*wa));
        if (!wa) { p->n = i + 1; tt_pool_destroy(p); tt_fail("sem memoria"); return NULL; }
        wa->pool = p;
        /* +1 porque a thread chamadora fica com cpus[0]. */
        wa->cpu = (affinity == TT_AFFINITY_NONE) ? -1 : cpus[(i + 1) % ncpu];

        if (tt_thread_start(&p->th[i], worker, wa) != 0) {
            free(wa);
            p->n = i + 1;
            tt_pool_destroy(p);
            tt_fail("nao consegui criar a thread %d", i + 1);
            return NULL;
        }
    }

    if (affinity != TT_AFFINITY_NONE) tt_thread_pin(cpus[0]);

    tt_dbg("pool: %d threads, afinidade %s", p->n,
           affinity == TT_AFFINITY_ALL ? "all" :
           affinity == TT_AFFINITY_NONE ? "none" : "p-cores");
    return p;
}

void tt_pool_destroy(tt_pool *p)
{
    if (!p) return;
    tt_at32_store(&p->stop, 1);
    tt_at32_add(&p->generation, 1);
    tt_futex_wake(&p->generation.v);
    for (int i = 0; i + 1 < p->n; i++) tt_thread_join(&p->th[i]);
    free(p);
}

int tt_pool_threads(const tt_pool *p) { return p ? p->n : 1; }

void tt_pool_run(tt_pool *p, tt_task_fn fn, void *arg, int64_t rows, int64_t grain)
{
    if (rows <= 0) return;
    if (grain <= 0) grain = 1;

    /* Trabalho pequeno nao paga a barreira.  Sem este corte, o roteador (64
     * linhas) e as normas viram milhares de dispatches por token. */
    if (!p || p->n <= 1 || rows <= grain) {
        fn(arg, 0, rows);
        return;
    }

    p->fn    = fn;
    p->arg   = arg;
    p->rows  = rows;
    p->grain = grain;
    tt_at64_store(&p->next, 0);
    tt_at32_store(&p->finished, 0);
    tt_at32_add(&p->generation, 1);

    /* Syscall so quando ha alguem dormindo -- no caminho quente o contador e
     * zero e isto e uma leitura atomica. */
    if (tt_at32_load(&p->sleepers) > 0) tt_futex_wake(&p->generation.v);

    drain(p);

    const int others = p->n - 1;
    int spins = 0;
    while (tt_at32_load(&p->finished) < others) {
        /* A chamadora NAO dorme: o trabalho esta em curso e vai acabar em
         * microssegundos. Dormir aqui adiaria a barreira. */
        if (++spins < 512) tt_cpu_relax(); else tt_yield();
    }
}

/* ----------------------------------------------------------- selftest */

typedef struct {
    int32_t *marks;
    int64_t  n;
} mark_arg;

static void mark_rows(void *raw, int64_t row0, int64_t row1)
{
    mark_arg *a = (mark_arg *)raw;
    for (int64_t i = row0; i < row1; i++) a->marks[i] += 1;
}

int tt_pool_selftest(void)
{
    const int64_t N = 10007;              /* primo: nao divide por grain */
    int32_t *marks = (int32_t *)calloc((size_t)N, sizeof(int32_t));
    if (!marks) return tt_fail("pool selftest: sem memoria");

    tt_pool *p = tt_pool_create(0, TT_AFFINITY_PCORES);
    if (!p) { free(marks); return -1; }

    mark_arg a = { marks, N };

    /* Cada linha tem de ser visitada exatamente uma vez, em qualquer
     * granularidade.  E a propriedade de que depende a promessa de que
     * particionar nao muda resultado: sem sobreposicao e sem buraco. */
    const int64_t grains[] = { 1, 7, 64, 4096 };
    for (size_t g = 0; g < sizeof(grains) / sizeof(grains[0]); g++) {
        memset(marks, 0, (size_t)N * sizeof(int32_t));
        tt_pool_run(p, mark_rows, &a, N, grains[g]);
        for (int64_t i = 0; i < N; i++) {
            if (marks[i] != 1) {
                int got = marks[i];
                long long where = (long long)i;
                long long gr = (long long)grains[g];
                tt_pool_destroy(p);
                free(marks);
                return tt_fail("pool: linha %lld visitada %d vez(es) com "
                               "grain %lld", where, got, gr);
            }
        }
    }

    int n = tt_pool_threads(p);
    tt_pool_destroy(p);
    free(marks);

    if (n < 1) return tt_fail("pool: %d threads", n);
    return 0;
}
