#include "budget.h"
#include "common.h"
#include "compat.h"
#include "kernels.h"
#include "pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* docs/09: the `auto` default leaves 8 GB of headroom.  It is not a numerical
 * safety margin -- it is the room in which the OS, the browser and the
 * person's compiler go on existing while the engine runs. */
#define TT_RAM_SLACK_GB 8.0

void tt_plan_compute(tt_plan *p, const tt_model *m, int64_t ctx, double ram_gb)
{
    memset(p, 0, sizeof(*p));

    p->ram_total_gb = tt_ram_total_gb();
    p->ram_avail_gb = tt_ram_avail_gb();
    p->budget_gb = ram_gb > 0 ? ram_gb : p->ram_total_gb - TT_RAM_SLACK_GB;
    if (p->budget_gb < 0) p->budget_gb = 0;

    p->weights = tt_model_weight_bytes(m);
    p->kv      = tt_kv_bytes(m, ctx);
    p->scratch = tt_scratch_bytes(m, ctx);
    p->total   = p->weights + p->kv + p->scratch;

    p->fits = ((double)p->total / 1e9) <= p->budget_gb;

    /* How much context the budget still pays for.  It is the number the
     * refusal message needs: saying "does not fit" without saying what to
     * problema em vez de resolver. */
    const uint64_t kv_per_token = tt_kv_bytes(m, 1);
    const int64_t  budget_bytes = (int64_t)(p->budget_gb * 1e9);
    const int64_t  fixed = (int64_t)(p->weights + tt_scratch_bytes(m, 1));
    p->ctx_max = kv_per_token ? (budget_bytes - fixed) / (int64_t)kv_per_token : 0;
    if (p->ctx_max < 0) p->ctx_max = 0;
}

void tt_plan_print(const tt_plan *p, const tt_model *m, int64_t ctx, int threads)
{
    char b[64];
    const tt_config *c = &m->cfg;

    printf("\n  contexto     %lld posicoes\n", (long long)ctx);
    printf("  atencao      %s\n",
           m->attn == TT_ATTN_ABSORBED ? "absorvida (KV comprimido)"
                                       : "expandida (formulacao do oraculo)");
    printf("  threads      %d\n\n", threads);

    printf("  pesos        %14s   residentes, mapeados do container\n",
           tt_bytes(b, sizeof(b), p->weights));
    char per[64];
    printf("  kv           %14s   %s per position\n",
           tt_bytes(b, sizeof(b), p->kv),
           tt_bytes(per, sizeof(per), tt_kv_bytes(m, 1)));
    printf("  scratch      %14s   pass buffers, allocated once\n",
           tt_bytes(b, sizeof(b), p->scratch));
    printf("  ------------------------------\n");
    printf("  total        %14s\n\n", tt_bytes(b, sizeof(b), p->total));

    printf("  orcamento    %14.1f GB\n", p->budget_gb);
    printf("  ram          %14.1f GB totais, %.1f GB disponiveis agora\n",
           p->ram_total_gb, p->ram_avail_gb);
    printf("  max context  %14lld positions within this budget\n",
           (long long)p->ctx_max);

    (void)c;
}

int tt_plan_check(const tt_plan *p, int64_t ctx)
{
    if (p->fits) return 0;

    char b[64], t[64];
    if (p->ctx_max > 0)
        return tt_fail("does not fit: %s requested against %.1f GB of budget. "
                       "Reduce --ctx from %lld to %lld, or raise --ram-gb "
                       "to %.1f",
                       tt_bytes(b, sizeof(b), p->total), p->budget_gb,
                       (long long)ctx, (long long)p->ctx_max,
                       (double)p->total / 1e9);

    return tt_fail("does not fit even at minimum context: the weights alone are %s "
                   "against %.1f GB of budget. This model does not run under this "
                   "limit -- raise --ram-gb to at least %s",
                   tt_bytes(b, sizeof(b), p->weights), p->budget_gb,
                   tt_bytes(t, sizeof(t), p->total));
}

int tt_budget_verify(const tt_plan *p)
{
    /* Estimar e prometer; medir e cumprir.  Se o RSS passou do teto, o plano
     * was wrong -- and knowing that now is worth more than finding out when the
     * comecar a paginar. */
    const double rss = tt_rss_gb();
    if (rss > p->budget_gb)
        return tt_fail("orcamento estourado em execucao: %.1f GB residentes "
                       "contra %.1f GB prometidos", rss, p->budget_gb);
    return 0;
}

/* ---------------------------------------------------------------- CLI */

int tt_cmd_plan(int argc, char **argv)
{
    const char *path = NULL;
    int64_t ctx = 4096;
    double  ram_gb = 0;
    int     threads = 0;
    tt_attn_mode attn = TT_ATTN_ABSORBED;
    tt_kv_prec kv_prec = TT_KV_F32;
    tt_affinity  affinity = TT_AFFINITY_PCORES;

    for (int i = 0; i < argc; i++) {
        const char *a = argv[i];
        int has_next = (i + 1 < argc);
        if (!strcmp(a, "--ctx") && has_next)          ctx = atoll(argv[++i]);
        else if (!strcmp(a, "--ram-gb") && has_next)  ram_gb = atof(argv[++i]);
        else if (!strcmp(a, "--threads") && has_next) threads = atoi(argv[++i]);
        /* Both names: `plan` was born with `--kv` and the other commands with
         * `--kv-prec`.  A name divergence between commands that answer the
         * same question is a trap -- accepting both costs one line. */
        else if ((!strcmp(a, "--kv") || !strcmp(a, "--kv-prec")) && has_next) {
            if (tt_kv_prec_parse(argv[++i], &kv_prec) != 0) return -1;
        } else if (!strcmp(a, "--attn") && has_next) {
            if (tt_attn_parse(argv[++i], &attn) != 0) return -1;
        } else if (!strcmp(a, "--affinity") && has_next) {
            if (tt_affinity_parse(argv[++i], &affinity) != 0) return -1;
        } else if (a[0] == '-') {
            return tt_fail("plan: opcao desconhecida '%s'", a);
        } else if (!path) {
            path = a;
        } else {
            return tt_fail("plan: argumento extra '%s'", a);
        }
    }

    if (!path) return tt_fail("usage: tootega plan <file.tootega> [--ctx N] "
                              "[--ram-gb X] [--attn absorbed|expanded]");
    if (ctx <= 0) return tt_fail("--ctx has to be positive");

    tt_model m;
    if (tt_model_open(&m, path, ctx, attn, kv_prec) != 0) return -1;

    /* How many threads the pool would use, without creating them. */
    int cpus[128];
    int ncpu = (affinity == TT_AFFINITY_ALL) ? tt_cpu_core_list(cpus, 128)
             : (affinity == TT_AFFINITY_NONE) ? tt_cpu_threads()
             : tt_cpu_pcore_list(cpus, 128);
    if (threads <= 0) threads = ncpu;

    tt_plan p;
    tt_plan_compute(&p, &m, ctx, ram_gb);

    printf("\n%s\n", path);
    tt_plan_print(&p, &m, ctx, threads);

    int rc = tt_plan_check(&p, ctx);
    if (rc == 0) printf("\n  it fits.\n\n");
    else         printf("\n");

    tt_model_close(&m);
    return rc;
}
