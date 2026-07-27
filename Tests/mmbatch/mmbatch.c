/* mmbatch -- what the batched matmul is worth, on the real tensors.
 *
 * Decode reads one weight per token and throws it away: it is memory-bound,
 * ~33 ms of memory against 4-7 ms of compute (docs/08).  Prefill and training
 * are the opposite regime -- many inputs for the same weight -- and the batched
 * kernel exists to stop paying the read `batch` times.
 *
 * Two numbers matter here and they are not the same:
 *
 *   speed-up   how much faster than the loop of single calls.  It is bounded
 *              by whether the weight was already in cache: a tensor that fits
 *              in L2 has nothing to save, and the gain there comes only from
 *              not re-unpacking int4.
 *
 *   GFLOP/s    the absolute rate.  It is the number the training estimate
 *              depends on, and the plan carried it as `[estimate] 137-300`
 *              until this tool ran.
 *
 * Bit-exactness is checked on every case, not assumed: the batched result has
 * to match the loop of singles BIT FOR BIT, or batched prefill changes tokens
 * and the 32/32 criterion falls by construction (docs/15).
 *
 *   Tests\mmbatch.exe models\glm47flash-v3\model.tootega
 */
#include "common.h"
#include "compat.h"
#include "container.h"
#include "kernels.h"
#include "pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BATCH 64

/* The shapes that decide the bill, from docs/04 and docs/08.  Named rather than
 * synthetic on purpose: a benchmark over invented shapes measures the
 * benchmark. */
static const char *TENSORS[] = {
    "model.layers.1.self_attn.q_b_proj.weight",       /* [5120, 768]   int8 */
    "model.layers.1.self_attn.kv_b_proj.weight",      /* [8960, 512]   int8 */
    "model.layers.1.self_attn.o_proj.weight",         /* [2048, 5120]  int8 */
    "model.layers.1.mlp.experts.0.gate_proj.weight",  /* [1536, 2048]  int4 */
    "model.layers.1.mlp.shared_experts.gate_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",            /* [10240, 2048] int4 */
    "lm_head.weight",                                 /* [154880,2048] int8 */
};

static double bytes_of(const tt_tensor *t)
{
    return (double)(t->data_bytes + t->scales_bytes);
}

static void fill_x(float *x, int64_t n, uint32_t seed)
{
    /* Deterministic and cheap; the values only have to exercise the FMA, not
     * be statistically anything. */
    uint32_t s = seed | 1u;
    for (int64_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = (float)((int32_t)(s >> 8) % 2001 - 1000) * 0.001f;
    }
}

static int run_tensor(const tt_container *c, const char *name,
                      const int *batches, int n_batches, int repeat)
{
    const tt_tensor *t = tt_container_find(c, name);
    if (!t) { tt_log("  %-46s AUSENTE", name); return 0; }
    if (t->ndim < 2) { tt_log("  %-46s nao e matriz", name); return 0; }

    const int64_t rows = t->rows, cols = t->cols;
    char kb[32];
    tt_log("");
    tt_log("  %s", name);
    tt_log("    %s [%lld, %lld], %s", tt_kind_name(t->kind),
           (long long)rows, (long long)cols,
           tt_bytes(kb, sizeof(kb), (uint64_t)bytes_of(t)));

    float *x = (float *)tt_alloc_aligned(sizeof(float) * (size_t)(cols * MAX_BATCH), 64);
    float *ya = (float *)tt_alloc_aligned(sizeof(float) * (size_t)(rows * MAX_BATCH), 64);
    float *yb = (float *)tt_alloc_aligned(sizeof(float) * (size_t)(rows * MAX_BATCH), 64);
    if (!x || !ya || !yb) {
        tt_free_aligned(x); tt_free_aligned(ya); tt_free_aligned(yb);
        return tt_fail("sem memoria para %lld x %d", (long long)cols, MAX_BATCH);
    }
    fill_x(x, cols * MAX_BATCH, 12345u);

    /* Residency first: the first pass would otherwise measure page faults.
     * docs/11 records the same effect at 5x on the first token. */
    tt_map_touch(&c->map, t->data_off, t->data_bytes);
    if (t->scales_bytes) tt_map_touch(&c->map, t->scales_off, t->scales_bytes);

    tt_log("    %5s %10s %10s %8s %9s %9s", "lote", "simples", "lote", "ganho",
           "GFLOP/s", "GB/s");

    for (int k = 0; k < n_batches; k++) {
        const int B = batches[k];
        if (B > MAX_BATCH) continue;

        /* The WHOLE-tensor entry points, not the row-range ones: those do not
         * touch the pool, and the first version of this tool measured one
         * thread twice and reported it as eight. */
        for (int b = 0; b < B; b++)
            tt_matmul(ya + (int64_t)b * rows, t, x + (int64_t)b * cols);
        tt_matmul_b(yb, rows, t, x, cols, B);

        int64_t bad = -1;
        for (int64_t i = 0; i < rows * B; i++)
            if (memcmp(&ya[i], &yb[i], sizeof(float)) != 0) { bad = i; break; }
        if (bad >= 0) {
            tt_free_aligned(x); tt_free_aligned(ya); tt_free_aligned(yb);
            return tt_fail("%s lote %d: y[%lld] difere (simples %.9g, lote %.9g)",
                           name, B, (long long)bad, (double)ya[bad], (double)yb[bad]);
        }

        double t0 = tt_now();
        for (int r = 0; r < repeat; r++)
            for (int b = 0; b < B; b++)
                tt_matmul(ya + (int64_t)b * rows, t, x + (int64_t)b * cols);
        const double ts = (tt_now() - t0) / repeat;

        t0 = tt_now();
        for (int r = 0; r < repeat; r++)
            tt_matmul_b(yb, rows, t, x, cols, B);
        const double tb = (tt_now() - t0) / repeat;

        const double flop = 2.0 * (double)rows * (double)cols * B;
        tt_log("    %5d %9.3f %9.3f %7.2fx %9.1f %9.1f",
               B, ts * 1e3, tb * 1e3, ts / (tb > 0 ? tb : 1e-9),
               flop / (tb > 0 ? tb : 1e-9) / 1e9,
               bytes_of(t) / (tb > 0 ? tb : 1e-9) / 1e9);
    }

    tt_free_aligned(x); tt_free_aligned(ya); tt_free_aligned(yb);
    return 0;
}

static void usage(void)
{
    fprintf(stderr,
      "mmbatch " TT_VERSION " -- batched matmul against the loop of single calls\n"
      "\n"
      "usage: Tests\\mmbatch.exe <model.tootega> [options]\n"
      "\n"
      "  --batch 1,2,4,8,16,32   batch sizes to measure (max %d)\n"
      "  --repeat N              timed passes per point (default: 5)\n"
      "  --threads N             default: physical P-cores; 1 measures the kernel alone\n"
      "  --affinity p-cores|all|none   default: p-cores\n"
      "  --kernel auto|scalar    default: auto (AVX2+FMA)\n"
      "  -v, --verbose\n", MAX_BATCH);
}

int main(int argc, char **argv)
{
    const char *path = NULL;
    int batches[16] = { 1, 2, 4, 8, 16, 32 };
    int n_batches = 6, repeat = 5, threads = 0;
    tt_impl impl = TT_IMPL_AUTO;
    /* docs/09 pins to P-cores because a MEMORY-bound matmul gains nothing from
     * the E-cores.  Batched matmul is compute-bound, so the premise does not
     * carry over and the flag exists to measure it instead of assuming it. */
    tt_affinity aff = TT_AFFINITY_PCORES;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has = (i + 1 < argc);
        if      (!strcmp(a, "-v") || !strcmp(a, "--verbose")) tt_verbose(1);
        else if (!strcmp(a, "--repeat") && has)  repeat = atoi(argv[++i]);
        else if (!strcmp(a, "--threads") && has) threads = atoi(argv[++i]);
        else if (!strcmp(a, "--affinity") && has) {
            if (tt_affinity_parse(argv[++i], &aff) != 0) { usage(); return 2; }
        }
        else if (!strcmp(a, "--kernel") && has) {
            const char *v = argv[++i];
            if      (!strcmp(v, "scalar")) impl = TT_IMPL_SCALAR;
            else if (!strcmp(v, "auto"))   impl = TT_IMPL_AUTO;
            else { usage(); return 2; }
        } else if (!strcmp(a, "--batch") && has) {
            n_batches = 0;
            for (const char *p = argv[++i]; *p && n_batches < 16; ) {
                batches[n_batches++] = atoi(p);
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
        } else if (a[0] == '-') { usage(); return 2; }
        else if (!path) path = a;
        else { usage(); return 2; }
    }
    if (!path) { usage(); return 2; }
    if (repeat < 1) repeat = 1;

    tt_kernels_set_impl(impl);

    tt_container c;
    if (tt_container_open(&c, path) != 0) {
        fprintf(stderr, "error: %s\n", tt_error());
        return 1;
    }

    tt_pool *pool = NULL;
    int used = 1;
    if (threads != 1) {
        pool = tt_pool_create(threads, aff);
        if (!pool) {
            fprintf(stderr, "error: %s\n", tt_error());
            tt_container_close(&c);
            return 1;
        }
        tt_kernels_set_pool(pool);
        used = tt_pool_threads(pool);
    }

    tt_log("mmbatch %s  build %s", TT_VERSION, tt_build_id());
    tt_log("  %s", path);
    tt_log("  kernel %s, %d thread(s), %d repeticao(oes)",
           tt_kernels_impl_name(), used, repeat);

    int rc = 0;
    for (size_t i = 0; i < sizeof(TENSORS) / sizeof(TENSORS[0]); i++)
        if ((rc = run_tensor(&c, TENSORS[i], batches, n_batches, repeat)) != 0)
            break;

    if (pool) { tt_kernels_set_pool(NULL); tt_pool_destroy(pool); }
    tt_container_close(&c);

    if (rc != 0) { fprintf(stderr, "error: %s\n", tt_error()); return 1; }
    return 0;
}
