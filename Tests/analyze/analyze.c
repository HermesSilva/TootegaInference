/* analyze -- measure first, change the format afterwards.
 *
 * Two ideas for reading fewer bytes per token are on the table, and both would
 * change the container.  Neither gets a line of engine code before this tool
 * says whether the bytes are there to be saved.
 *
 * 1. THE VOCABULARY PROJECTION WITH A SHORTLIST
 *
 *    `lm_head` is [154880][2048] in int8 g64: 337 MB read per token, 14% of the
 *    traffic, to find a handful of candidates.  Split every code into its top
 *    `b` bits and the rest:
 *
 *        q = t * 2^s + l        s = 8 - b,   l in [0, 2^s - 1]
 *
 *    A coarse pass reads only `t` (b bits per weight) and assumes `l` sits at
 *    the centre c = (2^s - 1) / 2.  What it leaves out is bounded, per row and
 *    WITHOUT ANY ASSUMPTION about the data:
 *
 *        |exact - coarse|  <=  c * sum_g |scale_g| * sum_{j in g} |x_j|
 *
 *    So a row whose upper bound is below the best lower bound cannot be the
 *    argmax, and never needs its low bits read.  The rows that survive are
 *    rescored exactly -- same kernel, same bits, same token.  The question this
 *    tool answers is HOW MANY survive, on real hidden states, for each `b`.
 *
 *    The same count is given for a 6-sigma band (treating the low bits as
 *    uniform noise).  That one is NOT a guarantee: it is there to show how much
 *    the rigorous bound leaves on the table, and what an approximate head would
 *    buy if an output change were ever on offer (docs/15: it is not, by default).
 *
 * 2. EXPERTS AS A COMMON BASE PLUS A SMALL DIFFERENCE
 *
 *    A token reads 4 of 64 experts per layer.  If the experts of one layer were
 *    close to each other, the layer could store their mean once and each expert
 *    as a narrow delta: one base read plus four cheap ones.  Whether they are
 *    close is a property of the weights, read straight from the container:
 *
 *        residual = sum_e |W_e - mean|^2 / sum_e |W_e|^2
 *
 *    1 - 1/64 = 0.984 is what 64 UNRELATED matrices give.  A delta needs about
 *    -0.5 * log2(residual) fewer bits per weight than the expert itself, so
 *    anything near 0.98 saves nothing.
 *
 *    The comparison is entry by entry, so it misses experts that are equal up
 *    to a permutation of their hidden units.  A residual near 1 says the cheap
 *    encoding does not exist; it does not say the experts compute different
 *    things.
 *
 *   Tests\analyze.exe models\glm47flash-v4\model.tootega ^
 *       --tokenizer models\v5-sft\tokenizer.json --text-file corpus_prosa.txt
 */
#include "common.h"
#include "compat.h"
#include "forward.h"
#include "kernels.h"
#include "model.h"
#include "pool.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================ lm_head */

#define NB 6                              /* top-bit widths tried: 1..6 */
static const int BITS[NB] = { 1, 2, 3, 4, 5, 6 };

typedef struct {
    const int8_t *w;                      /* [V][cols] */
    const float  *scales;                 /* [V][ng] */
    int64_t       cols, ng;
    int           group;

    const float  *x;                      /* the hidden state, [cols] */
    const double *S, *A, *Q;              /* per group: sum x, sum |x|, sum x^2 */

    float *coarse[NB];                    /* [V] each */
    float *exact, *bound1, *sigma1;       /* [V]; bound and sigma for c = 1 */
} head_job;

static void head_rows(void *raw, int64_t r0, int64_t r1)
{
    const head_job *j = (const head_job *)raw;
    const int G = j->group;

    for (int64_t r = r0; r < r1; r++) {
        const int8_t *wr = j->w + r * j->cols;
        const float  *sr = j->scales + r * j->ng;
        double ex = 0.0, bd = 0.0, sg = 0.0, co[NB] = { 0 };

        for (int64_t g = 0; g < j->ng; g++) {
            const int8_t *p  = wr + g * G;
            const float  *xv = j->x + g * G;
            const double  s  = (double)sr[g];

            double dot = 0.0, t[NB] = { 0 };
            for (int k = 0; k < G; k++) {
                const double xk = (double)xv[k];
                /* u = q + 128 is unsigned, so the shift is a true floor
                 * division and does not depend on how the compiler shifts a
                 * negative int8. */
                const unsigned u = (unsigned)((int)p[k] + 128);
                dot += (double)p[k] * xk;
                for (int b = 0; b < NB; b++)
                    t[b] += (double)(u >> (8 - BITS[b])) * xk;
            }
            ex += s * dot;
            bd += fabs(s) * j->A[g];
            sg += s * s * j->Q[g];

            for (int b = 0; b < NB; b++) {
                const int    sh   = 8 - BITS[b];
                const double step = (double)(1 << sh);
                const double c    = (step - 1.0) / 2.0;
                /* floor(q / 2^sh) = (u >> sh) - (128 >> sh), exactly. */
                const double tq = t[b] - (double)(128 >> sh) * j->S[g];
                co[b] += s * (step * tq + c * j->S[g]);
            }
        }

        j->exact[r]  = (float)ex;
        j->bound1[r] = (float)bd;
        j->sigma1[r] = (float)sqrt(sg);
        for (int b = 0; b < NB; b++) j->coarse[b][r] = (float)co[b];
    }
}

static int desc_float(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return (x < y) - (x > y);
}

static int asc_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* How many rows a band of half-width `half[i]` around `coarse[i]` cannot rule
 * out of the top `k`.  `scratch` is [V]. */
static int64_t survivors(const float *coarse, const float *unit, double c,
                         int64_t V, int k, float *scratch)
{
    for (int64_t i = 0; i < V; i++)
        scratch[i] = coarse[i] - (float)(c * (double)unit[i]);
    qsort(scratch, (size_t)V, sizeof(float), desc_float);
    const float thr = scratch[k - 1];           /* the k-th best LOWER bound */

    int64_t n = 0;
    for (int64_t i = 0; i < V; i++)
        if (coarse[i] + (float)(c * (double)unit[i]) >= thr) n++;
    return n;
}

typedef struct { double *v; int n; } series;

static void push(series *s, double v) { s->v[s->n++] = v; }

static void summary(series *s, double *mean, double *med, double *mx)
{
    double sum = 0;
    qsort(s->v, (size_t)s->n, sizeof(double), asc_double);
    for (int i = 0; i < s->n; i++) sum += s->v[i];
    *mean = s->n ? sum / s->n : 0.0;
    *med  = s->n ? s->v[s->n / 2] : 0.0;
    *mx   = s->n ? s->v[s->n - 1] : 0.0;
}

static int analyze_head(tt_state *s, const tt_model *m, tt_pool *pool,
                        const int32_t *ids, int64_t n_ids, int samples, int topk)
{
    const tt_tensor *t = m->lm_head;
    if (t->kind != TT_KIND_INT8 || !t->scales || t->group <= 0)
        return tt_fail("lm_head is %s; this measurement is written for int8 "
                       "with group scales", tt_kind_name(t->kind));

    const int64_t V = t->rows, cols = t->cols, ng = cols / t->group;
    if (topk < 1) topk = 1;
    if (topk > V) topk = (int)V;

    /* Positions: skip the first 16 (a state with almost no context is not what
     * decode looks like), then one sample every `gap`. */
    const int64_t first = 16;
    if (n_ids <= first + 1) return tt_fail("the text is too short: %lld tokens", (long long)n_ids);
    int64_t gap = (n_ids - first) / samples;
    if (gap < 1) { gap = 1; samples = (int)(n_ids - first); }

    head_job j;
    memset(&j, 0, sizeof(j));
    j.w = (const int8_t *)t->data; j.scales = t->scales;
    j.cols = cols; j.ng = ng; j.group = t->group;

    double *S = (double *)malloc((size_t)ng * 3 * sizeof(double));
    float  *buf = (float *)malloc((size_t)V * (NB + 4) * sizeof(float));
    double *store = (double *)malloc((size_t)samples * (NB * 5) * sizeof(double));
    if (!S || !buf || !store) return tt_fail("no memory");
    j.S = S; j.A = S + ng; j.Q = S + 2 * ng;
    for (int b = 0; b < NB; b++) j.coarse[b] = buf + (size_t)b * V;
    j.exact = buf + (size_t)NB * V;
    j.bound1 = buf + (size_t)(NB + 1) * V;
    j.sigma1 = buf + (size_t)(NB + 2) * V;
    float *scratch = buf + (size_t)(NB + 3) * V;

    /* [b][metric]: argmax rigorous, top-k rigorous, argmax 6s, top-k 6s, and
     * how much of the rigorous bound the real error used. */
    series ser[NB][5];
    for (int b = 0; b < NB; b++)
        for (int k = 0; k < 5; k++) {
            ser[b][k].v = store + (size_t)(b * 5 + k) * samples;
            ser[b][k].n = 0;
        }

    int64_t violations = 0, disagree = 0, missed = 0, done = 0;

    tt_state_reset(s);
    for (int64_t i = 0; i < n_ids && done < samples; i++) {
        const int take = (i >= first) && ((i - first) % gap == 0);
        if (!take) { if (tt_forward_prefill(s, ids[i]) != 0) return -1; continue; }

        float *logits = NULL;
        if (tt_forward(s, ids[i], &logits) != 0) return -1;

        /* After the pass `s->hn` is the input of lm_head: the final hidden
         * state, already through the last norm. */
        j.x = s->hn;
        for (int64_t g = 0; g < ng; g++) {
            double sx = 0, ax = 0, qx = 0;
            for (int k = 0; k < t->group; k++) {
                const double v = (double)s->hn[g * t->group + k];
                sx += v; ax += fabs(v); qx += v * v;
            }
            S[g] = sx; S[ng + g] = ax; S[2 * ng + g] = qx;
        }
        tt_pool_run(pool, head_rows, &j, V, 256);

        /* The engine's argmax and ours must be the same row: this recomputes
         * the projection in double, and a disagreement beyond a tie means the
         * tool is not reading the tensor the way the kernel does. */
        int64_t am = 0, am_engine = 0;
        for (int64_t r = 1; r < V; r++) {
            if (j.exact[r] > j.exact[am]) am = r;
            if (logits[r] > logits[am_engine]) am_engine = r;
        }
        if (am != am_engine) disagree++;

        for (int b = 0; b < NB; b++) {
            const int    sh = 8 - BITS[b];
            const double c  = ((double)(1 << sh) - 1.0) / 2.0;
            /* Standard deviation of a uniform integer on [0, 2^sh - 1]. */
            const double sd = sqrt(((double)(1 << sh) * (double)(1 << sh) - 1.0) / 12.0);

            double used = 0.0;
            for (int64_t r = 0; r < V; r++) {
                const double err = fabs((double)j.exact[r] - (double)j.coarse[b][r]);
                const double bd  = c * (double)j.bound1[r];
                if (err > bd * (1.0 + 1e-4) + 1e-3) violations++;
                if (bd > 0) used += err / bd;
            }
            push(&ser[b][4], used / (double)V);

            push(&ser[b][0], (double)survivors(j.coarse[b], j.bound1, c, V, 1, scratch));
            push(&ser[b][1], (double)survivors(j.coarse[b], j.bound1, c, V, topk, scratch));
            push(&ser[b][2], (double)survivors(j.coarse[b], j.sigma1, 6.0 * sd, V, 1, scratch));
            push(&ser[b][3], (double)survivors(j.coarse[b], j.sigma1, 6.0 * sd, V, topk, scratch));

            /* The rigorous shortlist must contain the true argmax, always. */
            {
                float best_lo = -INFINITY;
                for (int64_t r = 0; r < V; r++) {
                    const float lo = j.coarse[b][r] - (float)(c * (double)j.bound1[r]);
                    if (lo > best_lo) best_lo = lo;
                }
                if (j.coarse[b][am] + (float)(c * (double)j.bound1[am]) < best_lo) missed++;
            }
        }
        done++;
        fprintf(stderr, "\r  lm_head: sample %lld of %d", (long long)done, samples);
    }
    fprintf(stderr, "\n");

    const double row_bytes  = (double)cols;
    const double sc_bytes   = (double)V * (double)ng * sizeof(float);
    const double full_bytes = (double)V * row_bytes + sc_bytes;

    printf("\n1. lm_head with a shortlist  --  [%lld][%lld] int8 g%d, %.0f MB per token today\n",
           (long long)V, (long long)cols, t->group, full_bytes / 1e6);
    printf("   %lld real hidden states, one every %lld positions after the first %lld\n\n",
           (long long)done, (long long)gap, (long long)first);

    printf("   top    coarse    ---- rows that must be rescored (of %lld) ----    bound    bytes per token,\n", (long long)V);
    printf("   bits   pass      RIGOROUS            6-sigma (not exact)    used     rigorous argmax\n");
    printf("          MB        argmax    top-%-3d   argmax    top-%-3d              mean      vs today\n", topk, topk);

    for (int b = 0; b < NB; b++) {
        double mean[5], med[5], mx[5];
        for (int k = 0; k < 5; k++) summary(&ser[b][k], &mean[k], &med[k], &mx[k]);

        const double coarse_bytes = (double)V * row_bytes * BITS[b] / 8.0 + sc_bytes;
        /* A survivor needs the bits the coarse pass did not read. */
        const double rescore = mean[0] * row_bytes * (8 - BITS[b]) / 8.0;
        const double total = coarse_bytes + rescore;

        printf("   %d      %6.0f    %7.0f   %7.0f   %7.0f   %7.0f       %5.1f%%   %6.0f MB  %5.1f%%\n",
               BITS[b], coarse_bytes / 1e6, mean[0], mean[1], mean[2], mean[3],
               100.0 * mean[4], total / 1e6, 100.0 * total / full_bytes);
    }

    printf("\n   worst case of the rigorous argmax shortlist, per width:");
    for (int b = 0; b < NB; b++) {
        double mean, med, mx;
        summary(&ser[b][0], &mean, &med, &mx);
        printf("  %d:%.0f", BITS[b], mx);
    }
    printf("\n");

    printf("\n   checks   bound violated on %lld rows (must be 0)\n", (long long)violations);
    printf("            true argmax outside the rigorous shortlist %lld times (must be 0)\n", (long long)missed);
    printf("            argmax in double differs from the engine's on %lld of %lld samples "
           "(ties only)\n", (long long)disagree, (long long)done);
    printf("   \"bound used\" is the real error over the rigorous bound: the lower it "
           "is,\n   the more the guarantee costs compared with what the data needs.\n");

    free(S); free(buf); free(store);
    return (violations || missed) ? tt_fail("the bound did not hold: the tool is wrong, "
                                            "do not use these numbers") : 0;
}

/* ================================================================ experts */

typedef struct {
    const tt_tensor **experts;            /* [E] */
    const tt_tensor  *shared;
    int64_t           E, cols;
    const int64_t    *rows;               /* sampled row indices */

    /* Accumulated per chunk and merged under the caller: one slot per sampled
     * row, so the sum does not depend on how the pool split the work. */
    double *sum_w2, *sum_res, *sum_mean2, *sum_cos;
    int     failed;
} expert_job;

static void expert_rows(void *raw, int64_t r0, int64_t r1)
{
    expert_job *j = (expert_job *)raw;
    const int64_t E = j->E, C = j->cols;

    float  *w    = (float *)malloc((size_t)(E + 1) * (size_t)C * sizeof(float));
    double *mean = (double *)malloc((size_t)C * sizeof(double));
    if (!w || !mean) { j->failed = 1; free(w); free(mean); return; }
    float *sh = w + (size_t)E * C;

    for (int64_t i = r0; i < r1; i++) {
        const int64_t row = j->rows[i];
        for (int64_t e = 0; e < E; e++)
            if (tt_dequant_row(w + (size_t)e * C, j->experts[e], row) != 0) j->failed = 1;
        if (tt_dequant_row(sh, j->shared, row) != 0) j->failed = 1;

        for (int64_t c = 0; c < C; c++) mean[c] = 0.0;
        for (int64_t e = 0; e < E; e++)
            for (int64_t c = 0; c < C; c++) mean[c] += (double)w[(size_t)e * C + c];
        for (int64_t c = 0; c < C; c++) mean[c] /= (double)E;

        double sh2 = 0.0, m2 = 0.0, w2 = 0.0, res = 0.0, cs = 0.0;
        for (int64_t c = 0; c < C; c++) { sh2 += (double)sh[c] * sh[c]; m2 += mean[c] * mean[c]; }

        for (int64_t e = 0; e < E; e++) {
            const float *we = w + (size_t)e * C;
            double n2 = 0.0, r2 = 0.0, dsh = 0.0;
            for (int64_t c = 0; c < C; c++) {
                const double v = (double)we[c], d = v - mean[c];
                n2 += v * v; r2 += d * d; dsh += v * (double)sh[c];
            }
            w2 += n2; res += r2;
            if (n2 > 0 && sh2 > 0) cs += dsh / sqrt(n2 * sh2);
        }
        j->sum_w2[i] = w2; j->sum_res[i] = res; j->sum_mean2[i] = m2; j->sum_cos[i] = cs;
    }
    free(w); free(mean);
}

static int analyze_experts(const tt_model *m, tt_pool *pool, int n_rows, int layer_step)
{
    const tt_config *c = &m->cfg;
    const int64_t E = c->n_routed_experts, K = c->num_experts_per_tok;
    static const char *KIND[3] = { "gate", "up", "down" };

    printf("\n2. experts of one layer against their own mean  --  %lld experts, %lld read per token\n",
           (long long)E, (long long)K);
    printf("   %d rows sampled per matrix; unrelated experts give a residual of %.3f\n\n",
           n_rows, 1.0 - 1.0 / (double)E);
    printf("   layer   matrix   residual   pairwise corr   cos to shared   bits saved by a delta\n");

    double tot_res[3] = { 0 }, tot_w2[3] = { 0 };
    double worst = 1e9; int worst_layer = -1, worst_kind = -1;

    for (int64_t li = 0; li < c->num_hidden_layers; li++) {
        const tt_layer *L = &m->layers[li];
        if (!L->is_moe) continue;
        const int show = ((li % layer_step) == 0) || li == c->num_hidden_layers - 1;

        for (int k = 0; k < 3; k++) {
            const tt_tensor **ex = k == 0 ? L->e_gate : k == 1 ? L->e_up : L->e_down;
            const tt_tensor  *sh = k == 0 ? L->sh_gate : k == 1 ? L->sh_up : L->sh_down;

            const int64_t R = ex[0]->rows, C = ex[0]->cols;
            const int nr = n_rows < R ? n_rows : (int)R;

            int64_t *rows = (int64_t *)malloc((size_t)nr * sizeof(int64_t));
            double  *acc  = (double *)calloc((size_t)nr * 4, sizeof(double));
            if (!rows || !acc) return tt_fail("no memory");
            for (int i = 0; i < nr; i++) rows[i] = (int64_t)i * R / nr;   /* evenly spread */

            expert_job j;
            memset(&j, 0, sizeof(j));
            j.experts = ex; j.shared = sh; j.E = E; j.cols = C; j.rows = rows;
            j.sum_w2 = acc; j.sum_res = acc + nr; j.sum_mean2 = acc + 2 * nr; j.sum_cos = acc + 3 * nr;
            tt_pool_run(pool, expert_rows, &j, nr, 2);
            if (j.failed) { free(rows); free(acc); return tt_fail("could not read the experts of layer %lld", (long long)li); }

            double w2 = 0, res = 0, m2 = 0, cs = 0;
            for (int i = 0; i < nr; i++) { w2 += acc[i]; res += acc[nr + i]; m2 += acc[2 * nr + i]; cs += acc[3 * nr + i]; }
            free(rows); free(acc);

            const double rho = w2 > 0 ? res / w2 : 0.0;
            /* |sum_e W_e|^2 = E^2 |mean|^2; taking out the E self-products
             * leaves the E(E-1) cross ones, normalised by the mean |W_e|^2. */
            const double pair = w2 > 0
                ? ((double)(E * E) * m2 - w2) / ((double)(E * (E - 1)) * (w2 / (double)E)) : 0.0;
            const double bits = rho > 0 && rho < 1 ? -0.5 * log(rho) / log(2.0) : 0.0;

            tot_res[k] += res; tot_w2[k] += w2;
            if (rho < worst) { worst = rho; worst_layer = (int)li; worst_kind = k; }

            if (show)
                printf("   %5lld   %-6s   %8.4f   %+13.4f   %+13.4f   %8.3f\n",
                       (long long)li, KIND[k], rho, pair, cs / ((double)nr * (double)E), bits);
        }
        fprintf(stderr, "\r  experts: layer %lld", (long long)li);
    }
    fprintf(stderr, "\n");

    printf("\n   whole model");
    double all_res = 0, all_w2 = 0;
    for (int k = 0; k < 3; k++) {
        printf("   %s %.4f", KIND[k], tot_w2[k] > 0 ? tot_res[k] / tot_w2[k] : 0.0);
        all_res += tot_res[k]; all_w2 += tot_w2[k];
    }
    const double rho = all_w2 > 0 ? all_res / all_w2 : 1.0;
    const double bits = rho > 0 && rho < 1 ? -0.5 * log(rho) / log(2.0) : 0.0;
    printf("\n   the most similar: layer %d %s, residual %.4f\n", worst_layer,
           worst_kind >= 0 ? KIND[worst_kind] : "?", worst);

    /* What the encoding would read per layer: the base once, K deltas with
     * `bits` fewer bits each -- against K experts at 4 bits today.  Scales are
     * left out of both sides. */
    const double now = (double)K * 4.0, then = 4.0 + (double)K * (4.0 - bits);
    printf("\n   a delta would need about %.3f fewer bits per weight than the 4 it has now.\n", bits);
    printf("   base + %lld deltas: %.2f bits against %.2f today -- %+.1f%% of the routed-expert traffic.\n",
           (long long)K, then, now, 100.0 * (then - now) / now);
    printf("   (positive means the base costs more than the deltas save.)\n");
    return 0;
}

/* =================================================================== main */

int main(int argc, char **argv)
{
    const char *path = NULL, *tok_path = NULL, *text_file = NULL;
    int threads = 20, samples = 24, topk = 40, n_rows = 256, layer_step = 5;
    int do_head = 1, do_experts = 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = i + 1 < argc;
        if      (!strcmp(a, "--tokenizer") && has)  tok_path = argv[++i];
        else if (!strcmp(a, "--text-file") && has)  text_file = argv[++i];
        else if (!strcmp(a, "--samples") && has)    samples = atoi(argv[++i]);
        else if (!strcmp(a, "--topk") && has)       topk = atoi(argv[++i]);
        else if (!strcmp(a, "--rows") && has)       n_rows = atoi(argv[++i]);
        else if (!strcmp(a, "--layer-step") && has) layer_step = atoi(argv[++i]);
        else if (!strcmp(a, "--threads") && has)    threads = atoi(argv[++i]);
        else if (!strcmp(a, "--no-head"))           do_head = 0;
        else if (!strcmp(a, "--no-experts"))        do_experts = 0;
        else if (a[0] != '-')                       path = a;
    }
    if (!path || (do_head && (!tok_path || !text_file))) {
        fprintf(stderr,
            "usage: analyze <model.tootega> --tokenizer <tokenizer.json> --text-file <text>\n"
            "               [--samples 24] [--topk 40]    lm_head: hidden states sampled, k of top-k\n"
            "               [--rows 256] [--layer-step 5] experts: rows per matrix, layers printed\n"
            "               [--no-head] [--no-experts] [--threads 20]\n");
        return 1;
    }
    if (samples < 1) samples = 1;
    if (n_rows < 1) n_rows = 1;
    if (layer_step < 1) layer_step = 1;

    enum { MAX_IDS = 2048 };
    static int32_t ids[MAX_IDS];
    int64_t n_ids = 0;
    tt_tokenizer tk;

    if (do_head) {
        if (tt_tokenizer_load(&tk, tok_path, 154880) != 0) {
            fprintf(stderr, "error: %s\n", tt_error()); return 1;
        }
        FILE *f = fopen(text_file, "rb");
        if (!f) { fprintf(stderr, "error: cannot read %s\n", text_file); return 1; }
        static char text[65536];
        const size_t got = fread(text, 1, sizeof(text) - 1, f);
        fclose(f);
        text[got] = '\0';
        n_ids = tt_encode(&tk, text, got, ids, MAX_IDS);
        if (n_ids <= 0) { fprintf(stderr, "error: the text did not encode\n"); return 1; }

        /* No more context than the samples need. */
        const int64_t want = 16 + (int64_t)samples * 8;
        if (n_ids > want) n_ids = want;
    }

    const int64_t ctx = n_ids > 0 ? n_ids : 1;
    tt_model m;
    if (tt_model_open(&m, path, ctx, TT_ATTN_ABSORBED, TT_KV_F32) != 0) {
        fprintf(stderr, "error: %s\n", tt_error()); return 1;
    }

    tt_pool *pool = tt_pool_create(threads, TT_AFFINITY_ALL);
    tt_kernels_set_pool(pool);

    printf("\n%s\n  build %s, %d threads\n", path, tt_build_id(), tt_pool_threads(pool));

    int rc = 0;
    if (do_head) {
        tt_state s;
        if (tt_state_init(&s, &m, ctx) != 0) rc = -1;
        else {
            rc = analyze_head(&s, &m, pool, ids, n_ids, samples, topk);
            tt_state_free(&s);
        }
    }
    if (rc == 0 && do_experts) rc = analyze_experts(&m, pool, n_rows, layer_step);
    if (rc != 0) fprintf(stderr, "error: %s\n", tt_error());
    printf("\n");

    tt_kernels_set_pool(NULL);
    tt_pool_destroy(pool);
    tt_model_close(&m);
    return rc ? 1 : 0;
}
