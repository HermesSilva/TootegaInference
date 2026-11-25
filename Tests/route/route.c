/* route -- which experts each token actually goes through, and how much the
 * same token varies from one context to another.
 *
 * The traffic arithmetic of docs/08 says "4 of 64 experts per layer" and stops
 * there.  This tool shows WHICH four, layer by layer: the router's choice is the
 * only data-dependent branch in the whole forward, and it decides 41% of the
 * bytes a token reads.
 *
 * It runs the engine's own forward (`src/`), with the routing and the hidden
 * trace on, so what it prints is what `tootega` does and not a reimplementation
 * of the router.
 *
 * Two modes:
 *
 *   --text "..."      one text; JSON on stdout with the route of every token.
 *
 *   --prompts <file>  one prompt per line, EACH IN A FRESH CONTEXT.  Keeps the
 *                     hidden state of every token at every layer boundary and
 *                     compares the occurrences of the same id across prompts:
 *                     how close the states are (cosine) and how many of the
 *                     four experts they share, per layer -- against pairs of
 *                     DIFFERENT ids as the baseline, because a similarity
 *                     without a baseline says nothing.
 *
 * Layer boundary 0 is the embedding: a pure function of the id, so two
 * occurrences of one id start IDENTICAL there.  What the measurement shows is
 * how fast context pulls them apart.
 *
 *   Tests\route.exe models\glm47flash-v4\model.tootega ^
 *       --tokenizer models\v5-sft\tokenizer.json ^
 *       --prompts Tests\route\prompts.pt.txt --json route_vary.json
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

#define MAX_PROMPTS 64
#define MAX_IDS     1024

static void json_string(FILE *f, const char *s)
{
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
        else if (*p == '\n') fputs("\\n", f);
        else if (*p < 0x20) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
}

static void piece_of(const tt_tokenizer *tk, int32_t id, char *buf, size_t cap)
{
    int64_t w = tt_decode(tk, &id, 1, buf, (int64_t)cap - 1);
    buf[w > 0 ? w : 0] = '\0';
}

/* How many experts two rows of K have in common.  Rows are ascending. */
static int shared_experts(const int32_t *a, const int32_t *b, int64_t K)
{
    int n = 0;
    for (int64_t i = 0; i < K; i++)
        for (int64_t j = 0; j < K; j++)
            if (a[i] >= 0 && a[i] == b[j]) n++;
    return n;
}

/* ------------------------------------------------------------ one text */

static int run_text(tt_state *s, const tt_model *m, const tt_tokenizer *tk,
                    const char *path, const int32_t *ids, int64_t n)
{
    const tt_config *c = &m->cfg;
    const int64_t L = c->num_hidden_layers, K = c->num_experts_per_tok;
    char piece[256];

    printf("{\n  \"model\": "); json_string(stdout, path);
    printf(",\n  \"build\": \"%s\",\n  \"layers\": %lld, \"experts\": %lld, "
           "\"per_token\": %lld, \"shared\": %lld,\n  \"tokens\": [\n",
           tt_build_id(), (long long)L, (long long)c->n_routed_experts,
           (long long)K, (long long)c->n_shared_experts);

    for (int64_t t = 0; t < n; t++) {
        /* The routing does not depend on the vocabulary projection, so no
         * position pays for it. */
        if (tt_forward_prefill(s, ids[t]) != 0) return -1;
        piece_of(tk, ids[t], piece, sizeof(piece));

        printf("    { \"pos\": %lld, \"id\": %d, \"piece\": ", (long long)t, ids[t]);
        json_string(stdout, piece);
        printf(",\n      \"route\": [");
        for (int64_t li = 0; li < L; li++) {
            printf("%s[", li ? "," : "");
            for (int64_t i = 0; i < K; i++)
                printf("%s%d", i ? "," : "", s->route[li * K + i]);
            printf("]");
        }
        printf("],\n      \"weight\": [");
        for (int64_t li = 0; li < L; li++) {
            printf("%s[", li ? "," : "");
            for (int64_t i = 0; i < K; i++)
                printf("%s%.4f", i ? "," : "", (double)s->route_w[li * K + i]);
            printf("]");
        }
        printf("] }%s\n", t + 1 < n ? "," : "");
    }
    printf("  ]\n}\n");
    return 0;
}

/* ------------------------------------------------------- many contexts */

typedef struct { int prompt; int pos; int32_t id; } tokrec;

typedef struct {
    int32_t id;
    int     occ, prompts, last_prompt;
    double  pairs;
    double  cos_first, cos_mid, cos_last;   /* after layer 0, middle, final */
    double  sh_first, sh_all;               /* first MoE layer, all MoE layers */
} idstat;

static int by_occ(const void *a, const void *b)
{
    const idstat *x = (const idstat *)a, *y = (const idstat *)b;
    if (x->prompts != y->prompts) return y->prompts - x->prompts;
    if (x->occ != y->occ) return y->occ - x->occ;
    return (x->id > y->id) - (x->id < y->id);
}

static int run_prompts(tt_state *s, const tt_model *m, const tt_tokenizer *tk,
                       const char *path, const char *file, int32_t **pids,
                       const int64_t *plen, int np, const char *json_path,
                       const char *dump_path)
{
    const tt_config *c = &m->cfg;
    const int64_t L = c->num_hidden_layers, K = c->num_experts_per_tok;
    const int64_t Hd = c->hidden_size, B = L + 1;      /* boundaries */

    int64_t N = 0;
    for (int p = 0; p < np; p++) N += plen[p];

    tokrec  *tok    = (tokrec *)malloc((size_t)N * sizeof(tokrec));
    float   *states = (float *)malloc((size_t)(N * B * Hd) * sizeof(float));
    int32_t *routes = (int32_t *)malloc((size_t)(N * L * K) * sizeof(int32_t));
    float   *norms  = (float *)malloc((size_t)(N * B) * sizeof(float));
    if (!tok || !states || !routes || !norms) return tt_fail("no memory for %lld tokens", (long long)N);

    s->trace = (float *)malloc((size_t)(B * Hd) * sizeof(float));
    if (!s->trace) return tt_fail("no memory");

    int64_t n = 0;
    for (int p = 0; p < np; p++) {
        tt_state_reset(s);                 /* a NEW context: nothing carries over */
        for (int64_t t = 0; t < plen[p]; t++, n++) {
            if (tt_forward_prefill(s, pids[p][t]) != 0) return -1;
            tok[n].prompt = p; tok[n].pos = (int)t; tok[n].id = pids[p][t];
            memcpy(states + n * B * Hd, s->trace, (size_t)(B * Hd) * sizeof(float));
            memcpy(routes + n * L * K, s->route, (size_t)(L * K) * sizeof(int32_t));
            for (int64_t b = 0; b < B; b++) {
                const float *v = states + (n * B + b) * Hd;
                norms[n * B + b] = sqrtf(tt_dot(v, v, Hd));
            }
        }
        fprintf(stderr, "  prompt %d: %lld tokens\n", p + 1, (long long)plen[p]);
    }

    /* Layers that have a router at all. */
    int first_moe = -1;
    for (int64_t li = 0; li < L; li++)
        if (routes[li * K] >= 0) { first_moe = (int)li; break; }

    /* Unique ids, with how many prompts each one shows up in. */
    idstat *ids = (idstat *)calloc((size_t)N, sizeof(idstat));
    int32_t *slot = (int32_t *)malloc((size_t)N * sizeof(int32_t));
    if (!ids || !slot) return tt_fail("no memory");
    int nu = 0;
    for (int64_t i = 0; i < N; i++) {
        int k = 0;
        while (k < nu && ids[k].id != tok[i].id) k++;
        if (k == nu) { ids[nu].id = tok[i].id; ids[nu].last_prompt = -1; nu++; }
        ids[k].occ++;
        if (ids[k].last_prompt != tok[i].prompt) { ids[k].prompts++; ids[k].last_prompt = tok[i].prompt; }
        slot[i] = k;
    }

    /* Every pair of tokens from DIFFERENT prompts, split by "same id or not". */
    double *cos_sum = (double *)calloc((size_t)(2 * B), sizeof(double));
    double *sh_sum  = (double *)calloc((size_t)(2 * L), sizeof(double));
    double *eq_sum  = (double *)calloc((size_t)(2 * L), sizeof(double));
    double  pairs[2] = { 0, 0 };
    if (!cos_sum || !sh_sum || !eq_sum) return tt_fail("no memory");

    const int64_t mid = B / 2;

    for (int64_t a = 0; a < N; a++) {
        for (int64_t b = a + 1; b < N; b++) {
            if (tok[a].prompt == tok[b].prompt) continue;
            const int same = (tok[a].id == tok[b].id);
            pairs[same] += 1.0;
            idstat *st = same ? &ids[slot[a]] : NULL;
            if (st) st->pairs += 1.0;

            for (int64_t l = 0; l < B; l++) {
                const float *va = states + (a * B + l) * Hd;
                const float *vb = states + (b * B + l) * Hd;
                const double den = (double)norms[a * B + l] * (double)norms[b * B + l];
                const double cs = den > 0 ? (double)tt_dot(va, vb, Hd) / den : 0.0;
                cos_sum[same * B + l] += cs;
                if (st) {
                    if (l == 1)     st->cos_first += cs;
                    if (l == mid)   st->cos_mid   += cs;
                    if (l == B - 1) st->cos_last  += cs;
                }
            }
            for (int64_t li = 0; li < L; li++) {
                if (routes[a * L * K + li * K] < 0) continue;
                const int sh = shared_experts(routes + (a * L + li) * K,
                                              routes + (b * L + li) * K, K);
                sh_sum[same * L + li] += sh;
                if (sh == K) eq_sum[same * L + li] += 1.0;
                if (st) {
                    st->sh_all += sh;
                    if (li == first_moe) st->sh_first += sh;
                }
            }
        }
    }

    const int64_t n_moe = L - (first_moe < 0 ? L : first_moe);

    /* ------------------------------------------------------ the report */

    printf("\n%s\n%s\n\n", path, file);
    printf("  %d prompts, %lld tokens, %d distinct ids, build %s\n", np,
           (long long)N, nu, tt_build_id());
    printf("  pairs across prompts: %.0f with the SAME id, %.0f with different ids\n",
           pairs[1], pairs[0]);
    printf("  chance level for shared experts: %.2f of %lld  (K*K/E)\n\n",
           (double)(K * K) / (double)c->n_routed_experts, (long long)K);

    printf("  boundary        cosine of the hidden state       experts in common (of %lld)   identical set\n", (long long)K);
    printf("                  same id    different ids        same id    different ids      same id\n");
    for (int64_t l = 0; l < B; l++) {
        const int show = (l <= 4) || (l % 5 == 0) || (l == B - 1);
        if (!show) continue;
        char name[32];
        if (l == 0) snprintf(name, sizeof(name), "embedding");
        else        snprintf(name, sizeof(name), "after layer %lld", (long long)(l - 1));
        printf("  %-15s %7.4f    %7.4f", name,
               pairs[1] ? cos_sum[B + l] / pairs[1] : 0.0,
               pairs[0] ? cos_sum[l] / pairs[0] : 0.0);
        /* The router of layer l reads the state that came out of layer l-1. */
        if (l < L && routes[l * K] >= 0)
            printf("            %5.2f      %5.2f              %5.1f%%   <- router of layer %lld",
                   pairs[1] ? sh_sum[L + l] / pairs[1] : 0.0,
                   pairs[0] ? sh_sum[l] / pairs[0] : 0.0,
                   pairs[1] ? 100.0 * eq_sum[L + l] / pairs[1] : 0.0, (long long)l);
        printf("\n");
    }

    qsort(ids, (size_t)nu, sizeof(idstat), by_occ);
    char piece[256];
    printf("\n  the ids that recur most           occ  prompts   cos after L0   cos middle   cos final   shared L%d   shared all\n", first_moe);
    for (int k = 0; k < nu && k < 16; k++) {
        const idstat *st = &ids[k];
        if (st->pairs < 1.0) break;
        piece_of(tk, st->id, piece, sizeof(piece));
        printf("  %7d %-22.22s %5d  %5d      %7.4f      %7.4f     %7.4f      %5.2f       %5.2f\n",
               st->id, piece, st->occ, st->prompts,
               st->cos_first / st->pairs, st->cos_mid / st->pairs,
               st->cos_last / st->pairs, st->sh_first / st->pairs,
               n_moe ? st->sh_all / st->pairs / (double)n_moe : 0.0);
    }
    printf("\n");

    /* ------------------------------------------------------------ JSON */

    if (json_path) {
        FILE *f = fopen(json_path, "wb");
        if (!f) return tt_fail("cannot write %s", json_path);
        fprintf(f, "{\n  \"model\": "); json_string(f, path);
        fprintf(f, ",\n  \"build\": \"%s\", \"prompts\": %d, \"tokens\": %lld, "
                   "\"distinct_ids\": %d,\n  \"layers\": %lld, \"experts\": %lld, "
                   "\"per_token\": %lld, \"first_moe\": %d,\n"
                   "  \"pairs_same\": %.0f, \"pairs_diff\": %.0f,\n",
                tt_build_id(), np, (long long)N, nu, (long long)L,
                (long long)c->n_routed_experts, (long long)K, first_moe,
                pairs[1], pairs[0]);

        const char *names[2] = { "diff", "same" };
        for (int w = 0; w < 2; w++) {
            fprintf(f, "  \"cos_%s\": [", names[w]);
            for (int64_t l = 0; l < B; l++)
                fprintf(f, "%s%.5f", l ? "," : "", pairs[w] ? cos_sum[w * B + l] / pairs[w] : 0.0);
            fprintf(f, "],\n  \"shared_%s\": [", names[w]);
            for (int64_t li = 0; li < L; li++)
                fprintf(f, "%s%.4f", li ? "," : "",
                        routes[li * K] < 0 ? -1.0 : (pairs[w] ? sh_sum[w * L + li] / pairs[w] : 0.0));
            fprintf(f, "],\n  \"identical_%s\": [", names[w]);
            for (int64_t li = 0; li < L; li++)
                fprintf(f, "%s%.5f", li ? "," : "",
                        routes[li * K] < 0 ? -1.0 : (pairs[w] ? eq_sum[w * L + li] / pairs[w] : 0.0));
            fprintf(f, "],\n");
        }

        fprintf(f, "  \"ids\": [\n");
        int wrote = 0;
        for (int k = 0; k < nu; k++) {
            const idstat *st = &ids[k];
            if (st->pairs < 1.0) continue;
            piece_of(tk, st->id, piece, sizeof(piece));
            fprintf(f, "%s    { \"id\": %d, \"piece\": ", wrote++ ? ",\n" : "", st->id);
            json_string(f, piece);
            fprintf(f, ", \"occ\": %d, \"prompts\": %d, \"pairs\": %.0f, "
                       "\"cos_first\": %.5f, \"cos_mid\": %.5f, \"cos_last\": %.5f, "
                       "\"shared_first\": %.4f, \"shared_all\": %.4f }",
                    st->occ, st->prompts, st->pairs, st->cos_first / st->pairs,
                    st->cos_mid / st->pairs, st->cos_last / st->pairs,
                    st->sh_first / st->pairs,
                    n_moe ? st->sh_all / st->pairs / (double)n_moe : 0.0);
        }

        /* Every token, with the first state that carries context (the output
         * of layer 0) reduced to its norm, and the first routing decision. */
        fprintf(f, "\n  ],\n  \"token_list\": [\n");
        for (int64_t i = 0; i < N; i++) {
            piece_of(tk, tok[i].id, piece, sizeof(piece));
            fprintf(f, "    { \"prompt\": %d, \"pos\": %d, \"id\": %d, \"piece\": ",
                    tok[i].prompt + 1, tok[i].pos, tok[i].id);
            json_string(f, piece);
            fprintf(f, ", \"norm_embed\": %.4f, \"norm_l0\": %.4f, \"route_first\": [",
                    (double)norms[i * B], (double)norms[i * B + 1]);
            for (int64_t k2 = 0; k2 < K; k2++)
                fprintf(f, "%s%d", k2 ? "," : "",
                        first_moe < 0 ? -1 : routes[(i * L + first_moe) * K + k2]);
            fprintf(f, "] }%s\n", i + 1 < N ? "," : "");
        }
        fprintf(f, "  ]\n}\n");
        fclose(f);
    }

    /* Raw initial states, for whoever wants to look at the vectors themselves:
     * [tokens][2][hidden] f32 -- the embedding and the output of layer 0. */
    if (dump_path) {
        FILE *f = fopen(dump_path, "wb");
        if (!f) return tt_fail("cannot write %s", dump_path);
        for (int64_t i = 0; i < N; i++)
            fwrite(states + i * B * Hd, sizeof(float), (size_t)(2 * Hd), f);
        fclose(f);
    }

    free(cos_sum); free(sh_sum); free(eq_sum); free(ids); free(slot);
    free(s->trace); s->trace = NULL;
    free(tok); free(states); free(routes); free(norms);
    return 0;
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    const char *path = NULL, *tok_path = NULL, *text = NULL, *file = NULL;
    const char *json_path = NULL, *dump_path = NULL;
    int threads = 20;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = i + 1 < argc;
        if      (!strcmp(a, "--tokenizer") && has)   tok_path = argv[++i];
        else if (!strcmp(a, "--text") && has)        text = argv[++i];
        else if (!strcmp(a, "--prompts") && has)     file = argv[++i];
        else if (!strcmp(a, "--json") && has)        json_path = argv[++i];
        else if (!strcmp(a, "--dump-states") && has) dump_path = argv[++i];
        else if (!strcmp(a, "--threads") && has)     threads = atoi(argv[++i]);
        else if (a[0] != '-')                        path = a;
    }
    if (!path || !tok_path || (!text && !file)) {
        fprintf(stderr,
            "usage: route <model.tootega> --tokenizer <tokenizer.json>\n"
            "             --text \"...\"                 route of one text, JSON on stdout\n"
            "             --prompts <file> [--json <out>] [--dump-states <out>]\n"
            "                                          one prompt per line, each in a\n"
            "                                          fresh context; variation of the\n"
            "                                          same id across prompts\n"
            "             [--threads 20]\n");
        return 1;
    }

    tt_tokenizer tk;
    if (tt_tokenizer_load(&tk, tok_path, 154880) != 0) {
        fprintf(stderr, "error: %s\n", tt_error()); return 1;
    }

    int32_t *pids[MAX_PROMPTS];
    int64_t  plen[MAX_PROMPTS];
    int np = 0;
    int64_t longest = 0;

    if (text) {
        pids[0] = (int32_t *)malloc(MAX_IDS * sizeof(int32_t));
        plen[0] = tt_encode(&tk, text, strlen(text), pids[0], MAX_IDS);
        np = 1;
    } else {
        FILE *f = fopen(file, "rb");
        if (!f) { fprintf(stderr, "error: cannot read %s\n", file); return 1; }
        static char line[16384];
        while (np < MAX_PROMPTS && fgets(line, sizeof(line), f)) {
            char *p = line;
            if (np == 0 && (unsigned char)p[0] == 0xEF && (unsigned char)p[1] == 0xBB
                        && (unsigned char)p[2] == 0xBF) p += 3;        /* BOM */
            size_t len = strlen(p);
            while (len && (p[len - 1] == '\n' || p[len - 1] == '\r')) p[--len] = '\0';
            if (!len) continue;
            pids[np] = (int32_t *)malloc(MAX_IDS * sizeof(int32_t));
            plen[np] = tt_encode(&tk, p, len, pids[np], MAX_IDS);
            np++;
        }
        fclose(f);
    }
    for (int p = 0; p < np; p++) {
        if (plen[p] <= 0) { fprintf(stderr, "error: prompt %d did not encode\n", p + 1); return 1; }
        if (plen[p] > longest) longest = plen[p];
    }
    if (!np) { fprintf(stderr, "error: nothing to run\n"); return 1; }

    tt_model m;
    if (tt_model_open(&m, path, longest, TT_ATTN_ABSORBED, TT_KV_F32) != 0) {
        fprintf(stderr, "error: %s\n", tt_error()); return 1;
    }

    tt_pool *pool = tt_pool_create(threads, TT_AFFINITY_ALL);
    tt_kernels_set_pool(pool);

    tt_state s;
    if (tt_state_init(&s, &m, longest) != 0) {
        fprintf(stderr, "error: %s\n", tt_error()); return 1;
    }

    const int64_t LK = m.cfg.num_hidden_layers * m.cfg.num_experts_per_tok;
    s.route   = (int32_t *)malloc((size_t)LK * sizeof(int32_t));
    s.route_w = (float *)malloc((size_t)LK * sizeof(float));
    if (!s.route || !s.route_w) { fprintf(stderr, "error: no memory\n"); return 1; }

    const int rc = text
        ? run_text(&s, &m, &tk, path, pids[0], plen[0])
        : run_prompts(&s, &m, &tk, path, file, pids, plen, np, json_path, dump_path);
    if (rc != 0) fprintf(stderr, "error: %s\n", tt_error());

    free(s.route); free(s.route_w);
    s.route = NULL; s.route_w = NULL;
    tt_state_free(&s);
    tt_kernels_set_pool(NULL);
    tt_pool_destroy(pool);
    tt_model_close(&m);
    for (int p = 0; p < np; p++) free(pids[p]);
    return rc ? 1 : 0;
}
