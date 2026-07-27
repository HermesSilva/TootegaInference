/* fixture -- generates `src/fixture_kernels.h`, the kernels' cross reference.
 *
 * What the fixture is for: if the SIMD kernel drifts by one ulp, `tootega
 * selftest` says so here, and not forty minutes later inside a 47-layer oracle
 * where everything is a suspect.
 *
 * Two independences make that possible, and both are deliberate:
 *
 *   quantisation   the weights are quantised by `tools/shared/quant.c`, the
 *                  SAME code the converter writes containers with.  A fixture
 *                  built by a lookalike quantiser would freeze the lookalike.
 *
 *   expected value the outputs are accumulated here, in float32, in the same
 *                  order docs/15 (D4) fixes for the reference -- with a scalar
 *                  loop that shares nothing with `src/kernels.c`.  The
 *                  dequantisation also comes from `quant.c` and not from
 *                  `tt_dequant_row`, for the same reason.
 *
 * So the comparison in the selftest is between two implementations, and it is
 * of BITS, not of tolerance: every literal is emitted in hexadecimal, exact,
 * with no round trip through decimal.
 *
 *     tools\fixture.exe --out src\fixture_kernels.h
 *     tools\fixture.exe --check src\fixture_kernels.h
 */
#include "quant.h"

#include "common.h"
#include "container.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FX_O     8
#define FX_I     192            /* 3 grupos por linha; I nao e potencia de 2 */
#define FX_GROUP 64

/* ------------------------------------------------------------- aleatorio */

/* splitmix64 + Box-Muller.  Nao precisa ser um bom gerador -- precisa ser o
 * MESMO gerador amanha, para que regerar a fixture sem mudar nada produza o
 * mesmo arquivo. */
static uint64_t g_rng = 20260725u;

static uint64_t next_u64(void)
{
    uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static double next_unit(void)                    /* [0,1) */
{
    return (double)(next_u64() >> 11) * (1.0 / 9007199254740992.0);
}

static double next_normal(void)
{
    double u1;
    do { u1 = next_unit(); } while (u1 <= 0.0);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586476925286766559 * next_unit());
}

static double next_uniform(double lo, double hi)
{
    return lo + (hi - lo) * next_unit();
}

/* --------------------------------------------------------------- saida */

typedef struct { char *p; size_t len, cap; } obuf;

static void ob_init(obuf *b)
{
    b->cap = 4u << 20;
    b->len = 0;
    b->p = (char *)malloc(b->cap);
    if (!b->p) { fprintf(stderr, "sem memoria\n"); exit(1); }
    b->p[0] = '\0';
}

static void ob_put(obuf *b, const char *s)
{
    const size_t n = strlen(s);
    if (b->len + n + 1 > b->cap) { fprintf(stderr, "buffer estourou\n"); exit(1); }
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void ob_printf(obuf *b, const char *fmt, ...)
{
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    ob_put(b, line);
}

/* float32 -> literal hexadecimal C, exato.  `%a` de um float promovido a double
 * nao perde bit nenhum, e o compilador le de volta o mesmo valor -- que e o
 * ponto: um literal decimal com 9 digitos ja seria uma aposta. */
static void hexf(char *out, size_t n, float v)
{
    if (v != v)                       { snprintf(out, n, "(0.0f/0.0f)"); return; }
    if (v > 3.0e38f)                  { snprintf(out, n, "(1e30f*1e30f)"); return; }
    if (v < -3.0e38f)                 { snprintf(out, n, "(-1e30f*1e30f)"); return; }
    snprintf(out, n, "%af", (double)v);
}

static void emit_floats(obuf *b, const char *name, const float *v, size_t n)
{
    ob_printf(b, "static const float %s[%zu] = {\n", name, n);
    for (size_t i = 0; i < n; i += 4) {
        ob_put(b, "    ");
        for (size_t k = i; k < i + 4 && k < n; k++) {
            char lit[64];
            hexf(lit, sizeof(lit), v[k]);
            ob_printf(b, "%s%s", k > i ? ", " : "", lit);
        }
        ob_put(b, ",\n");
    }
    ob_put(b, "};\n\n");
}

static void emit_ints(obuf *b, const char *name, const char *ctype,
                      const long *v, size_t n)
{
    ob_printf(b, "static const %s %s[%zu] = {\n", ctype, name, n);
    for (size_t i = 0; i < n; i += 16) {
        ob_put(b, "    ");
        for (size_t k = i; k < i + 16 && k < n; k++)
            ob_printf(b, "%s%ld", k > i ? ", " : "", v[k]);
        ob_put(b, ",\n");
    }
    ob_put(b, "};\n\n");
}

/* --------------------------------------------------------- referencia */

/* Produto interno em float32, acumulador escalar, ordem 0..I-1.
 *
 * E a regra de docs/15 (D4): a mesma ordem que o kernel percorre, para que a
 * comparacao seja de bits e nao de tolerancia.  O build usa `/fp:precise` sem
 * contracao para FMA, entao cada operacao arredonda para float aqui tambem. */
static float dot_f32(const float *w, const float *x, int64_t n)
{
    float acc = 0.0f;
    for (int64_t i = 0; i < n; i++)
        acc = acc + w[i] * x[i];
    return acc;
}

int main(int argc, char **argv)
{
    const char *out = NULL, *check = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = (i + 1 < argc);
        if      (!strcmp(a, "--out") && has)   out = argv[++i];
        else if (!strcmp(a, "--check") && has) check = argv[++i];
        else { fprintf(stderr, "uso: fixture [--out <file.h>] [--check <file.h>]\n"); return 2; }
    }
    if (!out && !check) {
        fprintf(stderr, "uso: fixture [--out <file.h>] [--check <file.h>]\n");
        return 2;
    }

    /* A fixture congela o formato do container, e o container e escrito com a
     * busca de clipping ligada -- entao ela tambem tem de estar ligada aqui. */
    tt_clip_grid_init(13);

    static float w[FX_O * FX_I], x[FX_I], g[FX_I];

    /* Distribuicao com cauda: grupos com absmax muito diferente entre si sao o
     * caso que expoe erro de indexacao de escala. */
    for (int o = 0; o < FX_O; o++) {
        const double s = next_uniform(0.05, 3.0);
        for (int i = 0; i < FX_I; i++)
            w[o * FX_I + i] = (float)(next_normal() * s);
    }
    for (int i = 0; i < FX_GROUP; i++) w[i] = 0.0f;      /* grupo todo zero -> escala 1 */
    w[FX_I] = 6.0f;                                      /* absmax isolado na linha 1 */

    for (int i = 0; i < FX_I; i++) x[i] = (float)next_normal();
    for (int i = 0; i < FX_I; i++) g[i] = (float)next_uniform(0.5, 1.5);

    const int ngroups = FX_I / FX_GROUP;

    static uint8_t packed4[FX_O * (FX_I / 2)];
    static int8_t  codes8[FX_O * FX_I];
    static float   scales4[FX_O * (FX_I / FX_GROUP)];
    static float   scales8[FX_O * (FX_I / FX_GROUP)];
    static float   deq4[FX_O * FX_I], deq8[FX_O * FX_I], deq_bf16[FX_O * FX_I];
    static uint16_t bf16[FX_O * FX_I];

    for (int o = 0; o < FX_O; o++) {
        int8_t codes[FX_GROUP];
        for (int gi = 0; gi < ngroups; gi++) {
            tt_quant_group(w + o * FX_I + gi * FX_GROUP, FX_GROUP, 7,
                           &scales4[o * ngroups + gi], codes);
            tt_pack_int4(codes, FX_GROUP,
                         packed4 + o * (FX_I / 2) + gi * (FX_GROUP / 2));

            tt_quant_group(w + o * FX_I + gi * FX_GROUP, FX_GROUP, 127,
                           &scales8[o * ngroups + gi], codes);
            memcpy(codes8 + o * FX_I + gi * FX_GROUP, codes, FX_GROUP);
        }
        tt_quant_unpack_row(TT_KIND_INT4, packed4 + o * (FX_I / 2),
                            scales4 + o * ngroups, FX_GROUP, FX_I, deq4 + o * FX_I);
        tt_quant_unpack_row(TT_KIND_INT8, (const uint8_t *)(codes8 + o * FX_I),
                            scales8 + o * ngroups, FX_GROUP, FX_I, deq8 + o * FX_I);
    }

    /* bf16 = 16 bits altos de um float32 (docs/14).  Truncamento, nao
     * arredondamento -- e o que o conversor faz na ida. */
    for (int i = 0; i < FX_O * FX_I; i++) {
        union { float f; uint32_t u; } c;
        c.f = w[i];
        bf16[i] = (uint16_t)(c.u >> 16);
        c.u = (uint32_t)bf16[i] << 16;
        deq_bf16[i] = c.f;
    }

    float y_f32[FX_O], y_int4[FX_O], y_int8[FX_O], y_bf16[FX_O];
    for (int o = 0; o < FX_O; o++) {
        y_f32[o]  = dot_f32(w        + o * FX_I, x, FX_I);
        y_int4[o] = dot_f32(deq4     + o * FX_I, x, FX_I);
        y_int8[o] = dot_f32(deq8     + o * FX_I, x, FX_I);
        y_bf16[o] = dot_f32(deq_bf16 + o * FX_I, x, FX_I);
    }

    /* rmsnorm em float32, soma dos quadrados em ordem fixa (docs/15). */
    float ss = 0.0f;
    for (int i = 0; i < FX_I; i++) ss = ss + x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)FX_I + 1e-6f);
    static float y_norm[FX_I];
    for (int i = 0; i < FX_I; i++) y_norm[i] = (x[i] * inv) * g[i];

    /* ------------------------------------------------------------ emitir */

    obuf b;
    ob_init(&b);
    ob_printf(&b,
        "/* fixture_kernels.h -- GENERATED by tools/fixture.exe.  Do not edit.\n"
        " *\n"
        " * Cross reference for the kernels.  The weights are quantised by\n"
        " * `tools/shared/quant.c` -- the converter's own quantiser -- and the outputs\n"
        " * are accumulated in float32 in the same order the kernel uses, by a scalar\n"
        " * loop that shares nothing with `src/kernels.c`.  So the comparison is of\n"
        " * BITS, not of tolerance.\n"
        " *\n"
        " * O = %d, I = %d, group = %d\n"
        " */\n"
        "#ifndef TOOTEGA_FIXTURE_KERNELS_H\n"
        "#define TOOTEGA_FIXTURE_KERNELS_H\n"
        "\n"
        "#include <stdint.h>\n"
        "\n"
        "#define FX_O     %d\n"
        "#define FX_I     %d\n"
        "#define FX_GROUP %d\n"
        "\n", FX_O, FX_I, FX_GROUP, FX_O, FX_I, FX_GROUP);

    {
        static long tmp[FX_O * FX_I];
        emit_floats(&b, "fx_x", x, FX_I);
        emit_floats(&b, "fx_w_f32", w, FX_O * FX_I);

        for (int i = 0; i < FX_O * (FX_I / 2); i++) tmp[i] = packed4[i];
        emit_ints(&b, "fx_w_int4", "uint8_t", tmp, FX_O * (FX_I / 2));
        emit_floats(&b, "fx_scales_int4", scales4, FX_O * ngroups);
        emit_floats(&b, "fx_w_deq4", deq4, FX_O * FX_I);

        for (int i = 0; i < FX_O * FX_I; i++) tmp[i] = codes8[i];
        emit_ints(&b, "fx_w_int8", "int8_t", tmp, FX_O * FX_I);
        emit_floats(&b, "fx_scales_int8", scales8, FX_O * ngroups);
        emit_floats(&b, "fx_w_deq8", deq8, FX_O * FX_I);

        for (int i = 0; i < FX_O * FX_I; i++) tmp[i] = bf16[i];
        emit_ints(&b, "fx_w_bf16", "uint16_t", tmp, FX_O * FX_I);

        emit_floats(&b, "fx_y_f32", y_f32, FX_O);
        emit_floats(&b, "fx_y_int4", y_int4, FX_O);
        emit_floats(&b, "fx_y_int8", y_int8, FX_O);
        emit_floats(&b, "fx_y_bf16", y_bf16, FX_O);
        emit_floats(&b, "fx_norm_w", g, FX_I);
        emit_floats(&b, "fx_norm_y", y_norm, FX_I);
    }
    ob_put(&b, "#endif /* TOOTEGA_FIXTURE_KERNELS_H */\n");

    /* ------------------------------------------------------------ conta */

    double num = 0.0, den = 0.0, wnum = 0.0, wden = 0.0;
    for (int o = 0; o < FX_O; o++) {
        num += (double)(y_int4[o] - y_f32[o]) * (y_int4[o] - y_f32[o]);
        den += (double)y_f32[o] * y_f32[o];
    }
    for (int i = 0; i < FX_O * FX_I; i++) {
        wnum += (double)(deq4[i] - w[i]) * (deq4[i] - w[i]);
        wden += (double)w[i] * w[i];
    }
    double num8 = 0.0;
    for (int o = 0; o < FX_O; o++)
        num8 += (double)(y_int8[o] - y_f32[o]) * (y_int8[o] - y_f32[o]);

    printf("%dx%d, grupo %d, clip-grid %d\n",
           FX_O, FX_I, FX_GROUP, tt_clip_grid_points());
    printf("  erro relativo da saida int4: %.2f%%\n", 100.0 * sqrt(num / den));
    printf("  erro relativo da saida int8: %.2f%%\n", 100.0 * sqrt(num8 / den));
    printf("  erro relativo por peso int4: %.2f%%\n", 100.0 * sqrt(wnum / wden));

    int rc = 0;

    if (check) {
        FILE *f = fopen(check, "rb");
        if (!f) { fprintf(stderr, "nao consegui abrir '%s'\n", check); return 1; }
        char *cur = (char *)malloc(b.cap + 1);
        if (!cur) { fclose(f); fprintf(stderr, "sem memoria\n"); return 1; }
        const size_t got = fread(cur, 1, b.cap, f);
        cur[got] = '\0';
        fclose(f);

        /* Comeca no include guard: o cabecalho traz o nome do gerador e nao e o
         * que esta sendo conferido -- os NUMEROS sao. */
        const char *mark = "#ifndef TOOTEGA_FIXTURE_KERNELS_H";
        const char *a = strstr(b.p, mark), *c = strstr(cur, mark);
        if (!a || !c) {
            printf("\n  %s: nao achei o include guard\n", check);
            rc = 1;
        } else if (b.len - (size_t)(a - b.p) == got - (size_t)(c - cur) &&
                   memcmp(a, c, b.len - (size_t)(a - b.p)) == 0) {
            printf("\n  %s confere\n", check);
        } else {
            printf("\n  %s DIVERGE do que este gerador produz\n", check);
            rc = 1;
        }
        free(cur);
    }

    if (out) {
        FILE *f = fopen(out, "wb");
        if (!f) { fprintf(stderr, "nao consegui escrever '%s'\n", out); return 1; }
        fwrite(b.p, 1, b.len, f);
        fclose(f);
        printf("\n  %s escrito (%zu bytes)\n", out, b.len);
    }

    free(b.p);
    return rc;
}
