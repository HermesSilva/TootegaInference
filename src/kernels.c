#include "kernels.h"
#include "pool.h"

#include <math.h>
#include <string.h>

/* Compilamos com x86-64-v3 (AVX2+FMA) como base -- docs/10 -- entao AVX2 nao
 * precisa de deteccao em runtime: um binario sem ele nem chega aqui.  A
 * deteccao que vai existir e a de AVX-VNNI, quando o W4A8 for medido. */
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#  define TT_HAVE_AVX2 1
#  include <immintrin.h>
#endif

static tt_impl g_impl = TT_IMPL_AUTO;

void    tt_kernels_set_impl(tt_impl impl) { g_impl = impl; }
tt_impl tt_kernels_impl(void)             { return g_impl; }

const char *tt_kernels_impl_name(void)
{
#if defined(TT_HAVE_AVX2)
    return g_impl == TT_IMPL_SCALAR ? "escalar" : "avx2+fma";
#else
    return "escalar";
#endif
}

#if defined(TT_HAVE_AVX2)
/* Soma horizontal com ordem de combinacao FIXA e escrita aqui, nao deixada ao
 * compilador: 8 lanes -> 4 -> 2 -> 1, sempre nesta arvore (docs/15). */
static inline float hsum8(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);
}
#endif

/* ------------------------------------------------------------- matmul */

static void mm_f32_ref(float *y, const float *w, const float *x,
                       int64_t cols, int64_t row0, int64_t row1)
{
    for (int64_t o = row0; o < row1; o++) {
        const float *wr = w + o * cols;
        float acc = 0.0f;
        for (int64_t i = 0; i < cols; i++)
            acc += wr[i] * x[i];
        y[o] = acc;
    }
}

static void mm_bf16_ref(float *y, const uint16_t *w, const float *x,
                        int64_t cols, int64_t row0, int64_t row1)
{
    for (int64_t o = row0; o < row1; o++) {
        const uint16_t *wr = w + o * cols;
        float acc = 0.0f;
        for (int64_t i = 0; i < cols; i++)
            acc += tt_bf16_to_f32(wr[i]) * x[i];
        y[o] = acc;
    }
}

static void mm_int8_ref(float *y, const int8_t *w, const float *scales,
                        const float *x, int64_t cols, int group,
                        int64_t row0, int64_t row1)
{
    const int64_t ngroups = cols / group;

    for (int64_t o = row0; o < row1; o++) {
        const int8_t *wr = w + o * cols;
        const float  *sr = scales + o * ngroups;
        float acc = 0.0f;

        for (int64_t g = 0; g < ngroups; g++) {
            const float   s = sr[g];
            const int8_t *p = wr + g * group;
            const float  *xv = x + g * group;
            /* (codigo * escala) * ativacao, nesta ordem: e o que
             * `dequant_grouped` produz e o que o oraculo consome. */
            for (int k = 0; k < group; k++)
                acc += ((float)p[k] * s) * xv[k];
        }
        y[o] = acc;
    }
}

static void mm_int4_ref(float *y, const uint8_t *w, const float *scales,
                        const float *x, int64_t cols, int group,
                        int64_t row0, int64_t row1)
{
    const int64_t ngroups = cols / group;
    const int     half    = group / 2;
    const int64_t stride  = cols / 2;              /* bytes por linha */

    for (int64_t o = row0; o < row1; o++) {
        const uint8_t *wr = w + o * stride;
        const float   *sr = scales + o * ngroups;
        float acc = 0.0f;

        for (int64_t g = 0; g < ngroups; g++) {
            const float    s  = sr[g];
            const uint8_t *p  = wr + g * half;
            const float   *xv = x + g * group;

            /* Metade/metade (docs/14): os nibbles baixos sao os pesos
             * 0..half-1 e os altos sao half..group-1.  Percorrer nesta ordem
             * mantem a reducao em i crescente, que e a ordem do oraculo.
             *
             * O deslocamento de 8 e o zero-point implicito: o byte guarda
             * 0..15, o codigo real e armazenado-8, faixa -8..7. */
            for (int j = 0; j < half; j++)
                acc += ((float)((int)(p[j] & 0x0F) - 8) * s) * xv[j];
            for (int j = 0; j < half; j++)
                acc += ((float)((int)(p[j] >> 4) - 8) * s) * xv[half + j];
        }
        y[o] = acc;
    }
}

/* ---------------------------------------------------------- matmul SIMD */

#if defined(TT_HAVE_AVX2)

static void mm_f32_simd(float *y, const float *w, const float *x,
                        int64_t cols, int64_t row0, int64_t row1)
{
    const int64_t n8 = cols & ~(int64_t)7;

    for (int64_t o = row0; o < row1; o++) {
        const float *wr = w + o * cols;
        __m256 acc = _mm256_setzero_ps();
        for (int64_t i = 0; i < n8; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(wr + i),
                                  _mm256_loadu_ps(x + i), acc);
        float sum = hsum8(acc);
        for (int64_t i = n8; i < cols; i++) sum += wr[i] * x[i];
        y[o] = sum;
    }
}

static void mm_bf16_simd(float *y, const uint16_t *w, const float *x,
                         int64_t cols, int64_t row0, int64_t row1)
{
    const int64_t n8 = cols & ~(int64_t)7;

    for (int64_t o = row0; o < row1; o++) {
        const uint16_t *wr = w + o * cols;
        __m256 acc = _mm256_setzero_ps();
        for (int64_t i = 0; i < n8; i += 8) {
            /* bf16 -> f32 e um shift de 16 bits, nao uma conversao. */
            __m128i h = _mm_loadu_si128((const __m128i *)(wr + i));
            __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
            acc = _mm256_fmadd_ps(_mm256_castsi256_ps(u),
                                  _mm256_loadu_ps(x + i), acc);
        }
        float sum = hsum8(acc);
        for (int64_t i = n8; i < cols; i++) sum += tt_bf16_to_f32(wr[i]) * x[i];
        y[o] = sum;
    }
}

static void mm_int8_simd(float *y, const int8_t *w, const float *scales,
                         const float *x, int64_t cols, int group,
                         int64_t row0, int64_t row1)
{
    const int64_t ngroups = cols / group;
    const int     chunks  = group / 8;

    for (int64_t o = row0; o < row1; o++) {
        const int8_t *wr = w + o * cols;
        const float  *sr = scales + o * ngroups;
        __m256 acc = _mm256_setzero_ps();

        for (int64_t g = 0; g < ngroups; g++) {
            const __m256  s  = _mm256_set1_ps(sr[g]);
            const int8_t *p  = wr + g * group;
            const float  *xv = x + g * group;
            for (int j = 0; j < chunks; j++) {
                __m128i b = _mm_loadl_epi64((const __m128i *)(p + j * 8));
                __m256  f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
                acc = _mm256_fmadd_ps(_mm256_mul_ps(f, s),
                                      _mm256_loadu_ps(xv + j * 8), acc);
            }
        }
        y[o] = hsum8(acc);
    }
}

static void mm_int4_simd(float *y, const uint8_t *w, const float *scales,
                         const float *x, int64_t cols, int group,
                         int64_t row0, int64_t row1)
{
    const int64_t ngroups = cols / group;
    const int     half    = group / 2;
    const int     chunks  = half / 8;
    const int64_t stride  = cols / 2;

    const __m256i lomask = _mm256_set1_epi32(0x0F);
    const __m256  eight  = _mm256_set1_ps(8.0f);

    for (int64_t o = row0; o < row1; o++) {
        const uint8_t *wr = w + o * stride;
        const float   *sr = scales + o * ngroups;
        __m256 acc = _mm256_setzero_ps();

        for (int64_t g = 0; g < ngroups; g++) {
            const __m256   s  = _mm256_set1_ps(sr[g]);
            const uint8_t *p  = wr + g * half;
            const float   *xv = x + g * group;

            for (int j = 0; j < chunks; j++) {
                /* 8 bytes -> 8 pesos baixos (j*8..) e 8 altos (half + j*8..).
                 * O layout metade/metade de docs/14 existe exatamente para
                 * isto: os dois vetores saem contiguos, sem permutacao. */
                __m128i b   = _mm_loadl_epi64((const __m128i *)(p + j * 8));
                __m256i u32 = _mm256_cvtepu8_epi32(b);

                __m256 flo = _mm256_cvtepi32_ps(_mm256_and_si256(u32, lomask));
                __m256 fhi = _mm256_cvtepi32_ps(_mm256_srli_epi32(u32, 4));

                flo = _mm256_mul_ps(_mm256_sub_ps(flo, eight), s);
                fhi = _mm256_mul_ps(_mm256_sub_ps(fhi, eight), s);

                acc = _mm256_fmadd_ps(flo, _mm256_loadu_ps(xv + j * 8), acc);
                acc = _mm256_fmadd_ps(fhi, _mm256_loadu_ps(xv + half + j * 8), acc);
            }
        }
        y[o] = hsum8(acc);
    }
}

/* ------------------------------------------------- matmul em lote (SIMD)
 *
 * Cada kernel abaixo e o seu par nao-lote com UM laco a mais, o de `b`, posto
 * por dentro -- em volta do FMA e nunca em volta da soma.  Assim o elemento `b`
 * ve exatamente a mesma sequencia de operacoes que veria sozinho, e o resultado
 * bate BIT A BIT com o laco de chamadas simples.  Nao e detalhe: se batesse so
 * com tolerancia, o prefill em lote mudaria tokens e o 32/32 cairia por
 * construcao (docs/15).
 *
 * O que se ganha e o que sai de dentro do laco: a leitura do peso, e no int4 e
 * no int8 tambem o desempacotamento e a escala.  Sao 6 operacoes por 16 pesos
 * que passam a ser pagas uma vez para o lote inteiro.
 *
 * O ladrilho existe por pressao de registrador: sao 16 ymm no total, e alguns
 * ja estao ocupados pelo peso desempacotado, pela escala e pelas mascaras.
 *
 * O limite do laco de `b` e a CONSTANTE MM_TILE, nunca uma variavel.  A
 * primeira versao passava o resto do ladrilho como parametro, e o compilador,
 * sem saber o limite, punha os acumuladores na pilha: cada FMA virava
 * load-modify-store e o kernel rodava a 33 GFLOP/s em vez de 89.  O resto do
 * lote sai pelo kernel simples, que da os mesmos bits por definicao.
 *
 * Tamanho escolhido por medicao (Tests/mmbatch, 1 thread, tensores reais):
 *
 *     ladrilho   lote 4   lote 8   lote 32
 *            4     80,5     80,4      80,9      <- uniforme, o escolhido
 *            8     25,0     89,3      88,5      <- +11%, mas zera em 4..7
 *           16     25,5     25,8      72,6      <- derrama registrador
 *
 * O 8 e 11% mais rapido no lote grande e perde TUDO entre 4 e 7, que e faixa
 * real: com 1024 tokens e top-4 de 64, um expert raro recebe meia duzia.
 * Uniformidade vale mais que 11% num caso. */
#define MM_TILE 4

static void mm_f32_simd_b(float *y, int64_t ldy, const float *w,
                          const float *x, int64_t ldx,
                          int64_t cols, int64_t row0, int64_t row1)
{
    const int64_t n8 = cols & ~(int64_t)7;

    for (int64_t o = row0; o < row1; o++) {
        const float *wr = w + o * cols;
        __m256 acc[MM_TILE];
        for (int b = 0; b < MM_TILE; b++) acc[b] = _mm256_setzero_ps();

        for (int64_t i = 0; i < n8; i += 8) {
            const __m256 wv = _mm256_loadu_ps(wr + i);
            for (int b = 0; b < MM_TILE; b++)
                acc[b] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + b * ldx + i),
                                         acc[b]);
        }
        for (int b = 0; b < MM_TILE; b++) {
            const float *xb = x + b * ldx;
            float sum = hsum8(acc[b]);
            for (int64_t i = n8; i < cols; i++) sum += wr[i] * xb[i];
            y[b * ldy + o] = sum;
        }
    }
}

static void mm_bf16_simd_b(float *y, int64_t ldy, const uint16_t *w,
                           const float *x, int64_t ldx,
                           int64_t cols, int64_t row0, int64_t row1)
{
    const int64_t n8 = cols & ~(int64_t)7;

    for (int64_t o = row0; o < row1; o++) {
        const uint16_t *wr = w + o * cols;
        __m256 acc[MM_TILE];
        for (int b = 0; b < MM_TILE; b++) acc[b] = _mm256_setzero_ps();

        for (int64_t i = 0; i < n8; i += 8) {
            __m128i h = _mm_loadu_si128((const __m128i *)(wr + i));
            __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
            const __m256 wv = _mm256_castsi256_ps(u);
            for (int b = 0; b < MM_TILE; b++)
                acc[b] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x + b * ldx + i),
                                         acc[b]);
        }
        for (int b = 0; b < MM_TILE; b++) {
            const float *xb = x + b * ldx;
            float sum = hsum8(acc[b]);
            for (int64_t i = n8; i < cols; i++)
                sum += tt_bf16_to_f32(wr[i]) * xb[i];
            y[b * ldy + o] = sum;
        }
    }
}

static void mm_int8_simd_b(float *y, int64_t ldy, const int8_t *w,
                           const float *scales, const float *x, int64_t ldx,
                           int64_t cols, int group,
                           int64_t row0, int64_t row1)
{
    const int64_t ngroups = cols / group;
    const int     chunks  = group / 8;

    for (int64_t o = row0; o < row1; o++) {
        const int8_t *wr = w + o * cols;
        const float  *sr = scales + o * ngroups;
        __m256 acc[MM_TILE];
        for (int b = 0; b < MM_TILE; b++) acc[b] = _mm256_setzero_ps();

        for (int64_t g = 0; g < ngroups; g++) {
            const __m256  s = _mm256_set1_ps(sr[g]);
            const int8_t *p = wr + g * group;
            for (int j = 0; j < chunks; j++) {
                __m128i bb = _mm_loadl_epi64((const __m128i *)(p + j * 8));
                __m256  f  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bb));
                const __m256 wv = _mm256_mul_ps(f, s);
                const int64_t at = g * group + j * 8;
                for (int b = 0; b < MM_TILE; b++)
                    acc[b] = _mm256_fmadd_ps(wv,
                                 _mm256_loadu_ps(x + b * ldx + at), acc[b]);
            }
        }
        for (int b = 0; b < MM_TILE; b++) y[b * ldy + o] = hsum8(acc[b]);
    }
}

static void mm_int4_simd_b(float *y, int64_t ldy, const uint8_t *w,
                           const float *scales, const float *x, int64_t ldx,
                           int64_t cols, int group,
                           int64_t row0, int64_t row1)
{
    const int64_t ngroups = cols / group;
    const int     half    = group / 2;
    const int     chunks  = half / 8;
    const int64_t stride  = cols / 2;

    const __m256i lomask = _mm256_set1_epi32(0x0F);
    const __m256  eight  = _mm256_set1_ps(8.0f);

    for (int64_t o = row0; o < row1; o++) {
        const uint8_t *wr = w + o * stride;
        const float   *sr = scales + o * ngroups;
        __m256 acc[MM_TILE];
        for (int b = 0; b < MM_TILE; b++) acc[b] = _mm256_setzero_ps();

        for (int64_t g = 0; g < ngroups; g++) {
            const __m256   s = _mm256_set1_ps(sr[g]);
            const uint8_t *p = wr + g * half;

            for (int j = 0; j < chunks; j++) {
                __m128i bb  = _mm_loadl_epi64((const __m128i *)(p + j * 8));
                __m256i u32 = _mm256_cvtepu8_epi32(bb);

                __m256 flo = _mm256_cvtepi32_ps(_mm256_and_si256(u32, lomask));
                __m256 fhi = _mm256_cvtepi32_ps(_mm256_srli_epi32(u32, 4));
                flo = _mm256_mul_ps(_mm256_sub_ps(flo, eight), s);
                fhi = _mm256_mul_ps(_mm256_sub_ps(fhi, eight), s);

                const int64_t alo = g * group + j * 8;
                const int64_t ahi = g * group + half + j * 8;
                for (int b = 0; b < MM_TILE; b++) {
                    const float *xb = x + b * ldx;
                    acc[b] = _mm256_fmadd_ps(flo, _mm256_loadu_ps(xb + alo), acc[b]);
                    acc[b] = _mm256_fmadd_ps(fhi, _mm256_loadu_ps(xb + ahi), acc[b]);
                }
            }
        }
        for (int b = 0; b < MM_TILE; b++) y[b * ldy + o] = hsum8(acc[b]);
    }
}

#endif /* TT_HAVE_AVX2 */

/* ---------------------------------------------------------- despacho */

void tt_matmul_f32_rows(float *y, const float *w, const float *x,
                        int64_t cols, int64_t row0, int64_t row1)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR) { mm_f32_simd(y, w, x, cols, row0, row1); return; }
#endif
    mm_f32_ref(y, w, x, cols, row0, row1);
}

void tt_matmul_bf16_rows(float *y, const uint16_t *w, const float *x,
                         int64_t cols, int64_t row0, int64_t row1)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR) { mm_bf16_simd(y, w, x, cols, row0, row1); return; }
#endif
    mm_bf16_ref(y, w, x, cols, row0, row1);
}

void tt_matmul_int8_rows(float *y, const int8_t *w, const float *scales,
                         const float *x, int64_t cols, int group,
                         int64_t row0, int64_t row1)
{
#if defined(TT_HAVE_AVX2)
    /* O caminho vetorial assume grupo multiplo de 8; qualquer outro cai na
     * referencia em vez de virar um caso especial silencioso. */
    if (g_impl != TT_IMPL_SCALAR && group % 8 == 0) {
        mm_int8_simd(y, w, scales, x, cols, group, row0, row1);
        return;
    }
#endif
    mm_int8_ref(y, w, scales, x, cols, group, row0, row1);
}

void tt_matmul_int4_rows(float *y, const uint8_t *w, const float *scales,
                         const float *x, int64_t cols, int group,
                         int64_t row0, int64_t row1)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR && group % 16 == 0) {
        mm_int4_simd(y, w, scales, x, cols, group, row0, row1);
        return;
    }
#endif
    mm_int4_ref(y, w, scales, x, cols, group, row0, row1);
}

/* ------------------------------------------------- despacho, em lote
 *
 * Sem SIMD o "lote" e o laco de chamadas simples -- que e a definicao do
 * resultado esperado, entao o caminho escalar nao pode divergir nem em
 * principio.  E o mesmo arranjo de docs/15: a referencia ingenua continua
 * existindo, e e contra ela que o vetorial responde. */

void tt_matmul_f32_rows_b(float *y, int64_t ldy, const float *w,
                          const float *x, int64_t ldx, int64_t batch,
                          int64_t cols, int64_t row0, int64_t row1)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR) {
        int64_t b0 = 0;
        for (; b0 + MM_TILE <= batch; b0 += MM_TILE)
            mm_f32_simd_b(y + b0 * ldy, ldy, w, x + b0 * ldx, ldx,
                          cols, row0, row1);
        for (int64_t b = b0; b < batch; b++)
            mm_f32_simd(y + b * ldy, w, x + b * ldx, cols, row0, row1);
        return;
    }
#endif
    for (int64_t b = 0; b < batch; b++)
        mm_f32_ref(y + b * ldy, w, x + b * ldx, cols, row0, row1);
}

void tt_matmul_bf16_rows_b(float *y, int64_t ldy, const uint16_t *w,
                           const float *x, int64_t ldx, int64_t batch,
                           int64_t cols, int64_t row0, int64_t row1)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR) {
        int64_t b0 = 0;
        for (; b0 + MM_TILE <= batch; b0 += MM_TILE)
            mm_bf16_simd_b(y + b0 * ldy, ldy, w, x + b0 * ldx, ldx,
                           cols, row0, row1);
        for (int64_t b = b0; b < batch; b++)
            mm_bf16_simd(y + b * ldy, w, x + b * ldx, cols, row0, row1);
        return;
    }
#endif
    for (int64_t b = 0; b < batch; b++)
        mm_bf16_ref(y + b * ldy, w, x + b * ldx, cols, row0, row1);
}

void tt_matmul_int8_rows_b(float *y, int64_t ldy, const int8_t *w,
                           const float *scales, const float *x, int64_t ldx,
                           int64_t batch, int64_t cols, int group,
                           int64_t row0, int64_t row1)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR && group % 8 == 0) {
        int64_t b0 = 0;
        for (; b0 + MM_TILE <= batch; b0 += MM_TILE)
            mm_int8_simd_b(y + b0 * ldy, ldy, w, scales, x + b0 * ldx, ldx,
                           cols, group, row0, row1);
        for (int64_t b = b0; b < batch; b++)
            mm_int8_simd(y + b * ldy, w, scales, x + b * ldx, cols, group, row0, row1);
        return;
    }
#endif
    for (int64_t b = 0; b < batch; b++)
        mm_int8_ref(y + b * ldy, w, scales, x + b * ldx, cols, group, row0, row1);
}

void tt_matmul_int4_rows_b(float *y, int64_t ldy, const uint8_t *w,
                           const float *scales, const float *x, int64_t ldx,
                           int64_t batch, int64_t cols, int group,
                           int64_t row0, int64_t row1)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR && group % 16 == 0) {
        int64_t b0 = 0;
        for (; b0 + MM_TILE <= batch; b0 += MM_TILE)
            mm_int4_simd_b(y + b0 * ldy, ldy, w, scales, x + b0 * ldx, ldx,
                           cols, group, row0, row1);
        for (int64_t b = b0; b < batch; b++)
            mm_int4_simd(y + b * ldy, w, scales, x + b * ldx, cols, group, row0, row1);
        return;
    }
#endif
    for (int64_t b = 0; b < batch; b++)
        mm_int4_ref(y + b * ldy, w, scales, x + b * ldx, cols, group, row0, row1);
}

static void mm_rows_dispatch_b(float *y, int64_t ldy, const tt_tensor *t,
                               const float *x, int64_t ldx, int64_t batch,
                               int64_t row0, int64_t row1)
{
    switch (t->kind) {
    case TT_KIND_F32:
        tt_matmul_f32_rows_b(y, ldy, (const float *)t->data, x, ldx, batch,
                             t->cols, row0, row1);
        break;
    case TT_KIND_BF16:
        tt_matmul_bf16_rows_b(y, ldy, (const uint16_t *)t->data, x, ldx, batch,
                              t->cols, row0, row1);
        break;
    case TT_KIND_INT8:
        tt_matmul_int8_rows_b(y, ldy, (const int8_t *)t->data, t->scales,
                              x, ldx, batch, t->cols, t->group, row0, row1);
        break;
    default:
        tt_matmul_int4_rows_b(y, ldy, (const uint8_t *)t->data, t->scales,
                              x, ldx, batch, t->cols, t->group, row0, row1);
        break;
    }
}

int tt_matmul_rows_b(float *y, int64_t ldy, const tt_tensor *t,
                     const float *x, int64_t ldx, int64_t batch,
                     int64_t row0, int64_t row1)
{
    if (t->ndim < 2)
        return tt_fail("matmul: %.*s tem %d dimensao(oes)",
                       (int)t->name_len, t->name, t->ndim);
    if (t->kind >= TT_KIND_COUNT)
        return tt_fail("matmul: kind %d nao implementado", (int)t->kind);
    if (batch <= 0) return 0;
    mm_rows_dispatch_b(y, ldy, t, x, ldx, batch, row0, row1);
    return 0;
}

static void mm_rows_dispatch(float *y, const tt_tensor *t, const float *x,
                             int64_t row0, int64_t row1)
{
    switch (t->kind) {
    case TT_KIND_F32:
        tt_matmul_f32_rows(y, (const float *)t->data, x, t->cols, row0, row1);
        break;
    case TT_KIND_BF16:
        tt_matmul_bf16_rows(y, (const uint16_t *)t->data, x, t->cols, row0, row1);
        break;
    case TT_KIND_INT8:
        tt_matmul_int8_rows(y, (const int8_t *)t->data, t->scales, x,
                            t->cols, t->group, row0, row1);
        break;
    default:
        tt_matmul_int4_rows(y, (const uint8_t *)t->data, t->scales, x,
                            t->cols, t->group, row0, row1);
        break;
    }
}

int tt_matmul_rows(float *y, const tt_tensor *t, const float *x,
                   int64_t row0, int64_t row1)
{
    if (t->ndim < 2)
        return tt_fail("matmul: %.*s tem %d dimensao(oes)",
                       (int)t->name_len, t->name, t->ndim);

    switch (t->kind) {
    case TT_KIND_F32:
        tt_matmul_f32_rows(y, (const float *)t->data, x, t->cols, row0, row1);
        return 0;
    case TT_KIND_BF16:
        tt_matmul_bf16_rows(y, (const uint16_t *)t->data, x, t->cols, row0, row1);
        return 0;
    case TT_KIND_INT8:
        tt_matmul_int8_rows(y, (const int8_t *)t->data, t->scales, x,
                            t->cols, t->group, row0, row1);
        return 0;
    case TT_KIND_INT4:
        tt_matmul_int4_rows(y, (const uint8_t *)t->data, t->scales, x,
                            t->cols, t->group, row0, row1);
        return 0;
    default:
        return tt_fail("matmul: kind %d nao implementado", (int)t->kind);
    }
}

/* ------------------------------------------------------------ paralelo */

static struct tt_pool *g_pool = NULL;

void            tt_kernels_set_pool(struct tt_pool *p) { g_pool = p; }
struct tt_pool *tt_kernels_pool(void)                   { return g_pool; }

typedef struct {
    float           *y;
    const tt_tensor *t;
    const float     *x;
    /* so o caminho em lote usa os tres abaixo */
    int64_t          ldy, ldx, batch;
} mm_job;

static void mm_job_run(void *arg, int64_t row0, int64_t row1)
{
    const mm_job *j = (const mm_job *)arg;
    mm_rows_dispatch(j->y, j->t, j->x, row0, row1);
}

static void mm_job_run_b(void *arg, int64_t row0, int64_t row1)
{
    const mm_job *j = (const mm_job *)arg;
    mm_rows_dispatch_b(j->y, j->ldy, j->t, j->x, j->ldx, j->batch, row0, row1);
}

/* Abaixo disto a barreira custa mais que o trabalho.  O roteador (64x2048 =
 * 131k MACs) fica de fora de proposito; um dispatch por camada so para ele
 * seriam ~2.800 barreiras por token. */
#define TT_PARALLEL_MIN_MACS (256 * 1024)

typedef struct {
    float **y;
    const tt_tensor **w;
    const float      *x;
    int64_t           start[TT_BATCH_MAX + 1];  /* prefixo de linhas */
    int               n;
} batch_job;

static void batch_run(void *arg, int64_t r0, int64_t r1)
{
    const batch_job *j = (const batch_job *)arg;

    /* A faixa global [r0,r1) pode cair sobre mais de uma matriz; cada uma
     * recebe a sua interseccao, em coordenada local. */
    for (int i = 0; i < j->n; i++) {
        const int64_t s = j->start[i], e = j->start[i + 1];
        const int64_t lo = r0 > s ? r0 : s;
        const int64_t hi = r1 < e ? r1 : e;
        if (lo < hi) mm_rows_dispatch(j->y[i], j->w[i], j->x, lo - s, hi - s);
    }
}

int tt_matmul_batch(float **y, const tt_tensor **w, int n, const float *x)
{
    if (n <= 0) return 0;
    if (n > TT_BATCH_MAX) return tt_fail("batch de %d matmuls, maximo %d",
                                         n, TT_BATCH_MAX);

    batch_job job;
    job.y = y; job.w = w; job.x = x; job.n = n;
    job.start[0] = 0;

    int64_t macs = 0;
    for (int i = 0; i < n; i++) {
        if (w[i]->ndim < 2 || w[i]->kind >= TT_KIND_COUNT)
            return tt_fail("batch: %.*s nao e matriz utilizavel",
                           (int)w[i]->name_len, w[i]->name);
        job.start[i + 1] = job.start[i] + w[i]->rows;
        macs += w[i]->rows * w[i]->cols;
    }
    const int64_t rows = job.start[n];

    const int threads = tt_pool_threads(g_pool);
    if (!g_pool || threads <= 1 || macs < TT_PARALLEL_MIN_MACS) {
        batch_run(&job, 0, rows);
        return 0;
    }

    int64_t grain = 65536 / w[0]->cols;
    if (grain < 4) grain = 4;
    const int64_t cap = rows / (threads * 4);
    if (cap >= 1 && grain > cap) grain = cap;

    tt_pool_run(g_pool, batch_run, &job, rows, grain);
    return 0;
}

int tt_matmul(float *y, const tt_tensor *t, const float *x)
{
    if (t->ndim < 2)
        return tt_fail("matmul: %.*s tem %d dimensao(oes)",
                       (int)t->name_len, t->name, t->ndim);
    if (t->kind >= TT_KIND_COUNT)
        return tt_fail("matmul: kind %d nao implementado", (int)t->kind);

    const int threads = tt_pool_threads(g_pool);
    if (!g_pool || threads <= 1 || t->rows * t->cols < TT_PARALLEL_MIN_MACS) {
        mm_rows_dispatch(y, t, x, 0, t->rows);
        return 0;
    }

    /* Bloco com trabalho suficiente para amortizar o fetch_add, mas pequeno o
     * bastante para haver varios por thread -- e o que absorve o
     * desbalanceamento P-core/E-core. */
    int64_t grain = 65536 / t->cols;
    if (grain < 4) grain = 4;
    const int64_t cap = t->rows / (threads * 4);
    if (cap >= 1 && grain > cap) grain = cap;

    mm_job job = { y, t, x, 0, 0, 1 };
    tt_pool_run(g_pool, mm_job_run, &job, t->rows, grain);
    return 0;
}

int tt_matmul_b(float *y, int64_t ldy, const tt_tensor *t,
                const float *x, int64_t ldx, int64_t batch)
{
    if (t->ndim < 2)
        return tt_fail("matmul: %.*s tem %d dimensao(oes)",
                       (int)t->name_len, t->name, t->ndim);
    if (t->kind >= TT_KIND_COUNT)
        return tt_fail("matmul: kind %d nao implementado", (int)t->kind);
    if (batch <= 0) return 0;

    /* A particao continua sendo POR LINHA de saida, nunca por elemento do
     * lote: e o que mantem a reducao inteira dentro de uma thread e o
     * resultado independente de --threads (docs/15).  Particionar pelo lote
     * seria mais simples e jogaria fora o reuso do peso, que e o motivo de
     * isto existir. */
    const int threads = tt_pool_threads(g_pool);
    if (!g_pool || threads <= 1
        || t->rows * t->cols * batch < TT_PARALLEL_MIN_MACS) {
        mm_rows_dispatch_b(y, ldy, t, x, ldx, batch, 0, t->rows);
        return 0;
    }

    int64_t grain = 65536 / (t->cols * batch);
    if (grain < 1) grain = 1;
    const int64_t cap = t->rows / (threads * 4);
    if (cap >= 1 && grain > cap) grain = cap;

    mm_job job = { y, t, x, ldy, ldx, batch };
    tt_pool_run(g_pool, mm_job_run_b, &job, t->rows, grain);
    return 0;
}

/* ---------------------------------------------------------- ativacoes */

void tt_rmsnorm(float *y, const float *x, const float *weight,
                int64_t n, float eps)
{
    float ss = 0.0f;
    for (int64_t i = 0; i < n; i++)
        ss += x[i] * x[i];

    /* mean -> rsqrt em float32.  `1.0f/sqrtf` e nao `rsqrtf`: a aproximacao
     * rapida diverge do oraculo no ultimo bit. */
    const float inv = 1.0f / sqrtf(ss / (float)n + eps);

    if (weight)
        for (int64_t i = 0; i < n; i++) y[i] = (x[i] * inv) * weight[i];
    else
        for (int64_t i = 0; i < n; i++) y[i] = x[i] * inv;
}

void tt_softmax(float *x, int64_t n)
{
    if (n <= 0) return;

    float max = x[0];
    for (int64_t i = 1; i < n; i++)
        if (x[i] > max) max = x[i];

    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }

    const float inv = 1.0f / sum;
    for (int64_t i = 0; i < n; i++) x[i] *= inv;
}

void tt_silu_mul(float *dst, const float *gate, const float *up, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        const float g = gate[i];
        dst[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

float tt_dot(const float *a, const float *b, int64_t n)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR) {
        const int64_t n8 = n & ~(int64_t)7;
        __m256 acc = _mm256_setzero_ps();
        for (int64_t i = 0; i < n8; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                                  _mm256_loadu_ps(b + i), acc);
        float sum = hsum8(acc);
        for (int64_t i = n8; i < n; i++) sum += a[i] * b[i];
        return sum;
    }
#endif
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void tt_axpy(float *dst, float k, const float *src, int64_t n)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR) {
        const int64_t n8 = n & ~(int64_t)7;
        const __m256 kv = _mm256_set1_ps(k);
        for (int64_t i = 0; i < n8; i += 8)
            _mm256_storeu_ps(dst + i,
                _mm256_fmadd_ps(kv, _mm256_loadu_ps(src + i),
                                _mm256_loadu_ps(dst + i)));
        for (int64_t i = n8; i < n; i++) dst[i] += k * src[i];
        return;
    }
#endif
    for (int64_t i = 0; i < n; i++) dst[i] += k * src[i];
}

float tt_dot_bf16(const float *a, const uint16_t *b, int64_t n)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR) {
        const int64_t n8 = n & ~(int64_t)7;
        __m256 acc = _mm256_setzero_ps();
        for (int64_t i = 0; i < n8; i += 8) {
            __m128i h = _mm_loadu_si128((const __m128i *)(b + i));
            __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                                  _mm256_castsi256_ps(u), acc);
        }
        float sum = hsum8(acc);
        for (int64_t i = n8; i < n; i++) sum += a[i] * tt_bf16_to_f32(b[i]);
        return sum;
    }
#endif
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) sum += a[i] * tt_bf16_to_f32(b[i]);
    return sum;
}

void tt_axpy_bf16(float *dst, float k, const uint16_t *src, int64_t n)
{
#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR) {
        const int64_t n8 = n & ~(int64_t)7;
        const __m256 kv = _mm256_set1_ps(k);
        for (int64_t i = 0; i < n8; i += 8) {
            __m128i h = _mm_loadu_si128((const __m128i *)(src + i));
            __m256i u = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
            _mm256_storeu_ps(dst + i,
                _mm256_fmadd_ps(kv, _mm256_castsi256_ps(u),
                                _mm256_loadu_ps(dst + i)));
        }
        for (int64_t i = n8; i < n; i++) dst[i] += k * tt_bf16_to_f32(src[i]);
        return;
    }
#endif
    for (int64_t i = 0; i < n; i++) dst[i] += k * tt_bf16_to_f32(src[i]);
}

void tt_store_bf16(uint16_t *dst, const float *src, int64_t n)
{
    /* Arredondamento para o par mais proximo, nao truncamento.  O container
     * trunca porque la o bf16 e a fonte; aqui e destino de uma conta, e
     * truncar enviesaria o cache inteiro para baixo em magnitude. */
    for (int64_t i = 0; i < n; i++) {
        union { float f; uint32_t u; } c;
        c.f = src[i];
        const uint32_t lsb = (c.u >> 16) & 1u;
        dst[i] = (uint16_t)((c.u + 0x7FFFu + lsb) >> 16);
    }
}

float tt_dot_int8g(const float *a, const int8_t *w, const float *scales,
                   int64_t n, int group)
{
    const int64_t ngroups = n / group;
    float sum = 0.0f;

#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR && group % 8 == 0) {
        __m256 acc = _mm256_setzero_ps();
        for (int64_t g = 0; g < ngroups; g++) {
            const __m256  s = _mm256_set1_ps(scales[g]);
            const int8_t *p = w + g * group;
            const float  *x = a + g * group;
            for (int j = 0; j < group; j += 8) {
                __m128i b = _mm_loadl_epi64((const __m128i *)(p + j));
                __m256  f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
                acc = _mm256_fmadd_ps(_mm256_mul_ps(f, s),
                                      _mm256_loadu_ps(x + j), acc);
            }
        }
        return hsum8(acc);
    }
#endif
    for (int64_t g = 0; g < ngroups; g++) {
        const float   s = scales[g];
        const int8_t *p = w + g * group;
        const float  *x = a + g * group;
        for (int j = 0; j < group; j++) sum += ((float)p[j] * s) * x[j];
    }
    return sum;
}

void tt_axpy_int8g(float *dst, float k, const int8_t *w, const float *scales,
                   int64_t n, int group)
{
    const int64_t ngroups = n / group;

#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR && group % 8 == 0) {
        for (int64_t g = 0; g < ngroups; g++) {
            /* A escala do grupo e o fator externo entram juntos: um multiply
             * a menos por elemento no laco mais quente da atencao. */
            const __m256  ks = _mm256_set1_ps(k * scales[g]);
            const int8_t *p = w + g * group;
            float        *d = dst + g * group;
            for (int j = 0; j < group; j += 8) {
                __m128i b = _mm_loadl_epi64((const __m128i *)(p + j));
                __m256  f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
                _mm256_storeu_ps(d + j,
                    _mm256_fmadd_ps(ks, f, _mm256_loadu_ps(d + j)));
            }
        }
        return;
    }
#endif
    for (int64_t g = 0; g < ngroups; g++) {
        const float   ks = k * scales[g];
        const int8_t *p = w + g * group;
        float        *d = dst + g * group;
        for (int j = 0; j < group; j++) d[j] += ks * (float)p[j];
    }
}

/* --------------------------------------------------- int4 fundido (linha)
 *
 * O desempacotamento e o de `mm_int4_simd`, byte a byte: 8 bytes viram 8 pesos
 * baixos e 8 altos, e o layout metade/metade de docs/14 e o que faz os dois
 * vetores sairem contiguos sem permutacao.
 *
 * A ORDEM DAS OPERACOES E A DELE TAMBEM, e nao por elegancia: `(w - 8) * s`
 * primeiro, FMA contra o vetor depois.  Dobrar a escala para o outro lado
 * (multiplicar o acumulador, ou juntar `k * s` como faz a versao int8) daria o
 * mesmo numero em algebra exata e bits diferentes em fp32 -- e o criterio de
 * saida de docs/15 e 32/32 tokens IDENTICOS, nao proximos. */

/* Este substitui `tt_dequant_row` seguido de `tt_dot`, e tem de dar os MESMOS
 * BITS que os dois davam juntos -- nao um numero proximo.
 *
 * Por isso a ordem de acumulacao aqui NAO e a de `mm_int4_simd`.  Aquele laco
 * varre o grupo em metade/metade (8 pesos baixos, depois os 8 altos, que moram
 * 32 floats adiante), o que atribui cada produto a uma pista diferente do
 * acumulador.  `tt_dot` varre a linha em ordem linear -- i, i+8, i+16 -- e a
 * soma em fp32 nao e associativa: mudar de qual pista sai cada parcela muda o
 * ultimo bit do resultado.
 *
 * Entao desempacotamos meio grupo por vez para um buffer minusculo em pilha, na
 * ordem NATURAL da linha, e acumulamos dele.  O buffer tem 64 floats, cabe em
 * registradores/L1 e nunca vai a memoria -- o custo que este kernel veio
 * eliminar era o dos 512 floats por linha escritos e relidos, nao este. */
float tt_dot_int4g(const float *a, const uint8_t *w, const float *scales,
                   int64_t n, int group)
{
    const int64_t ngroups = n / group;
    const int     half    = group / 2;

#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR && group % 8 == 0 && group <= 256) {
        const __m256i lomask = _mm256_set1_epi32(0x0F);
        const __m256  eight  = _mm256_set1_ps(8.0f);

        float  buf[256];
        __m256 acc = _mm256_setzero_ps();

        for (int64_t g = 0; g < ngroups; g++) {
            const __m256   s  = _mm256_set1_ps(scales[g]);
            const uint8_t *p  = w + g * half;
            const float   *xv = a + g * group;

            /* Desempacota o grupo inteiro na ordem da linha. */
            for (int j = 0; j < half; j += 8) {
                __m128i b   = _mm_loadl_epi64((const __m128i *)(p + j));
                __m256i u32 = _mm256_cvtepu8_epi32(b);

                __m256 flo = _mm256_cvtepi32_ps(_mm256_and_si256(u32, lomask));
                __m256 fhi = _mm256_cvtepi32_ps(_mm256_srli_epi32(u32, 4));

                _mm256_storeu_ps(buf + j,
                                 _mm256_mul_ps(_mm256_sub_ps(flo, eight), s));
                _mm256_storeu_ps(buf + half + j,
                                 _mm256_mul_ps(_mm256_sub_ps(fhi, eight), s));
            }

            /* ... e acumula linearmente, como `tt_dot` faria sobre ele. */
            for (int j = 0; j < group; j += 8)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(buf + j),
                                      _mm256_loadu_ps(xv + j), acc);
        }
        return hsum8(acc);
    }
#endif
    /* Via escalar: soma na ordem da linha, que e a que o `tt_dot` escalar faria
     * sobre a linha dequantizada. */
    float sum = 0.0f;
    for (int64_t g = 0; g < ngroups; g++) {
        const float    s  = scales[g];
        const uint8_t *p  = w + g * half;
        const float   *xv = a + g * group;
        for (int j = 0; j < half; j++)
            sum += ((float)((int)(p[j] & 0x0F) - 8) * s) * xv[j];
        for (int j = 0; j < half; j++)
            sum += ((float)((int)(p[j] >> 4) - 8) * s) * xv[j + half];
    }
    return sum;
}

void tt_axpy_int4g(float *dst, float k, const uint8_t *w, const float *scales,
                   int64_t n, int group)
{
    const int64_t ngroups = n / group;
    const int     half    = group / 2;

#if defined(TT_HAVE_AVX2)
    if (g_impl != TT_IMPL_SCALAR && half % 8 == 0) {
        const int      chunks = half / 8;
        const __m256i  lomask = _mm256_set1_epi32(0x0F);
        const __m256   eight  = _mm256_set1_ps(8.0f);
        const __m256   kv     = _mm256_set1_ps(k);

        for (int64_t g = 0; g < ngroups; g++) {
            const __m256   s = _mm256_set1_ps(scales[g]);
            const uint8_t *p = w + g * half;
            float         *d = dst + g * group;

            for (int j = 0; j < chunks; j++) {
                __m128i b   = _mm_loadl_epi64((const __m128i *)(p + j * 8));
                __m256i u32 = _mm256_cvtepu8_epi32(b);

                __m256 flo = _mm256_cvtepi32_ps(_mm256_and_si256(u32, lomask));
                __m256 fhi = _mm256_cvtepi32_ps(_mm256_srli_epi32(u32, 4));

                /* Peso primeiro, exatamente como o dequant o entregaria; `k`
                 * entra depois, no FMA, como faria o `tt_axpy` que vinha em
                 * seguida. A cadeia de arredondamento e a mesma das duas
                 * chamadas que isto substitui. */
                flo = _mm256_mul_ps(_mm256_sub_ps(flo, eight), s);
                fhi = _mm256_mul_ps(_mm256_sub_ps(fhi, eight), s);

                _mm256_storeu_ps(d + j * 8,
                    _mm256_fmadd_ps(kv, flo, _mm256_loadu_ps(d + j * 8)));
                _mm256_storeu_ps(d + half + j * 8,
                    _mm256_fmadd_ps(kv, fhi,
                                    _mm256_loadu_ps(d + half + j * 8)));
            }
        }
        return;
    }
#endif
    for (int64_t g = 0; g < ngroups; g++) {
        const float    s = scales[g];
        const uint8_t *p = w + g * half;
        float         *d = dst + g * group;
        for (int j = 0; j < half; j++) {
            d[j]        += k * ((float)((int)(p[j] & 0x0F) - 8) * s);
            d[j + half] += k * ((float)((int)(p[j] >> 4)   - 8) * s);
        }
    }
}

void tt_store_int8g(int8_t *dst, float *scales, const float *src,
                    int64_t n, int group)
{
    const int64_t ngroups = n / group;
    for (int64_t g = 0; g < ngroups; g++) {
        const float *x = src + g * group;

        float absmax = 0.0f;
        for (int j = 0; j < group; j++) {
            const float v = fabsf(x[j]);
            if (v > absmax) absmax = v;
        }
        /* absmax == 0 usa escala 1: evita divisao por zero e reconstroi
         * exatamente, igual ao conversor (docs/14). */
        const float s = absmax > 0.0f ? absmax / 127.0f : 1.0f;
        scales[g] = s;

        const float inv = 1.0f / s;
        int8_t *d = dst + g * group;
        for (int j = 0; j < group; j++) {
            float q = x[j] * inv;
            q = (q >= 0.0f) ? floorf(q + 0.5f) : ceilf(q - 0.5f);
            if (q > 127.0f) q = 127.0f;
            if (q < -128.0f) q = -128.0f;
            d[j] = (int8_t)q;
        }
    }
}

void tt_add(float *dst, const float *src, int64_t n)
{
    for (int64_t i = 0; i < n; i++) dst[i] += src[i];
}

void tt_scale(float *dst, float k, int64_t n)
{
    for (int64_t i = 0; i < n; i++) dst[i] *= k;
}

void tt_copy(float *dst, const float *src, int64_t n)
{
    memcpy(dst, src, (size_t)n * sizeof(float));
}

void tt_zero(float *dst, int64_t n)
{
    memset(dst, 0, (size_t)n * sizeof(float));
}

int64_t tt_argmax(const float *x, int64_t n)
{
    int64_t best = 0;
    for (int64_t i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;      /* `>` estrito: empate fica no menor */
    return best;
}

void tt_topk(const float *x, int64_t n, int k, int32_t *idx_out, float *val_out)
{
    /* Selecao por insercao.  Com k=4 e n=64 (o roteador do GLM-4.7-Flash) sao
     * ~256 comparacoes -- ordenar os 64 custaria mais. */
    for (int i = 0; i < k; i++) {
        idx_out[i] = -1;
        val_out[i] = -INFINITY;
    }

    for (int64_t i = 0; i < n; i++) {
        const float v = x[i];
        int pos = k;
        /* `>` estrito preserva o indice menor no empate: com dois experts de
         * mesmo score, qual entra no top-4 nao pode depender da ordem de
         * varredura. */
        while (pos > 0 && v > val_out[pos - 1]) pos--;
        if (pos >= k) continue;

        for (int j = k - 1; j > pos; j--) {
            val_out[j] = val_out[j - 1];
            idx_out[j] = idx_out[j - 1];
        }
        val_out[pos] = v;
        idx_out[pos] = (int32_t)i;
    }
}

/* ------------------------------------------------------------ dequant */

int tt_dequant_row(float *dst, const tt_tensor *t, int64_t row)
{
    if (row < 0 || row >= t->rows)
        return tt_fail("dequant: linha %lld fora de [0, %lld)",
                       (long long)row, (long long)t->rows);

    const int64_t I = t->cols;

    switch (t->kind) {
    case TT_KIND_F32:
        memcpy(dst, (const float *)t->data + row * I, (size_t)I * sizeof(float));
        return 0;

    case TT_KIND_BF16: {
        const uint16_t *p = (const uint16_t *)t->data + row * I;
        for (int64_t i = 0; i < I; i++) dst[i] = tt_bf16_to_f32(p[i]);
        return 0;
    }

    case TT_KIND_INT8: {
        const int8_t *p = (const int8_t *)t->data + row * I;
        const float  *s = t->scales + row * (I / t->group);
        for (int64_t i = 0; i < I; i++)
            dst[i] = (float)p[i] * s[i / t->group];
        return 0;
    }

    case TT_KIND_INT4: {
        const int      group = t->group, half = group / 2;
        const uint8_t *p = (const uint8_t *)t->data + row * (I / 2);
        const float   *s = t->scales + row * (I / group);
        for (int64_t g = 0; g < I / group; g++) {
            const float    sc = s[g];
            const uint8_t *pg = p + g * half;
            float         *dg = dst + g * group;
            for (int j = 0; j < half; j++) {
                dg[j]        = (float)((int)(pg[j] & 0x0F) - 8) * sc;
                dg[j + half] = (float)((int)(pg[j] >> 4)   - 8) * sc;
            }
        }
        return 0;
    }

    default:
        return tt_fail("dequant: kind %d nao implementado", (int)t->kind);
    }
}

/* ----------------------------------------------------------- selftest */

#include "fixture_kernels.h"

#define CHECK(cond, ...) do {                                   \
        if (!(cond)) {                                          \
            tt_kernels_set_impl(saved);                         \
            return tt_fail(__VA_ARGS__);                        \
        }                                                       \
    } while (0)

/* Comparacao de BITS.  Tolerancia esconderia exatamente o erro que estamos
 * caçando: um desvio de 1 ulp por termo vira um token diferente 47 camadas
 * depois. */
static int bits_equal(const float *a, const float *b, int64_t n, int64_t *where)
{
    for (int64_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            if (where) *where = i;
            return 0;
        }
    }
    return 1;
}

int tt_kernels_selftest(void)
{
    float y[FX_O], deq[FX_O * FX_I];
    int64_t bad = -1;

    /* A fixture e a referencia ESCALAR.  O SIMD e conferido depois, contra ela
     * e com tolerancia, porque acumular em 8 lanes com FMA nao da os mesmos
     * bits -- e nao deve dar. */
    const tt_impl saved = tt_kernels_impl();
    tt_kernels_set_impl(TT_IMPL_SCALAR);

    /* 1. Desempacotamento: a linha materializada tem de bater bit a bit com o
     *    que `dequant_grouped` produziu em Python. */
    {
        tt_tensor t = {0};
        t.name = "fixture.int4"; t.name_len = 12;
        t.kind = TT_KIND_INT4; t.ndim = 2;
        t.shape[0] = FX_O; t.shape[1] = FX_I;
        t.rows = FX_O; t.cols = FX_I; t.group = FX_GROUP;
        t.data = fx_w_int4; t.scales = fx_scales_int4;

        for (int64_t o = 0; o < FX_O; o++)
            if (tt_dequant_row(deq + o * FX_I, &t, o) != 0) return -1;

        CHECK(bits_equal(deq, fx_w_deq4, FX_O * FX_I, &bad),
              "int4: peso [%lld][%lld] difere do dequant_grouped do Python "
              "(C %.9g, py %.9g) -- empacotamento metade/metade errado?",
              (long long)(bad / FX_I), (long long)(bad % FX_I),
              (double)deq[bad], (double)fx_w_deq4[bad]);

        /* 2. O matmul int4 tem de dar o MESMO resultado que o matmul f32 sobre
         *    a matriz desempacotada.  E o invariante que separa erro de
         *    desempacotamento de erro de acumulacao. */
        if (tt_matmul(y, &t, fx_x) != 0) return -1;
        CHECK(bits_equal(y, fx_y_int4, FX_O, &bad),
              "int4: y[%lld] = %.9g, esperado %.9g",
              (long long)bad, (double)y[bad], (double)fx_y_int4[bad]);

        float y2[FX_O];
        tt_matmul_f32_rows(y2, deq, fx_x, FX_I, 0, FX_O);
        CHECK(bits_equal(y, y2, FX_O, &bad),
              "int4: kernel quantizado divergiu do f32 sobre a mesma matriz "
              "em y[%lld] -- a ordem de acumulacao nao casa", (long long)bad);

        /* 3. Faixa de linhas: particionar nao pode mudar valor.  E o teste que
         *    protege a promessa de determinismo com --threads. */
        float y3[FX_O];
        for (int64_t r = 0; r < FX_O; r++) y3[r] = 12345.0f;
        tt_matmul_rows(y3, &t, fx_x, 0, 3);
        tt_matmul_rows(y3, &t, fx_x, 3, 5);
        tt_matmul_rows(y3, &t, fx_x, 5, FX_O);
        CHECK(bits_equal(y, y3, FX_O, &bad),
              "int4: particionar em faixas mudou y[%lld]", (long long)bad);
    }

    /* 4. int8 */
    {
        tt_tensor t = {0};
        t.name = "fixture.int8"; t.name_len = 12;
        t.kind = TT_KIND_INT8; t.ndim = 2;
        t.shape[0] = FX_O; t.shape[1] = FX_I;
        t.rows = FX_O; t.cols = FX_I; t.group = FX_GROUP;
        t.data = fx_w_int8; t.scales = fx_scales_int8;

        for (int64_t o = 0; o < FX_O; o++)
            if (tt_dequant_row(deq + o * FX_I, &t, o) != 0) return -1;
        CHECK(bits_equal(deq, fx_w_deq8, FX_O * FX_I, &bad),
              "int8: peso [%lld] difere do Python", (long long)bad);

        if (tt_matmul(y, &t, fx_x) != 0) return -1;
        CHECK(bits_equal(y, fx_y_int8, FX_O, &bad),
              "int8: y[%lld] = %.9g, esperado %.9g",
              (long long)bad, (double)y[bad], (double)fx_y_int8[bad]);
    }

    /* 5. f32 e bf16 */
    {
        tt_tensor t = {0};
        t.name = "fixture.f32"; t.name_len = 11;
        t.kind = TT_KIND_F32; t.ndim = 2;
        t.shape[0] = FX_O; t.shape[1] = FX_I;
        t.rows = FX_O; t.cols = FX_I;
        t.data = fx_w_f32;
        if (tt_matmul(y, &t, fx_x) != 0) return -1;
        CHECK(bits_equal(y, fx_y_f32, FX_O, &bad),
              "f32: y[%lld] = %.9g, esperado %.9g",
              (long long)bad, (double)y[bad], (double)fx_y_f32[bad]);

        t.name = "fixture.bf16"; t.name_len = 12;
        t.kind = TT_KIND_BF16;
        t.data = fx_w_bf16;
        if (tt_matmul(y, &t, fx_x) != 0) return -1;
        CHECK(bits_equal(y, fx_y_bf16, FX_O, &bad),
              "bf16: y[%lld] = %.9g, esperado %.9g",
              (long long)bad, (double)y[bad], (double)fx_y_bf16[bad]);
    }

    /* 6. rmsnorm contra a mesma conta feita em Python */
    {
        float out[FX_I];
        tt_rmsnorm(out, fx_x, fx_norm_w, FX_I, 1e-6f);
        CHECK(bits_equal(out, fx_norm_y, FX_I, &bad),
              "rmsnorm: out[%lld] = %.9g, esperado %.9g",
              (long long)bad, (double)out[bad], (double)fx_norm_y[bad]);
    }

    /* 7. softmax: soma 1, monotonico, e imune a deslocamento constante --
     *    a subtracao do maximo existe para isso. */
    {
        float a[5] = { 1.0f, 3.0f, 2.0f, -1.0f, 0.5f };
        float b[5];
        for (int i = 0; i < 5; i++) b[i] = a[i] + 100.0f;
        tt_softmax(a, 5);
        tt_softmax(b, 5);

        float sum = 0.0f;
        for (int i = 0; i < 5; i++) sum += a[i];
        CHECK(fabsf(sum - 1.0f) < 1e-6f, "softmax: soma %.9g", (double)sum);
        CHECK(tt_argmax(a, 5) == 1, "softmax: maximo mudou de lugar");
        CHECK(bits_equal(a, b, 5, &bad),
              "softmax: deslocamento constante mudou a[%lld]", (long long)bad);
    }

    /* 8. top-k: valores, ordem e desempate pelo indice menor */
    {
        const float s[8] = { 0.1f, 0.9f, 0.5f, 0.9f, 0.2f, 0.7f, 0.5f, 0.0f };
        int32_t idx[4];
        float   val[4];
        tt_topk(s, 8, 4, idx, val);

        CHECK(idx[0] == 1 && idx[1] == 3, "topk: empate de 0,9 deu %d,%d",
              idx[0], idx[1]);
        CHECK(idx[2] == 5, "topk: terceiro deu %d, esperava 5", idx[2]);
        CHECK(idx[3] == 2, "topk: empate de 0,5 deu %d, esperava 2", idx[3]);
        CHECK(val[0] == 0.9f && val[3] == 0.5f, "topk: valores errados");
    }

    /* 9. argmax com empate fica no indice menor -- amostragem greedy tem de
     *    ser reprodutivel. */
    {
        const float v[4] = { 1.0f, 2.0f, 2.0f, 0.0f };
        CHECK(tt_argmax(v, 4) == 1, "argmax: empate nao ficou no menor indice");
    }

    /* 10. SIMD contra a referencia escalar.  Nao ha comparacao de bits aqui:
     *     8 lanes e FMA arredondam diferente, e o que se exige e que a
     *     diferenca fique no nivel do ruido de fp32, nao em zero. */
    tt_kernels_set_impl(TT_IMPL_AUTO);
    if (tt_kernels_impl() != TT_IMPL_SCALAR) {
        struct { const char *name; tt_kind kind; const void *data;
                 const float *scales; const float *want; } cases[] = {
            { "int4", TT_KIND_INT4, fx_w_int4, fx_scales_int4, fx_y_int4 },
            { "int8", TT_KIND_INT8, fx_w_int8, fx_scales_int8, fx_y_int8 },
            { "f32",  TT_KIND_F32,  fx_w_f32,  NULL,           fx_y_f32  },
            { "bf16", TT_KIND_BF16, fx_w_bf16, NULL,           fx_y_bf16 },
        };

        for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
            tt_tensor t = {0};
            t.name = cases[ci].name; t.name_len = strlen(cases[ci].name);
            t.kind = cases[ci].kind; t.ndim = 2;
            t.shape[0] = FX_O; t.shape[1] = FX_I;
            t.rows = FX_O; t.cols = FX_I;
            t.group = cases[ci].scales ? FX_GROUP : 0;
            t.data = cases[ci].data; t.scales = cases[ci].scales;

            if (tt_matmul(y, &t, fx_x) != 0) { tt_kernels_set_impl(saved); return -1; }
            for (int64_t o = 0; o < FX_O; o++) {
                const float want = cases[ci].want[o];
                const float tol  = 1e-4f * (fabsf(want) > 1.0f ? fabsf(want) : 1.0f);
                CHECK(fabsf(y[o] - want) <= tol,
                      "%s SIMD: y[%lld] = %.9g, escalar da %.9g (delta %.3g > %.3g)",
                      cases[ci].name, (long long)o, (double)y[o], (double)want,
                      (double)fabsf(y[o] - want), (double)tol);
            }
        }
    }

    /* 11. Lote contra o laco de chamadas simples, BIT A BIT, nos dois kernels.
     *
     *     Aqui nao ha tolerancia e nao pode haver: o lote existe para o
     *     prefill e para o treino, e um desvio de 1 ulp por termo vira um
     *     token diferente 47 camadas adiante.  O laco de chamadas simples e a
     *     definicao do resultado -- o kernel em lote responde a ele, nunca o
     *     contrario. */
    {
        /* 11 nao e multiplo do ladrilho de 8, de proposito: o resto do
         * ladrilho e onde um erro de indice se esconde. */
        enum { NB = 11 };
        static float xb[NB * FX_I], yb[NB * FX_O], ys[NB * FX_O];

        for (int b = 0; b < NB; b++)
            for (int64_t i = 0; i < FX_I; i++)
                xb[b * FX_I + i] = fx_x[i] * (1.0f + 0.125f * (float)b)
                                 - 0.03125f * (float)((b * 7) % 5);

        struct { const char *name; tt_kind kind; const void *data;
                 const float *scales; } cases[] = {
            { "int4", TT_KIND_INT4, fx_w_int4, fx_scales_int4 },
            { "int8", TT_KIND_INT8, fx_w_int8, fx_scales_int8 },
            { "f32",  TT_KIND_F32,  fx_w_f32,  NULL           },
            { "bf16", TT_KIND_BF16, fx_w_bf16, NULL           },
        };
        const tt_impl impls[2] = { TT_IMPL_SCALAR, TT_IMPL_AUTO };

        for (int ii = 0; ii < 2; ii++) {
            tt_kernels_set_impl(impls[ii]);
            const char *iname = tt_kernels_impl_name();

            for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
                tt_tensor t = {0};
                t.name = cases[ci].name; t.name_len = strlen(cases[ci].name);
                t.kind = cases[ci].kind; t.ndim = 2;
                t.shape[0] = FX_O; t.shape[1] = FX_I;
                t.rows = FX_O; t.cols = FX_I;
                t.group = cases[ci].scales ? FX_GROUP : 0;
                t.data = cases[ci].data; t.scales = cases[ci].scales;

                for (int64_t b = 0; b < NB; b++)
                    if (tt_matmul_rows(ys + b * FX_O, &t, xb + b * FX_I,
                                       0, FX_O) != 0) {
                        tt_kernels_set_impl(saved); return -1;
                    }

                /* lote inteiro de uma vez */
                if (tt_matmul_rows_b(yb, FX_O, &t, xb, FX_I, NB, 0, FX_O) != 0) {
                    tt_kernels_set_impl(saved); return -1;
                }
                CHECK(bits_equal(yb, ys, NB * FX_O, &bad),
                      "%s/%s em lote: y[%lld][%lld] = %.9g, a chamada simples da "
                      "%.9g -- a ordem de acumulacao mudou com o lote",
                      cases[ci].name, iname, (long long)(bad / FX_O),
                      (long long)(bad % FX_O), (double)yb[bad], (double)ys[bad]);

                /* e particionar o lote tambem nao pode mudar nada: o prefill
                 * vai chamar isto em pedacos conforme o KV enche */
                for (int64_t i = 0; i < NB * FX_O; i++) yb[i] = 12345.0f;
                if (tt_matmul_rows_b(yb, FX_O, &t, xb, FX_I, 3, 0, FX_O) != 0
                 || tt_matmul_rows_b(yb + 3 * FX_O, FX_O, &t, xb + 3 * FX_I,
                                     FX_I, NB - 3, 0, FX_O) != 0) {
                    tt_kernels_set_impl(saved); return -1;
                }
                CHECK(bits_equal(yb, ys, NB * FX_O, &bad),
                      "%s/%s: quebrar o lote em 3+%d mudou y[%lld]",
                      cases[ci].name, iname, NB - 3, (long long)bad);

                /* faixa de linhas, agora com lote */
                for (int64_t i = 0; i < NB * FX_O; i++) yb[i] = 12345.0f;
                if (tt_matmul_rows_b(yb, FX_O, &t, xb, FX_I, NB, 0, 3) != 0
                 || tt_matmul_rows_b(yb, FX_O, &t, xb, FX_I, NB, 3, FX_O) != 0) {
                    tt_kernels_set_impl(saved); return -1;
                }
                CHECK(bits_equal(yb, ys, NB * FX_O, &bad),
                      "%s/%s: faixas de linha com lote mudaram y[%lld]",
                      cases[ci].name, iname, (long long)bad);
            }
        }
    }

    /* 12. Os int4 fundidos contra a sequencia que substituem, BIT A BIT.
     *
     *     `tt_dot_int4g` e `tt_axpy_int4g` existem para a atencao absorvida,
     *     onde `tt_dequant_row` para um buffer seguido de `tt_dot`/`tt_axpy`
     *     escrevia 822 MB de f32 por token que ninguem mais lia.  Fundir e uma
     *     otimizacao de trafego, e uma otimizacao de trafego que muda o
     *     resultado nao e uma otimizacao -- e um bug com ganho de desempenho.
     *
     *     Sem tolerancia, e nas DUAS implementacoes: o escalar e a referencia,
     *     e o SIMD tem de bater com o SIMD que ele substitui (nao com o
     *     escalar -- esse e o teste 10 e tem tolerancia por bons motivos). */
    {
        tt_tensor t = {0};
        t.name = "fixture.int4"; t.name_len = 12;
        t.kind = TT_KIND_INT4; t.ndim = 2;
        t.shape[0] = FX_O; t.shape[1] = FX_I;
        t.rows = FX_O; t.cols = FX_I; t.group = FX_GROUP;
        t.data = fx_w_int4; t.scales = fx_scales_int4;

        const uint8_t *w4 = (const uint8_t *)fx_w_int4;
        const tt_impl impls[2] = { TT_IMPL_SCALAR, TT_IMPL_AUTO };

        for (int ii = 0; ii < 2; ii++) {
            tt_kernels_set_impl(impls[ii]);
            const char *iname = tt_kernels_impl_name();

            for (int64_t o = 0; o < FX_O; o++) {
                const uint8_t *wr = w4 + o * (FX_I / 2);
                const float   *sr = fx_scales_int4 + o * (FX_I / FX_GROUP);

                /* dot: dequant + tt_dot  ==  tt_dot_int4g */
                float row[FX_I];
                if (tt_dequant_row(row, &t, o) != 0) {
                    tt_kernels_set_impl(saved); return -1;
                }
                const float want = tt_dot(row, fx_x, FX_I);
                const float got  = tt_dot_int4g(fx_x, wr, sr, FX_I, FX_GROUP);
                CHECK(bits_equal(&got, &want, 1, &bad),
                      "int4/%s dot fundido: linha %lld deu %.9g, dequant+dot da "
                      "%.9g -- a ordem de acumulacao nao casa",
                      iname, (long long)o, (double)got, (double)want);

                /* axpy: dequant + tt_axpy  ==  tt_axpy_int4g.  O acumulador
                 * comeca sujo de proposito: axpy SOMA, e um kernel que
                 * escrevesse em vez de somar passaria com ele zerado. */
                const float k = 0.31830987f;
                float acc_a[FX_I], acc_b[FX_I];
                for (int64_t i = 0; i < FX_I; i++)
                    acc_a[i] = acc_b[i] = 0.5f - 0.001f * (float)i;

                tt_axpy(acc_a, k, row, FX_I);
                tt_axpy_int4g(acc_b, k, wr, sr, FX_I, FX_GROUP);
                CHECK(bits_equal(acc_b, acc_a, FX_I, &bad),
                      "int4/%s axpy fundido: linha %lld, elemento %lld deu %.9g, "
                      "dequant+axpy da %.9g",
                      iname, (long long)o, (long long)bad,
                      (double)acc_b[bad], (double)acc_a[bad]);
            }
        }
    }

    tt_kernels_set_impl(saved);
    return 0;
}
