#include "kernels.h"
#include "backend.h"
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
} mm_job;

static void mm_job_run(void *arg, int64_t row0, int64_t row1)
{
    const mm_job *j = (const mm_job *)arg;
    mm_rows_dispatch(j->y, j->t, j->x, row0, row1);
}

/* Abaixo disto a barreira custa mais que o trabalho.  O roteador (64x2048 =
 * 131k MACs) fica de fora de proposito; um dispatch por camada so para ele
 * seriam ~2.800 barreiras por token. */
#define TT_PARALLEL_MIN_MACS (256 * 1024)

typedef struct {
    float           **y;
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

    /* Residentes na VRAM vao juntos, numa ida so; o resto segue na CPU.
     *
     * A separacao e por tensor e nao por camada porque a colocacao e por
     * prioridade de trafego (docs/18): a atencao de todas as camadas sobe
     * antes do primeiro expert, entao uma camada pode ter metade de cada
     * lado. */
    const tt_backend *be = tt_backend_get();
    if (be && be->matmul_many) {
        float        *dev_y[TT_BATCH_MAX];
        tt_dev_tensor dev_w[TT_BATCH_MAX];
        const tt_tensor *cpu_w[TT_BATCH_MAX];
        float           *cpu_y[TT_BATCH_MAX];
        int nd = 0, nc = 0;

        for (int i = 0; i < n; i++) {
            if (w[i]->dev) { dev_y[nd] = y[i]; dev_w[nd++] = w[i]->dev; }
            else           { cpu_y[nc] = y[i]; cpu_w[nc++] = w[i];      }
        }
        if (nd) {
            if (be->matmul_many(dev_y, dev_w, nd, x) != 0)
                return tt_fail("backend falhou no batch de %d", nd);
            if (!nc) return 0;
            y = cpu_y;
            w = cpu_w;
            n = nc;
        }
    }

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

    /* Residente na VRAM: o peso nem e tocado aqui.  Se o backend falhar, cai
     * para a CPU em silencio -- uma placa que sumiu no meio da geracao nao
     * pode derrubar a resposta. */
    if (t->dev) {
        const tt_backend *be = tt_backend_get();
        if (be && be->matmul(y, t->dev, x) == 0) return 0;
    }

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

    mm_job job = { y, t, x };
    tt_pool_run(g_pool, mm_job_run, &job, t->rows, grain);
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

    tt_kernels_set_impl(saved);
    return 0;
}
