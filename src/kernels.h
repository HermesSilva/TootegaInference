/* kernels.h -- the arithmetic.  Deliberately naive, for now.
 *
 * docs/15-kernels-and-numerics.md fixes three things this file obeys:
 *
 *   W4A32   Unpack the weight to float and multiply with FMA.  Decode is
 *           memory-bound (~33 ms of memory against 4-7 ms of compute), so
 *           spending the compute slack to buy precision is the right trade.
 *
 *   fp32    A float accumulator, nothing in double.  The goal is to MATCH the
 *           oracle, not to beat it.
 *
 *   fixed   The reduction along I happens entirely inside one thread, in the
 *   order   order 0..I-1, with a scalar accumulator per row.  Threads never sum
 *           each other's partials -- if they did, the result would change with
 *           --threads and the determinism test would fail by construction.
 *
 * That is why every matmul function works over a ROW RANGE: it is how the pool
 * partitions the work without touching the reduction order.
 *
 * Implementation order (docs/15): naive -> oracle passing -> SIMD -> threads.
 * We are on the first step, and it stays as the executable reference for the
 * next ones.
 */
#ifndef TOOTEGA_KERNELS_H
#define TOOTEGA_KERNELS_H

#include <stdint.h>

#include "container.h"

/* ----------------------------------------------------- implementation */

/* The vectorised kernel and the naive one do NOT produce the same bits: SIMD
 * accumulates in 8 lanes and uses FMA (one rounding instead of two).  Both are
 * valid -- neither is "the right one" -- but only one can be the reference, and
 * the reference is the naive one, which matches Python.  Being able to switch at
 * runtime is what turns "SIMD diverged" into a diagnosis that takes minutes
 * (docs/15). */
typedef enum {
    TT_IMPL_AUTO = 0,      /* SIMD when available */
    TT_IMPL_SCALAR,        /* the naive reference, always */
} tt_impl;

void        tt_kernels_set_impl(tt_impl impl);
tt_impl     tt_kernels_impl(void);
const char *tt_kernels_impl_name(void);

/* Pool used by the matmuls.  NULL = everything on the calling thread.  The
 * partition is always by output row, so turning the pool on or off does not
 * change a single bit of the result -- which is what the determinism test
 * verifies. */
struct tt_pool;
void           tt_kernels_set_pool(struct tt_pool *p);
struct tt_pool *tt_kernels_pool(void);

/* ------------------------------------------------------------- matmul */

/* y[o] = sum_i W[o][i] * x[i], for o in [row0, row1).
 *
 * W is always [O, I] with I contiguous -- the reduction dimension.  The
 * quantised variants group along I, `group` weights per scale. */

void tt_matmul_f32_rows (float *y, const float *w, const float *x,
                         int64_t cols, int64_t row0, int64_t row1);

void tt_matmul_bf16_rows(float *y, const uint16_t *w, const float *x,
                         int64_t cols, int64_t row0, int64_t row1);

void tt_matmul_int8_rows(float *y, const int8_t *w, const float *scales,
                         const float *x, int64_t cols, int group,
                         int64_t row0, int64_t row1);

/* int4 packed half/half: inside each group, byte[j] holds weight j in the low
 * nibble and weight j + group/2 in the high one (docs/14). */
void tt_matmul_int4_rows(float *y, const uint8_t *w, const float *scales,
                         const float *x, int64_t cols, int group,
                         int64_t row0, int64_t row1);

/* Dispatch by the tensor's `kind`.  Returns -1 if the tensor is not a
 * multipliable matrix (1-dimensional, for instance). */
int tt_matmul_rows(float *y, const tt_tensor *w, const float *x,
                   int64_t row0, int64_t row1);
int tt_matmul(float *y, const tt_tensor *w, const float *x);

/* ------------------------------------------------------- matmul in batch */

/* The SAME weight against `batch` different inputs:
 *
 *     y[b*ldy + o] = sum_i W[o][i] * x[b*ldx + i]
 *
 * Decode reads one weight per token and throws it away, which is why it is
 * memory-bound (docs/08).  Prefill and training are not: they have many inputs
 * for the same weight, and reading it once instead of `batch` times is the
 * whole difference.  For int4 there is a second saving, larger than it looks --
 * unpacking a group costs an `and`, a `shift`, two converts and two multiplies,
 * and batched it happens ONCE for the whole batch instead of once per input.
 *
 * The reduction order is untouched.  The batch loop sits INSIDE, around the
 * FMA, so element `b` sees exactly the sequence of operations it would see
 * alone -- which is what lets prefill be batched without changing a single
 * token (docs/15: the order is fixed and independent of how the work is split).
 * `tt_kernels_selftest` checks that bit for bit against the loop of singles.
 *
 * `batch == 1` is the same as the non-batched call, and costs the same. */

void tt_matmul_f32_rows_b (float *y, int64_t ldy, const float *w,
                           const float *x, int64_t ldx, int64_t batch,
                           int64_t cols, int64_t row0, int64_t row1);

void tt_matmul_bf16_rows_b(float *y, int64_t ldy, const uint16_t *w,
                           const float *x, int64_t ldx, int64_t batch,
                           int64_t cols, int64_t row0, int64_t row1);

void tt_matmul_int8_rows_b(float *y, int64_t ldy, const int8_t *w,
                           const float *scales, const float *x, int64_t ldx,
                           int64_t batch, int64_t cols, int group,
                           int64_t row0, int64_t row1);

void tt_matmul_int4_rows_b(float *y, int64_t ldy, const uint8_t *w,
                           const float *scales, const float *x, int64_t ldx,
                           int64_t batch, int64_t cols, int group,
                           int64_t row0, int64_t row1);

int tt_matmul_rows_b(float *y, int64_t ldy, const tt_tensor *w,
                     const float *x, int64_t ldx, int64_t batch,
                     int64_t row0, int64_t row1);

/* Whole-tensor version, threaded through the pool like `tt_matmul`. */
int tt_matmul_b(float *y, int64_t ldy, const tt_tensor *w,
                const float *x, int64_t ldx, int64_t batch);

/* Several INDEPENDENT matmuls over the same input, in a single dispatch.
 *
 * It exists because of a barrier calculation: a 1536x2048 expert takes 26 us
 * with 20 threads, and synchronising 20 threads costs the same order.  One MoE
 * layer does 15 of those (4 experts + shared, gate/up/down) -- 15 barriers for
 * 400 us of work.  Fused, that is 2 barriers, and the threads see a row queue
 * large enough to balance themselves.
 *
 * The docs/15 rule still holds: each thread owns whole output rows and the
 * reduction along I happens inside it.  Fusing changes who takes which row,
 * never the order of the sum. */
#define TT_BATCH_MAX 16
int tt_matmul_batch(float **y, const tt_tensor **w, int n, const float *x);

/* -------------------------------------------------------- activations */

/* rmsnorm: y[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i].
 * The sum of squares is float32 in fixed order. */
void tt_rmsnorm(float *y, const float *x, const float *weight,
                int64_t n, float eps);

/* max-subtracted softmax, in place. */
void tt_softmax(float *x, int64_t n);

/* SwiGLU: dst[i] = silu(gate[i]) * up[i], with silu(v) = v / (1 + exp(-v)). */
void tt_silu_mul(float *dst, const float *gate, const float *up, int64_t n);

/* Dot product and dst += k*src.  They are attention's loop: with 4k of context,
 * 47 layers and 20 heads, that is ~2 GFLOP per token in here alone. */
float tt_dot(const float *a, const float *b, int64_t n);
void  tt_axpy(float *dst, float k, const float *src, int64_t n);

/* Versions against a bf16 vector -- the KV cache.  bf16 has fp32's same 8
 * exponent bits and 7 of mantissa: it does not change the range, only the
 * precision, and it halves the traffic. */
float tt_dot_bf16(const float *a, const uint16_t *b, int64_t n);
void  tt_axpy_bf16(float *dst, float k, const uint16_t *src, int64_t n);
void  tt_store_bf16(uint16_t *dst, const float *src, int64_t n);

/* The same against int8 with a per-group scale -- the container's symmetric
 * quantisation (docs/14): scale = absmax/127, no zero point.  It errs 0.59%
 * against bf16's ~1.5% AND takes less space: 576 int8 + 9 scales = 612 B against
 * bf16's 1,152 B and f32's 2,304 B. */
float tt_dot_int8g(const float *a, const int8_t *w, const float *scales,
                   int64_t n, int group);
void  tt_axpy_int8g(float *dst, float k, const int8_t *w, const float *scales,
                    int64_t n, int group);
void  tt_store_int8g(int8_t *dst, float *scales, const float *src,
                     int64_t n, int group);

void tt_add(float *dst, const float *src, int64_t n);      /* residual */
void tt_scale(float *dst, float k, int64_t n);
void tt_copy(float *dst, const float *src, int64_t n);
void tt_zero(float *dst, int64_t n);

int64_t tt_argmax(const float *x, int64_t n);

/* top-k by partial selection, descending and stable on ties (lower index
 * first).  It is the MoE router: k = 4 of 64 (docs/04). */
void tt_topk(const float *x, int64_t n, int k, int32_t *idx_out, float *val_out);

/* ------------------------------------------------------------ dequant */

static inline float tt_bf16_to_f32(uint16_t h)
{
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)h << 16;
    return c.f;
}

/* Materialises one row of the tensor in float32.  It is the executable
 * definition of the layout: if this and the kernel diverge, the kernel is
 * wrong. */
int tt_dequant_row(float *dst, const tt_tensor *t, int64_t row);

int tt_kernels_selftest(void);

#endif /* TOOTEGA_KERNELS_H */
