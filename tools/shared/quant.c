#include "quant.h"

#include <math.h>
#include <string.h>

/* Clip search grid: fractions of the group's absmax, from 1.00 down to 0.70.
 *
 * `absmax/qmax` -- the obvious choice -- optimises the group's WORST weight
 * instead of the group.  Shrinking the scale clips the extremes and in exchange
 * tightens the step for every other weight; with 16 levels that trade pays off
 * almost always.  docs/04 measured it: output error of int4 g64 goes from 10.89%
 * to 9.63%, and the format does not change -- still one f32 scale per group, and
 * the kernel cannot tell the difference.  In int8 the gain is zero (255 levels
 * already make absmax essentially optimal), and the search stays there only to
 * avoid opening a special case.
 *
 * Side effect that pays for itself: with ratio < 1 the code -8 becomes
 * reachable.  With the exact absmax/7 scale, round-to-nearest never emits -8 and
 * int4 uses 15 of the 16 levels the dequantiser already knows how to read. */
static float g_clip[TT_CLIP_MAX] = { 1.0f };
static int   g_nclip = 1;

void tt_clip_grid_init(int points)
{
    if (points < 1) points = 1;
    if (points > TT_CLIP_MAX) points = TT_CLIP_MAX;

    g_nclip = points;
    if (points == 1) { g_clip[0] = 1.0f; return; }
    for (int i = 0; i < points; i++)
        g_clip[i] = (float)(1.0 - 0.30 * (double)i / (double)(points - 1));
    g_clip[points - 1] = 0.70f;      /* exact endpoint, not 0.7000001 */
}

int tt_clip_grid_points(void) { return g_nclip; }

/* `rintf` and not `floor(x+0.5)`: round-to-nearest-even is what the dequantiser
 * assumes and what `fixture_kernels.h` freezes.
 *
 * The search keeps the FIRST scale that attains the minimum, and the grid is
 * walked from 1.00 downwards -- so a tie keeps the larger ratio, the one that
 * clips less.  Error accumulates in double: the grid decides between candidates
 * that differ in the fourth digit, and a float32 sum of 64 squares is not always
 * able to order them. */
void tt_quant_group(const float *src, int group, int qmax,
                    float *scale_out, int8_t *codes)
{
    float absmax = 0.0f;
    for (int j = 0; j < group; j++) {
        const float v = fabsf(src[j]);
        if (v > absmax) absmax = v;
    }

    /* absmax == 0: scale 1 avoids the division by zero and reconstructs
     * exactly -- every code is 0. */
    if (absmax <= 0.0f) {
        *scale_out = 1.0f;
        memset(codes, 0, (size_t)group);
        return;
    }

    const float base = absmax / (float)qmax;
    float  best_s   = base;
    double best_err = -1.0;

    for (int c = 0; c < g_nclip; c++) {
        const float s   = base * g_clip[c];
        const float inv = 1.0f / s;
        double err = 0.0;
        for (int j = 0; j < group; j++) {
            float q = rintf(src[j] * inv);
            if (q > (float)qmax) q = (float)qmax;
            if (q < -(float)(qmax + 1)) q = -(float)(qmax + 1);
            const double d = (double)(q * s) - (double)src[j];
            err += d * d;
        }
        if (best_err < 0.0 || err < best_err) { best_err = err; best_s = s; }
    }

    *scale_out = best_s;

    const float inv = 1.0f / best_s;
    for (int j = 0; j < group; j++) {
        float q = rintf(src[j] * inv);
        if (q > (float)qmax) q = (float)qmax;
        if (q < -(float)(qmax + 1)) q = -(float)(qmax + 1);
        codes[j] = (int8_t)q;
    }
}

void tt_pack_int4(const int8_t *codes, int group, uint8_t *out)
{
    const int half = group / 2;
    for (int j = 0; j < half; j++) {
        const uint8_t lo = (uint8_t)((codes[j] + 8) & 0x0F);
        const uint8_t hi = (uint8_t)((codes[j + half] + 8) & 0x0F);
        out[j] = (uint8_t)(lo | (hi << 4));
    }
}

/* Deliberadamente independente de `tt_dequant_row` de `src/kernels.c`: e com
 * esta volta que a fixture do selftest e calculada, e uma fixture gerada pelo
 * proprio kernel que ela deveria vigiar nao vigia nada. */
void tt_quant_unpack_row(tt_kind kind, const uint8_t *row, const float *scales,
                         int group, int64_t cols, float *dst)
{
    const int64_t ngroups = cols / group;

    if (kind == TT_KIND_INT8) {
        const int8_t *q = (const int8_t *)row;
        for (int64_t g = 0; g < ngroups; g++)
            for (int j = 0; j < group; j++)
                dst[g * group + j] = (float)q[g * group + j] * scales[g];
        return;
    }

    const int half = group / 2;
    for (int64_t g = 0; g < ngroups; g++) {
        const uint8_t *p = row + g * (int64_t)half;
        for (int j = 0; j < half; j++) {
            dst[g * group + j]        = (float)((int)(p[j] & 0x0F) - 8) * scales[g];
            dst[g * group + half + j] = (float)((int)(p[j] >> 4)   - 8) * scales[g];
        }
    }
}
