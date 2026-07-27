/* quant.h -- the quantiser, shared by the converter and the fixture generator.
 *
 * It lives in `tools/shared/` and not in `src/`: the engine READS quantised
 * weights and never produces them, so a quantiser inside it would be surface
 * nobody exercises (CLAUDE.md).
 *
 * It is shared because `src/fixture_kernels.h` only proves anything if it comes
 * out of the SAME quantiser the container was written with.  Two lookalike
 * implementations agreeing is not a check -- it is a coincidence waiting to
 * break (docs/13, D3).
 *
 * Format: symmetric, per group, no zero-point -- docs/14.
 */
#ifndef TOOTEGA_TOOLS_QUANT_H
#define TOOTEGA_TOOLS_QUANT_H

#include <stdint.h>

#include "container.h"

/* Clip search grid: how many fractions of absmax to try between 0.70 and 1.00.
 * 1 means pure absmax.  Default is 13, the number docs/04 measured. */
#define TT_CLIP_MAX 64

void tt_clip_grid_init(int points);
int  tt_clip_grid_points(void);

/* `qmax` is 7 for int4 and 127 for int8; `codes` receives `group` values in
 * [-(qmax+1), qmax]. */
void tt_quant_group(const float *src, int group, int qmax,
                    float *scale_out, int8_t *codes);

/* Half/half inside the group: byte[j] carries weight j in the low nibble and
 * j + group/2 in the high one.  Unpacking with SIMD then yields two contiguous
 * vectors with no activation shuffle (docs/14, D1). */
void tt_pack_int4(const int8_t *codes, int group, uint8_t *out);

/* One row back to float, the way the kernel would read it.  `row` points at the
 * row's codes (packed, for int4) and `scales` at its group scales. */
void tt_quant_unpack_row(tt_kind kind, const uint8_t *row, const float *scales,
                         int group, int64_t cols, float *dst);

#endif /* TOOTEGA_TOOLS_QUANT_H */
