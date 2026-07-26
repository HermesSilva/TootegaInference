/* sample.h -- sampling.  Settles D9 of docs/13.
 *
 * Greedy, temperature, top-k and top-p.  Trivial to write and easy to get wrong
 * in the two places that matter:
 *
 *   - padding ids.  `vocab_size` is 154,880 against 154,856 in use; the slack is
 *     `lm_head` padding and must NEVER be sampled (docs/16).
 *   - reproducibility.  The generator is explicit and seeded; without that, when
 *     a run "came out different" you never know whether it was the sampler or
 *     the engine.
 */
#ifndef TOOTEGA_SAMPLE_H
#define TOOTEGA_SAMPLE_H

#include <stdint.h>

typedef struct {
    float    temperature;      /* <= 0 -> greedy, and the rest is ignored */
    float    top_p;            /* <= 0 or >= 1 turns it off */
    int      top_k;            /* <= 0 turns it off */
    uint64_t state;
} tt_sampler;

void tt_sampler_init(tt_sampler *s, float temperature, float top_p, int top_k,
                     uint64_t seed);

/* `limit` is the first id that does NOT exist -- everything from there up is
 * padding.  `logits` is modified. */
int32_t tt_sample(tt_sampler *s, float *logits, int64_t limit);

int tt_sample_selftest(void);

#endif /* TOOTEGA_SAMPLE_H */
