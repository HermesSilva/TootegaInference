/* sample.h -- amostragem.  Resolve D9 de docs/13.
 *
 * Greedy, temperatura, top-k e top-p.  Trivial de escrever e facil de errar em
 * dois pontos que importam:
 *
 *   - ids de padding.  `vocab_size` e 154.880 contra 154.856 em uso; a folga e
 *     preenchimento do `lm_head` e NUNCA deve ser amostrada (docs/16).
 *   - reprodutibilidade.  O gerador e explicito e semeado; sem isso, "rodou
 *     diferente" nunca se sabe se foi o sampler ou o engine.
 */
#ifndef TOOTEGA_SAMPLE_H
#define TOOTEGA_SAMPLE_H

#include <stdint.h>

typedef struct {
    float    temperature;      /* <= 0 -> greedy, e o resto e ignorado */
    float    top_p;            /* <= 0 ou >= 1 desliga */
    int      top_k;            /* <= 0 desliga */
    uint64_t state;
} tt_sampler;

void tt_sampler_init(tt_sampler *s, float temperature, float top_p, int top_k,
                     uint64_t seed);

/* `limit` e o primeiro id que NAO existe -- tudo dali para cima e padding.
 * `logits` e modificado. */
int32_t tt_sample(tt_sampler *s, float *logits, int64_t limit);

int tt_sample_selftest(void);

#endif /* TOOTEGA_SAMPLE_H */
