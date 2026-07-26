#include "sample.h"
#include "common.h"
#include "kernels.h"

#include <math.h>
#include <string.h>

#define TT_MAX_CAND 256

void tt_sampler_init(tt_sampler *s, float temperature, float top_p, int top_k,
                     uint64_t seed)
{
    s->temperature = temperature;
    s->top_p = top_p;
    s->top_k = top_k;
    /* xorshift64* must never start from zero. */
    s->state = seed ? seed : 0x9E3779B97F4A7C15ull;
}

static float next_unit(tt_sampler *s)
{
    uint64_t x = s->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    s->state = x;
    /* 24 mantissa bits: enough to choose among 256 candidates and
     * exatamente representavel em float. */
    return (float)((x * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}

int32_t tt_sample(tt_sampler *s, float *logits, int64_t limit)
{
    if (limit <= 0) return 0;

    if (s->temperature <= 0.0f)
        return (int32_t)tt_argmax(logits, limit);

    /* Working over the best K instead of over 154,880 logits: top-p without
     * ordenacao total, e o custo deixa de depender do vocabulario. */
    int k = s->top_k > 0 && s->top_k < TT_MAX_CAND ? s->top_k : TT_MAX_CAND;
    if ((int64_t)k > limit) k = (int)limit;

    int32_t idx[TT_MAX_CAND];
    float   val[TT_MAX_CAND];
    tt_topk(logits, limit, k, idx, val);

    /* Softmax over the candidates, temperature already applied.
     *
     * `max` is copied BEFORE the loop.  Using `val[0]` inside it was a real
     * bug: at iteration 0, `val[0]` itself becomes exp(0) = 1, and the rest
     * then subtract 1 instead of the maximum.  With logits around 100 that
     * exp(330) -> infinito -> soma infinita -> probabilidades NaN -> nenhuma
     * the draw comparison is always true, and the sampler always returned the
     * candidato do top-k.
     *
     * Symptom: temperature 0 perfect (it returns before here) and any
     * temperature > 0 producing text plausible word by word and with no
     * sentido nenhum. */
    const float max = val[0];
    const float inv_t = 1.0f / s->temperature;
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        val[i] = expf((val[i] - max) * inv_t);
        sum += val[i];
    }
    for (int i = 0; i < k; i++) val[i] /= sum;

    /* top-p: keep the shortest prefix whose mass exceeds p.  Always at least
     * one candidate, or a too-small p would leave nothing. */
    int n = k;
    if (s->top_p > 0.0f && s->top_p < 1.0f) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) {
            acc += val[i];
            if (acc >= s->top_p) { n = i + 1; break; }
        }
    }

    float mass = 0.0f;
    for (int i = 0; i < n; i++) mass += val[i];

    const float r = next_unit(s) * mass;
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += val[i];
        if (r < acc) return idx[i];
    }
    return idx[n - 1];
}

/* ----------------------------------------------------------- selftest */

int tt_sample_selftest(void)
{
    float logits[8];
    tt_sampler s;

    /* Greedy ignora a semente e o resto. */
    tt_sampler_init(&s, 0.0f, 0.9f, 5, 42);
    for (int i = 0; i < 8; i++) logits[i] = (float)i * 0.5f;
    logits[3] = 10.0f;
    if (tt_sample(&s, logits, 8) != 3)
        return tt_fail("sample: greedy did not pick the maximum");

    /* Padding never comes out: the largest logit is beyond the limit. */
    for (int i = 0; i < 8; i++) logits[i] = 1.0f;
    logits[7] = 100.0f;
    tt_sampler_init(&s, 0.0f, 0.0f, 0, 1);
    if (tt_sample(&s, logits, 6) >= 6)
        return tt_fail("sample: amostrou id de padding");

    /* top-k = 1 is deterministic even with temperature. */
    for (int i = 0; i < 8; i++) logits[i] = (float)(8 - i);
    for (int trial = 0; trial < 16; trial++) {
        float tmp[8];
        memcpy(tmp, logits, sizeof(tmp));
        tt_sampler_init(&s, 1.0f, 1.0f, 1, (uint64_t)trial + 1);
        if (tt_sample(&s, tmp, 8) != 0)
            return tt_fail("sample: top-k=1 was not deterministic");
    }

    /* A low temperature has to converge to greedy, and the probabilities have
     * to be finite.
     *
     * This test exists because the earlier ones did not catch the bug there
     * was here: greedy returns before the softmax, and top-k=1 only looks at
     * val[0].  The defect lived exactly between the two -- k > 1 with
     * temperature > 0 -- and with LARGE logits, the real regime (~100 here).
     *
     * With temperature 0.05 and a 5-point logit lead, the top has to come out
     * practically every time. */
    {
        float big[8];
        for (int i = 0; i < 8; i++) big[i] = 100.0f - (float)i * 5.0f;

        int top = 0;
        for (int trial = 0; trial < 64; trial++) {
            float tmp[8];
            memcpy(tmp, big, sizeof(tmp));
            tt_sampler_init(&s, 0.05f, 0.95f, 8, (uint64_t)trial + 7);
            const int32_t got = tt_sample(&s, tmp, 8);
            if (got == 0) top++;
            if (got < 0 || got >= 8)
                return tt_fail("sample: index %d out of [0,8)", got);
        }
        if (top < 60)
            return tt_fail("sample: temperature 0.05 chose the top only "
                           "%d/64 vezes -- softmax ou normalizacao errada", top);
    }

    /* Mesma semente, mesma sequencia. */
    int32_t a[8], b[8];
    for (int pass = 0; pass < 2; pass++) {
        tt_sampler_init(&s, 1.0f, 0.95f, 40, 12345);
        for (int i = 0; i < 8; i++) {
            float tmp[8];
            for (int j = 0; j < 8; j++) tmp[j] = (float)((i + j) % 5);
            int32_t v = tt_sample(&s, tmp, 8);
            if (pass == 0) a[i] = v; else b[i] = v;
        }
    }
    if (memcmp(a, b, sizeof(a)) != 0)
        return tt_fail("sample: mesma semente deu sequencia diferente");

    return 0;
}
