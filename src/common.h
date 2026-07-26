/* common.h -- error, log and arena.  The three pieces everything else uses.
 *
 * The arena is the answer to docs/11-memory.md: working memory is sized once at
 * init and after that only a pointer moves.  Zero `malloc` in the token loop is
 * an M1 exit requirement, not a style preference.
 */
#ifndef TOOTEGA_COMMON_H
#define TOOTEGA_COMMON_H

#include <stddef.h>
#include <stdint.h>

/* -------------------------------------------------------------- error */

/* Records the message and returns -1, always.  Allows `return tt_fail(...)`. */
int         tt_fail(const char *fmt, ...);
const char *tt_error(void);        /* last message, "" if none */
void        tt_error_clear(void);

/* ---------------------------------------------------------------- log */

void tt_log(const char *fmt, ...);      /* stderr, always */
void tt_dbg(const char *fmt, ...);      /* stderr, only with verbose on */
void tt_verbose(int on);
int  tt_is_verbose(void);

/* -------------------------------------------------------------- arena */

/* Bump allocator.  There is no individual free: you either reset everything or
 * destroy it.  `peak` exists for the budget report in docs/09. */
typedef struct {
    unsigned char *base;
    size_t         size;
    size_t         used;
    size_t         peak;
    const char    *name;
} tt_arena;

int   tt_arena_init(tt_arena *a, size_t bytes, const char *name);
void  tt_arena_free(tt_arena *a);
void  tt_arena_reset(tt_arena *a);

/* Alignment in bytes, a power of two.  Returns NULL and records an error on
 * overflow -- arena overflow is a sizing bug, never a normal condition, so the
 * message says how much was missing. */
void *tt_arena_alloc(tt_arena *a, size_t bytes, size_t align);

/* Shortcuts with the 64-byte alignment docs/14 requires of the blobs and the
 * SIMD kernels assume. */
#define TT_ALIGN 64
#define tt_arena_new(a, type, n) \
    ((type *)tt_arena_alloc((a), sizeof(type) * (size_t)(n), TT_ALIGN))

/* --------------------------------------------------------------- util */

static inline size_t tt_align_up(size_t v, size_t align)
{
    return (v + align - 1) & ~(align - 1);
}

/* Formats bytes as "15.63 GB" / "812.4 MB".  Caller's buffer, >= 32. */
const char *tt_bytes(char *buf, size_t buflen, uint64_t bytes);

/* Hash of the `src/` sources this binary was built from (docs/10).
 *
 * It exists because a measurement tool links `src/` statically: after fixing the
 * forward, an `.exe` that was not rebuilt keeps measuring the old code -- and a
 * wrong number does not announce itself.  With the stamp printed in every
 * report, the divergence becomes visible. */
const char *tt_build_id(void);

/* Engine version.  It lives here and not in `main.c` because the server
 * publishes it in `/stats` and the UI shows it in the header -- two copies of
 * the same constant would diverge at the first bump. */
#define TT_VERSION "0.1.0-m1"

#endif /* TOOTEGA_COMMON_H */
