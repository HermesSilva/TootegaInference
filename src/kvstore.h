/* kvstore.h -- the KV of a fixed prefix, on disk (docs/12, section 4).
 *
 * The KV cache is the aggregate of everything the model has read so far: 576
 * values per position per layer that stand for the whole forward of that
 * position.  Recomputing it costs a full pass over the weights per token --
 * 2.38 GB of memory traffic each; reading it back costs its own size once,
 * 108 KB per position.  For the prefix every session starts with -- system
 * prompt, tool schemas -- that is the difference between minutes and a disk
 * read, and it changes no bit: what comes back is what was computed.
 *
 * The rule that governs it was paid for elsewhere (docs/12): a KV persisted
 * from another conversation, partially matching the new prompt, was resumed and
 * produced text for a prompt nobody sent.  So:
 *
 *   a file is usable only if it was produced by EXACTLY the token sequence it
 *   is being asked to stand for, by this model, by this build, in this KV
 *   layout.  There is no partial match.  Anything else is ignored, with one
 *   log line naming the field that diverged.
 *
 * Only the ABSORBED formulation is stored: it is the default and the only one
 * whose cache is small enough to be worth a file (docs/13, D11).
 */
#ifndef TOOTEGA_KVSTORE_H
#define TOOTEGA_KVSTORE_H

#include <stddef.h>
#include <stdint.h>

#include "forward.h"

/* Word-wise hash, ours.  Integrity against truncation and bit rot, and file
 * naming -- not a defence against an adversary. */
uint64_t tt_hash64(uint64_t seed, const void *data, size_t bytes);

/* What identifies the weights a KV was computed with: the container's size, its
 * whole index, and a sample from the middle of every tensor.
 *
 * A sample and not the full 19 GB on purpose -- hashing everything would cost
 * more than the prefill it saves on a short prefix.  It is enough to tell apart
 * two containers of the same shape (v4 and a fine-tune of v4 have identical
 * indexes apart from offsets), because requantising a tensor moves essentially
 * every byte of it. */
uint64_t tt_model_fingerprint(const tt_model *m);

/* Where the KV of `ids[0..n)` lives inside `dir`: "<dir>/<hash>.ttkv". */
void tt_kv_path(char *out, size_t cap, const char *dir,
                const int32_t *ids, int64_t n);

/* Writes the first `n` positions of the state.  Atomic: a temporary file,
 * flushed to the device, then renamed -- a crash leaves either the old file or
 * the new one, never half of one.  The state must hold at least `n` positions. */
int tt_kv_save(const tt_state *s, const char *path, uint64_t fingerprint,
               int64_t n);

/* Restores the KV of `ids[0..n)` if `path` holds exactly that.
 *
 * Returns `n` and leaves the state at `len == n`, or returns 0 and says why in
 * the log.  A file that fails AFTER the cache was written to -- short read, bad
 * checksum -- leaves the state EMPTY (`len == 0`), never half restored.
 *
 * The logits are not part of a KV: after a load the caller still has to run the
 * last position, which is what `tt_state_match_prefix` already arranges. */
int64_t tt_kv_load(tt_state *s, const char *path, uint64_t fingerprint,
                   const int32_t *ids, int64_t n);

/* Bytes the file of an `n`-position prefix takes on disk. */
uint64_t tt_kv_file_bytes(const tt_state *s, int64_t n);

/* Makes room in `dir` for a file of `incoming` bytes under a total of `budget`.
 *
 * Least recently USED goes first: a load refreshes the file's modification
 * time, so the prefix a session keeps coming back to is the last to go.  A
 * leftover `.tmp` -- a write that a crash interrupted -- is removed outright;
 * it is nobody's cache.
 *
 * Returns 0 when the new file fits, -1 when it would not fit even in an empty
 * directory (and then nothing is removed).  `removed`, if not NULL, receives
 * how many files went. */
int tt_kv_evict(const char *dir, uint64_t budget, uint64_t incoming, int *removed);

int tt_kvstore_selftest(void);

#endif /* TOOTEGA_KVSTORE_H */
