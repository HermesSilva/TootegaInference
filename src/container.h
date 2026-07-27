/* container.h -- the `.tootega` reader.
 *
 * The normative specification is docs/14-container-format.md; the writer is
 * `tools/convert`.  This reader has to match it byte for byte, and so it
 * validates everything the spec promises instead of trusting: magic, version,
 * the 64-byte alignment of every blob, and the relationship between `shape`,
 * `group` and each blob's size.  A container that gets past here can be read by
 * the kernels with no further checking.
 *
 * Zero copy: every pointer returned points inside the file's read-only mapping.
 * Nothing is read into a buffer of our own, not even the index.
 */
#ifndef TOOTEGA_CONTAINER_H
#define TOOTEGA_CONTAINER_H

#include <stdint.h>

#include "common.h"
#include "compat.h"
#include "json.h"

typedef enum {
    TT_KIND_INT4 = 0,
    TT_KIND_INT8,
    TT_KIND_F32,
    TT_KIND_BF16,
    TT_KIND_COUNT
} tt_kind;

const char *tt_kind_name(tt_kind k);

#define TT_MAX_DIMS 4

typedef struct {
    const char *name;          /* inside the index, not terminated */
    size_t      name_len;
    tt_kind     kind;

    int64_t     shape[TT_MAX_DIMS];
    int         ndim;
    int64_t     rows;          /* O -- product of the dims minus the last */
    int64_t     cols;          /* I -- last dim, the reduction one */

    const void *data;          /* inside the mapping, aligned to 64 */
    uint64_t    data_off, data_bytes;

    const float *scales;       /* NULL for f32/bf16 */
    uint64_t     scales_off, scales_bytes;

    int32_t     group;         /* 0 when there is no scale */
} tt_tensor;

typedef struct {
    tt_map      map;
    tt_arena    arena;         /* index DOM + tensor table */

    uint32_t    format_version;
    uint64_t    index_off, index_len;

    const tt_json *root;       /* {meta, tensors} */
    const tt_json *meta;
    const tt_json *config;     /* meta.config -- the architecture comes from here */

    tt_tensor  *tensors;
    size_t      n_tensors;

    uint32_t   *buckets;       /* open hashing: 0 empty, i+1 = tensors[i] */
    size_t      n_buckets;

    /* The accounting the docs/09 budget uses. */
    uint64_t    bytes_by_kind[TT_KIND_COUNT];
    size_t      count_by_kind[TT_KIND_COUNT];
} tt_container;

int  tt_container_open(tt_container *c, const char *path);
void tt_container_close(tt_container *c);

/* Exact-name lookup, O(1).  NULL if it does not exist -- the caller decides
 * whether the absence is fatal, because for an optional tensor (MTP) it is
 * not. */
const tt_tensor *tt_container_find(const tt_container *c, const char *name);

/* The same, building the name with printf.  Avoids a temporary buffer at every
 * call site in the weight loader. */
const tt_tensor *tt_container_findf(const tt_container *c, const char *fmt, ...);

/* Bytes that stay resident if everything is touched.  `embed_tokens` is left
 * out by the decision in docs/04: it is read one row per token and lives
 * file-backed. */
uint64_t tt_container_resident_bytes(const tt_container *c);

int tt_cmd_inspect(int argc, char **argv);

#endif /* TOOTEGA_CONTAINER_H */
