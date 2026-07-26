/* json.h -- our own JSON parser, a DOM over an arena.
 *
 * It exists because docs/10-implementation.md forbids a JSON library: the engine
 * consumes third-party model files and every dependency is a surface we do not
 * control.  It serves two consumers, and both are large:
 *
 *   - the container index (~2 MB, 9,703 tensors)      docs/14
 *   - `tokenizer.json` (~20 MB, 321,649 merges)       docs/16
 *
 * Deliberate properties:
 *
 *   zero copy    Strings without escapes point into the source buffer.  Only
 *                escaped ones are copied (and unescaped) into the arena.
 *   no malloc    All memory comes from a caller-provided arena.
 *   no mutation  The input buffer is const -- the index lives inside the
 *                container's read-only mapping.
 *
 * Key lookup is LINEAR.  For objects with thousands of members (the tensor map,
 * the vocabulary), iterate once and build your own table; do not call
 * tt_json_get in a loop.
 */
#ifndef TOOTEGA_JSON_H
#define TOOTEGA_JSON_H

#include <stddef.h>
#include <stdint.h>

#include "common.h"

typedef enum {
    TT_JSON_NULL = 0,
    TT_JSON_BOOL,
    TT_JSON_INT,        /* an integer that fit in int64 -- offsets and shapes */
    TT_JSON_REAL,       /* everything else that is a number */
    TT_JSON_STRING,
    TT_JSON_ARRAY,
    TT_JSON_OBJECT
} tt_json_type;

typedef struct tt_json tt_json;

struct tt_json {
    tt_json_type type;

    /* Member name, when this node is a member of an object.  It is NOT
     * terminated; use key_len or tt_json_key_eq. */
    const char *key;
    size_t      key_len;

    tt_json *child;     /* first member (object) or element (array) */
    tt_json *next;      /* next sibling */
    size_t   count;     /* members or elements */

    union {
        int      boolean;
        int64_t  i;
        double   r;
        struct { const char *ptr; size_t len; } str;  /* not terminated */
    } v;
};

/* Upper bound in arena bytes for parsing this buffer.  Computed in a cheap
 * scan; use it to size the arena and turn "it overflowed" into an
 * impossibility, not a runtime error. */
size_t tt_json_bound(const char *data, size_t len);

/* NULL on error, with tt_error() explaining where (offset and byte). */
tt_json *tt_json_parse(const char *data, size_t len, tt_arena *arena);

/* -------------------------------------------------------------- access */

const tt_json *tt_json_get(const tt_json *obj, const char *key);
const tt_json *tt_json_at(const tt_json *arr, size_t index);

int      tt_json_key_eq(const tt_json *node, const char *key);
int      tt_json_str_eq(const tt_json *node, const char *s);

/* The accessors tolerate NULL and the wrong type, returning the default.  That
 * is deliberate: the caller validates the shape once, with its own message,
 * instead of checking types on every read. */
int64_t     tt_json_int(const tt_json *node, int64_t def);
double      tt_json_real(const tt_json *node, double def);
int         tt_json_bool(const tt_json *node, int def);
const char *tt_json_str(const tt_json *node, size_t *len);   /* NULL if not a string */

/* Copies a string from the DOM into a terminated C buffer.  Returns -1 if it
 * does not fit or if the node is not a string. */
int tt_json_strcpy(const tt_json *node, char *dst, size_t dstlen);

int tt_json_selftest(void);

#endif /* TOOTEGA_JSON_H */
