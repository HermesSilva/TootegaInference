/* retrieve.h -- BM25 over a corpus, inside the engine.
 *
 * Why the engine retrieves at all.  A container fine-tuned with RAFT (docs/22)
 * is trained to answer WITH the passage in the context, and the measurement that
 * settles it is not subtle: 26 of 29 answers doctrinally correct with the
 * passage, 0 of 11 without it.  Serving such a container without retrieval is
 * serving half of it -- so the retrieval is part of the engine, not scaffolding
 * around it.
 *
 * Why BM25 and not embeddings.  Not retrieval quality: embeddings would very
 * likely retrieve better.  BM25 is an inverted index and two divisions, which
 * fits the rule this project exists to keep -- an embedding retriever would put
 * a SECOND neural model inside `src/`, and there is no version of that which
 * stays auditable in one reading session.
 *
 * The unit is the SECTION.  An essay is 5-14k tokens and would swamp the
 * context; a section is one argument under its own roman numeral, which is also
 * what a citation points at.
 *
 * The reference implementation is `finetune/scripts/retrieve.py`, the same code
 * that chose the passages every training pair was written against.  If this file
 * ranks differently, the model sees at inference something other than what it
 * saw in training -- which is why `tools/index` links this file instead of
 * carrying its own copy of the tokeniser.
 */
#ifndef TOOTEGA_RETRIEVE_H
#define TOOTEGA_RETRIEVE_H

#include <stdint.h>

#include "common.h"
#include "compat.h"

/* --------------------------------------------------------------- format */

/* `.ttindex`, version 1.  Same shape as the container of docs/14 and the
 * `.ttpack` of `tools/finetune`: magic, a fixed header, sections aligned to 64,
 * little-endian throughout.  Everything is read straight out of the mapping --
 * nothing is copied into a buffer of ours, not even the text.
 *
 *   0   8   "TTINDEX\1"
 *   8   4   version
 *   12  4   n_docs
 *   16  4   n_terms
 *   20  4   flags (0)
 *   24  8   total_len          sum of document lengths, in terms -> avgdl
 *   32  8   terms_off/len      tt_ix_term[n_terms], sorted by the term bytes
 *   48  8   strings_off/len    the terms themselves, NUL-terminated
 *   64  8   post_off/len       tt_ix_post[], grouped by term, doc ascending
 *   80  8   docs_off/len       tt_ix_doc[n_docs]
 *   96  8   text_off/len       the section text, UTF-8, NUL-terminated
 *   112 16  reserved, zero
 */
#define TT_INDEX_MAGIC   "TTINDEX\1"
#define TT_INDEX_VERSION 1u
#define TT_INDEX_HEADER  128u
#define TT_INDEX_ALIGN   64u

/* Offsets are u32 and relative to their own section.  The corpus this was built
 * for is 3 MB; a 4 GB ceiling on a plain-text index is not a limit anyone will
 * meet, and halving the table keeps it in cache. */
typedef struct {
    uint32_t str_off;     /* into `strings` */
    uint32_t df;          /* documents containing the term */
    uint32_t post_off;    /* into `post`, in records */
    uint32_t post_n;      /* == df */
} tt_ix_term;

typedef struct {
    uint32_t title_off;   /* into `strings` -- the document's `# ` heading */
    uint32_t sect_off;    /* the `## ` heading, minus its roman numeral */
    uint32_t cite_off;    /* "A Estrutura da Existencia, Ensaio I, SII" */
    uint32_t text_off;    /* into `text` */
    uint32_t text_len;    /* bytes, NUL not counted */
    uint32_t n_terms;     /* the BM25 |D| -- terms kept, not bytes */
} tt_ix_doc;

typedef struct {
    uint32_t doc;
    uint32_t tf;
} tt_ix_post;

/* ---------------------------------------------------------------- index */

typedef struct {
    tt_map   map;
    uint32_t n_docs, n_terms;
    double   avgdl;

    const tt_ix_term *terms;
    const tt_ix_doc  *docs;
    const tt_ix_post *post;
    const char       *strings;
    const char       *text;

    uint64_t n_post;
    uint32_t strings_len, text_len;

    /* Scratch for the scoring pass, sized once at open.  It lives here and not
     * on the stack because it is O(n_docs) doubles and the search is not on the
     * token path -- and it is why `tt_index_search` takes a mutable index.  The
     * server answers one request at a time (see server.h); a second caller would
     * need a second index or a lock, and there is no second caller. */
    double *scores;
} tt_index;

typedef struct {
    int32_t doc;
    float   score;
} tt_hit;

int  tt_index_open(tt_index *ix, const char *path);
void tt_index_close(tt_index *ix);

/* Ranks the corpus against `query` and writes at most `max` hits, best first.
 * Returns how many were written, or -1.  Documents scoring zero are left out --
 * a query sharing no term with the corpus retrieves NOTHING, which the caller
 * must treat as "answer without a passage" and not as "here are three bad
 * passages".
 *
 * Ties break towards the higher document id, which is what Python's
 * `sorted(reverse=True)` over (score, index) does in the reference. */
int tt_index_search(tt_index *ix, const char *query, tt_hit *out, int max);

const char *tt_index_title(const tt_index *ix, int32_t doc);
const char *tt_index_section(const tt_index *ix, int32_t doc);
const char *tt_index_cite(const tt_index *ix, int32_t doc);
const char *tt_index_text(const tt_index *ix, int32_t doc, uint32_t *len);

/* ------------------------------------------------------------ tokeniser */

/* Calls `fn` once per term, in reading order.  A term is a maximal run of
 * [a-z0-9] over the lowercased, accent-folded text, longer than two bytes and
 * not a Portuguese function word.
 *
 * The whole point of exporting it is that `tools/index` uses THIS function.  A
 * builder that folds "ação" differently from the searcher produces an index in
 * which the query term never matches the indexed one, and the failure is silent:
 * the search simply returns nothing, or returns the wrong section, and looks
 * like a bad ranking rather than a bug. */
void tt_tokenize(const char *text, size_t len,
                 void (*fn)(const char *term, size_t n, void *ud), void *ud);

/* One term, folded, for a caller that already has a word.  Returns its length,
 * 0 if it folds to nothing.  Exposed for `tools/index inspect`. */
size_t tt_fold_term(const char *s, size_t n, char *out, size_t cap);

#endif /* TOOTEGA_RETRIEVE_H */
