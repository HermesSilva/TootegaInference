/* safetensors.h -- leitor do checkpoint BF16 original, para ferramentas.
 *
 * Fica em `tools/shared/` e nao em `src/`: o engine le o container `.tootega` e
 * so ele.  Um segundo formato de peso dentro do engine seria superficie que ele
 * nao precisa (CLAUDE.md), e um caminho de peso a mais para auditar.
 *
 * Entrega `tt_tensor` direto -- o mesmo tipo do container -- porque os kernels
 * ja sabem consumir `TT_KIND_BF16` e `TT_KIND_F32`.  E o que permite a
 * `ablate` trocar um grupo de tensores sem tocar no forward.
 */
#ifndef TOOTEGA_TOOLS_SAFETENSORS_H
#define TOOTEGA_TOOLS_SAFETENSORS_H

#include "common.h"
#include "compat.h"
#include "container.h"

typedef struct {
    const char *name;
    size_t      name_len;
    const void *data;
    uint64_t    bytes;
    int64_t     rows, cols;
    int         ndim;
    int64_t     shape[TT_MAX_DIMS];
    tt_kind     kind;          /* BF16 ou F32 -- e o que este checkpoint tem */
    int         shard;
} st_entry;

typedef struct {
    tt_arena   arena;
    tt_map    *maps;
    size_t     n_maps;
    st_entry  *entries;
    size_t     n_entries;
} st_file;

/* `folder` e a pasta com `model.safetensors.index.json` e os shards. */
int  st_open(st_file *st, const char *folder);
void st_close(st_file *st);

const st_entry *st_find(const st_file *st, const char *name, size_t len);

/* Config bruta do modelo, para copiar integral para dentro do container
 * (docs/14: a arquitetura vem do container, nao de um arquivo ao lado). */
char *st_read_config(const char *folder, size_t *len);

#endif /* TOOTEGA_TOOLS_SAFETENSORS_H */
