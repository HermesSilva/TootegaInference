/* container.h -- leitor do `.tootega`.
 *
 * A especificacao normativa e docs/14-formato-container.md; o escritor e
 * `convert_to_int4.py`.  Este leitor tem de casar byte a byte com ele, e por
 * isso valida tudo o que a spec promete em vez de confiar: magic, versao,
 * alinhamento de 64 de cada blob, e a relacao entre `shape`, `group` e o
 * tamanho de cada blob.  Um container que passa daqui pode ser lido pelos
 * kernels sem nenhuma outra checagem.
 *
 * Zero copia: todo ponteiro devolvido aponta para dentro do mapeamento
 * somente-leitura do arquivo.  Nada e lido para buffer proprio, nem o indice.
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
    const char *name;          /* dentro do indice, sem terminador */
    size_t      name_len;
    tt_kind     kind;

    int64_t     shape[TT_MAX_DIMS];
    int         ndim;
    int64_t     rows;          /* O -- produto das dims menos a ultima */
    int64_t     cols;          /* I -- ultima dim, a de reducao */

    const void *data;          /* dentro do mapeamento, alinhado a 64 */
    uint64_t    data_off, data_bytes;

    const float *scales;       /* NULL para f32/bf16 */
    uint64_t     scales_off, scales_bytes;

    int32_t     group;         /* 0 quando nao ha escala */
} tt_tensor;

typedef struct {
    tt_map      map;
    tt_arena    arena;         /* DOM do indice + tabela de tensores */

    uint32_t    format_version;
    uint64_t    index_off, index_len;

    const tt_json *root;       /* {meta, tensors} */
    const tt_json *meta;
    const tt_json *config;     /* meta.config -- a arquitetura vem daqui */

    tt_tensor  *tensors;
    size_t      n_tensors;

    uint32_t   *buckets;       /* hash aberto: 0 vazio, i+1 = tensors[i] */
    size_t      n_buckets;

    /* Contabilidade que o orcamento de docs/09 usa. */
    uint64_t    bytes_by_kind[TT_KIND_COUNT];
    size_t      count_by_kind[TT_KIND_COUNT];
} tt_container;

int  tt_container_open(tt_container *c, const char *path);
void tt_container_close(tt_container *c);

/* Busca por nome exato, O(1).  NULL se nao existe -- e o chamador que decide
 * se a ausencia e fatal, porque para tensor opcional (MTP) nao e. */
const tt_tensor *tt_container_find(const tt_container *c, const char *name);

/* Idem, montando o nome com printf.  Evita o buffer temporario em cada
 * chamada do carregador de pesos. */
const tt_tensor *tt_container_findf(const tt_container *c, const char *fmt, ...);

/* Bytes que ficam residentes se tudo for tocado.  `embed_tokens` fica de fora
 * por decisao de docs/04: e lido uma linha por token e vive file-backed. */
uint64_t tt_container_resident_bytes(const tt_container *c);

int tt_cmd_inspect(int argc, char **argv);

#endif /* TOOTEGA_CONTAINER_H */
