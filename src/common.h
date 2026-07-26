/* common.h -- erro, log e arena.  As tres pecas que todo o resto usa.
 *
 * Arena e a resposta a docs/11-memoria.md: a memoria de trabalho e dimensionada
 * uma vez na init e depois so se avanca um ponteiro.  Zero `malloc` no laco de
 * token e requisito de saida do M1, nao preferencia de estilo.
 */
#ifndef TOOTEGA_COMMON_H
#define TOOTEGA_COMMON_H

#include <stddef.h>
#include <stdint.h>

/* --------------------------------------------------------------- erro */

/* Registra a mensagem e retorna -1, sempre.  Permite `return tt_fail(...)`. */
int         tt_fail(const char *fmt, ...);
const char *tt_error(void);        /* ultima mensagem, "" se nenhuma */
void        tt_error_clear(void);

/* ---------------------------------------------------------------- log */

void tt_log(const char *fmt, ...);      /* stderr, sempre */
void tt_dbg(const char *fmt, ...);      /* stderr, so com verbose ligado */
void tt_verbose(int on);
int  tt_is_verbose(void);

/* -------------------------------------------------------------- arena */

/* Bump allocator.  Nao ha free individual: ou se reseta tudo, ou se destroi.
 * `peak` existe para o relatorio de orcamento de docs/09. */
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

/* Alinhamento em bytes, potencia de dois.  Retorna NULL e registra erro se
 * estourar -- estouro de arena e bug de dimensionamento, nunca condicao
 * normal, entao a mensagem diz quanto faltou. */
void *tt_arena_alloc(tt_arena *a, size_t bytes, size_t align);

/* Atalhos com o alinhamento de 64 que docs/14 exige dos blobs e que os
 * kernels SIMD assumem. */
#define TT_ALIGN 64
#define tt_arena_new(a, type, n) \
    ((type *)tt_arena_alloc((a), sizeof(type) * (size_t)(n), TT_ALIGN))

/* --------------------------------------------------------------- util */

static inline size_t tt_align_up(size_t v, size_t align)
{
    return (v + align - 1) & ~(align - 1);
}

/* Formata bytes como "15,63 GB" / "812,4 MB".  Buffer do chamador, >= 32. */
const char *tt_bytes(char *buf, size_t buflen, uint64_t bytes);

/* Hash das fontes de `src/` com que este binario foi construido (docs/10).
 *
 * Existe porque uma ferramenta de medicao linka `src/` estaticamente: depois
 * de consertar o forward, um `.exe` nao reconstruido continua medindo o codigo
 * velho -- e o numero errado nao se anuncia. Com o carimbo impresso em todo
 * relatorio, a divergencia fica visivel. */
const char *tt_build_id(void);

#endif /* TOOTEGA_COMMON_H */
