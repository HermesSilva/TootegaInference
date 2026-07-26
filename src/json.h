/* json.h -- parser JSON proprio, DOM sobre arena.
 *
 * Existe porque docs/10-implementacao.md proibe biblioteca de JSON: o engine
 * consome arquivos de modelo de terceiros e cada dependencia e superficie que
 * nao controlamos.  Serve a dois consumidores, e os dois sao grandes:
 *
 *   - o indice do container (~2 MB, 9.703 tensores)   docs/14
 *   - `tokenizer.json` (~20 MB, 321.649 merges)       docs/16
 *
 * Propriedades deliberadas:
 *
 *   zero copia   Strings sem escape apontam para o buffer de origem.  So as
 *                que tem escape sao copiadas (e desescapadas) na arena.
 *   sem malloc   Toda a memoria sai de uma arena do chamador.
 *   sem mutacao  O buffer de entrada e const -- o indice vive dentro do
 *                mapeamento somente-leitura do container.
 *
 * Busca por chave e LINEAR.  Para objetos com milhares de membros (o mapa de
 * tensores, o vocabulario), itere uma vez e construa sua propria tabela; nao
 * chame tt_json_get em laco.
 */
#ifndef TOOTEGA_JSON_H
#define TOOTEGA_JSON_H

#include <stddef.h>
#include <stdint.h>

#include "common.h"

typedef enum {
    TT_JSON_NULL = 0,
    TT_JSON_BOOL,
    TT_JSON_INT,        /* inteiro que coube em int64 -- offsets e shapes */
    TT_JSON_REAL,       /* tudo o mais que e numero */
    TT_JSON_STRING,
    TT_JSON_ARRAY,
    TT_JSON_OBJECT
} tt_json_type;

typedef struct tt_json tt_json;

struct tt_json {
    tt_json_type type;

    /* Nome do membro, quando este no e membro de um objeto.  NAO tem
     * terminador; use key_len ou tt_json_key_eq. */
    const char *key;
    size_t      key_len;

    tt_json *child;     /* primeiro membro (objeto) ou elemento (array) */
    tt_json *next;      /* proximo irmao */
    size_t   count;     /* membros ou elementos */

    union {
        int      boolean;
        int64_t  i;
        double   r;
        struct { const char *ptr; size_t len; } str;  /* sem terminador */
    } v;
};

/* Limite superior de bytes de arena para parsear este buffer.  Calculado numa
 * varredura barata; use-o para dimensionar a arena e transformar "estourou" em
 * impossibilidade, nao em erro de execucao. */
size_t tt_json_bound(const char *data, size_t len);

/* NULL em erro, com tt_error() explicando onde (offset e byte). */
tt_json *tt_json_parse(const char *data, size_t len, tt_arena *arena);

/* ------------------------------------------------------------- acesso */

const tt_json *tt_json_get(const tt_json *obj, const char *key);
const tt_json *tt_json_at(const tt_json *arr, size_t index);

int      tt_json_key_eq(const tt_json *node, const char *key);
int      tt_json_str_eq(const tt_json *node, const char *s);

/* Os acessores toleram NULL e tipo errado, devolvendo o default.  E
 * deliberado: o chamador valida a forma uma vez, com mensagem propria, em vez
 * de checar tipo em toda leitura. */
int64_t     tt_json_int(const tt_json *node, int64_t def);
double      tt_json_real(const tt_json *node, double def);
int         tt_json_bool(const tt_json *node, int def);
const char *tt_json_str(const tt_json *node, size_t *len);   /* NULL se nao-string */

/* Copia uma string do DOM para buffer C com terminador.  Retorna -1 se nao
 * couber ou se o no nao for string. */
int tt_json_strcpy(const tt_json *node, char *dst, size_t dstlen);

int tt_json_selftest(void);

#endif /* TOOTEGA_JSON_H */
