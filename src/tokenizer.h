/* tokenizer.h -- BPE tiktoken/GPT-4, lendo `tokenizer.json` do HuggingFace.
 *
 * Implementa docs/16-tokenizer.md.  Quatro etapas, e as quatro importam:
 *
 *   1. pre-tokenizacao pelo regex do cl100k -- maquina de estados a mao, sem
 *      motor de regex (docs/16: "nao implementar um motor de regex")
 *   2. ByteLevel: cada byte 0..255 vira um code point pela tabela do GPT-2
 *   3. BPE com `ignore_merges = true` -- se o pre-token inteiro esta no
 *      vocabulario, usa o id direto e nao aplica merge nenhum
 *   4. tokens especiais reconhecidos ANTES do regex, por casamento exato
 *
 * Truque de representacao: em vez de converter cada pre-token para o alfabeto
 * ByteLevel e casar contra o vocabulario nessa forma, o vocabulario e
 * convertido de volta para BYTES CRUS na carga.  O BPE passa a operar sobre os
 * bytes da entrada direto -- uma conversao por vocabulario em vez de uma por
 * token, e nenhuma alocacao no caminho quente.
 *
 * Erro aqui nao estoura: produz ids diferentes e saida diferente do oraculo,
 * com o forward perfeitamente correto.  Por isso o selftest compara contra ids
 * gerados pelo `tokenizers` da HF, e nao so contra si mesmo.
 */
#ifndef TOOTEGA_TOKENIZER_H
#define TOOTEGA_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#include "common.h"

typedef struct {
    const uint8_t *ptr;        /* bytes crus, dentro da arena */
    uint32_t       len;
} tt_token;

typedef struct {
    tt_arena  arena;

    tt_token *tokens;          /* id -> bytes */
    int32_t   n_tokens;        /* ids 0..n_tokens-1 do vocabulario base */
    int32_t   vocab_size;      /* da config; ha folga no fim (docs/16) */

    /* bytes -> id, hash aberto */
    int32_t  *vocab_slots;
    size_t    n_vocab_slots;

    /* (id_esq << 32 | id_dir) -> rank e id resultante, hash aberto.  E o laco
     * quente do encode: varredura linear aqui custa segundos. */
    uint64_t *merge_keys;
    int32_t  *merge_rank;
    int32_t  *merge_id;
    size_t    n_merge_slots;
    int32_t   n_merges;

    int32_t   byte_id[256];    /* id do token de um byte isolado */

    /* especiais, casados antes do regex */
    tt_token *special;
    int32_t  *special_id;
    int32_t   n_special;
    int32_t   special_min_id;
    /* Primeiro byte de algum especial.  Evita 36 memcmp por byte de texto na
     * varredura que fatia a entrada. */
    uint8_t   special_first[256];
} tt_tokenizer;

int  tt_tokenizer_load(tt_tokenizer *t, const char *path, int32_t vocab_size);
void tt_tokenizer_free(tt_tokenizer *t);

/* Devolve quantos ids foram escritos, ou -1 se nao couberam em `max`. */
int64_t tt_encode(const tt_tokenizer *t, const char *text, size_t len,
                  int32_t *out, int64_t max);

/* Bytes escritos (sem terminador), ou -1 se nao couberam. */
int64_t tt_decode(const tt_tokenizer *t, const int32_t *ids, int64_t n,
                  char *out, size_t max);

/* Ids acima do vocabulario em uso existem so como padding do `lm_head` e nunca
 * devem ser amostrados (docs/16). */
int tt_token_is_valid(const tt_tokenizer *t, int32_t id);

int tt_tokenizer_selftest(void);

#endif /* TOOTEGA_TOKENIZER_H */
