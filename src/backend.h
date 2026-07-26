/* backend.h -- a fronteira entre o engine e um acelerador.
 *
 * CUDA e C++ e arrasta um runtime pesado. Ele fica FORA do binario
 * (docs/10): o engine carrega `tootega_cuda.dll` em execucao e fala com ela
 * por C ABI. Se a DLL nao existe, nao carrega, ou a placa sumiu, o engine roda
 * em CPU e diz isso -- nunca falha por causa dela.
 *
 * O contrato tem uma regra que decide o desenho inteiro:
 *
 *   **Peso nao atravessa o PCIe.** A 12 GB/s, mandar os 2,87 GB que um token
 *   toca daria 4 tok/s -- pior que a CPU sozinha. O peso que a GPU usa mora na
 *   VRAM, carregado uma vez; o que cruza e ativacao: 2.048 floats, 8 KB.
 *
 * Por isso a interface nao tem "multiplique esta matriz". Ela tem "carregue
 * este tensor" e "rode esta camada", que sao as operacoes compativeis com a
 * restricao.
 */
#ifndef TOOTEGA_BACKEND_H
#define TOOTEGA_BACKEND_H

#include <stdint.h>

#define TT_BACKEND_ABI 2

/* Identificador opaco de um tensor ja residente na VRAM. */
typedef uint32_t tt_dev_tensor;
#define TT_DEV_NONE ((tt_dev_tensor)0)

typedef struct {
    uint32_t abi;                  /* TT_BACKEND_ABI; incompatibilidade recusa */
    char     name[64];             /* "CUDA sm_75", para o log */
    uint64_t vram_total;
    uint64_t vram_free;

    /* Ciclo de vida.  `init` devolve 0 em sucesso; qualquer outra coisa faz o
     * engine seguir em CPU. */
    int  (*init)(int device);
    void (*shutdown)(void);

    /* Sobe um tensor para a VRAM.  `kind` usa os valores de `tt_kind`, e o
     * layout e o mesmo do container (docs/14) -- a DLL desempacota igual.
     * Devolve TT_DEV_NONE se nao couber, e ai o chamador deixa essa camada na
     * CPU. */
    tt_dev_tensor (*upload)(const void *data, uint64_t bytes,
                            const float *scales, uint64_t scales_bytes,
                            int kind, int64_t rows, int64_t cols, int group);
    void (*release)(tt_dev_tensor t);

    /* y[rows] = W . x[cols].  `x` e `y` sao ponteiros de HOST: a travessia dos
     * 8 KB acontece aqui dentro, e e o unico trafego que cruza o PCIe. */
    int (*matmul)(float *y, tt_dev_tensor w, const float *x);

    /* Barreira: garante que tudo o que foi enfileirado terminou. */
    void (*sync)(void);

    /* Milissegundos por chamada do KERNEL, sem transferencia -- `reps` vezes,
     * cronometrado no device.
     *
     * `matmul` mede o ida-e-volta inteiro, e o piso de ~0,15 ms de lancamento
     * mais copia nao tem nada a ver com a velocidade do kernel.  Como o desenho
     * do M2 e uma camada inteira no device, decidir pela medida do ida-e-volta
     * seria decidir pelo numero errado.  Devolve -1 se nao souber medir. */
    double (*bench)(tt_dev_tensor w, int reps);
} tt_backend;

/* Assinatura do simbolo exportado pela DLL. */
typedef const tt_backend *(*tt_backend_entry)(void);
#define TT_BACKEND_ENTRY "tootega_backend"

/* Tenta carregar o backend.  NULL significa "siga em CPU" -- nunca e erro
 * fatal. A razao vai para `tt_error()` e para o log. */
const tt_backend *tt_backend_load(const char *path, int device);
void              tt_backend_unload(void);

/* O backend ativo, ou NULL. */
const tt_backend *tt_backend_get(void);

#endif /* TOOTEGA_BACKEND_H */
