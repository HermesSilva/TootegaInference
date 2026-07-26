# 14 — Formato do container `.tootega`

Especificação normativa. O escritor é `convert_to_int4.py`; o leitor em C tem de
casar **byte a byte**. Enquanto isso for só convenção entre dois programas em
linguagens diferentes, é onde nascem bugs de alinhamento.

Tudo little-endian. Resolve o **D3** de [13](13-decisoes-pendentes.md).

## Layout geral

```
offset 0        cabecalho fixo, 128 bytes
offset 128..    blobs de dados, cada um alinhado a 64 bytes
                (ordem de escrita = ordem dos shards de origem)
index_off       indice JSON, alinhado a 64
EOF
```

## Cabeçalho (128 bytes)

| offset | tipo | campo | valor |
|---|---|---|---|
| 0 | `u8[8]` | magic | `54 4F 4F 54 45 47 41 01` (`"TOOTEGA\x01"`) |
| 8 | `u32` | `format_version` | `1` |
| 12 | `u32` | `flags` | `0` (reservado) |
| 16 | `u64` | `n_tensors` | número de entradas no índice |
| 24 | `u64` | `index_off` | offset absoluto do índice |
| 32 | `u64` | `index_len` | bytes do índice |
| 40..127 | — | reservado | zeros |

Magic diferente → recusar. `format_version` desconhecida → recusar, **nunca**
ler "na esperança".

## Índice

JSON UTF-8 em `[index_off, index_off + index_len)`:

```json
{
  "meta": {
    "model": "GLM-4.7-Flash", "source": "...", "group": 64,
    "num_hidden_layers": 47, "mtp_layer": 47,
    "config": { ...config.json integral... }
  },
  "tensors": {
    "model.layers.1.mlp.experts.0.gate_proj.weight": {
      "kind": "int4",
      "shape": [1536, 2048],
      "data":   [1234560, 1572864],
      "scales": [2807424, 196608],
      "group": 64
    }
  }
}
```

`config` integral é deliberado: o engine lê a arquitetura do container, não de um
`config.json` ao lado que pode divergir.

| campo | significado |
|---|---|
| `kind` | `int4` \| `int8` \| `f32` \| `bf16` |
| `shape` | dimensões lógicas, **antes** do empacotamento |
| `data` | `[offset, bytes]`, offset múltiplo de 64 |
| `scales` | `[offset, bytes]` ou `null` para `f32`/`bf16` |
| `group` | pesos por escala, ou `null` |

## Layout dos dados

Convenção de forma: `[O, I]` — `O` linhas de saída, `I` de entrada. O
agrupamento é sempre ao longo de **`I`** (a dimensão de redução do matmul).
`I % group == 0` é invariante; o conversor recusa antes de escrever se falhar.

### `f32` e `bf16`

Linha-maior, contíguo, sem escalas. `f32` = `float32[O][I]`;
`bf16` = `uint16[O][I]` com os 16 bits altos de um float32.

### `int8`

```
data   : int8[O][I]            escalas simetricas, sem zero-point
scales : float32[O][I/group]
peso[o][i] = data[o][i] * scales[o][i / group]
```

### `int4` — empacotamento metade/metade

```
data   : uint8[O][I/2]
scales : float32[O][I/group]
```

Dentro de cada grupo de `group` pesos, o byte `j` (com `0 <= j < group/2`)
carrega:

```
nibble baixo (bits 0-3) -> peso j
nibble alto  (bits 4-7) -> peso j + group/2
```

Os códigos são **deslocados**: valor armazenado `0..15`, código real
`armazenado - 8`, faixa `-8..7`.

Desempacotar o grupo `g` da linha `o`:

```c
const uint8_t *p = data + o*(I/2) + g*(GROUP/2);
const float    s = scales[o*(I/GROUP) + g];
for (int j = 0; j < GROUP/2; j++) {
    w[j]           = (int)(p[j] & 0x0F) - 8;   /* pesos 0..GROUP/2-1 */
    w[j + GROUP/2] = (int)(p[j] >> 4)   - 8;   /* pesos GROUP/2..GROUP-1 */
}
/* peso real = w[k] * s */
```

**Por que metade/metade e não par/ímpar.** Carregando 32 bytes com AVX2,
`v & 0x0F` entrega `group/2` pesos **contíguos** e `v >> 4` os `group/2`
seguintes — dois vetores prontos para o produto contra as ativações
correspondentes. Com par/ímpar, `v & 0x0F` entregaria `0,2,4,…`, forçando a
permutar as ativações. É a convenção do `Q4_0` do llama.cpp, e a escolha está
registrada como D1 em [13](13-decisoes-pendentes.md).

## Escalas

Simétricas, sem zero-point:

```
qmax   = 7   (int4)  |  127 (int8)
scale  = absmax(grupo) / qmax        , e 1.0 se absmax == 0
codigo = clamp(round(peso / scale), -(qmax+1), qmax)
```

O caso `absmax == 0` usa escala `1.0` para evitar divisão por zero; todos os
códigos ficam `0` e a reconstrução é exata.

Custo de armazenamento: `4/group` bytes por peso. Com `group=64`, int4 custa
**0,5625 B/peso** — é o número que a conta de teto em
[08](08-teto-de-desempenho.md) assume.

**Grupos válidos neste modelo: 32, 64, 128 ou 256.** O limite superior é 256
porque `q_b_proj` tem dimensão de entrada 768.

## Alinhamento

Todo blob começa em múltiplo de **64** — a linha de cache e a largura de um
registrador AVX-512. É o que permite carregar direto do arquivo mapeado sem
caminho de cópia ([11](11-memoria.md)). O `--verify` do conversor checa isso em
todos os blobs e falha se algum escapar.

O preenchimento entre blobs é zero.

## Escrita atômica

Escreve em `<nome>.part`, `fsync`, `rename`. Uma queda no meio não pode deixar
um container truncado com aparência de válido.

## Verificação

```powershell
python convert_to_int4.py --verify --out <caminho>
```

Confere magic, versão, e o alinhamento de todos os blobs. Não relê os dados —
integridade de conteúdo é responsabilidade do checksum previsto em
[12](12-estado-persistente.md), ainda não implementado.

## Referência executável

`quant_grouped` e `dequant_grouped` em `convert_to_int4.py` são a definição
executável do layout. O kernel em C tem de reproduzir `dequant_grouped`
exatamente; qualquer divergência aparece como falha no oráculo token-exato do
M1, não como erro de leitura.
