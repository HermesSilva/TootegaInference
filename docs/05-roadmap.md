# 05 — Roadmap

Cada marco tem **critério de saída mensurável**. Sem número, não fecha.

Todas as medições em: GLM-4.7-Flash int4, contexto de 4k, prompt fixo, greedy,
esta máquina, orçamento de RAM fixo. A lição de
[07-baseline.md](07-baseline.md) é que comparar duas medições com configuração
diferente não significa nada.

---

## M0 — Converter e validar o modelo

Sem isto não há o que rodar.

- Conversor BF16 → int4 g64, streaming shard a shard (nunca os dois lados em
  memória)
- **Tensores alinhados a 64 bytes na escrita** — sem isso não há leitura
  zero-copy com SIMD ([11](11-memoria.md))
- Oráculo `transformers`: 32 posições teacher-forcing, ids salvos em `ref.json`
- Ablação de precisão por tensor (a tabela de [04](04-modelo-alvo.md) é hipótese)

**Saída:** container int4 de ~19,3 GB em disco, alinhado; oráculo gerado;
perplexidade do int4 medida contra o BF16 num conjunto fixo, com o custo
documentado.

**Risco:** a quantização degradar além do aceitável em algum tensor. Mitigação: a
ablação existe justamente para achar isso antes de escrever kernel.

---

## M1 — Forward CPU-only, token-exato

O marco que decide se o projeto é viável.

- Carregador de safetensors: mmap único, pesos lidos in-place, zero cópia
- Config → IR (atenção e roteamento como variantes, só MLA + `n_group=1` no dia 1)
- Kernels CPU: matmul int4 g64 com AVX-VNNI, atenção MLA, rmsnorm, RoPE parcial
- Arena de scratch dimensionada na init — **zero `malloc` no laço de token**
- Config imutável lida uma vez — **zero `getenv` no laço de camadas**
- KV com casamento de prefixo e truncamento no divergente
- Orçamentos `--ram-gb`, `--threads`, `--affinity` e o comando `--plan`
  ([09](09-limites-de-recursos.md))

**Saída:**
1. **32/32 posições idênticas ao oráculo** — bloqueia tudo o mais;
2. **≥ 10 tok/s** a 4k de contexto (≥ 37% do roofline de 27);
3. RSS estável entre turnos — sem crescimento por token;
4. **`--ram-gb` respeitado e verificado em execução**; `--plan` recusa o que não
   cabe com mensagem que diz o que reduzir.

**Meta desejável:** ≥ 15 tok/s (~55% do roofline).

**Corte:** abaixo de 10 tok/s a premissa de banda está errada. Parar e refazer a
conta antes de escrever mais código.

---

## M2 — Partição por camada GPU/CPU

Valida a hipótese central de [02](02-hardware.md) e [03](03-arquitetura.md).

- Planner: medir banda de cada link na init, resolver `k` (quantas camadas na GPU)
- Kernels sm_75 espelhando os da CPU, mesmos resultados
- Fronteira: 8 KB de ativação por travessia, pesos **nunca** cruzam PCIe

**Saída:**
1. Saída **byte-idêntica** entre CPU-only e split — a colocação não pode alterar
   tokens;
2. **≥ 20% acima do M1** no mesmo teste;
3. `--vram-gb` e `--gpu-layers` respeitados, pico verificado em execução;
4. **Teste de regressão do `CUDA_DENSE`**: pedir mais VRAM do que existe tem de
   sair com erro acionável, não render 0,054 tok/s.

**Corte:** ganho abaixo de 20% → M2 é **cancelado**, não adiado. Dois backends é
um custo permanente de manutenção e precisa se pagar.

---

## M3 — Servidor e batching contínuo

Onde vive a segunda meta de [01](01-objetivos.md).

- Slots de KV independentes, admissão por slot
- Batching contínuo no decode: cada slot ativo contribui uma linha
- API compatível OpenAI; loopback por padrão, chave obrigatória fora dele

**Saída:**
1. Throughput agregado com K=4 **≥ 2,5× o de K=1** (contra `N^0,27` medido sob
   streaming, que dá 1,45×);
2. Latência por requisição em K=4 não pior que 2× a de K=1;
3. Casamento de prefixo entre turnos com **prefill 0** ao continuar conversa.

---

## M4 — CLI agêntico

- Loop de ferramentas, execução local, aprovação para ação destrutiva
- Montagem de prompt seguindo a regra de ordenação de [03](03-arquitetura.md):
  estável primeiro, volátil por último
- Canal para injetar estado tardio **no fim** do prefixo

- Estado persistente: índice, plano, tokenizer e blocos de KV de prefixo
  ([12](12-estado-persistente.md))

**Saída:**
1. Um passe de SWE-bench-lite (ou subconjunto próprio) concluído fim a fim;
2. **Taxa de acerto de cache de prefixo ≥ 90%** em sessão multi-turno — mede se
   a ordenação do prompt está certa, e é o que separa um CLI utilizável de um
   que recomputa 100k tokens por turno;
3. **Abertura quente ≤ 3 s** contra ~60 s a frio, com o prefixo de 20k já
   carregado;
4. Sem vazamento de estado entre sessões — bloco de KV endereçado pelo hash da
   sequência exata, nunca por slot.

---

## M5 — Otimização, com número

Só depois do M4, e cada item entra com antes/depois na mesma configuração:

- Especulação MTP — a experiencia anterior diz que só paga perto de residência total;
  aqui estamos nesse regime, então **é a primeira a testar**
- Sobreposição compute/leitura, se o M1 mostrar que não sobrepõe
- Kernels tensor-core W4A16 em sm_75
- Reuso de KV entre sessões que compartilham prefixo

---

## Fora de escopo

Multi-GPU. Multi-host. Treino. Outras arquiteturas além de MLA + `n_group=1` —
os pontos de variação existem no IR desde o M1, mas implementar GQA só quando
houver um modelo concreto que valha.
