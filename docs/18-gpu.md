# 18 — GPU: portão do M2, e o que comprar

O M2 de [05](05-roadmap.md) tem um corte explícito: **ganho abaixo de 20% e o
M2 é cancelado, não adiado.** Dois backends são custo permanente de manutenção
e precisam se pagar. Este documento existe para que esse corte seja julgado
contra número, não contra expectativa.

**Veredito: o M2 passa.** O caminho até esse veredito é mais interessante que
ele, porque a primeira medição dizia o contrário — por três defeitos meus.

## Linha de base — CPU

`Tests/bench.exe`, container v3, 20 threads:

```
banda de memoria     68,7 GB/s     medida nesta maquina, nao especificada
trafego por token    2,87 GB       tensores que um decode realmente toca
teto do roofline     24,0 tok/s    banda / trafego

decode quente        74 ms/token   = 13,5 tok/s   (56% do teto)
```

**O alvo do M2 é 16,2 tok/s** — 13,5 mais 20%.

## O hardware

```
RTX 2060 SUPER, 8 GB, sm_75, 448 GB/s de VRAM, driver 560.94
CUDA 12.6 (host compiler VS 2022 -- o CUDA 12.6 recusa o toolset de 2026)
PCIe gen 3 x16 -- ~12 GB/s, o elo mais lento da maquina
VRAM utilizavel  ~7,5 GB (8 menos driver e desktop)
```

## O que estava errado

A primeira versão dos kernels media **0,13×** — a GPU sete vezes mais lenta que
a CPU. Não era a placa; era o kernel, e por três motivos somados:

1. **Um bloco de 256 threads por linha de saída.** Com `cols=2048` a linha tem
   16 grupos: **16 threads trabalhavam e 240 esperavam na barreira.**
2. **Leitura byte a byte, sem coalescer.** Cada thread lia 32 bytes
   consecutivos e a vizinha lia 32 bytes muito adiante — o warp emitia dezenas
   de transações de 1 byte em vez de uma de 128.
3. **`x` relido em escalar, oito vezes por vetor de peso.** Para cada `uint4`
   de peso saíam 32 leituras de 4 bytes do vetor de ativação. O kernel ficava
   preso em *emissão de instrução*, não em banda.

O conserto, em três passos, cada um medido:

| versão | GB/s | ganho |
|---|---|---|
| bloco por linha, byte a byte | **5,9** | — |
| warp por linha, `uint4` coalescido, `__shfl_down_sync` | **35** | 5,9× |
| `x` em `float4` | **122** | 3,5× |
| 4 linhas por warp (amortiza `x`) | **~185** | 1,5× |

**31× no total, na mesma placa.** E o resultado continua **bit a bit igual** ao
da CPU nos cinco tensores (`erro rel 0,0000%`).

## Três armadilhas de medição, e o conserto de cada uma

Estas custaram mais tempo que os kernels, e são o motivo de o portão inicial ter
mentido.

**1. O ida-e-volta não é o kernel.** `be_matmul` paga H2D, lançamento, D2H e
sincronização a cada chamada — um piso de ~20 µs que não tem nada a ver com a
velocidade do kernel, e que o desenho do M2 (`run_layer` no device) não paga por
matmul. A ABI do backend ganhou `bench`, que cronometra o kernel sozinho com
eventos CUDA. Sem isso a decisão seria tomada pelo número errado.

**2. Média mede a máquina, não o kernel.** Com o editor compilando ao lado, o
mesmo matmul de CPU mediu **0,033 ms e 0,612 ms na mesma sessão** — 18× de
diferença vinda de fora. O portão passou a reportar o **menor tempo observado**:
o kernel é determinista, então o mínimo é o custo dele e interferência só
adiciona.

**3. A CPU estava lendo do L3.** Depois de 400 repetições o tensor de 5 MB mora
no L3 de 36 MB, e a CPU mede **104 GB/s** — mas o decode real toca 2,87 GB por
token, vindos da RAM a 68,7 GB/s. Comparar a GPU (lendo da VRAM de verdade)
contra a CPU (lendo do cache) infla a CPU em 1,5×. A projeção usa a banda de
**streaming**, e a tabela mostra as duas.

## Onde estamos

```
tensor                              CPU ms  CPU GB/s   GPU ms  kernel     GB/s  erro
o_proj            (5,2 MB int4)      0.080     140.1    0.100   0.081    137.8  0.0000%
kv_b_proj         (4,6 MB int4)      0.031     155.7    0.049   0.028    176.3  0.0000%
expert.gate_proj  (1,5 MB int4)      0.025      71.9    0.029   0.010    171.0  0.0000%
expert.down_proj  (1,5 MB int4)      0.024      74.3    0.030   0.011    162.3  0.0000%
dense.gate_proj   (10,5 MB int4)     0.143      82.8    0.062   0.046    257.5  0.0000%

CPU quente (L3)        0.302 ms -- 104 GB/s
CPU streaming (RAM)    68,7 GB/s
ida e volta por matmul 1.11x
kernel sozinho         1.72x  -- 179 GB/s
piso por chamada       0.019 ms
```

**179–190 GB/s contra 68,7 da CPU: 2,6× a 2,8×.** São 40% do pico teórico da
placa (448 GB/s) — bom para um kernel escrito do zero, e ainda com folga.

O tensor grande chega a **257 GB/s** (57% do pico); os pequenos ficam em ~165,
limitados por lançamento e paralelismo. Isso importa para o desenho: um MoE
decode é dominado por matmuls **pequenos** de expert, e a saída é batê-los num
lançamento só, como o `tt_matmul_batch` já faz na CPU.

## A conta da partição

Um expert só ajuda se **a camada inteira** dele estiver residente: qualquer um
dos 64 pode ser roteado no próximo token. Por isso a unidade é a camada.

```
atencao, 47 camadas         0,51 GB    28% do trafego de pesos
lm_head                     0,16 GB     9%
experts, 46 x 302 MB       13,90 GB    61%
camada densa                0,03 GB     2%
KV a 16k de contexto        1,77 GB
```

Com `f` de fração do tráfego na GPU e ganho 2,6×, o tempo cai para
`(1−f) + f/2,6`:

| VRAM | o que cabe | f | ganho | tok/s |
|---|---|---|---|---|
| 8 GB (hoje) | atenção + 15 das 46 camadas MoE | 0,57 | **1,54×** | **20,8** |
| 12 GB | atenção + 27 camadas | 0,73 | 1,82× | 24,5 |
| 16 GB | atenção + 42 camadas | 0,92 | 2,30× | 31,1 |
| 24 GB | **tudo** (18,5 de pesos + 1,8 de KV) | 1,00 | 2,60× | 35,1 |

**Todos passam do corte de 16,2 tok/s** — inclusive a placa que já está na
máquina. O M2 está aprovado.

## Se for comprar placa

Duas coisas decidem, e a ordem importa:

**Banda, não teraflops.** Isto é um decode memory-bound dos dois lados. Núcleo
CUDA sobrando não acelera nada; GB/s aceleram tudo, linearmente.

**VRAM decide a *fração*, banda decide o *ganho*.** 16 GB já leva f a 0,92 —
daí para cima o retorno de mais VRAM é pequeno, e o de mais banda continua
linear.

Assumindo a eficiência que medimos (**~41% do pico**) e o mesmo modelo:

| placa | VRAM | banda | efetiva | f | tok/s previstos |
|---|---|---|---|---|---|
| RTX 2060 SUPER (atual) | 8 GB | 448 | 184 | 0,57 | 21 |
| RTX 4060 Ti | 16 GB | **288** | 118 | 0,92 | 22 |
| RTX 5060 Ti | 16 GB | 448 | 184 | 0,92 | 32 |
| RTX 4070 Ti SUPER | 16 GB | 672 | 276 | 0,92 | 44 |
| RTX 4080 SUPER | 16 GB | 736 | 302 | 0,92 | 46 |
| RTX 3090 (usada) | 24 GB | 936 | 384 | 1,00 | 75 |
| RTX 4090 | 24 GB | 1008 | 413 | 1,00 | 81 |
| RTX 5090 | 32 GB | 1792 | 735 | 1,00 | 144 |

**A 4060 Ti 16 GB é a armadilha da lista.** Tem a VRAM que interessa e a banda
de 288 GB/s — daria **22 tok/s**, praticamente o mesmo que a 2060 SUPER de
8 GB que já está aqui. Comprar VRAM sem banda não compra velocidade.

**Melhor relação: 3090 usada.** 24 GB põem o modelo inteiro residente com
espaço para 16k de KV, e 936 GB/s é banda de placa topo de linha de geração
anterior. É a única faixa em que se sai de "acelerar um pedaço" para "a CPU não
participa mais".

Conferir na compra: a banda do modelo exato (fabricantes vendem variantes com
barramento menor) e o comprimento físico da placa.

Ressalva honesta: a coluna `tok/s previstos` é **teto de projeto**, não medida.
Ela assume que a eficiência de 41% se mantém na placa nova e que a atenção, o
softmax e o roteamento também migram. O que está **medido** é a linha da 2060
SUPER, e mesmo essa em kernel isolado — não em decode completo.

## O que ainda precisa ser escrito

O ganho acima **não existe hoje**: com a interface atual, um matmul por
chamada, o ida-e-volta come quase tudo (1,11×).

1. **`run_layer` no device.** Uma camada inteira por lançamento, ativação
   cruzando o PCIe duas vezes por camada em vez de trinta. É o desenho de
   [03](03-arquitetura.md) e é o que transforma 1,11× em 2,6×.
2. **Experts em lançamento único.** Os cinco experts ativos de uma camada num
   kernel só, como o `tt_matmul_batch` da CPU. É onde estão 61% do tráfego e os
   matmuls mais lentos por lançamento.
3. **Planner de partição.** Medir a VRAM livre na init e escolher quais camadas
   sobem, respeitando `--vram-gb`.
4. **Atenção no device.** Enquanto o KV ficar na RAM, a atenção das camadas
   residentes puxa 108 KB por posição pelo PCIe. O KV cabe folgado na VRAM.
5. **Critério 1 do M2**: saída **byte-idêntica** entre CPU-only e split. A
   colocação não pode alterar tokens — e hoje a ordem de redução da GPU é uma
   árvore por warp, a da CPU é um acumulador por linha. Os cinco tensores deram
   `0,0000%` de erro relativo, mas isso é igualdade numérica, não bit a bit.
   Esse critério pode exigir fixar a ordem de redução dos dois lados
   ([15](15-kernels-e-numerica.md)).
