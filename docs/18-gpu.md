# 18 — GPU: linha de base e desenho

O M2 de [05](05-roadmap.md) tem um corte explícito: **ganho abaixo de 20% e o
M2 é cancelado, não adiado.** Dois backends são custo permanente de manutenção
e precisam se pagar. Este documento existe para que esse corte seja julgado
contra número, não contra expectativa.

## Linha de base — CPU, medida antes de escrever qualquer CUDA

`Tests/bench.exe`, container v3, 20 threads, build `dbb6ba4600b2`:

```
banda de memoria     68,9 GB/s     medida nesta maquina, nao especificada
trafego por token    2,87 GB       tensores que um decode realmente toca
teto do roofline     24,0 tok/s    banda / trafego

decode quente        74 ms/token   = 13,5 tok/s   (56% do teto)
decode frio         293 ms/token   = 3,4 tok/s    (4,0x pior)
aquecimento         18,54 GB em 5,3 s
prefill             ~11 tok/s
RSS                 16,8 GB de pico, 36,3 GB livres
```

**O alvo do M2 é 16,2 tok/s** — 13,5 mais 20%. Abaixo disso, cancela.

## O hardware

```
RTX 2060 SUPER, 8 GB, sm_75, driver 560.94
CUDA 12.6 instalada (fora do PATH; nvcc em C:\Program Files\NVIDIA GPU
                     Computing Toolkit\CUDA\v12.6\bin)
PCIe gen 3 x16 -- ~12 GB/s, o elo mais lento da maquina
VRAM utilizavel  ~6,5 GB (8 menos driver e desktop)
```

## A restrição que decide o desenho

**Peso não atravessa PCIe.** A 12 GB/s, mandar 2,87 GB por token pelo
barramento daria 4 tok/s — pior que a CPU sozinha. O que atravessa é ativação:
2.048 floats, 8 KB.

Logo, o peso que a GPU usa tem de **morar na VRAM**. E 6,5 GB não comportam os
18,5 GB do modelo. A pergunta do M2 é: **quais 6,5 GB?**

## Duas partições, e a conta de cada uma

### A — camadas inteiras (o plano original de docs/03)

`k` camadas na GPU, o resto na CPU. Uma camada MoE precisa dos **64 experts
residentes** — qualquer um pode ser roteado —, o que dá ~340 MB por camada.

```
6,5 GB / 0,34 GB  ->  ~19 das 47 camadas  (40%)
```

Essas 19 rodam a 448 GB/s em vez de 68,9 — 6,5× mais rápido, por serem
memory-bound dos dois lados:

```
tempo = 0,60·T + 0,40·T/6,5 = 0,66·T   ->  ~1,5x   ->  ~20 tok/s
```

Passa do corte, mas com folga pequena e **desperdiçando VRAM**: guarda 64
experts por camada para usar 4.

### B — atenção de todas as camadas

A atenção roda **inteira, em toda camada, para todo token** — nunca há parte
ociosa. Os pesos de atenção das 47 camadas cabem folgados em VRAM, e sobra
espaço para experts das primeiras camadas.

Custo: a ativação cruza o PCIe duas vezes por camada, 94 travessias por token.
Em banda isso é irrelevante (94 × 8 KB = 752 KB, 63 µs a 12 GB/s); o que pesa é
a **latência** de cada travessia, ~10–20 µs, dando 1–2 ms por token contra os
74 ms atuais — 2%.

**B usa a VRAM por utilidade, não por proximidade.** É a partição que um MoE
pede, e ela não estava no plano original porque o plano foi escrito antes de
existir a medição de tráfego por tensor.

## Ordem de trabalho

1. **Interface de backend e carregador** — C ABI, `LoadLibrary`/`dlopen`,
   queda para CPU com mensagem que distingue "DLL não achada" de "dependência
   da DLL não achada" ([10](10-implementacao.md)). Não precisa de nvcc.
2. **DLL mínima com um kernel** — matmul int4 g64, espelhando o da CPU. Prova
   a toolchain e mede a travessia de verdade.
3. **Medir** uma camada na GPU contra a mesma na CPU. Se a razão não chegar
   perto de 6×, o desenho está errado e nada mais precisa ser escrito.
4. Só então o planner que escolhe a partição.

O passo 3 é o portão. Ele custa pouco e mata o M2 cedo se tiver de morrer.
