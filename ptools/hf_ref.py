#!/usr/bin/env python3
"""Referencia EXTERNA: o transformers rodando o modelo de verdade.

Esta e a peca que faltou o projeto inteiro, e a ausencia dela custou caro.

O engine em C e o `oracle.py` sao duas implementacoes da MESMA leitura minha do
`modeling_glm4_moe_lite.py`. Elas concordam 32/32 -- e concordavam tambem
quando a variante de RoPE estava errada. Concordancia entre implementacoes
proprias valida aritmetica, nunca interpretacao.

Aqui quem roda e o codigo da HuggingFace. Se os logits baterem, a arquitetura
esta entendida. Se nao baterem, a posicao em que divergem diz onde procurar.

Os 62 GB nao cabem nos 64 de RAM -- por isso `device_map='auto'` com
`offload_folder`: o accelerate mantem em memoria so o que esta em uso e le o
resto do disco. E lento (minutos para um passe de 32 posicoes) e nao importa:
sao alguns minutos contra dias de procurar bug no lugar errado.

    python ptools/hf_ref.py --ids-file ids32.txt --out hf_logits.npy
    python ptools/hf_ref.py --ids-file ids32.txt --compare l.bin
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "models" / "glm47flash-bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--ids", help="ids separados por virgula")
    ap.add_argument("--ids-file")
    ap.add_argument("--corpus", help="arquivo de texto; usa a primeira janela")
    ap.add_argument("--tokens", type=int, default=384, help="tamanho da janela")
    ap.add_argument("--offload", default=str(ROOT / "offload"),
                    help="pasta de offload; precisa de ~62 GB livres")
    ap.add_argument("--out", help="salva os logits (.npy)")
    ap.add_argument("--compare", help="logits do engine (f32 cru, [T][vocab])")
    ap.add_argument("--layers", help="traco do engine (--dump-layers): bissecao"
                                     " camada a camada")
    ap.add_argument("--final", help="hidden final do engine (--dump-hidden):"
                                    " isola norma final e lm_head")
    args = ap.parse_args()

    if args.corpus:
        # Mesma janela que o `ablate` mede, com o mesmo tokenizer: e o unico
        # jeito de os dois numeros serem comparaveis.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.src)
        text = Path(args.corpus).read_text(encoding="utf-8")
        ids = tok(text)["input_ids"][:args.tokens]
    elif args.ids_file:
        raw = Path(args.ids_file).read_text(encoding="utf-8")
        ids = [int(v) for v in raw.replace(",", " ").split()]
    elif args.ids:
        ids = [int(v) for v in args.ids.replace(",", " ").split()]
    else:
        sys.exit("passe --ids, --ids-file ou --corpus")

    import torch
    from transformers import AutoModelForCausalLM

    Path(args.offload).mkdir(parents=True, exist_ok=True)

    print(f"carregando {args.src} com offload em {args.offload}")
    print("(a primeira vez copia ~62 GB para a pasta de offload)\n", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.src,
        dtype=torch.float32,          # fp32: o engine acumula em fp32
        device_map="auto",
        offload_folder=args.offload,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # `attn_implementation` eager: e a formulacao que o engine reproduz, e a
    # unica que nao depende de kernel externo (docs/15).
    model.config._attn_implementation = "eager"

    print(f"forward teacher-forcing em {len(ids)} posicoes ...", flush=True)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], dtype=torch.long),
                    output_hidden_states=bool(args.layers or args.final))

    logits = out.logits[0].float().numpy()
    print(f"logits {logits.shape}\n")

    # Perplexidade: o numero que o `Tests/ablate.exe` reporta, calculado do
    # mesmo jeito e sobre as mesmas posicoes. E a comparacao que separa
    # arquitetura de quantizacao, porque aqui os pesos sao os originais.
    x = logits.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    logZ = np.log(np.exp(x).sum(axis=-1))
    nll = np.mean([logZ[i] - x[i, ids[i + 1]] for i in range(len(ids) - 1)])
    print(f"  nll {nll:.4f}   perplexidade {np.exp(nll):.2f}   "
          f"({len(ids) - 1} previsoes)\n")

    if args.corpus:
        print("  compare com o `Tests/ablate.exe --bf16` na mesma janela:\n"
              "    numeros proximos -> arquitetura entendida\n"
              "    ablate MUITO pior -> bug no forward do engine\n")

    # Lista posicao a posicao so quando cabe na tela.
    show = len(ids) if len(ids) <= 40 else 0
    if not show:
        print("  (lista por posicao omitida: janela longa)\n")

    for i in range(show):
        top = int(logits[i].argmax())
        part = np.partition(logits[i], -2)
        print(f"  {i:3d} {ids[i]:>8} -> {top:>8}   logit {logits[i, top]:8.3f}"
              f"   margem {part[-1] - part[-2]:6.3f}")

    if args.out:
        np.save(args.out, logits)
        print(f"\n{args.out} escrito")

    if args.final:
        # `hidden_states[-1]` JA passou pela norma final -- e exatamente o que
        # o engine grava com `--dump-hidden`. Comparar os dois separa o que
        # esta antes da norma do que esta depois:
        #
        #   final bate, logits nao  -> lm_head
        #   final tambem nao bate   -> norma final (ou a ultima camada)
        ref = out.hidden_states[-1][0].float().numpy()
        got = np.fromfile(args.final, dtype=np.float32).reshape(ref.shape)

        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        print(f"\n  hidden final (pos-norma): erro relativo {rel * 100:.4f}%")
        print(f"    norma  engine {np.linalg.norm(got):10.3f}   "
              f"transformers {np.linalg.norm(ref):10.3f}")
        print(f"    escala engine/ref = {np.linalg.norm(got) / np.linalg.norm(ref):.6f}")
        if rel < 0.05:
            print("    -> as camadas e a norma estao certas; o bug e no lm_head")
        else:
            print("    -> o erro ja existe antes do lm_head")

    if args.layers:
        # `hidden_states` e [camadas+1][batch][T][hidden]: entrada 0 e o
        # embedding, e a ULTIMA ja passou pela norma final -- enquanto o
        # `--dump-layers` grava a saida crua da ultima camada. Comparar as
        # duas na ultima posicao da erro gigante que nao existe; por isso ela
        # fica de fora.
        ref = np.stack([h[0].float().numpy() for h in out.hidden_states])
        n_lay, T, hid = ref.shape

        raw = np.fromfile(args.layers, dtype=np.float32)
        got = raw.reshape(T, n_lay, hid).transpose(1, 0, 2)

        print(f"\n  bissecao por camada ({n_lay - 2} de {n_lay - 1}, {T} posicoes)")
        print("  (a ultima entrada e pos-norma no transformers; use --final)\n")
        print(f"    {'camada':>8} {'erro rel':>10}   {'salto':>8}")

        prev = 0.0
        for L in range(n_lay - 1):    # ultima fica de fora: e pos-norma
            num = np.linalg.norm(got[L] - ref[L])
            den = np.linalg.norm(ref[L])
            rel = num / den if den else 0.0
            jump = rel - prev
            mark = ""
            # A primeira camada que introduz erro grande e onde o bug mora:
            # camadas seguintes so propagam o que receberam.
            if jump > 0.05 and rel > 0.1:
                mark = "  <-- introduz o erro"
            name = "embed" if L == 0 else f"{L - 1}"
            print(f"    {name:>8} {rel * 100:9.3f}% {jump * 100:8.3f}%{mark}")
            prev = rel

    if args.compare:
        raw = np.fromfile(args.compare, dtype=np.float32)
        got = raw.reshape(len(ids), -1)

        same = int((got.argmax(-1) == logits.argmax(-1)).sum())
        rel = np.linalg.norm(got - logits) / np.linalg.norm(logits)

        print(f"\nengine contra transformers:")
        print(f"  tokens iguais   {same}/{len(ids)}")
        print(f"  erro relativo   {rel * 100:.4f}%")
        print(f"  maior diferenca {np.abs(got - logits).max():.4f}")

        # Onde diverge importa mais que quanto: erro que CRESCE com a posicao
        # aponta para posicao/RoPE/cache; erro constante aponta para peso ou
        # norma.
        print("\n  erro relativo por posicao:")
        for i in range(len(ids)):
            r = np.linalg.norm(got[i] - logits[i]) / np.linalg.norm(logits[i])
            flag = "  <-- diverge" if got[i].argmax() != logits[i].argmax() else ""
            print(f"    {i:3d}  {r * 100:7.3f}%{flag}")

        sys.exit(0 if same == len(ids) else 1)


if __name__ == "__main__":
    main()
