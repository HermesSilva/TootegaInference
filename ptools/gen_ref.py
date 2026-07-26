#!/usr/bin/env python3
"""Geracao gulosa de REFERENCIA, em numpy, independente do engine em C.

Existe por causa de um resultado que nao fecha: no `Tests/gentest.exe`, os
pesos BF16 originais degeneram tanto quanto os quantizados -- `*   *   *   *`
em loop, 52,7% de 4-gramas repetidos, 1 de 12 prompts terminando em EOS.

Se a quantizacao esta inocente, sobram tres suspeitos: o forward em C, o
template de chat, ou a amostragem. Esta ferramenta separa o primeiro dos
outros dois, porque refaz a conta por um caminho que nao compartilha uma linha
com o engine -- `oracle.forward`, ja validado token a token contra ele.

  numpy coerente + C degenerado  -> bug no engine
  numpy degenerado tambem        -> template ou amostragem; engine limpo

E lento de proposito: ~1,5 s por token. Sao 40 tokens para responder a
pergunta, nao um benchmark.

    python ptools/gen_ref.py --prompt "Qual e a capital do Brasil?" -n 40
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import Container, forward          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "glm47flash-int4" / "model.tootega"
DEFAULT_SRC = ROOT / "models" / "glm47flash-bf16"

# O mesmo template que o engine monta, conferido contra `apply_chat_template`.
PRE, POST = "[gMASK]<sop><|user|>", "<|assistant|><think>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--prompt", default="Qual e a capital do Brasil? Responda em uma frase.")
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--raw", action="store_true",
                    help="sem template: gera a partir do texto cru")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src)

    text = args.prompt if args.raw else PRE + args.prompt + POST
    ids = list(tok(text, add_special_tokens=False)["input_ids"])

    eos = set()
    import json
    cfg = json.loads((Path(args.src) / "config.json").read_text(encoding="utf-8"))
    e = cfg.get("eos_token_id")
    eos = set(e) if isinstance(e, list) else {e}

    c = Container(Path(args.model))

    print(f"prompt : {text!r}")
    print(f"ids    : {len(ids)} tokens")
    print(f"modelo : {args.model}")
    print(f"gerando {args.n} tokens, guloso, ~1,5 s cada\n", flush=True)

    out = []
    for step in range(args.n):
        # Sem cache: recomputa a sequencia inteira a cada passo.  E O(n^2) e
        # nao importa -- o ponto e nao compartilhar NADA com o engine, nem o
        # KV, para que um bug de cache no C nao se reproduza aqui.
        logits, _, _ = forward(c, ids)
        nxt = int(logits[-1].argmax())

        if nxt in eos:
            print(f"\n[EOS em {step} tokens]")
            break

        ids.append(nxt)
        out.append(nxt)
        print(tok.decode([nxt]), end="", flush=True)

    print("\n")
    print("texto completo:")
    print(tok.decode(out))

    # Degeneracao, medida do mesmo jeito que o gentest mede.
    if len(out) >= 8:
        grams = [tuple(out[i:i+4]) for i in range(len(out) - 3)]
        seen, rep = set(), 0
        for g in grams:
            if g in seen:
                rep += 1
            seen.add(g)
        print(f"\nrep4: {100.0 * rep / len(grams):.1f}%")


if __name__ == "__main__":
    main()
