#!/usr/bin/env python3
"""Confere os logits do engine contra um `ref.json` do oraculo.

Separado de `oracle.py` porque roda em milissegundos: depois que a referencia
existe, cada mudanca de kernel (SIMD, threads) precisa ser verificada, e
recalcular o oraculo de 47 camadas a cada vez seria o que impede de verificar.

    tootega forward modelo.tootega --ids-file ids32.txt --dump-logits out.bin
    python tools/check_ref.py out.bin ref32.json
"""

import json
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    logits_path, ref_path = Path(sys.argv[1]), Path(sys.argv[2])
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    want = ref["argmax"]

    raw = np.fromfile(logits_path, dtype=np.float32)
    if raw.size % len(want):
        sys.exit(f"{logits_path}: {raw.size} floats nao divide por {len(want)} posicoes")
    got = raw.reshape(len(want), -1)
    pred = got.argmax(-1)

    same = int((pred == np.array(want)).sum())
    print(f"tokens iguais  {same}/{len(want)}")

    # Margem: quanto o vencedor ganha do segundo.  Onde ela e pequena, uma
    # mudanca de ordem de acumulacao pode virar o token -- e util saber qual
    # posicao esta por um fio antes de mexer em kernel.
    part = np.partition(got, -2, axis=-1)
    margin = part[:, -1] - part[:, -2]
    print(f"margem minima  {margin.min():.4f} na posicao {int(margin.argmin())}")

    for i, (a, b) in enumerate(zip(pred, want)):
        if a != b:
            print(f"  pos {i}: engine {a} (logit {got[i, a]:.4f}), "
                  f"oraculo {b} (logit {got[i, b]:.4f}), margem {margin[i]:.4f}")

    sys.exit(0 if same == len(want) else 1)


if __name__ == "__main__":
    main()
