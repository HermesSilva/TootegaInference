#!/usr/bin/env python3
"""Read the audit file and show WHY pairs were rejected.

Three rounds of tuning against a drop counter moved retention from 0% to 17%
and explained nothing.  The counter says "unfaithful: 8"; it does not say
whether the answer was wrong, or the judge was confused, or -- as is possible
here -- the citation line the generator now appends automatically is what the
judge is objecting to.  Those three have opposite fixes.

Usage:
    python scripts/audit.py data/sample.audit.jsonl          # summary
    python scripts/audit.py data/sample.audit.jsonl --full   # every rejection
"""
import argparse
import json
import re
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.path, encoding="utf-8")]
    kept = [r for r in rows if r["kept"]]
    rej = [r for r in rows if not r["kept"]]
    print(f"{len(rows)} pairs: {len(kept)} kept, {len(rej)} rejected "
          f"({100*len(kept)/max(1,len(rows)):.0f} %)\n")

    print("by family:", dict(Counter(r["family"] for r in rows)))
    print("by mode:  ", dict(Counter(r["mode"] for r in rows)))
    print("kept by mode:", dict(Counter(r["mode"] for r in kept)))

    # What kind of complaint is the judge making?
    kinds = {
        "citacao": r"\b(atribui|erroneamente|§|Ensaio|Livro)\b.*\b(secao|ensaio|livro|numero)\b",
        "contradiz": r"\bcontradiz\b",
        "acrescenta": r"\b(acrescenta|atribui|introduz|afirma que)\b",
    }
    print("\nnatureza das reprovacoes:")
    for name, pat in kinds.items():
        n = sum(1 for r in rej if re.search(pat, r["verdict"], re.I))
        print(f"  {name:12s} {n}")

    print(f"\n{'='*78}\nREPROVADOS (primeiros {a.n if not a.full else len(rej)})")
    for r in (rej if a.full else rej[:a.n]):
        print("=" * 78)
        print(f"[{r['family']}/{r['mode']}] {r['question']}")
        print(f"cite anexada: {r['cite']}")
        print(f"contexto: " + " | ".join(
            f"{c['cite']}" for c in r["context"]))
        print(f"\nVEREDICTO COMPLETO:\n{r['verdict']}")
        print(f"\nRESPOSTA:\n{r['answer'][:1500]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
