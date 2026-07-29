#!/usr/bin/env python3
"""Flag definition answers that describe the wrong term.

Judging 872 pairs by hand is possible but wasteful: the measured rate in
`oracle` mode is ~90% clean, and every error found so far in the clean mode was
the same shape -- the answer defines term X using the gloss of a neighbouring
term Y:

  "Que elemento é fonte da gênese da Mente individual?"  -> answered with the
     Dicionário, which the index defines as the interface for accessing values
  "O que é Interface bidirecional?"  -> answered with the Dicionário's gloss
     verbatim ("a porta, a medida")
  "O que é Primeira projeção?"  -> answered with generic metaphysics
  "O que é Valores?"            -> same

That shape is measurable.  For a `definicao` pair we know which term was asked,
so we know which gloss is the right one.  Compare the answer's content words
against the correct gloss and against every other gloss: when some other term
matches better than the one asked, the answer is describing the wrong thing.

This does not replace judgement -- it ranks pairs so judgement goes where the
errors are, instead of reading 872 in order.
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
MANIFEST = os.path.join(ROOT, "data", "manifest.json")

STOP = set("""a o as os um uma de do da dos das em no na nos nas por para com sem
sobre entre e ou mas que se nao nem como quando onde qual ao aos pelo pela este
esta esse essa aquele isto isso seu sua seus suas seus e sao era eram ser sendo
sido tem tinha ha foi foram sera ser mais menos muito todo toda todos todas
outro outra mesmo ja tambem so apenas ainda entao assim porque pois lhe ele ela
eles elas nao cf ver""".split())


def fold(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def words(s):
    return {w for w in re.findall(r"[a-z]{4,}", fold(s)) if w not in STOP}


def load_terms():
    man = json.load(open(MANIFEST, encoding="utf-8"))
    out = []
    for d in man["docs"]:
        if d["kind"] != "indice_de_termos":
            continue
        for line in open(os.path.join(ROOT, d["path"]), encoding="utf-8"):
            m = re.match(r"^\*\*(.+?)\*\*\s*[—–-]+\s*(.+)$", line.strip())
            if m:
                out.append({"head": m.group(1).strip(), "gloss": m.group(2).strip(),
                            "w": words(m.group(2))})
    return out


def asked_term(question):
    m = re.search(r'"(.+?)"', question)
    return m.group(1).strip() if m else None


def jaccardish(a, b):
    """Overlap normalised by the gloss, not by the union.

    The answer is far longer than the gloss, so a symmetric measure would score
    every pair near zero.  What matters is how much OF THE GLOSS the answer
    covers.
    """
    return len(a & b) / max(1, len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "suspects.jsonl"))
    ap.add_argument("--margin", type=float, default=0.0,
                    help="flag when another term beats the asked one by this much")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.infile, encoding="utf-8")]
    terms = load_terms()
    by_head = {fold(t["head"]): t for t in terms}

    defs = [r for r in rows if r.get("family") == "definicao"]
    flagged, checked, nomatch = [], 0, 0
    for r in defs:
        t = asked_term(r["question"])
        if not t:
            continue
        key = fold(t)
        target = by_head.get(key)
        if target is None:
            cand = [v for k, v in by_head.items() if key in k or k in key]
            target = cand[0] if cand else None
        if target is None:
            nomatch += 1
            continue
        checked += 1
        aw = words(r["answer"])
        own = jaccardish(aw, target["w"])
        best, best_t = own, target
        for other in terms:
            if other["head"] == target["head"]:
                continue
            s = jaccardish(aw, other["w"])
            if s > best:
                best, best_t = s, other
        if best > own + a.margin and best_t["head"] != target["head"]:
            flagged.append({**r, "asked": target["head"], "own_score": round(own, 3),
                            "best_other": best_t["head"], "best_score": round(best, 3)})

    with open(a.out, "w", encoding="utf-8") as f:
        for r in flagged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(defs)} definicao pairs | {checked} with a matched index entry "
          f"| {nomatch} unmatched")
    print(f"flagged {len(flagged)} ({100*len(flagged)/max(1,checked):.0f} %) "
          f"-> {a.out}")
    print("\nmais suspeitos (a resposta cobre melhor a glosa de OUTRO termo):")
    for r in sorted(flagged, key=lambda r: r["own_score"])[:12]:
        print(f"  {r['asked']!r:38s} own={r['own_score']:.2f} "
              f"-> parece {r['best_other']!r} ({r['best_score']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
