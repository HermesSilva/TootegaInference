#!/usr/bin/env python3
"""Re-apply the mechanical filters to an audit file, without regenerating.

Filtering is separated from generation for one reason: a filter bug should cost
seconds, not two hours of GPU.  The audit file holds every answer the generator
produced; this replays the rules over it.

The language filter is the reason this exists.  It required 5 absolute hits of
common Portuguese function words, which a short, dense answer does not reach --
147 of 1210 perfectly good answers were discarded, including

    "Na Cosmologia Composicional, a precedência é a ordem que rege a
     existência: o necessário precede o que diverte."

The generator had been told to prefer being short and exact over long and
ornamented, and the filter then punished precisely that.  The fix is to compare
Portuguese against English density rather than count Portuguese in absolute
terms: that detects the failure that actually happens (the model answering in
English) and is indifferent to length.

Usage:
    python scripts/refilter.py data/raft_raw.audit.jsonl --out data/candidates.jsonl
"""
import argparse
import json
import os
import re
import sys

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")

PT = re.compile(r"\b(que|n[ãa]o|para|uma?|dos|das|com|por|como|mas|ele|ela|isso|"
                r"[ée]|de|em|n[ao]|se|ao|os|as|do|da|pelo|pela|seu|sua)\b", re.I)
EN = re.compile(r"\b(the|of|and|is|to|in|that|it|for|with|as|this|are|be|on|"
                r"from|which|by|an)\b", re.I)

BAD = re.compile(
    r"\b(o|no|neste|deste|nesse|desse|do)\s+(trecho|excerto|texto\s+acima|"
    r"material\s+fornecido|passagem\s+acima)\b|\bconforme\s+(o\s+)?(trecho|"
    r"excerto|acima)\b|\bsegundo\s+o\s+trecho\b|"
    r"\b(trecho|ensaio|passagem|item|documento|se[çc][ãa]o)s?\s*\**\s*\[\d+\]|"
    r"\[\d+\]\s*\**\s*(—|-|:|diz|afirma|estabelece|trata)", re.I)


def not_portuguese(text):
    """True when the answer is not in Portuguese.

    Density, not count: a 300-character answer in good Portuguese can carry
    four function words and is not suspect for it.  English text carries近zero
    of the Portuguese set and many of the English one, so the comparison
    separates them cleanly at any length.
    """
    pt, en = len(PT.findall(text)), len(EN.findall(text))
    return en > pt or (pt + en) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "candidates.jsonl"))
    ap.add_argument("--min-chars", type=int, default=180)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.infile, encoding="utf-8")]
    kept, drop = [], {"empty": 0, "short": 0, "refers": 0, "lang": 0, "nocite": 0}
    for r in rows:
        ans = (r.get("answer") or "").strip()
        if not ans:
            drop["empty"] += 1; continue
        if len(ans) < a.min_chars:
            drop["short"] += 1; continue
        if BAD.search(ans):
            drop["refers"] += 1; continue
        if not_portuguese(ans):
            drop["lang"] += 1; continue
        if r.get("mode") == "oracle" and r.get("cite") and "(Fonte:" not in ans:
            ans = ans.rstrip() + f"\n\n(Fonte: {r['cite']})"
            r = {**r, "answer": ans}
        kept.append(r)

    with open(a.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(rows)} -> {len(kept)} kept ({100*len(kept)/max(1,len(rows)):.0f} %) "
          f"-> {a.out}")
    print(f"  dropped: {drop}")
    for key in ("family", "mode"):
        c = {}
        for r in kept:
            c[r.get(key)] = c.get(r.get(key), 0) + 1
        print(f"  by {key}: {dict(sorted(c.items(), key=lambda kv: -kv[1]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
