#!/usr/bin/env python3
"""Export candidate pairs for judging in-session, instead of through the API.

The judge needs two things per pair: the answer, and the canonical statement it
could contradict.  Shipping all 223 index entries with every pair is what the
API judge did (cheap there, because the block was cached); in-session that would
be repeated verbatim on every page and waste the whole budget.

So each pair carries only the glosses of the terms it actually mentions.  A
claim about the Mente Universal is judged against the Mente Universal entry;
that is where the measured failures lived -- agency attributed to a totality the
index defines as "sem agência", the Absoluto made to precede its constituents,
"germe de origem" turned into a primitive stage.

Answers are clipped, not summarized: the opening is where the false relation is
asserted, and a truncated answer that reads clean is judged clean.

Usage:
    python scripts/export_review.py data/candidates.jsonl --page 1 --per-page 60
"""
import argparse
import json
import os
import re
import sys
import unicodedata

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
MANIFEST = os.path.join(ROOT, "data", "manifest.json")


def fold(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def load_terms():
    man = json.load(open(MANIFEST, encoding="utf-8"))
    terms = []
    for d in man["docs"]:
        if d["kind"] != "indice_de_termos":
            continue
        for line in open(os.path.join(ROOT, d["path"]), encoding="utf-8"):
            m = re.match(r"^\*\*(.+?)\*\*\s*[—–-]+\s*(.+)$", line.strip())
            if m:
                head = m.group(1).strip()
                # "Mente individual / individualizada" -> match either half
                keys = [k.strip(" ()") for k in re.split(r"\s*/\s*", head)]
                keys = [k for k in keys if len(k) > 3]
                terms.append({"head": head, "gloss": m.group(2).strip(),
                              "keys": [fold(k) for k in keys]})
    return terms


def match_terms(answer, terms):
    a = fold(answer)
    hits = []
    for t in terms:
        for k in t["keys"]:
            if re.search(r"\b" + re.escape(k) + r"\b", a):
                hits.append(t)
                break
    return hits


def relevant(answer, terms, freq, limit=4):
    """Index entries the answer mentions, rarest first.

    Rarity, not string length.  "Mente", "Arena" and "Horizonte" match almost
    every answer and almost never carry the error; "Planeja sóbria, executa
    enevoada" matches one and is exactly where the falsehood was.  Ranking by
    how many answers in the corpus mention the term puts the discriminating
    gloss at the top and lets the budget cover more pairs per page.
    """
    hits = match_terms(answer, terms)
    hits.sort(key=lambda t: freq.get(t["head"], 0))
    out, seen = [], set()
    for t in hits:
        if t["head"] in seen:
            continue
        seen.add(t["head"])
        out.append(t)
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--per-page", type=int, default=60)
    ap.add_argument("--answer-chars", type=int, default=900)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.infile, encoding="utf-8")]
    terms = load_terms()

    # How many answers mention each term -- the rarity ranking above needs it.
    freq = {}
    for r in rows:
        for t in match_terms(r["answer"], terms):
            freq[t["head"]] = freq.get(t["head"], 0) + 1

    if a.stats:
        n_pages = (len(rows) + a.per_page - 1) // a.per_page
        print(f"{len(rows)} pairs, {a.per_page} per page -> {n_pages} pages")
        from collections import Counter
        print("family:", dict(Counter(r.get("family") for r in rows)))
        print("mode:  ", dict(Counter(r.get("mode") for r in rows)))
        return 0

    start = (a.page - 1) * a.per_page
    chunk = rows[start:start + a.per_page]
    if not chunk:
        sys.exit(f"page {a.page} is empty ({len(rows)} pairs total)")

    out = a.out or os.path.join(ROOT, "review", f"page{a.page:02d}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Revisão — página {a.page} (pares {start+1}–{start+len(chunk)} "
                f"de {len(rows)})\n\n")
        for i, r in enumerate(chunk, start=start + 1):
            ans = r["answer"].strip()
            clipped = len(ans) > a.answer_chars
            if clipped:
                ans = ans[:a.answer_chars].rsplit(" ", 1)[0] + " […]"
            f.write(f"## {i}. [{r.get('family')}/{r.get('mode')}]\n")
            f.write(f"**P:** {r['question']}\n\n")
            if r.get("cite"):
                f.write(f"*fonte declarada: {r['cite']}*\n\n")
            f.write(f"**R:** {ans}\n\n")
            gl = relevant(r["answer"], terms, freq)
            if gl:
                f.write("**Glosas:**\n")
                for t in gl:
                    g = t["gloss"]
                    if len(g) > 240:
                        g = g[:240].rsplit(" ", 1)[0] + " […]"
                    f.write(f"- **{t['head']}** — {g}\n")
            f.write("\n---\n\n")

    size = os.path.getsize(out)
    print(f"page {a.page}: {len(chunk)} pairs -> {out} ({size/1024:.0f} KB, "
          f"~{size//4} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
