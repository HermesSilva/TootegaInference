#!/usr/bin/env python3
"""Doctrine checks that do not need a judge, derived from measured failures.

Three rules, each written because the generator broke it and a reader had to
catch it by hand:

  agencia   The index defines the Mente Universal as "a totalidade real, SEM
            AGÊNCIA".  The raw continued-pretraining run invented a creator
            Fonte above it; the RAFT run had it "converting incompletude into
            certo realizado".  Any sentence giving it a verb of will, decision
            or judgement contradicts the entry that defines it.

  recusado  Four terms are marked *termo recusado* in the index -- Alma, Alma
            (viajante), Encarnação, Espírito.  Using one affirmatively as the
            system's own vocabulary is an error by construction.  Mentioning it
            in order to REFUSE it is the correct behaviour, so the rule fires
            only when no refusal marker is nearby.

  dissolve  The Mente individual "persiste"; the index lists "Alma (viajante)"
            precisely to deny a substance that migrates or dissolves.  An
            answer that dissolves the Mente individual into the whole at death
            contradicts it -- the animal's temporary mind does that, not the
            individual Mente.

These are cheap, exact, and they rank pairs for a reader; they are not a
substitute for reading the flagged ones.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")

MU = r"(?:a\s+)?Mente\s+Universal"
AGENCY = re.compile(
    MU + r"\s+(?:é\s+quem\s+|ela\s+)?"
    r"(quer|deseja|decide|escolhe|julga|pune|premia|ordena|manda|determina|"
    r"pretende|planeja|delibera|intervém|intervem|converte|realiza|avalia|"
    r"cuida|zela|protege|responde|escuta|perdoa)\b", re.I)

REFUSED = ["alma", "espírito", "espirito", "encarnação", "encarnacao"]
REFUSAL_MARK = re.compile(
    r"\b(recusa|recusado|rejeita|não\s+(usa|aceita|emprega|admite)|nao\s+(usa|aceita)|"
    r"termo\s+recusado|não\s+é\s+(o\s+)?termo|evita|impróprio|improprio|"
    r"em\s+vez\s+de|no\s+lugar\s+de|ao\s+contrário\s+d|tradição|tradicao|"
    r"Platão|Platao|Avicena|Ved[āa]nta|Kardec)\b", re.I)

DISSOLVE = re.compile(
    r"(a\s+)?Mente\s+individual[^.]{0,80}?\b(dissolve|dissolu|funde|fund[ei]|"
    r"reintegra|retorna\s+ao\s+todo|volta\s+ao\s+todo|se\s+perde)\b", re.I)


def refused_hits(text):
    out = []
    for w in REFUSED:
        for m in re.finditer(r"\b" + w + r"\w*\b", text, re.I):
            lo = max(0, m.start() - 220)
            window = text[lo:m.end() + 220]
            if not REFUSAL_MARK.search(window):
                out.append(m.group(0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "doctrine_flags.jsonl"))
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.infile, encoding="utf-8")]
    flags, counts = [], Counter()
    for i, r in enumerate(rows, 1):
        ans = r["answer"]
        why = []
        m = AGENCY.search(ans)
        if m:
            why.append(("agencia", m.group(0)))
        rh = refused_hits(ans)
        if rh:
            why.append(("recusado", ", ".join(sorted(set(rh))[:3])))
        m = DISSOLVE.search(ans)
        if m:
            why.append(("dissolve", m.group(0)[:80]))
        if why:
            for k, _ in why:
                counts[k] += 1
            flags.append({**r, "n": i, "flags": why})

    with open(a.out, "w", encoding="utf-8") as f:
        for r in flags:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(rows)} pairs | {len(flags)} flagged -> {a.out}")
    print(f"  por regra: {dict(counts)}")
    for r in flags[:10]:
        kinds = ", ".join(f"{k}: {v!r}" for k, v in r["flags"])
        print(f"\n  #{r['n']} [{r.get('family')}] {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
