#!/usr/bin/env python3
"""Generate instruction pairs from the books, using the model itself.

Self-instruct in two passes, because asking for a finished pair in one shot
trades away the thing that matters: diversity comes from the question, fidelity
comes from the answer being written with the source paragraph in the context
window.  Pass A writes questions from a section; pass B answers each one with
that same section in front of it.

Six families, and each has a reason:

  definicao   From the term index -- 223 entries, already glossed by the author
              and already carrying the citation.  Four are marked *termo
              recusado* ("Alma"), and those are the most valuable pairs in the
              set: they teach the model to decline a word and say why.
  exposicao   From an essay section: state the thesis and develop it.
  ligacao     Two sections from different essays: how do these connect.
  objecao     The defence.  The books already argue against materialism, theism
              and idealism, and mark the exact point where their own analogies
              break ("A analogia da sociedade e os seus limites").  This is
              extraction, not invention.
  aplicacao   A concrete case examined through the system.
  recusa      Questions the work does not answer.  Without these the model
              invents, which is exactly what the baseline shows it doing today.

Holdout is excluded as context.  The six essays being used to measure the
adapter must not be the source of a single training pair, or the measurement
measures itself.

Two failure modes are filtered, not hoped away: answers that refer to a passage
the reader cannot see ("conforme o trecho acima"), which is an artefact of how
the answer was produced and would be nonsense at inference; and answers that
drift out of Portuguese.

Usage:
    python scripts/gen_sft.py --limit 20 --out data/sample.jsonl   # inspect first
    python scripts/gen_sft.py --out data/sft.jsonl                 # the real run
"""
import argparse
import json
import os
import random
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")
MANIFEST = os.path.join(ROOT, "data", "manifest.json")

HOLDOUT = ["livro_1_ensaio_9_", "livro_1_ensaio_16_", "livro_2_ensaio_1_",
           "livro_2_ensaio_15_", "livro_3_ensaio_6_", "livro_3_ensaio_23_"]

SYSTEM = (
    "Voce e um estudioso da Cosmologia Composicional, a obra em tres livros de "
    "que os trechos abaixo fazem parte. Escreve em portugues, com precisao e "
    "elegancia, e defende as teses da obra pelos seus proprios argumentos."
)

# Answers must stand alone.  The section is in the context only so the answer is
# faithful; the reader will never see it.
STANDALONE = (
    "A resposta deve ser auto-suficiente: NUNCA se refira a 'o trecho', 'o texto "
    "acima', 'o excerto' ou 'o material fornecido'. Escreva como quem conhece a "
    "obra e a expoe de memoria."
)

ASK = {
    "exposicao": "Escreva {n} perguntas que um leitor interessado faria sobre a "
                 "tese exposta nesta secao. Perguntas abertas, que pecam "
                 "explicacao e nao apenas um dado.",
    "objecao": "Escreva {n} OBJECOES que um critico culto levantaria contra o que "
               "esta secao afirma. Formule-as como perguntas ou desafios diretos, "
               "na voz do critico, sem responder a elas.",
    "aplicacao": "Escreva {n} perguntas sobre casos concretos da vida em que esta "
                 "secao ajudaria a decidir ou a compreender. Perguntas praticas.",
}

ANSWER = {
    "exposicao": "Responda a pergunta expondo a tese de modo completo e ordenado.",
    "objecao": "Responda a objecao DEFENDENDO a obra: reconheca o que ha de "
               "legitimo na critica, e mostre por que a tese resiste, usando os "
               "argumentos da propria obra.",
    "aplicacao": "Responda aplicando o sistema ao caso, sem moralizar e sem "
                 "prometer mais do que a obra sustenta.",
    "definicao": "Defina o termo com precisao, diga o que ele NAO e, e situe-o no "
                 "sistema.",
    "ligacao": "Responda mostrando como as duas ideias se articulam no sistema.",
}

BAD = re.compile(
    r"\b(o|no|neste|deste|nesse|desse|do)\s+(trecho|excerto|texto\s+acima|"
    r"material\s+fornecido|passagem\s+acima)\b|\bcomo\s+(o\s+)?texto\s+(acima|"
    r"fornecido)\b|\bconforme\s+(o\s+)?(trecho|excerto|acima)\b|\bsegundo\s+o\s+"
    r"trecho\b", re.I)
PT = re.compile(r"\b(que|nao|para|uma|dos|das|com|por|como|mas|ele|ela|isso)\b", re.I)
NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$")


def is_holdout(fname):
    return any(fname.startswith(h) for h in HOLDOUT)


def sections(text):
    """Split on ## headings, keeping the roman numeral the citation points at."""
    parts = re.split(r"^##\s+", text, flags=re.M)[1:]
    out = []
    for p in parts:
        head, _, body = p.partition("\n")
        m = re.match(r"([IVXLC]+)\s*[.)-]\s*(.+)", head.strip())
        roman, title = (m.group(1), m.group(2)) if m else (None, head.strip())
        body = body.strip()
        if len(body) > 400:
            out.append({"roman": roman, "title": title, "body": body})
    return out


def terms(text):
    """Entries of the term index: **Termo** — glosa. *II*; cf. ...

    The separator is an em dash, the character the author actually typed.  A
    hyphen class silently matches nothing here and the whole term index --
    223 entries, the best definition source in the corpus -- vanishes from the
    dataset without an error.
    """
    out = []
    for line in text.splitlines():
        m = re.match(r"^\*\*(.+?)\*\*\s*[—–-]+\s*(.+)$", line.strip())
        if m and len(m.group(2)) > 60:
            out.append({"term": m.group(1).strip(), "gloss": m.group(2).strip(),
                        "refused": "termo recusado" in m.group(2).lower()})
    return out


class Runner:
    def __init__(self, model, tok, bs, max_new, temp):
        self.m, self.t, self.bs, self.max_new, self.temp = model, tok, bs, max_new, temp

    def __call__(self, prompts):
        outs = []
        for i in range(0, len(prompts), self.bs):
            chunk = prompts[i:i + self.bs]
            texts = [self.t.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False,
                enable_thinking=False) for p in chunk]
            enc = self.t(texts, return_tensors="pt", padding=True,
                         padding_side="left", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                gen = self.m.generate(**enc, max_new_tokens=self.max_new,
                                      do_sample=self.temp > 0, temperature=self.temp or None,
                                      top_p=0.9 if self.temp > 0 else None,
                                      pad_token_id=self.t.pad_token_id)
            n_in = enc["input_ids"].shape[1]
            outs.extend(self.t.decode(g[n_in:], skip_special_tokens=True).strip()
                        for g in gen)
            done = min(i + self.bs, len(prompts))
            print(f"    {done}/{len(prompts)}", end="\r", flush=True)
        print(" " * 24, end="\r")
        return outs


def parse_list(text, limit):
    out = []
    for line in text.splitlines():
        m = NUMBERED.match(line)
        if not m:
            continue
        q = m.group(1).strip().strip('"').strip()
        if 15 < len(q) < 300:
            out.append(q)
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sft.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N source items")
    ap.add_argument("--per-section", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--moe", default="batched_mm", choices=["batched_mm", "eager"])
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)

    man = json.load(open(MANIFEST, encoding="utf-8"))
    docs = [d for d in man["docs"] if not is_holdout(d["file"])]
    essays = [d for d in docs if d["kind"] == "ensaio"]
    indexes = [d for d in docs if d["kind"] == "indice_de_termos"]
    print(f"corpus: {len(essays)} essays usable, {len(indexes)} term indexes "
          f"({len(man['docs']) - len(docs)} documents held out and excluded)")

    tok = AutoTokenizer.from_pretrained(BASE)
    # batched_mm, not eager: this script decodes one token at a time, where
    # eager's Python loop over 64 experts x 46 layers dominates and halves the
    # rate.  eager belongs in training, where a step covers thousands of tokens.
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", experts_implementation=a.moe)
    model.eval()
    run = Runner(model, tok, a.bs, 640, a.temp)

    # ---- build the source items --------------------------------------------
    items = []
    for d in essays:
        text = open(os.path.join(ROOT, d["path"]), encoding="utf-8").read()
        for s in sections(text):
            cite = d["cite"] + (f", SS{s['roman']}" if s["roman"] else "")
            items.append({"kind": "section", "cite": cite, "file": d["file"],
                          "title": d["title"], "section": s["title"],
                          "body": s["body"][:6000]})
    for d in indexes:
        text = open(os.path.join(ROOT, d["path"]), encoding="utf-8").read()
        for t in terms(text):
            items.append({"kind": "term", "cite": d["book_title"] + ", Indice de Termos",
                          "file": d["file"], "title": t["term"],
                          "section": None, "body": t["gloss"], "refused": t["refused"]})
    random.shuffle(items)
    if a.limit:
        items = items[:a.limit]
    n_sec = sum(1 for i in items if i["kind"] == "section")
    print(f"source items: {n_sec} sections + {len(items)-n_sec} terms")

    # ---- pass A: questions ---------------------------------------------------
    tasks = []
    for it in items:
        if it["kind"] == "term":
            q = (f"Qual e a definicao do termo \"{it['title']}\" na Cosmologia "
                 f"Composicional?" if not it.get("refused") else
                 f"Por que a Cosmologia Composicional recusa o termo "
                 f"\"{it['title']}\"?")
            tasks.append({"item": it, "type": "definicao", "question": q})
            continue
        fam = random.choice(["exposicao", "objecao", "aplicacao"])
        prompt = (f"Secao \"{it['section']}\" de \"{it['title']}\".\n\n"
                  f"{it['body']}\n\n---\n"
                  + ASK[fam].format(n=a.per_section) +
                  "\nUma por linha, numeradas. Apenas as perguntas.")
        tasks.append({"item": it, "type": fam, "ask": prompt})

    asks = [t for t in tasks if "ask" in t]
    print(f"\npass A: {len(asks)} question prompts (batch {a.bs})")
    t0 = time.time()
    for t, text in zip(asks, run([t["ask"] for t in asks])):
        t["questions"] = parse_list(text, a.per_section)
    print(f"  {time.time()-t0:.0f}s, "
          f"{sum(len(t.get('questions', [])) for t in asks)} questions parsed")

    # ---- pass B: answers -----------------------------------------------------
    pairs = []
    for t in tasks:
        for q in ([t["question"]] if "question" in t else t.get("questions", [])):
            pairs.append({"item": t["item"], "type": t["type"], "question": q})
    print(f"\npass B: {len(pairs)} answer prompts")
    prompts = []
    for p in pairs:
        it = p["item"]
        src = (f"Verbete do indice: **{it['title']}** -- {it['body']}"
               if it["kind"] == "term" else
               f"Secao \"{it['section']}\" de \"{it['title']}\":\n\n{it['body']}")
        prompts.append(
            f"{src}\n\n---\nPergunta: {p['question']}\n\n"
            f"{ANSWER[p['type']]}\n{STANDALONE}\n"
            f"Termine a resposta com a linha exata: (Fonte: {it['cite']})")
    t0 = time.time()
    answers = run(prompts)
    print(f"  {time.time()-t0:.0f}s")

    # ---- filter and write ----------------------------------------------------
    kept, drop = [], {"short": 0, "refers": 0, "lang": 0, "nocite": 0}
    with open(a.out, "w", encoding="utf-8") as f:
        for p, ans in zip(pairs, answers):
            ans = ans.strip()
            if len(ans) < 200:
                drop["short"] += 1
                continue
            if BAD.search(ans):
                drop["refers"] += 1
                continue
            if len(PT.findall(ans)) < 5:
                drop["lang"] += 1
                continue
            if "(Fonte:" not in ans:
                drop["nocite"] += 1
                continue
            rec = {"type": p["type"], "question": p["question"], "answer": ans,
                   "cite": p["item"]["cite"], "source": p["item"]["file"],
                   "section": p["item"].get("section")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept.append(rec)

    print(f"\nkept {len(kept)} of {len(pairs)} -> {a.out}")
    print(f"  dropped: {drop}")
    by = {}
    for r in kept:
        by[r["type"]] = by.get(r["type"], 0) + 1
    print(f"  by family: {by}")
    if kept:
        ex = random.choice(kept)
        print(f"\n--- sample [{ex['type']}]\nQ: {ex['question']}\nA: {ex['answer'][:700]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
