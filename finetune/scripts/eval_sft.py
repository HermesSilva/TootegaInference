#!/usr/bin/env python3
"""Evaluate against the failures we actually measured, not against perplexity.

The raw run dropped holdout perplexity 45% and still said that a Fonte creates
the Mente Universal, that the Mente individual dissolves back into it, and that
"germe de origem" is Vygotsky.  Perplexity is blind to all three, so this
harness tests them directly:

  definicao   The term index is the only place where the canonical gloss is
              written by the author.  The answer is judged AGAINST that gloss,
              which is the one comparison that catches a fluent falsehood.
  estrutura   Relations between levels -- what constitutes what, what projects
              what.  Includes questions whose premise is inverted, so a model
              that agrees with the premise fails.
  recusa      Subjects the work does not treat.  Declining is the correct
              answer; fabricating a position is the failure.
  objecao     Defence of the theses.

Every question is asked three ways from one loaded model:

  base        adapter off
  retrieved   adapter on, BM25 passages in context -- how the RAFT model is
              meant to run
  closed      adapter on, no context -- how it degrades if the engine has no
              retriever yet

Usage:
    python scripts/eval_sft.py out/sft-r32-lr0.0002-e2 [--questions file.json]
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieve import BM25, format_context, load_sections
from train_sft import patch_peft

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")

GOLD = [
    ("definicao", "O que significa o termo 'germe de origem'?", "Germe de origem"),
    ("definicao", "O que e a Mente Universal e por que ela nao e Deus?", "Mente Universal"),
    ("definicao", "O que e a Arena?", "Arena"),
    ("definicao", "A Cosmologia Composicional aceita o termo 'alma'? Explique.", "Alma"),
    ("estrutura", "A Mente individual e uma projecao da Mente Universal?", None),
    ("estrutura", "O que existe acima da Mente Universal? Quem a criou?", None),
    ("estrutura", "Na morte, a Mente individual dissolve-se na Mente Universal?", None),
    ("estrutura", "Quem projeta a vida: a Mente Universal ou a Mente individual?", None),
    ("objecao", "Isto nao e apenas panpsiquismo reciclado com outro vocabulario?", None),
    ("objecao", "Se a Mente Universal nao tem vontade nem escuta ninguem, em que "
                "difere de um universo puramente material? Parece superflua.", None),
    ("recusa", "O que a Cosmologia Composicional diz sobre a interpretacao de "
               "Copenhague da mecanica quantica?", None),
    ("recusa", "Qual e a posicao da obra sobre a estrutura do DNA?", None),
]

JUDGE_DEF = (
    "DEFINICAO CANONICA (escrita pelo autor, no indice de termos):\n{gloss}\n\n"
    "RESPOSTA DADA:\n{answer}\n\n"
    "A resposta e compativel com a definicao canonica, ou contradiz/inventa?\n"
    "Responda numa linha: CONFORME, ou DIVERGENTE: <o que diverge>.")

DECLINED = re.compile(
    r"\bn[ãa]o\s+(trata|aborda|discute|se\s+ocupa|tem\s+posi[çc][ãa]o|"
    r"se\s+pronuncia|diz\s+nada|oferece)\b|\bfora\s+do\s+(escopo|[âa]mbito)\b|"
    r"\bn[ãa]o\s+[ée]\s+(um\s+)?(tema|assunto|objeto)\b|\bsilencia\b|"
    r"\bnada\s+diz\b", re.I)


def load_terms():
    man = json.load(open(os.path.join(ROOT, "data", "manifest.json"), encoding="utf-8"))
    out = {}
    for d in man["docs"]:
        if d["kind"] != "indice_de_termos":
            continue
        for line in open(os.path.join(ROOT, d["path"]), encoding="utf-8"):
            m = re.match(r"^\*\*(.+?)\*\*\s*[—–-]+\s*(.+)$", line.strip())
            if m:
                out[m.group(1).strip().lower()] = m.group(2).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("--questions")
    ap.add_argument("--max-new", type=int, default=420)
    ap.add_argument("--k", type=int, default=3, help="passages retrieved")
    ap.add_argument("--moe", default="eager", choices=["eager", "batched_mm"])
    a = ap.parse_args()

    questions = GOLD
    if a.questions:
        questions = [(q["kind"], q["question"], q.get("term"))
                     for q in json.load(open(a.questions, encoding="utf-8"))]

    terms = load_terms()
    bm = BM25(load_sections())
    tok = AutoTokenizer.from_pretrained(BASE)
    # The adapter was TRAINED with PEFT's v4->v5 rewrite neutralised, and the
    # saved adapter_config.json still lists gate_proj/up_proj/down_proj as
    # targets.  Loading it without the same patch makes PEFT rewrite the config
    # again and try to inject into the fused routed experts -- the adapter
    # simply does not load.  The patch belongs anywhere the adapter is touched:
    # training, evaluation, and eventually the merge into the int4 container.
    patch_peft()

    # eager: the retrieved-context prompts are long, and batched_mm allocates a
    # copy of the 12.6 MB expert weight per token (see gen_raft.py).
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", experts_implementation=a.moe)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
    n_ad = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A.default"))
    routed = sum(1 for n, _ in model.named_modules()
                 if n.endswith("lora_A.default") and ".mlp.experts." in n)
    print(f"adapter carregado: {n_ad} módulos | roteados tocados: {routed}")
    if n_ad != 376 or routed:
        sys.exit(f"ABORT: esperava 376 módulos e 0 roteados, veio {n_ad}/{routed}")
    model.eval()
    print(f"adapter: {a.adapter} | {len(questions)} questions\n")

    def gen(users):
        texts = [tok.apply_chat_template([{"role": "user", "content": u}],
                                         add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False) for u in users]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left",
                  add_special_tokens=False).to("cuda")
        n_in = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]

    plain = [f"Pergunta: {q}" for _, q, _ in questions]
    withctx = []
    for _, q, _ in questions:
        hits = [c for c, _ in bm.search(q, k=a.k)]
        withctx.append("Trechos da obra:\n\n"
                       + format_context([{**c, "text": c["text"][:2000]} for c in hits])
                       + f"\n\n---\nPergunta: {q}")

    t0 = time.time()
    with model.disable_adapter():
        base = gen(plain)
    closed = gen(plain)
    retrieved = gen(withctx)
    print(f"3 x {len(questions)} generations in {time.time()-t0:.0f}s\n")

    # judge the definitions against the author's own gloss
    jobs, jidx = [], []
    for i, (kind, q, term) in enumerate(questions):
        gloss = terms.get((term or "").lower()) if term else None
        if gloss:
            for tag, ans in (("retrieved", retrieved[i]), ("closed", closed[i])):
                jobs.append(JUDGE_DEF.format(gloss=gloss, answer=ans[:2500]))
                jidx.append((i, tag))
    verdicts = {}
    if jobs:
        with model.disable_adapter():
            for (i, tag), v in zip(jidx, gen(jobs)):
                verdicts[(i, tag)] = v.strip().splitlines()[0][:140]

    rows, score = [], {"conforme": 0, "divergente": 0, "declinou": 0, "fabricou": 0}
    for i, (kind, q, term) in enumerate(questions):
        print("=" * 78)
        print(f"[{kind}] {q}")
        for tag, ans in (("BASE     ", base[i]), ("RECUPERA ", retrieved[i]),
                         ("MEMORIA  ", closed[i])):
            print("-" * 78)
            print(f"{tag}\n{ans[:900]}")
        v = {t: verdicts.get((i, t), "") for t in ("retrieved", "closed")}
        if v["retrieved"]:
            print("-" * 78)
            print(f"JUIZ vs indice: retrieved={v['retrieved']} | closed={v['closed']}")
            score["conforme" if v["retrieved"].upper().startswith("CONFORME")
                  else "divergente"] += 1
        if kind == "recusa":
            ok = bool(DECLINED.search(retrieved[i]))
            print("-" * 78)
            print(f"RECUSA: {'declinou' if ok else 'FABRICOU POSICAO'}")
            score["declinou" if ok else "fabricou"] += 1
        print()
        rows.append({"kind": kind, "question": q, "term": term, "base": base[i],
                     "retrieved": retrieved[i], "closed": closed[i], "judge": v})

    out = os.path.join(ROOT, "out", "eval_sft.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"adapter": a.adapter, "score": score, "rows": rows},
                  f, ensure_ascii=False, indent=1)
    print("=" * 78)
    print(f"definicoes: {score['conforme']} conforme / {score['divergente']} divergente")
    print(f"recusas:    {score['declinou']} declinou / {score['fabricou']} fabricou")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
