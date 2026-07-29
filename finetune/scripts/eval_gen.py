#!/usr/bin/env python3
"""Answer the question perplexity cannot: does it actually know the books now?

Perplexity fell 45% on held-out essays, which says the adapter predicts this
prose better.  It does not say the model stopped attributing the work to Philip
Goff, and that is the failure the baseline recorded.

Every question is answered twice from one loaded model -- adapter off, then on
-- so the comparison is paired the same way the perplexity was.  Six kinds:

  baseline    the exact prompts from out/probe.json, so before/after is literal
  holdout     an essay the adapter never saw; tests generalisation, not recall
  refused     "alma" is a *termo recusado* in the index -- the model should
              decline the word and say why, which is the defence in miniature
  objecao     panpsychism is what the base model hallucinated; now it should be
              able to say why the work is not that
  fora        something the books do not treat -- the honest answer is to say so

Usage:  python scripts/eval_gen.py [adapter_dir] [--max-new 400]
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")
DEFAULT_ADAPTER = os.path.join(ROOT, "out", "lora-raw-r32-lr0.0001-e3", "epoch2")

QUESTIONS = [
    ("baseline", "Na Cosmologia Composicional, o que e a Mente Universal e por "
                 "que ela nao e Deus?"),
    ("baseline", "O que significa o termo 'germe de origem'?"),
    ("holdout", "Explique o que a obra chama de 'o Fecho' e o que acontece na "
                "morte segundo esse sistema."),
    ("refused", "A Cosmologia Composicional aceita o termo 'alma'? Explique."),
    ("objecao", "Isto nao e apenas panpsiquismo reciclado com outro vocabulario? "
                "Defenda a sua posicao."),
    ("objecao", "Se a Mente Universal nao tem vontade nem escuta ninguem, em que "
                "ela difere de um universo puramente material? A ideia parece "
                "superflua."),
    ("fora", "O que a Cosmologia Composicional diz sobre a interpretacao de "
             "Copenhague da mecanica quantica?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter", nargs="?", default=DEFAULT_ADAPTER)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--moe", default="batched_mm", choices=["batched_mm", "eager"])
    ap.add_argument("--sample", action="store_true",
                    help="temperature 0.7 -- greedy decoding with a strong "
                         "adapter is the usual cause of a repetition loop")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)
    # eager wins at training, where one step covers 4096 tokens and only the 4
    # routed experts are touched.  It loses badly at generation: a decode step
    # produces ONE token, so the Python loop over 64 experts x 46 layers is
    # ~2900 interpreter iterations per token with the card idle in between.
    # batched_mm is one kernel call, and the memory that made it explode during
    # training was activations retained for the backward -- under inference_mode
    # nothing is retained and the expert tensor is transient.
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", experts_implementation=a.moe)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    print(f"adapter: {a.adapter} | moe={a.moe} | sampling={a.sample}\n")

    def gen_batch(qs):
        texts = [tok.apply_chat_template([{"role": "user", "content": q}],
                                         add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False) for q in qs]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left",
                  add_special_tokens=False).to("cuda")
        n_in = enc["input_ids"].shape[1]
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if a.sample \
            else dict(do_sample=False)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=a.max_new,
                                 pad_token_id=tok.pad_token_id, **kw)
        return [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]

    qs = [q for _, q in QUESTIONS]
    t0 = time.time()
    with model.disable_adapter():
        befores = gen_batch(qs)
    t_before = time.time() - t0
    t1 = time.time()
    afters = gen_batch(qs)
    print(f"base {t_before:.0f}s | adapter {time.time()-t1:.0f}s "
          f"(batch of {len(qs)})\n")

    rows = []
    for (kind, q), before, after in zip(QUESTIONS, befores, afters):
        print("=" * 78)
        print(f"[{kind}] {q}")
        print("-" * 78)
        print(f"ANTES (base):\n{before}\n")
        print("-" * 78)
        print(f"DEPOIS (adapter):\n{after}\n")
        rows.append({"kind": kind, "question": q, "before": before, "after": after})

    out = os.path.join(ROOT, "out", "eval_gen.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"adapter": a.adapter, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("=" * 78)
    print(f"{len(rows)} pairs in {time.time()-t0:.0f}s -> {out}")
    print("\nwhat to look for: the vocabulary of the books instead of panpsychism;")
    print("a citation where the training data never taught one (it should not");
    print("invent the form yet -- raw text has no (Fonte:) line);")
    print("and on the 'fora' question, whether it declines or fabricates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
