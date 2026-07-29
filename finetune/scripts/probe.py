#!/usr/bin/env python3
"""Attach the adapter and measure one real training step.

Three things this settles that no amount of arithmetic could:

  target_modules   Passing suffixes is a trap here.  "down_proj" is the suffix
                   of the shared expert's nn.Linear AND of the 64 routed
                   experts' packed 3-D tensor, so PEFT reaches for both and
                   fails on a ParamWrapper.  The list is built from the model:
                   every nn.Linear except lm_head, which is exactly attention +
                   shared expert + the dense layer 0, and nothing else.

  think            The engine's server.c emits `</think>` closed immediately so
                   the model answers directly.  Generation here uses the same
                   form, otherwise the baseline records a reasoning draft that
                   the engine will never produce.

  the step         Peak VRAM and seconds per step with real windows from the
                   .ttpack, which is what turns "3 epochs" into a wall-clock
                   number instead of a guess.
"""
import json
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttpack import TTPack

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")
PACK = os.path.join(ROOT, "data", "cosmologia-87.ttpack")
OUT = os.path.join(ROOT, "out", "probe.json")

QUESTIONS = [
    ("geral", "Explique, em portugues, o que distingue uma totalidade composta "
              "de uma simples soma das partes."),
    ("no-corpus", "Na Cosmologia Composicional, o que e a Mente Universal e por "
                  "que ela nao e Deus?"),
    ("termo", "O que significa o termo 'germe de origem'?"),
]


def gb(n):
    return n / (1024 ** 3)


def main():
    moe = os.environ.get("COSMO_MOE", "batched_mm")
    cfg = AutoConfig.from_pretrained(BASE)
    tok = AutoTokenizer.from_pretrained(BASE)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", experts_implementation=moe)
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"loaded in {time.time()-t0:.0f}s | weights {gb(torch.cuda.memory_allocated()):.1f} GB "
          f"| free {gb(total - torch.cuda.memory_allocated()):.1f} GB")

    # ---- baseline generation, in the form the engine actually emits ---------
    model.eval()
    gens = []
    for tag, q in QUESTIONS:
        enc = tok.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True, enable_thinking=False)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        n_in = enc["input_ids"].shape[-1]
        t = time.time()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=320, do_sample=False,
                                 pad_token_id=cfg.pad_token_id)
        dt = time.time() - t
        text = tok.decode(out[0][n_in:], skip_special_tokens=True)
        n = out.shape[-1] - n_in
        print(f"\n=== [{tag}] {q}\n--- {n} tok, {n/dt:.1f} tok/s\n{text}")
        gens.append({"tag": tag, "prompt": q, "output": text,
                     "n_tokens": int(n), "tok_per_s": n / dt})

    # ---- adapter: attention only, and here is why ---------------------------
    # PEFT 0.20 rewrites the config for transformers >= 5: any target ending in
    # .gate_proj / .up_proj / .down_proj is moved out of target_modules and into
    # target_parameters, pointing at the FUSED routed experts.  Asking for the
    # shared expert hands back the 64 routed experts instead, and the shared
    # expert gets no adapter at all.  With lora_dropout=0 that swap is silent.
    # The five attention names survive the rewrite untouched.
    targets = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
    expected = sum(1 for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in targets)
    print(f"\ntarget modules: {targets}")
    print(f"  matrices in the model: {expected}")

    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                    task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lc)
    leaked = sorted(getattr(model.peft_config["default"], "target_parameters", None) or [])
    attached = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  attached {len(attached)} of {expected} | trainable "
          f"{trainable/1e6:.1f} M | after attach {gb(torch.cuda.memory_allocated()):.1f} GB")
    print(f"  target_parameters after PEFT's rewrite: {leaked}  "
          f"{'(clean)' if not leaked else '(LEAKED INTO THE ROUTED EXPERTS)'}")
    if len(attached) != expected or leaked:
        print("  WARNING: the adapter is not what was asked for")

    # ---- one real step, with real windows -----------------------------------
    pack = TTPack(PACK)
    train = pack.split(holdout=False)
    print(f"\npack: {len(train)} training windows of {pack.meta['window']}")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    steps = []
    for bs, sl in ((1, 1024), (2, 1024), (4, 1024), (2, 2048)):
        need = bs * sl
        rows = []
        i = 0
        while len(rows) < bs:
            ids = []
            while len(ids) < sl and i < len(train):
                ids.extend(pack.window_ids(train[i]))
                i += 1
            if i >= len(train) and len(ids) < sl:
                break
            rows.append(ids[:sl])
        if len(rows) < bs:
            print(f"  bs={bs} sl={sl}: not enough windows, skipped")
            continue
        x = torch.tensor(rows, dtype=torch.long, device="cuda")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        try:
            out = model(input_ids=x, labels=x)
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t
            peak = torch.cuda.max_memory_allocated()
            tps = need / dt
            print(f"  bs={bs} sl={sl:5d} -> loss {out.loss.item():.3f} | {dt:.2f}s | "
                  f"{tps:7.0f} tok/s | peak {gb(peak):.1f} GB "
                  f"({gb(total-peak):.1f} free)")
            steps.append({"batch": bs, "seq": sl, "loss": out.loss.item(),
                          "seconds": dt, "tokens_per_s": tps, "peak_gb": gb(peak)})
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs} sl={sl:5d} -> OOM")
            steps.append({"batch": bs, "seq": sl, "oom": True})
            torch.cuda.empty_cache()

    if steps:
        best = max((s for s in steps if "tokens_per_s" in s),
                   key=lambda s: s["tokens_per_s"], default=None)
        if best:
            corpus = pack.meta["train_ids"]
            ep = corpus / best["tokens_per_s"]
            print(f"\nat {best['tokens_per_s']:.0f} tok/s (bs={best['batch']} "
                  f"sl={best['seq']}): one epoch over {corpus} tokens = "
                  f"{ep/60:.1f} min, three epochs = {3*ep/60:.1f} min")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"moe": moe, "targets": targets, "matrices": expected,
                   "attached": len(attached), "target_parameters_leaked": leaked,
                   "trainable": trainable, "steps": steps, "generations": gens,
                   "vram_total_gb": gb(total)}, f, ensure_ascii=False, indent=1)
    print(f"saved -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
