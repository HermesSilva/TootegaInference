#!/usr/bin/env python3
"""Find the training step that fits, and say why the others do not.

The first probe reported OOM at batch 1, sequence 1024, with 39.2 GB free --
and swallowed the allocator's message, which is the one number that explains
it.  This one keeps the message, checks that gradient checkpointing is really
on, and sweeps upward from a sequence short enough that it cannot fail.

The suspect is the MoE kernel.  batched_mm is dense: it runs all 64 experts over
every token and keeps [64, tokens, 3072] alive per layer, so 46 layers of that
is tens of GB unless checkpointing discards it.  eager loops over experts and
touches only the tokens each one was routed, which is 16x less work at top-4 of
64.  Whichever wins, it should win on a measurement.

Usage:  python scripts/probe_train.py [batched_mm|eager]
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttpack import TTPack

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")
PACK = os.path.join(ROOT, "data", "cosmologia-87.ttpack")
ATTN = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]


def gb(n):
    return n / (1024 ** 3)


def main():
    moe = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("COSMO_MOE", "batched_mm")
    out_path = os.path.join(ROOT, "out", f"probe_train_{moe}.json")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", experts_implementation=moe)
    total = torch.cuda.get_device_properties(0).total_memory
    w = torch.cuda.memory_allocated()
    print(f"[{moe}] loaded in {time.time()-t0:.0f}s | weights {gb(w):.1f} GB | "
          f"free {gb(total-w):.1f} GB")

    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                    task_type="CAUSAL_LM", target_modules=ATTN)
    model = get_peft_model(model, lc)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Checkpointing has to be on the module that owns the layers, and reentrant
    # checkpointing does not see the LoRA inputs as requiring grad.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    inner = model.base_model.model.model
    on = getattr(inner, "gradient_checkpointing", None)
    print(f"  trainable {trainable/1e6:.1f} M | gradient_checkpointing={on} | "
          f"use_cache={model.config.use_cache}")
    if not on:
        print("  WARNING: checkpointing is OFF -- every layer's activations stay alive")

    pack = TTPack(PACK)
    train = pack.split(holdout=False)
    flat = []
    for win in train[:64]:
        flat.extend(pack.window_ids(win))

    results = []
    for bs, sl in ((1, 256), (1, 512), (1, 1024), (2, 1024), (1, 2048), (4, 1024)):
        need = bs * sl
        if need > len(flat):
            continue
        x = torch.tensor([flat[i * sl:(i + 1) * sl] for i in range(bs)],
                         dtype=torch.long, device="cuda")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        t = time.time()
        try:
            loss = model(input_ids=x, labels=x).loss
            fwd_peak = torch.cuda.max_memory_allocated()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t
            peak = torch.cuda.max_memory_allocated()
            print(f"  bs={bs} sl={sl:5d} -> loss {loss.item():.3f} | {dt:5.2f}s | "
                  f"{need/dt:7.0f} tok/s | fwd {gb(fwd_peak-base_mem):5.1f} GB | "
                  f"peak {gb(peak):.1f} GB | free {gb(total-peak):.1f} GB")
            results.append({"batch": bs, "seq": sl, "loss": loss.item(),
                            "seconds": dt, "tokens_per_s": need / dt,
                            "peak_gb": gb(peak),
                            "activations_gb": gb(fwd_peak - base_mem)})
        except torch.cuda.OutOfMemoryError as e:
            msg = str(e).split("\n")[0]
            print(f"  bs={bs} sl={sl:5d} -> OOM  {msg}")
            results.append({"batch": bs, "seq": sl, "oom": msg})
        finally:
            del opt
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    ok = [r for r in results if "tokens_per_s" in r]
    if ok:
        best = max(ok, key=lambda r: r["tokens_per_s"])
        corpus = pack.meta["train_ids"]
        ep = corpus / best["tokens_per_s"]
        print(f"\nbest: bs={best['batch']} sl={best['seq']} at "
              f"{best['tokens_per_s']:.0f} tok/s")
        print(f"one epoch over {corpus} tokens = {ep/60:.1f} min | "
              f"three epochs = {3*ep/60:.1f} min")
    else:
        print("\nnothing fit -- see the OOM messages above")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"moe": moe, "trainable": trainable,
                   "gradient_checkpointing": bool(on),
                   "weights_gb": gb(w), "vram_total_gb": gb(total),
                   "results": results}, f, ensure_ascii=False, indent=1)
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
