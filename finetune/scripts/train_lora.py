#!/usr/bin/env python3
"""Continued pre-training of a LoRA adapter on the raw corpus, measured.

This is the cheap end-to-end run: no synthetic instruction data yet, just the
books, so the pipeline gets proven and the first real number arrives before the
expensive part of the plan starts.

The measurement is PAIRED, per window.  The same 61 holdout windows are scored
with the adapter disabled and again with it enabled, and what is reported is the
distribution of per-window differences -- not two global means.  docs/04 records
what an unpaired comparison did to the first perplexity measurement on this
project: it swallowed the effect in the variance between windows and came out
with the sign inverted.  PEFT's disable_adapter() makes the pairing exact,
because both scores come from one model, one load, one code path.

Windows are padded to the batch maximum with -100 labels rather than
concatenated, so no training example ever spans two documents.

Usage:
    python scripts/train_lora.py [--epochs 3] [--lr 1e-4] [--r 32] [--bs 4]
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ttpack import TTPack

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")
PACK = os.path.join(ROOT, "data", "cosmologia-87.ttpack")
ATTN = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
PAD = -100


def gb(n):
    return n / (1024 ** 3)


def batches(pack, windows, bs, shuffle_seed=None):
    """Group windows into padded batches, longest-first to waste less padding."""
    idx = list(range(len(windows)))
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        idx = [idx[i] for i in torch.randperm(len(idx), generator=g).tolist()]
    idx.sort(key=lambda i: -windows[i].len)
    out = []
    for i in range(0, len(idx), bs):
        group = [pack.window_ids(windows[j]) for j in idx[i:i + bs]]
        n = max(len(r) for r in group)
        ids = torch.full((len(group), n), 0, dtype=torch.long)
        lab = torch.full((len(group), n), PAD, dtype=torch.long)
        for k, r in enumerate(group):
            ids[k, :len(r)] = torch.tensor(r, dtype=torch.long)
            lab[k, :len(r)] = torch.tensor(r, dtype=torch.long)
        out.append((ids, lab))
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed + 1)
        out = [out[i] for i in torch.randperm(len(out), generator=g).tolist()]
    return out


@torch.no_grad()
def score(model, pack, windows):
    """Mean loss per window -- one row at a time, so nothing is averaged twice."""
    model.eval()
    losses = []
    for w in windows:
        ids = torch.tensor([pack.window_ids(w)], dtype=torch.long, device="cuda")
        losses.append(model(input_ids=ids, labels=ids).loss.item())
    model.train()
    return losses


def report_paired(before, after, label):
    d = [a - b for b, a in zip(before, after)]
    n = len(d)
    mean = sum(d) / n
    sd = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    better = sum(1 for x in d if x < 0)
    ppl_b = math.exp(sum(before) / n)
    ppl_a = math.exp(sum(after) / n)
    print(f"\n--- {label} (paired over {n} windows)")
    print(f"  perplexity   {ppl_b:8.3f} -> {ppl_a:8.3f}   ({100*(ppl_a/ppl_b-1):+.2f} %)")
    print(f"  mean d(loss) {mean:+.4f}  sd {sd:.4f}  se {se:.4f}  "
          f"t={mean/se if se else float('nan'):+.2f}")
    print(f"  windows improved {better}/{n}  ({100*better/n:.0f} %)")
    return {"n": n, "ppl_before": ppl_b, "ppl_after": ppl_a, "mean_delta": mean,
            "sd": sd, "se": se, "improved": better,
            "deltas": [round(x, 5) for x in d]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="raw")
    a = ap.parse_args()

    outdir = os.path.join(ROOT, "out", f"lora-{a.tag}-r{a.r}-lr{a.lr:g}-e{a.epochs}")
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(a.seed)

    pack = TTPack(PACK)
    train_w = pack.split(holdout=False)
    hold_w = pack.split(holdout=True)
    print(f"pack: {len(train_w)} train windows, {len(hold_w)} holdout windows")

    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", experts_implementation="eager")
    total = torch.cuda.get_device_properties(0).total_memory

    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none",
                    task_type="CAUSAL_LM", target_modules=ATTN)
    model = get_peft_model(model, lc)
    leaked = sorted(getattr(model.peft_config["default"], "target_parameters", None) or [])
    if leaked:
        sys.exit(f"ABORT: PEFT redirected the adapter into {leaked}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"adapter r={a.r} on {ATTN}: {trainable/1e6:.1f} M trainable, clean")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    # Baseline with the adapter switched off: same weights, same code path.
    print("\nscoring holdout with the adapter disabled ...")
    t = time.time()
    with model.disable_adapter():
        base_hold = score(model, pack, hold_w)
        base_train = score(model, pack, train_w[:len(hold_w)])
    print(f"  base holdout ppl {math.exp(sum(base_hold)/len(base_hold)):.3f} "
          f"({time.time()-t:.0f}s)")

    steps_per_epoch = math.ceil(len(train_w) / a.bs)
    total_steps = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, max(10, total_steps // 20), total_steps)
    print(f"training {a.epochs} epochs, {steps_per_epoch} steps/epoch, "
          f"{total_steps} total, lr={a.lr:g}")

    history, step, t0 = [], 0, time.time()
    for ep in range(1, a.epochs + 1):
        run, n_tok = 0.0, 0
        for ids, lab in batches(pack, train_w, a.bs, shuffle_seed=a.seed + ep):
            ids, lab = ids.to("cuda"), lab.to("cuda")
            loss = model(input_ids=ids, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run += loss.item()
            n_tok += int((lab != PAD).sum())
            if step % 20 == 0:
                el = time.time() - t0
                print(f"  step {step:4d}/{total_steps} ep{ep} loss {run/20:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} | {n_tok/el:.0f} tok/s | "
                      f"peak {gb(torch.cuda.max_memory_allocated()):.1f} GB")
                run = 0.0
        ep_hold = score(model, pack, hold_w)
        r = report_paired(base_hold, ep_hold, f"epoch {ep}")
        history.append({"epoch": ep, **{k: v for k, v in r.items() if k != "deltas"}})
        model.save_pretrained(os.path.join(outdir, f"epoch{ep}"))

    final_hold = score(model, pack, hold_w)
    final_train = score(model, pack, train_w[:len(hold_w)])
    rh = report_paired(base_hold, final_hold, "FINAL holdout")
    rt = report_paired(base_train, final_train, "FINAL train sample (overfit check)")

    model.save_pretrained(outdir)
    with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "trainable": trainable,
                   "minutes": (time.time() - t0) / 60,
                   "history": history, "holdout": rh, "train_sample": rt,
                   "pack_build": pack.meta["build"]}, f, ensure_ascii=False, indent=1)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {outdir}")
    print("the gap between holdout and train-sample deltas is the overfit signal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
