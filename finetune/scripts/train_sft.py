#!/usr/bin/env python3
"""Supervised fine-tuning on the RAFT pairs.  Replaces the raw-text run.

What changed from train_lora.py, and why each change has a source:

  no CPT      "The Finetuner's Fallacy" finds continued pre-training HURTS
              versus going straight to SFT when the domain corpus is under
              ~10M tokens.  This corpus is 0.59M.  The raw run we measured
              confirmed it from the other side: perplexity -45% and the
              propositions still wrong.  So this trains from the BASE model.

  all linear  Targeting attention alone is the inferior configuration; the
              standard is every linear layer, and for MoE specifically:
              attention + shared expert, routed experts frozen.  PEFT 0.20
              blocks that -- it rewrites any target ending in .gate_proj /
              .up_proj / .down_proj into target_parameters pointing at the
              FUSED routed experts, silently when dropout is 0.  Neutralising
              that rewrite gives 376 adapters / 60.4 M instead of 235 / 42.1 M,
              with zero adapters on the routed experts (verified on meta).

  lr 2e-4     the standard for LoRA; the raw run used 1e-4.
  alpha 2r    crucial at higher ranks.
  2 epochs    1-3 is the range; epoch 3 regressed on holdout in the raw run.
  masked loss loss only on the answer.  Training on the prompt teaches the
              model to generate retrieved passages, which is the opposite of
              reading them.

Usage:
    python scripts/train_sft.py --data data/raft.jsonl
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")
PAD = -100


def patch_peft():
    """Stop PEFT from redirecting MLP targets into the fused routed experts.

    The rewrite exists to load adapters trained on transformers v4.  We have
    none, so it only does harm here: it moves shared_experts.gate_proj (a real
    nn.Linear) onto mlp.experts.gate_up_proj (a packed 3-D tensor of all 64
    routed experts), which is 17x the parameters and the thing we deliberately
    freeze.  Verified on meta device: with this off, 0 adapters land on routed
    experts; with it on, the shared expert gets none at all.
    """
    import peft.utils.transformers_weight_conversion as twc
    twc.convert_peft_config_for_transformers = lambda *a, **k: None


def gb(n):
    return n / (1024 ** 3)


def passage_text(c, by_cite):
    """Full section text, rehydrated from the corpus when the record is clipped.

    The audit file stores context as {cite, section, head} with `head` cut to
    400 characters -- fine for reading a verdict, wrong for training: the model
    would learn on a fifth of the passage it will actually be given at
    inference.  The citation identifies the section uniquely, so the full text
    is recoverable.
    """
    full = by_cite.get(c.get("cite"))
    if full:
        return full[:2000]
    return c.get("text") or c.get("head") or ""


def build(tok, rows, max_len, by_cite=None):
    """Chat format the engine emits: <|assistant|></think> then the answer."""
    by_cite = by_cite or {}
    out = []
    for r in rows:
        parts = []
        if r.get("context"):
            blocks = []
            for i, c in enumerate(r["context"]):
                head = c.get("title") or (c.get("cite") or "").split(", §")[0]
                sec = f" — {c['section']}" if c.get("section") else ""
                blocks.append(f"[{i+1}] {head}{sec}\n{passage_text(c, by_cite)}")
            parts.append("Trechos da obra:\n\n" + "\n\n".join(blocks) + "\n\n---\n")
        parts.append(f"Pergunta: {r['question']}")
        user = "".join(parts)
        prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                         add_generation_prompt=True, tokenize=False,
                                         enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(r["answer"], add_special_tokens=False).input_ids + [tok.eos_token_id]
        if len(p_ids) + len(a_ids) > max_len:
            keep = max_len - len(a_ids)
            if keep < 64:
                continue
            p_ids = p_ids[:keep // 2] + p_ids[-(keep - keep // 2):]
        out.append({"ids": p_ids + a_ids,
                    "labels": [PAD] * len(p_ids) + a_ids,
                    "mode": r.get("mode"), "family": r.get("family")})
    return out


def collate(batch, pad_id):
    n = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), PAD, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["ids"])
        ids[i, :L] = torch.tensor(b["ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return ids, lab, att


@torch.no_grad()
def evaluate(model, rows, pad_id):
    model.eval()
    tot, n = 0.0, 0
    per_mode = {}
    for r in rows:
        ids, lab, att = collate([r], pad_id)
        loss = model(input_ids=ids.cuda(), labels=lab.cuda(),
                     attention_mask=att.cuda()).loss.item()
        tot += loss
        n += 1
        per_mode.setdefault(r["mode"], []).append(loss)
    model.train()
    return tot / max(1, n), {k: sum(v) / len(v) for k, v in per_mode.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "raft.jsonl"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--micro-bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--warmup", type=float, default=0.08)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--attention-only", action="store_true",
                    help="the old, inferior configuration -- for comparison")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    random.shuffle(rows)
    n_val = max(8, int(len(rows) * a.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"data: {len(rows)} pairs -> {len(train_rows)} train / {len(val_rows)} val")
    modes = {}
    for r in rows:
        modes[r.get("mode")] = modes.get(r.get("mode"), 0) + 1
    print(f"  modes: {modes}")

    # cite -> full section text, so clipped context records can be rehydrated
    try:
        from retrieve import load_sections
        by_cite = {s["cite"]: s["text"] for s in load_sections(include_holdout=True)}
        print(f"corpus index: {len(by_cite)} sections available for rehydration")
    except Exception as e:
        by_cite = {}
        print(f"WARNING: no corpus index ({type(e).__name__}); "
              f"context will use whatever the records carry")

    tok = AutoTokenizer.from_pretrained(BASE)
    train = build(tok, train_rows, a.max_len, by_cite)
    val = build(tok, val_rows, a.max_len, by_cite)
    lens = sorted(len(t["ids"]) for t in train)
    sup = sum(sum(1 for x in t["labels"] if x != PAD) for t in train)
    print(f"  tokens: {sum(lens)} total, {sup} supervised "
          f"({100*sup/max(1,sum(lens)):.0f} %) | len p50={lens[len(lens)//2]} "
          f"p95={lens[int(len(lens)*0.95)]} max={lens[-1]}")

    patch_peft()
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", experts_implementation="eager")
    total_vram = torch.cuda.get_device_properties(0).total_memory

    attn = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
    targets = attn if a.attention_only else attn + ["gate_proj", "up_proj", "down_proj"]
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets))
    cfg = model.peft_config["default"]
    leaked = sorted(getattr(cfg, "target_parameters", None) or [])
    ad = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    routed = [n for n in ad if ".mlp.experts." in n]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nadapter r={a.r} alpha={2*a.r} on {targets}")
    print(f"  {len(ad)} modules, {trainable/1e6:.1f} M trainable | "
          f"routed experts touched: {len(routed)} | target_parameters: {leaked}")
    if routed or leaked:
        sys.exit("ABORT: adapter leaked into the routed experts")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    steps = math.ceil(len(train) / (a.micro_bs * a.accum)) * a.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, max(5, int(steps * a.warmup)), steps)
    tag = f"sft-r{a.r}-lr{a.lr:g}-e{a.epochs}" + ("-attn" if a.attention_only else "")
    outdir = os.path.join(ROOT, "out", tag)
    os.makedirs(outdir, exist_ok=True)

    v0, v0m = evaluate(model, val, tok.pad_token_id)
    print(f"\nval loss before training: {v0:.4f}  {v0m}")
    print(f"training {a.epochs} epochs, {steps} optimiser steps, "
          f"effective batch {a.micro_bs*a.accum}")

    hist, step, t0, micro = [], 0, time.time(), 0
    run_loss, run_tok = 0.0, 0
    for ep in range(1, a.epochs + 1):
        order = list(range(len(train)))
        random.shuffle(order)
        for i in range(0, len(order), a.micro_bs):
            batch = [train[j] for j in order[i:i + a.micro_bs]]
            ids, lab, att = collate(batch, tok.pad_token_id)
            loss = model(input_ids=ids.cuda(), labels=lab.cuda(),
                         attention_mask=att.cuda()).loss
            (loss / a.accum).backward()
            run_loss += loss.item()
            run_tok += int((lab != PAD).sum())
            micro += 1
            if micro % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0:
                    el = time.time() - t0
                    print(f"  step {step:4d}/{steps} ep{ep} "
                          f"loss {run_loss/(5*a.accum):.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} | "
                          f"{run_tok/el:.0f} sup-tok/s | "
                          f"peak {gb(torch.cuda.max_memory_allocated()):.1f} GB")
                    run_loss = 0.0
        v, vm = evaluate(model, val, tok.pad_token_id)
        print(f"--- epoch {ep}: val loss {v0:.4f} -> {v:.4f} "
              f"({100*(v/v0-1):+.1f} %)  {vm}")
        hist.append({"epoch": ep, "val_loss": v, "by_mode": vm})
        model.save_pretrained(os.path.join(outdir, f"epoch{ep}"))

    model.save_pretrained(outdir)
    with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(a), "trainable": trainable, "modules": len(ad),
                   "val_before": v0, "history": hist,
                   "minutes": (time.time() - t0) / 60}, f,
                  ensure_ascii=False, indent=1)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {outdir}")
    print("val loss by mode matters more than the mean: 'distractors' rising "
          "means the model is losing the ability to decline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
