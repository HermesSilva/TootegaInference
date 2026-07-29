#!/usr/bin/env python3
"""Prove the environment, and record the model's behaviour before training.

Four things, each of which would otherwise be found out the expensive way:

  1. What the checkpoint actually costs on the card.  The first run answered a
     question nobody had asked: 29.94 B parameters load, not the 31.22 B the
     safetensors hold.  Glm4MoeLiteForCausalLM does not build the MTP layer, so
     1.28 B stay on disk -- which is why the weights are 55.8 GB and not 62.4.

  2. The real module names and, just as important, which parameters are *not*
     modules.  The routed experts are packed 3-D tensors here, not nn.Linear, so
     PEFT cannot attach to them by name at all.

  3. That the proposed adapter attaches to the count of matrices expected.  A
     target_modules list that matches nothing trains air and reports success.

  4. What the model says today about these books.  It has never seen them, so
     the in-corpus answer is invention -- and invention is the baseline the
     fine-tune has to replace.  Recorded now or not at all.
"""
import json
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")
OUT = os.path.join(ROOT, "out", "baseline_smoke.json")

ATTN = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]

PROMPTS = [
    ("geral",
     "Explique, em portugues, o que distingue uma totalidade composta de uma "
     "simples soma das partes."),
    ("no-corpus",
     "Na Cosmologia Composicional, o que e a Mente Universal e por que ela nao "
     "e Deus?"),
    ("termo",
     "O que significa o termo 'germe de origem'?"),
]


def gb(n):
    return n / (1024 ** 3)


def main():
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(BASE)
    tok = AutoTokenizer.from_pretrained(BASE)
    print(f"config: {cfg.model_type} layers={cfg.num_hidden_layers} "
          f"experts={cfg.n_routed_experts} topk={cfg.num_experts_per_tok} "
          f"mtp={cfg.num_nextn_predict_layers}")
    print(f"tokenizer: {len(tok)} tokens (config vocab_size={cfg.vocab_size})")

    # transformers 5.x routes MoE through torch._grouped_mm by default, and that
    # kernel is compute-capability 9.0 only -- it raises on this Blackwell card
    # (12.0) in the middle of the first forward, not at load time.  batched_mm
    # is plain bmm and runs anywhere; eager is the slow fallback.
    moe = os.environ.get("COSMO_MOE", "batched_mm")
    print(f"loading bf16 onto cuda:0, experts_implementation={moe} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        experts_implementation=moe)
    model.eval()
    load_s = time.time() - t0

    n_all = sum(p.numel() for p in model.parameters())
    alloc = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    ckpt_params = 31.22e9   # measured by scripts/map_tensors.py
    print(f"loaded in {load_s:.0f}s | {n_all/1e9:.2f} B params in memory "
          f"({(ckpt_params-n_all)/1e9:.2f} B of the checkpoint not built: MTP)")
    print(f"weights {gb(alloc):.1f} GB / {gb(total):.1f} GB -> "
          f"{gb(total-alloc):.1f} GB free for activations")

    # ---- 2. modules, with full paths, and what is not a module --------------
    linear, by_suffix = {}, {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            linear[name] = tuple(mod.weight.shape)
            by_suffix.setdefault(name.split(".")[-1], []).append(name)
    print("\nnn.Linear by suffix:")
    for s, names in sorted(by_suffix.items(), key=lambda kv: -len(kv[1])):
        shapes = {linear[n] for n in names}
        print(f"  {s:22s} x{len(names):<4d} shapes={sorted(shapes)}")
        print(f"      {names[0]}")
        if len(names) > 1:
            print(f"      {names[-1]}")

    big = [(n, tuple(p.shape), p.numel()) for n, p in model.named_parameters()
           if p.dim() == 3]
    print(f"\n3-D parameters (packed experts, unreachable by PEFT): {len(big)}")
    for n, sh, ne in big[:3]:
        print(f"  {n}  {sh}  {ne/1e6:.1f} M")
    if big:
        print(f"  ... {sum(ne for _, _, ne in big)/1e9:.2f} B parameters in total")

    # ---- 4. baseline generation ---------------------------------------------
    results = []
    for tag, q in PROMPTS:
        enc = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        n_in = enc["input_ids"].shape[-1]
        t = time.time()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=200, do_sample=False,
                                 pad_token_id=cfg.pad_token_id)
        dt = time.time() - t
        new = out[0][n_in:]
        text = tok.decode(new, skip_special_tokens=True)
        print(f"\n=== [{tag}] {q}")
        print(f"--- {len(new)} tok em {dt:.1f}s ({len(new)/dt:.1f} tok/s)")
        print(text)
        results.append({"tag": tag, "prompt": q, "output": text,
                        "n_tokens": int(len(new)), "tok_per_s": len(new)/dt})

    peak_gen = torch.cuda.max_memory_allocated()

    # ---- 3. the proposed adapter --------------------------------------------
    lora_report = {}
    try:
        from peft import LoraConfig, get_peft_model
        targets = [t for t in ATTN + MLP if t in by_suffix]
        expected = sum(len(by_suffix[t]) for t in targets)
        lc = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                        task_type="CAUSAL_LM", target_modules=targets)
        pm = get_peft_model(model, lc)
        tr = sum(p.numel() for p in pm.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in pm.parameters())
        attached = [n for n, _ in pm.named_modules() if n.endswith("lora_A.default")]
        print(f"\nLoRA r=32 targets={targets}")
        print(f"  matrices expected {expected}, adapters attached {len(attached)}")
        print(f"  trainable {tr/1e6:.1f} M / {tot/1e9:.2f} B ({100*tr/tot:.3f} %)")
        print(f"  adapter+optimiser ~{tr*14/1e9:.2f} GB -> activations get "
              f"{gb(total)-gb(alloc)-tr*14/1e9:.1f} GB")
        if len(attached) != expected:
            print("  WARNING: attached != expected -- a target matched nothing")
        lora_report = {"targets": targets, "expected": expected,
                       "attached": len(attached), "trainable": int(tr), "r": 32}
        model = pm.unload()
    except Exception as e:
        print(f"\nPEFT check failed: {type(e).__name__}: {e}")
        lora_report = {"error": f"{type(e).__name__}: {e}"}

    report = {
        "base": BASE, "model_type": cfg.model_type,
        "params_in_memory_b": n_all / 1e9, "params_in_checkpoint_b": ckpt_params / 1e9,
        "mtp_not_built_b": (ckpt_params - n_all) / 1e9,
        "load_seconds": load_s,
        "vram_weights_gb": gb(alloc), "vram_peak_gen_gb": gb(peak_gen),
        "vram_total_gb": gb(total), "vram_free_gb": gb(total - alloc),
        "linear_by_suffix": {k: len(v) for k, v in by_suffix.items()},
        "packed_3d_params": len(big),
        "experts_implementation": moe,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "lora": lora_report, "generations": results,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {OUT}")
    print(f"peak VRAM {gb(peak_gen):.1f} GB, headroom {gb(total-peak_gen):.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
