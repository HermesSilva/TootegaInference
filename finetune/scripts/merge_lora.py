#!/usr/bin/env python3
"""Merge the LoRA adapter into the bf16 checkpoint, tensor by tensor.

Not via model.merge_and_unload(): loading the checkpoint through
Glm4MoeLiteForCausalLM drops the MTP layer (1.28 B parameters -- measured:
29.94 B load out of 31.22 B on disk), so saving the loaded model would silently
produce a checkpoint the converter cannot turn into a v4 container, which
declares mtp_layer: 47.  Reading and rewriting the shards keeps every tensor
that is not a LoRA target exactly as it was, MTP included.

It also runs on CPU, needs no GPU, and touches only the shards that actually
contain a target -- the rest are copied.

The merge itself:  W' = W + (B @ A) * (alpha / r)

accumulated in float32 and cast back to bfloat16 at the end.  Doing it in bf16
throughout loses low bits of the delta, and the delta is the whole point.

Usage:
    python scripts/merge_lora.py out/sft-r32-lr0.0002-e2 --out base/glm47flash-sft
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")


def load_adapter(adapter_dir):
    """LoRA pairs keyed by the base tensor name they modify."""
    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]
    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    pairs = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            # base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight
            if ".lora_A" not in key and ".lora_B" not in key:
                continue
            which = "A" if ".lora_A" in key else "B"
            stem = key.split(".lora_")[0]
            stem = stem.removeprefix("base_model.model.")
            name = stem + ".weight"
            pairs.setdefault(name, {})[which] = f.get_tensor(key)
    full = {k: v for k, v in pairs.items() if "A" in v and "B" in v}
    if len(full) != len(pairs):
        sys.exit(f"adapter has {len(pairs)-len(full)} unpaired A/B entries")
    return full, scale, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("--out", default=os.path.join(ROOT, "base", "glm47flash-sft"))
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    deltas, scale, cfg = load_adapter(a.adapter)
    print(f"adapter: {len(deltas)} target tensors | r={cfg['r']} "
          f"alpha={cfg['lora_alpha']} scale={scale:g}")

    index_path = os.path.join(a.base, "model.safetensors.index.json")
    index = json.load(open(index_path))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    missing = [n for n in deltas if n not in weight_map]
    if missing:
        sys.exit(f"ABORT: {len(missing)} adapter targets absent from the "
                 f"checkpoint, e.g. {missing[:3]}")

    touched = {}
    for name in deltas:
        touched.setdefault(weight_map[name], []).append(name)
    print(f"{len(shards)} shards, {len(touched)} contain a target")

    if a.dry_run:
        for s in sorted(touched):
            print(f"  {s}: {len(touched[s])} tensors")
        return 0

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    n_merged = 0
    for i, shard in enumerate(shards, 1):
        src = os.path.join(a.base, shard)
        dst = os.path.join(a.out, shard)
        if shard not in touched:
            shutil.copy2(src, dst)
            print(f"  [{i}/{len(shards)}] {shard} copied", end="\r", flush=True)
            continue
        tensors, meta = {}, {}
        with safe_open(src, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            for key in f.keys():
                t = f.get_tensor(key)
                if key in deltas:
                    A = deltas[key]["A"].to(torch.float32)
                    B = deltas[key]["B"].to(torch.float32)
                    delta = (B @ A) * scale
                    if delta.shape != t.shape:
                        sys.exit(f"ABORT: {key} delta {tuple(delta.shape)} "
                                 f"!= weight {tuple(t.shape)}")
                    t = (t.to(torch.float32) + delta).to(torch.bfloat16)
                    n_merged += 1
                tensors[key] = t
        save_file(tensors, dst, metadata=meta or {"format": "pt"})
        print(f"  [{i}/{len(shards)}] {shard} merged "
              f"({len(touched[shard])} tensors)      ", end="\r", flush=True)
    print(" " * 78, end="\r")

    for extra in ("config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "chat_template.jinja",
                  "model.safetensors.index.json"):
        p = os.path.join(a.base, extra)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(a.out, extra))

    if n_merged != len(deltas):
        sys.exit(f"ABORT: merged {n_merged} of {len(deltas)} targets")

    size = sum(os.path.getsize(os.path.join(a.out, f))
               for f in os.listdir(a.out)) / 1e9
    print(f"merged {n_merged} tensors into {a.out} "
          f"({size:.1f} GB, {(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
