#!/usr/bin/env python3
"""Map every tensor in the checkpoint without loading a byte of weight data.

The LoRA target_modules have to be the model's real names.  Guessing them from
an architecture diagram is the failure that costs a whole run: PEFT matches
nothing, attaches zero adapters, trains happily and produces a no-op.

A safetensors file starts with an 8-byte little-endian header length followed by
a JSON header carrying every tensor's name, dtype and shape.  Reading that is a
few kilobytes per shard -- no GPU, no weights, no allocation -- and it is the
same source of truth the loader uses.
"""
import json
import os
import re
import struct
import sys
from collections import OrderedDict

ROOT = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
BASE = os.path.join(ROOT, "base", "glm47flash-bf16")
OUT = os.path.join(ROOT, "data", "tensor_map.json")

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1,
               "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    idx_path = os.path.join(BASE, "model.safetensors.index.json")
    shards = sorted({v for v in json.load(open(idx_path))["weight_map"].values()})
    print(f"{len(shards)} shards")

    tensors = {}
    for s in shards:
        for name, meta in read_header(os.path.join(BASE, s)).items():
            if name == "__metadata__":
                continue
            tensors[name] = (meta["dtype"], tuple(meta["shape"]))

    def numel(shape):
        n = 1
        for d in shape:
            n *= d
        return n

    total = sum(numel(sh) for _, sh in tensors.values())
    nbytes = sum(numel(sh) * DTYPE_BYTES.get(dt, 2) for dt, sh in tensors.values())
    print(f"{len(tensors)} tensors, {total/1e9:.2f} B params, {nbytes/1e9:.1f} GB\n")

    # Collapse layer indices and expert indices so the pattern shows up.
    groups = OrderedDict()
    for name, (dt, sh) in tensors.items():
        pat = re.sub(r"\.\d+\.", ".N.", name)
        g = groups.setdefault(pat, {"n": 0, "params": 0, "dtype": dt,
                                    "shape": sh, "example": name})
        g["n"] += 1
        g["params"] += numel(sh)

    print(f"{'pattern':62s} {'count':>6s} {'shape':>18s} {'params':>10s}")
    print("-" * 100)
    for pat, g in sorted(groups.items(), key=lambda kv: -kv[1]["params"]):
        if not pat.endswith(".weight") and not pat.endswith(".bias"):
            continue
        print(f"{pat:62s} {g['n']:6d} {str(g['shape']):>18s} "
              f"{g['params']/1e6:9.1f}M")

    # What the plan proposes to train, priced from the real shapes.
    print("\n--- LoRA sizing from the real shapes ---")
    R = 32

    def lora_params(pat_sub, r=R):
        tot = 0
        for name, (dt, sh) in tensors.items():
            if pat_sub in name and name.endswith(".weight") and len(sh) == 2:
                out_f, in_f = sh
                tot += (in_f + out_f) * r
        return tot

    plan = {
        "attention": ["q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj",
                      "self_attn.o_proj"],
        "shared_expert": ["shared_experts"],
        "dense_layer0": ["layers.0.mlp"],
        "routed_experts": ["experts."],
    }
    report = {}
    for label, subs in plan.items():
        p = sum(lora_params(s) for s in subs)
        matched = sum(1 for s in subs for n in tensors
                      if s in n and n.endswith(".weight") and len(tensors[n][1]) == 2)
        report[label] = {"lora_params": p, "matched_matrices": matched}
        adam = p * 12 / 1e9   # fp32 m + v + master
        print(f"  {label:16s} {matched:5d} matrices  LoRA r={R}: {p/1e6:8.1f} M  "
              f"(+{adam:.2f} GB optimiser)")

    proposed = sum(report[k]["lora_params"] for k in
                   ("attention", "shared_expert", "dense_layer0"))
    print(f"\n  PROPOSED (attn + shared + dense0): {proposed/1e6:.1f} M trainable")
    print(f"  weights bf16 62.4 GB + adapter/optimiser "
          f"{proposed*14/1e9:.2f} GB -> activations get "
          f"{95.6 - 62.4 - proposed*14/1e9:.1f} GB")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"n_tensors": len(tensors), "params": total, "bytes": nbytes,
                   "patterns": {k: {"n": v["n"], "params": v["params"],
                                    "shape": list(v["shape"]),
                                    "dtype": v["dtype"], "example": v["example"]}
                                for k, v in groups.items()},
                   "lora_plan": report}, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
