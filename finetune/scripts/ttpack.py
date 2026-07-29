#!/usr/bin/env python3
"""Read a .ttpack -- the format tools/finetune.exe writes.

The trainer consumes token ids that the engine's own tokenizer produced and
verified by round-trip, instead of re-tokenising the Markdown with a different
library.  Two tokenisers that disagree on one merge do not crash; they train the
model on a slightly different text than the one the engine will ever see, and
nothing reports it (docs/16).

Layout, little-endian, from tools/finetune/finetune.c:

    0   char[8]  "TTPACK\0\1"
    8   u32      version (1)
    12  u32      flags
    16  u64      n_docs
    24  u64      n_windows
    32  u64      n_ids
    40  u64      ids_offset      -> int32 x n_ids
    48  u64      win_offset      -> 16 bytes x n_windows: doc, off, len, flags
    56  u64      index_offset    -> JSON: meta + docs[]
    64  u64      index_length

    window flag 1 = holdout
"""
import json
import struct

MAGIC = b"TTPACK\0\1"
WIN_HOLDOUT = 1


class Window:
    __slots__ = ("doc", "off", "len", "flags")

    def __init__(self, doc, off, n, flags):
        self.doc, self.off, self.len, self.flags = doc, off, n, flags

    @property
    def holdout(self):
        return bool(self.flags & WIN_HOLDOUT)


class TTPack:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            hdr = f.read(128)
            if len(hdr) != 128 or hdr[:8] != MAGIC:
                raise ValueError(f"{path}: not a .ttpack (bad magic or truncated)")
            ver, _flags = struct.unpack_from("<II", hdr, 8)
            if ver != 1:
                raise ValueError(f"{path}: pack version {ver}, this reader knows 1")
            (n_docs, n_wins, n_ids, ids_off, win_off,
             idx_off, idx_len) = struct.unpack_from("<7Q", hdr, 16)

            f.seek(ids_off)
            raw = f.read(4 * n_ids)
            if len(raw) != 4 * n_ids:
                raise ValueError(f"{path}: ids truncated")
            self.ids = memoryview(raw).cast("i")

            f.seek(win_off)
            wraw = f.read(16 * n_wins)
            self.windows = [Window(*struct.unpack_from("<4I", wraw, 16 * i))
                            for i in range(n_wins)]

            f.seek(idx_off)
            index = json.loads(f.read(idx_len).decode("utf-8"))

        self.meta = index["meta"]
        self.docs = index["docs"]
        if len(self.docs) != n_docs:
            raise ValueError(f"{path}: index lists {len(self.docs)} docs, header says {n_docs}")

        vocab = self.meta.get("vocab", 0)
        bad = [i for i in range(0, len(self.ids), 997)
               if not (0 <= self.ids[i] < vocab)]
        if bad:
            raise ValueError(f"{path}: id out of vocabulary at position {bad[0]}")

    def window_ids(self, w):
        return list(self.ids[w.off:w.off + w.len])

    def split(self, holdout):
        return [w for w in self.windows if w.holdout == holdout]

    def summary(self):
        tr = self.split(False)
        ho = self.split(True)
        return (f"{self.path}\n"
                f"  {len(self.docs)} docs, {len(self.windows)} windows, "
                f"{len(self.ids)} ids, window={self.meta['window']}\n"
                f"  train   {len(tr):5d} windows  {sum(w.len for w in tr):7d} ids\n"
                f"  holdout {len(ho):5d} windows  {sum(w.len for w in ho):7d} ids  "
                f"({self.meta['n_holdout_docs']} docs)\n"
                f"  tokenizer {self.meta['tokenizer']}\n"
                f"  verified  {self.meta['verified']}  build {self.meta['build']}")


if __name__ == "__main__":
    import os
    import sys

    root = os.environ.get("COSMO_ROOT", "/workspace/cosmologia")
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "data", "cosmologia-87.ttpack")
    pack = TTPack(p)
    print(pack.summary())

    print("\nholdout documents:")
    for d in pack.docs:
        if d["holdout"]:
            print(f"  {d['ids']:6d} ids  {d['title']}")

    # Decode one window, as a sanity check that ids and tokenizer agree.
    try:
        from transformers import AutoTokenizer
        base = os.path.join(root, "base", "glm47flash-bf16")
        tok = AutoTokenizer.from_pretrained(base)
        w = pack.split(False)[0]
        text = tok.decode(pack.window_ids(w))
        print(f"\nfirst training window (doc {w.doc}, {w.len} ids), first 300 chars:")
        print("  " + text[:300].replace("\n", "\n  "))
        n_over = sum(1 for i in range(0, len(pack.ids), 1013)
                     if pack.ids[i] >= len(tok))
        print(f"\nids beyond the tokenizer vocabulary (sampled): {n_over}")
    except Exception as e:
        print(f"\n(decode check skipped: {type(e).__name__}: {e})")
