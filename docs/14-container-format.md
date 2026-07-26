# 14 — The `.tootega` container format

A normative specification. The writer is `convert_to_int4.py`; the C reader has
to match it **byte for byte**. As long as this is only a convention between two
programs in different languages, it is where alignment bugs are born.

Everything little-endian. It settles **D3** of [13](13-open-decisions.md).

## General layout

```
offset 0        fixed header, 128 bytes
offset 128..    data blobs, each aligned to 64 bytes
                (write order = source shard order)
index_off       JSON index, aligned to 64
EOF
```

## Header (128 bytes)

| offset | type | field | value |
|---|---|---|---|
| 0 | `u8[8]` | magic | `54 4F 4F 54 45 47 41 01` (`"TOOTEGA\x01"`) |
| 8 | `u32` | `format_version` | `1` |
| 12 | `u32` | `flags` | `0` (reserved) |
| 16 | `u64` | `n_tensors` | number of index entries |
| 24 | `u64` | `index_off` | absolute offset of the index |
| 32 | `u64` | `index_len` | index size in bytes |
| 40..127 | — | reserved | zeros |

A different magic → refuse. An unknown `format_version` → refuse, **never** read
"in hope".

## Index

UTF-8 JSON in `[index_off, index_off + index_len)`:

```json
{
  "meta": {
    "model": "GLM-4.7-Flash", "source": "...", "group": 64,
    "num_hidden_layers": 47, "mtp_layer": 47,
    "config": { ...the whole config.json... }
  },
  "tensors": {
    "model.layers.1.mlp.experts.0.gate_proj.weight": {
      "kind": "int4",
      "shape": [1536, 2048],
      "data":   [1234560, 1572864],
      "scales": [2807424, 196608],
      "group": 64
    }
  }
}
```

Embedding the whole `config` is deliberate: the engine reads the architecture
from the container, not from a `config.json` sitting next to it that may have
drifted.

| field | meaning |
|---|---|
| `kind` | `int4` \| `int8` \| `f32` \| `bf16` |
| `shape` | logical dimensions, **before** packing |
| `data` | `[offset, bytes]`, offset a multiple of 64 |
| `scales` | `[offset, bytes]` or `null` for `f32`/`bf16` |
| `group` | weights per scale, or `null` |

## Data layout

Shape convention: `[O, I]` — `O` output rows, `I` input columns. Grouping is
always along **`I`** (the matmul's reduction dimension). `I % group == 0` is an
invariant; the converter refuses before writing if it fails.

### `f32` and `bf16`

Row-major, contiguous, no scales. `f32` = `float32[O][I]`;
`bf16` = `uint16[O][I]` with the high 16 bits of a float32.

### `int8`

```
data   : int8[O][I]            symmetric scales, no zero point
scales : float32[O][I/group]
weight[o][i] = data[o][i] * scales[o][i / group]
```

### `int4` — half/half packing

```
data   : uint8[O][I/2]
scales : float32[O][I/group]
```

Inside each group of `group` weights, byte `j` (with `0 <= j < group/2`) carries:

```
low nibble  (bits 0-3) -> weight j
high nibble (bits 4-7) -> weight j + group/2
```

The codes are **offset**: stored value `0..15`, real code `stored - 8`, range
`-8..7`.

Unpacking group `g` of row `o`:

```c
const uint8_t *p = data + o*(I/2) + g*(GROUP/2);
const float    s = scales[o*(I/GROUP) + g];
for (int j = 0; j < GROUP/2; j++) {
    w[j]           = (int)(p[j] & 0x0F) - 8;   /* weights 0..GROUP/2-1 */
    w[j + GROUP/2] = (int)(p[j] >> 4)   - 8;   /* weights GROUP/2..GROUP-1 */
}
/* real weight = w[k] * s */
```

**Why half/half and not even/odd.** Loading 32 bytes with AVX2, `v & 0x0F`
delivers `group/2` **contiguous** weights and `v >> 4` the next `group/2` — two
vectors ready for the product against the corresponding activations. With
even/odd, `v & 0x0F` would deliver `0,2,4,…`, forcing a permutation of the
activations. It is llama.cpp's `Q4_0` convention, and the choice is recorded as
D1 in [13](13-open-decisions.md).

## Scales

Symmetric, no zero point:

```
qmax  = 7   (int4)  |  127 (int8)
scale = absmax(group) / qmax        , and 1.0 if absmax == 0
code  = clamp(round(weight / scale), -(qmax+1), qmax)
```

The `absmax == 0` case uses a scale of `1.0` to avoid dividing by zero; all codes
end up `0` and reconstruction is exact.

Storage cost: `4/group` bytes per weight. With `group=64`, int4 costs
**0.5625 B/weight** — the number the ceiling arithmetic in
[08](08-performance-ceiling.md) assumes.

**Valid groups for this model: 32, 64, 128 or 256.** The upper limit is 256
because `q_b_proj` has an input dimension of 768.

## Alignment

Every blob starts at a multiple of **64** — the cache line and the width of an
AVX-512 register. It is what allows loading straight from the mapped file with no
copying path ([11](11-memory.md)). The converter's `--verify` checks this on
every blob and fails if any escapes.

Padding between blobs is zero.

## Atomic write

Writes to `<name>.part`, `fsync`, `rename`. A crash midway must not leave a
truncated container that looks valid.

## Verification

```powershell
python convert_to_int4.py --verify --out <path>
```

Checks the magic, the version and the alignment of every blob. It does not re-read
the data — content integrity is the job of the checksum planned in
[12](12-persistent-state.md), not yet implemented.

## Executable reference

`quant_grouped` and `dequant_grouped` in `convert_to_int4.py` are the executable
definition of the layout. The C kernel has to reproduce `dequant_grouped`
exactly; any divergence shows up as a failure in M1's token-exact oracle, not as
a read error.
