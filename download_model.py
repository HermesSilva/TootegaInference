#!/usr/bin/env python3
"""Download GLM-4.7-Flash, the target model for TootegaEngine.

Why this model, measured on this machine (i9-14900K, 64 GB DDR5-4800 dual
channel, RTX 2060 SUPER 8 GB, RAID0 NVMe on F:):

  * 31.2B total / ~1.1 GB active per token -> 19.3 GB once quantized to int4
    (measured after conversion; the 15.6 GB figure that circulated earlier left
    out group scales, the int8 MTP layer and the bf16 embeddings), so the whole
    model is RAM-resident and the disk leaves the decode path entirely. For
    comparison, GLM-5.2 needs 363 GB of experts and measured 0.93 tok/s here
    because ~85% of every token came off the disk.
  * Agentic scores that hold up against much larger models: SWE-bench Verified
    59.2 and tau2-Bench 79.5, with tool calling in the chat template and a
    202,752-token context -- and 19.3 GB of weights leaves ~34 GB of RAM for
    the KV cache an agent loop actually consumes.
  * MLA attention with kv_lora_rank=512 and n_group=1, the narrow architectural
    window the colibri engine accepts (it exits hard otherwise), so the model
    is testable on the existing engine before TootegaEngine exists.
  * num_nextn_predict_layers=1, so the MTP speculation path is testable.

The BF16 source is 62.4 GB and lands next to this script, under models/.
Source dtype is bfloat16, NOT the FP8 that GLM-5.2 ships, so the int4
conversion step needs a converter that accepts BF16 input -- verify that
before relying on this download.

Usage:
    python download_model.py                     # download to models/
    python download_model.py --dry-run           # show the plan, download nothing
    python download_model.py --verify            # check an existing download
    python download_model.py --wait-space        # poll until the disk has room
    python download_model.py --out F:\\models\\x  # somewhere else
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Both must be set BEFORE huggingface_hub is imported, which is why the import
# is deferred into main(): the symlink warning fires at import time on Windows
# without developer mode, and it is pure noise for a local_dir download.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# A plain Windows console is cp1252; any non-ASCII in a print() raises
# UnicodeEncodeError and kills the run. Force UTF-8 the way the colibri
# converters do (c/tools/convert_olmoe.py).
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

REPO = "zai-org/GLM-4.7-Flash"
DEFAULT_OUT = str(Path(__file__).resolve().parent / "models" / "glm47flash-bf16")
HEADROOM_GB = 5.0            # index files, partial writes, filesystem slack


def human(gb):
    return f"{gb:,.1f} GB"


def local_bytes(out: Path) -> int:
    """Bytes already materialized, ignoring hf_hub's staging dir."""
    if not out.exists():
        return 0
    return sum(f.stat().st_size for f in out.rglob("*")
               if f.is_file() and ".cache" not in f.parts)


def staged_bytes(out: Path) -> int:
    """Bytes sitting in hf_hub's staging dir as partial shards. These are real
    progress -- a resumed run continues from them -- but they are invisible to
    local_bytes(), so a run that fetched 2 GB of half-shards would otherwise
    report 0% and look like nothing was saved."""
    staging = out / ".cache" / "huggingface" / "download"
    if not staging.exists():
        return 0
    return sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())


def free_space_gb(target: Path) -> float:
    """Free space on the volume that will actually hold the files."""
    probe = target
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def check_space(target: Path, need_gb: float, wait: bool) -> float:
    """Refuse to start without room. With --wait-space, poll instead of dying:
    a 62 GB download that aborts an hour in for lack of a few GB is worse than
    waiting for a cleanup that is already under way."""
    deadline = time.time() + 3600
    warned = False
    while True:
        free_gb = free_space_gb(target)
        if free_gb >= need_gb:
            return free_gb
        deficit = need_gb - free_gb
        if not wait:
            sys.exit(f"espaco insuficiente em {target}: {human(free_gb)} livres, "
                     f"{human(need_gb)} necessarios (faltam {human(deficit)}).\n"
                     f"libere espaco, use --out para outro volume, ou --wait-space "
                     f"para aguardar aqui.")
        if not warned:
            print(f"aguardando espaco: {human(free_gb)} livres, faltam {human(deficit)}")
            print("  checando a cada 30s por ate 1h (Ctrl+C para sair)")
            warned = True
        if time.time() > deadline:
            sys.exit(f"espaco nao apareceu em 1h (faltam {human(deficit)})")
        time.sleep(30)


def report_progress(out: Path, have_gb: float, expected_gb: float):
    """Say what a resume would actually skip: completed files and partial ones."""
    staged_gb = staged_bytes(out) / 1e9
    if have_gb <= 0 and staged_gb <= 0:
        return
    pct = 100 * (have_gb + staged_gb) / expected_gb if expected_gb else 0
    detail = f"{human(have_gb)} completos"
    if staged_gb > 0:
        detail += f" + {human(staged_gb)} parciais"
    print(f"  retoma: {detail} ({pct:.0f}%)")


def remote_manifest(repo: str):
    """{filename: size} straight from the Hub. This is the only thing that can
    tell a finished download from a short one."""
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    return {f.rfilename: (f.size or 0) for f in info.siblings}


def verify(out: Path, manifest: dict) -> bool:
    """Compare every file against the remote size. snapshot_download returning
    without raising is NOT proof of completeness -- a truncated or missing
    shard looks exactly like success until the engine fails to load it."""
    missing, short = [], []
    for name, size in sorted(manifest.items()):
        path = out / name
        if not path.exists():
            missing.append(name)
        elif size and path.stat().st_size != size:
            short.append((name, path.stat().st_size, size))

    total = sum(manifest.values())
    print(f"\n  esperado: {len(manifest)} arquivos, {human(total / 1e9)}")
    print(f"  em disco: {len(manifest) - len(missing)} arquivos, "
          f"{human(local_bytes(out) / 1e9)}")

    for name in missing[:10]:
        print(f"  AUSENTE   {name}")
    if len(missing) > 10:
        print(f"  ... e mais {len(missing) - 10} ausentes")
    for name, got, want in short[:10]:
        print(f"  TRUNCADO  {name}: {got:,} de {want:,} bytes")

    for required in ("config.json", "tokenizer.json", "generation_config.json"):
        mark = "ok" if (out / required).exists() else "AUSENTE"
        print(f"  {required:<24} {mark}")

    ok = not missing and not short
    print(f"\n  {'COMPLETO' if ok else 'INCOMPLETO - reexecute para retomar'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel downloads (default 8)")
    parser.add_argument("--retries", type=int, default=50,
                        help="restarts on network failure; each resumes (default 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the plan, download nothing")
    parser.add_argument("--verify", action="store_true",
                        help="check an existing download against the Hub and exit")
    parser.add_argument("--wait-space", action="store_true",
                        help="poll until the disk has room instead of exiting")
    args = parser.parse_args()

    out = Path(args.out)
    print(f"repo    {args.repo}")
    print(f"destino {out}")

    try:
        manifest = remote_manifest(args.repo)
    except Exception as error:
        sys.exit(f"nao consegui ler o repo no Hub: {type(error).__name__}: {error}")

    expected_gb = sum(manifest.values()) / 1e9
    have_gb = local_bytes(out) / 1e9
    # Only the remainder has to fit: re-running after a partial download must
    # not demand room for bytes that are already on disk.
    need = max(0.0, expected_gb - have_gb) + HEADROOM_GB

    if args.verify:
        return 0 if verify(out, manifest) else 1

    if args.dry_run:
        # A dry-run must always report, never exit early: its whole job is to
        # tell you what is missing, including the space.
        free = free_space_gb(out)
        short = "" if free >= need else f"  <-- faltam {human(need - free)}"
        print(f"espaco  {human(free)} livres, precisa de ~{human(need)}{short}")
        print(f"\n  {len(manifest)} arquivos, {human(expected_gb)} no repo")
        report_progress(out, have_gb, expected_gb)
        print("  (dry-run: nada baixado)")
        return 0

    free = check_space(out, need, args.wait_space)
    print(f"espaco  {human(free)} livres, precisa de ~{human(need)}")
    report_progress(out, have_gb, expected_gb)

    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    try:
        for attempt in range(1, args.retries + 1):
            try:
                snapshot_download(repo_id=args.repo, local_dir=str(out),
                                  max_workers=args.workers)
                break
            except Exception as error:
                # 62 GB over a long connection: transient failures are expected
                # and every retry resumes from the partial files on disk.
                wait = min(30 * attempt, 300)
                print(f"[tentativa {attempt}/{args.retries}] "
                      f"{type(error).__name__}: {error}")
                print(f"  repetindo em {wait}s (continua de onde parou)")
                time.sleep(wait)
        else:
            print(f"falhou apos {args.retries} tentativas")
            return 1
    except KeyboardInterrupt:
        # Wraps the sleep too, so Ctrl+C during a backoff exits cleanly instead
        # of dumping a traceback.
        print("\ninterrompido - reexecute para continuar de onde parou")
        return 130

    elapsed = time.time() - started
    # Rate is bytes fetched THIS run, not everything on disk: a resumed run
    # would otherwise report a wildly inflated speed.
    fetched_gb = max(0.0, local_bytes(out) / 1e9 - have_gb)
    rate = fetched_gb * 1000 / elapsed if elapsed > 0 else 0
    print(f"\n  baixados {human(fetched_gb)} em {elapsed / 60:.1f} min "
          f"({rate:.0f} MB/s)")

    if not verify(out, manifest):
        return 1

    print("\nproximo passo: quantizar para int4. A fonte e bfloat16, entao o "
          "conversor precisa aceitar BF16 (o do colibri e FP8->int4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
