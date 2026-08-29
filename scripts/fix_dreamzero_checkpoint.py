#!/usr/bin/env python
"""Make the DreamZero-DROID checkpoint usable on this machine.

    python scripts/fix_dreamzero_checkpoint.py [--dry-run]

Two problems, both from the checkpoint being packaged on someone else's filesystem.

1. The eval harness expects the checkpoint at third_party/dreamzero/checkpoints/DreamZero-DROID
   (scripts/eval_common.py's POLICIES table). HF puts it in the shared hub cache. Symlink,
   don't copy -- it is ~43 GB and the weights are not part of an environment.

2. Its bundled config.json hardcodes ABSOLUTE paths to three Wan2.1-I2V-14B-480P base-model
   components -- the CLIP image encoder, the umt5-xxl text encoder and the VAE -- under a
   ~/Workspace/robot-prompt-opt/checkpoints/ tree belonging to the machine it was built on.
   Those paths resolved there and do not exist here, so the server dies at load time. This
   rewrites them to the local HF snapshot of Wan-AI/Wan2.1-I2V-14B-480P.

The rewrite is deliberately conservative: it only touches string values that both point at a
nonexistent absolute path AND have a basename present in the local Wan2.1 snapshot, and it
reports every substitution. A path that is already valid is left alone, so re-running is safe.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HUB = Path.home() / ".cache/huggingface/hub"
DEST = REPO_ROOT / "third_party/dreamzero/checkpoints/DreamZero-DROID"


def snapshot(repo_id: str) -> Path | None:
    d = HUB / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not d.is_dir():
        return None
    snaps = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return snaps[0] if snaps else None


def index_wan(root: Path) -> dict[str, Path]:
    """basename -> real path, for every file in the Wan2.1 snapshot."""
    return {p.name: p for p in root.rglob("*") if p.is_file()}


def rewrite(obj, wan: dict[str, Path], subs: list[tuple[str, str]]):
    if isinstance(obj, dict):
        return {k: rewrite(v, wan, subs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v, wan, subs) for v in obj]
    if isinstance(obj, str) and obj.startswith("/") and not Path(obj).exists():
        cand = wan.get(Path(obj).name)
        if cand is not None:
            subs.append((obj, str(cand)))
            return str(cand)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    dz, wan_root = snapshot("GEAR-Dreams/DreamZero-DROID"), snapshot("Wan-AI/Wan2.1-I2V-14B-480P")
    if dz is None:
        sys.exit("DreamZero-DROID not in the HF cache; run: hf download GEAR-Dreams/DreamZero-DROID")
    if wan_root is None:
        sys.exit("Wan2.1-I2V-14B-480P not in the HF cache; run: hf download Wan-AI/Wan2.1-I2V-14B-480P")
    print(f"  DreamZero snapshot: {dz}\n  Wan2.1 snapshot:    {wan_root}")

    # 1. symlink into the path eval_common.py expects
    if not a.dry_run:
        DEST.parent.mkdir(parents=True, exist_ok=True)
        if DEST.is_symlink() or DEST.exists():
            (DEST.unlink() if DEST.is_symlink() else shutil.rmtree(DEST))
        DEST.symlink_to(dz)
    print(f"  symlink: {DEST} -> {dz}")

    # 2. repoint the three absolute base-model paths
    cfg = dz / "config.json"
    if not cfg.exists():
        sys.exit(f"no config.json in {dz}")
    original = json.loads(cfg.read_text())
    subs: list[tuple[str, str]] = []
    fixed = rewrite(original, index_wan(wan_root), subs)

    if not subs:
        print("  config.json: no dangling absolute paths -- nothing to rewrite")
        return
    print(f"  config.json: {len(subs)} path(s) to repoint")
    for old, new in subs:
        print(f"    - {old}\n    + {new}")
    if a.dry_run:
        print("  (dry run, not written)")
        return
    backup = cfg.with_suffix(".json.orig")
    if not backup.exists():
        # The snapshot dir is HF-cache-managed; keep the untouched original beside it so the
        # edit is reversible without a 43 GB re-download.
        backup.write_text(cfg.read_text())
        print(f"  backed up original -> {backup}")
    cfg.write_text(json.dumps(fixed, indent=2))
    print("  config.json rewritten")


if __name__ == "__main__":
    main()
