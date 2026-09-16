#!/usr/bin/env python
"""Turn a fine-tune smoke run's log into the baseline a later run is diffed against.

    python scripts/record_finetune_smoke.py --log ~/openpi-data/logs/<config>__<exp>.log \
        --config pi05_droid_jointpos_from_base --dataset data/droid_100 \
        --out reference/finetune_smoke.json

`reference/reproduced_cells.json` answers "do we still get the success rate we got"; this
answers the cheaper question underneath it -- "does the fine-tune path still run at all, and
does the loss still fall" -- which is the one worth asking after a change to a data pipeline
or a config, because it needs minutes rather than a campaign.

What it records and why each field is here rather than inferred later:

* `first_loss` / `last_loss` and the step count, because "loss decreased" is the whole
  assertion. Read from the log rather than from a metrics file: openpi logs the per-step loss
  through `logging.info` (that was a fix in this stack -- `pbar.write` never reached a file),
  and there is no other durable record on a run with wandb offline.
* `peak_rss_gb`, because host memory is what actually bounds this on a 2x48 GB machine. A
  single-card attempt OOMed on the optimizer state and a CPU attempt was OOM-killed outright;
  the number that decides whether a future run fits is this one, not VRAM.
* the wall clock and the settled step rate, taken from the LAST interval rather than the
  first. openpi's first sample after a compile or a restore is an order of magnitude slow --
  56 s/it falling to ~4 -- and reading the wrong one has already misled this campaign twice.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
from pathlib import Path

LOSS_RE = re.compile(r"loss=([0-9.eE+-]+)")
STEP_RE = re.compile(r"step[= ]+(\d+)", re.IGNORECASE)
RATE_RE = re.compile(r"([0-9.]+)\s*s/it")
RSS_RE = re.compile(r"rss=([0-9.]+)GB")


def numbers(pattern, text, cast=float):
    return [cast(m.group(1)) for m in pattern.finditer(text)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True, help="the training log")
    parser.add_argument("--watch-log", type=Path, default=None, help="optional rss=NNGB watcher output")
    parser.add_argument("--config", required=True, help="the openpi TrainConfig name")
    parser.add_argument("--dataset", required=True, help="what it trained on")
    parser.add_argument("--checkpoint", type=Path, default=None, help="where the run wrote checkpoints")
    parser.add_argument("--out", type=Path, default=Path("reference/finetune_smoke.json"))
    args = parser.parse_args()

    if not args.log.exists():
        raise SystemExit(f"{args.log} does not exist")
    text = args.log.read_text(errors="replace")
    losses = numbers(LOSS_RE, text)
    steps = numbers(STEP_RE, text, int)
    rates = numbers(RATE_RE, text)

    if not losses:
        raise SystemExit(
            f"{args.log} records no loss. The run did not reach a training step -- record the\n"
            "failure in docs/eval_reproduction.md instead of writing a baseline that says it ran."
        )

    peak_rss = None
    if args.watch_log and args.watch_log.exists():
        seen = numbers(RSS_RE, args.watch_log.read_text(errors="replace"))
        peak_rss = max(seen) if seen else None

    payload = {
        "recorded": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(),
        "config": args.config,
        "dataset": args.dataset,
        "steps_logged": len(losses),
        "last_step": steps[-1] if steps else None,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        # The last interval, not the first: see the module docstring.
        "settled_s_per_step": rates[-1] if rates else None,
        "peak_rss_gb": peak_rss,
        "checkpoints": sorted(p.name for p in args.checkpoint.iterdir()) if args.checkpoint and args.checkpoint.is_dir() else [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {args.out}")
    for key in ("config", "steps_logged", "first_loss", "last_loss", "loss_decreased", "settled_s_per_step", "peak_rss_gb"):
        print(f"  {key:22s} {payload[key]}")
    if not payload["loss_decreased"]:
        raise SystemExit("FAILED: the loss did not decrease. Recorded anyway, so the run can be compared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
