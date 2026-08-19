#!/usr/bin/env python
"""Provenance smoke test: `python scripts/check_provenance.py runs/`.

Walks every runs/<policy>/<task>/<date>/ cell and asserts it's actually reproducible: a
provenance.json exists with every required field populated, the pinned-assets file it
references still hashes the same, the CSV/output paths it points at still exist, and the
CSV's episode count is consistent with what was requested. A run without all of this is not
reproducible and should not count toward the acceptance criteria in plans/BENCHMARK.md.

Exits non-zero and prints the offending cell paths on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED_FIELDS = [
    "molmospaces_git_sha",
    "pinned_assets_file",
    "pinned_assets_sha256",
    "policy",
    "policy_checkpoint",
    "task",
    "benchmark_dir",
    "eval_command",
    "eval_to_csv_command",
    "success_condition",
    "task_horizon_steps",
    "max_episodes",
    "seed",
    "n_episodes",
    "results_csv",
    "raw_output_dir",
    "timestamp",
]
# May legitimately be None: task_horizon_steps (benchmark's own task_horizon_sec can be used
# instead), max_episodes (None means "all episodes", the common case), and policy_checkpoint
# (server-managed policies like tiptop have no client-side checkpoint path at all).
OPTIONAL_NULLABLE_FIELDS = {"task_horizon_steps", "max_episodes", "policy_checkpoint"}

# Runs retired for a known reason are renamed rather than deleted, so the numbers stay
# auditable and nobody re-derives a conclusion from them:
#   <policy>_INVALID_<reason>_<date>  -- produced by a since-fixed integration bug
#   <policy>_STALE_<reason>_<date>    -- produced before a since-fixed harness bug
#   _superseded_max_episodes/         -- valid code, but truncated/oversampled coverage that
#                                        is not comparable to the leaderboard
# Their provenance deliberately still points at the original pre-rename paths, so validating
# them would always fail -- skip them by convention, and report the count so a retired run can
# never be silently forgotten.
QUARANTINE_MARKERS = ("_INVALID", "_STALE", "_superseded")


def check_cell(provenance_path: Path) -> list[str]:
    errors = []
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"{provenance_path}: unreadable ({e})"]

    for field_name in REQUIRED_FIELDS:
        if field_name in OPTIONAL_NULLABLE_FIELDS:
            continue
        if provenance.get(field_name) in (None, ""):
            errors.append(f"{provenance_path}: missing required field '{field_name}'")
    if errors:
        return errors  # can't meaningfully continue without the fields above

    pinned_assets_file = Path(provenance["pinned_assets_file"])
    if not pinned_assets_file.exists():
        errors.append(f"{provenance_path}: pinned_assets_file {pinned_assets_file} no longer exists")
    else:
        live_hash = hashlib.sha256(pinned_assets_file.read_bytes()).hexdigest()
        if live_hash != provenance["pinned_assets_sha256"]:
            errors.append(
                f"{provenance_path}: pinned_assets_file has changed since this run "
                f"(recorded {provenance['pinned_assets_sha256']}, now {live_hash})"
            )

    results_csv = Path(provenance["results_csv"])
    raw_output_dir = Path(provenance["raw_output_dir"])
    if not results_csv.exists():
        errors.append(f"{provenance_path}: results_csv {results_csv} does not exist")
    if not raw_output_dir.exists():
        errors.append(f"{provenance_path}: raw_output_dir {raw_output_dir} does not exist")
    if errors:
        return errors

    df = pd.read_csv(results_csv, comment="#")
    overall = df[df["category"] == "OVERALL"]
    if overall.empty:
        errors.append(f"{provenance_path}: {results_csv} has no OVERALL row")
        return errors
    csv_total = int(overall.iloc[0]["total"])

    n_episodes = provenance["n_episodes"]
    max_episodes = provenance.get("max_episodes")
    if max_episodes is not None:
        if csv_total > n_episodes:
            errors.append(
                f"{provenance_path}: CSV total ({csv_total}) exceeds the benchmark's episode "
                f"count ({n_episodes}) even though --max_episodes={max_episodes} was set"
            )
    elif csv_total != n_episodes:
        errors.append(
            f"{provenance_path}: CSV total ({csv_total}) != benchmark episode count "
            f"({n_episodes}) with no --max_episodes documented -- undocumented short count"
        )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", type=Path, help="Root runs/ directory to walk.")
    args = parser.parse_args()

    all_provenance = sorted(args.runs_dir.glob("*/*/*/provenance.json"))
    provenance_files = [
        p for p in all_provenance
        if not any(m in p.parts[-4] for m in QUARANTINE_MARKERS)
    ]
    quarantined = len(all_provenance) - len(provenance_files)
    if not provenance_files:
        print(f"No provenance.json files found under {args.runs_dir}")
        sys.exit(1)

    all_errors = []
    for p in provenance_files:
        all_errors.extend(check_cell(p))

    print(f"Checked {len(provenance_files)} cells.")
    if quarantined:
        quarantine_dirs = sorted(
            {p.parts[-4] for p in all_provenance if any(m in p.parts[-4] for m in QUARANTINE_MARKERS)}
        )
        print(
            f"Skipped {quarantined} quarantined cell(s) in {len(quarantine_dirs)} retired "
            f"run dir(s) (kept for audit, not part of any result): {', '.join(quarantine_dirs)}"
        )
    if all_errors:
        print(f"{len(all_errors)} provenance failures:")
        for e in all_errors:
            print(f"  {e}")
        sys.exit(1)
    print("All cells have valid provenance.")


if __name__ == "__main__":
    main()
