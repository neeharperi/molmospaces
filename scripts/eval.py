#!/usr/bin/env python
"""Matrix runner: dispatches molmo_spaces/evaluation/eval_main.py over (policy, task) cells.

    python scripts/eval.py --policy pi05_droid --task Pick-v1.5
    python scripts/eval.py --policy pi05_droid --all-tasks
    python scripts/eval.py --all-policies --all-tasks
    python scripts/eval.py --policy pi05_droid --task Pick-v1.5 --dry-run

Each cell writes runs/<policy>/<task>/<date>/{eval_stdout.log,results.csv,provenance.json}
and is resumable: a cell with both provenance.json and results.csv already present is
skipped unless --force is passed. See plans/BENCHMARK.md and docs/eval_reproduction.md for
the design this implements.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import POLICIES, TASKS, PolicySpec, TaskSpec, check_env_matches, resolve_benchmark_dir

REPO_ROOT = Path(__file__).resolve().parent.parent

# eval_main.py's main() logs these exact lines (see molmo_spaces/evaluation/eval_main.py);
# we parse them back out of the captured subprocess output rather than re-deriving them,
# since eval_main.py writes no manifest of its own.
_OUTPUT_DIR_RE = re.compile(r"Output directory:\s*(\S+)")
# pipeline.py's own "Success count: N, Total count: M" line gives the actual number of
# episodes evaluated in *this* run (post --max_episodes truncation) -- the same number
# eval_to_csv.py's OVERALL row will report as "total", so provenance's n_episodes always
# matches the CSV exactly. (main()'s own "Total episodes: N" line reports the *benchmark's*
# full size instead, which is the wrong semantic value here and is also currently emitted
# before logging.basicConfig ever runs, so it was silently dropped -- fixed separately in
# eval_main.py, but this regex doesn't depend on that fix anyway.)
_TOTAL_EPISODES_RE = re.compile(r"Success count:\s*\d+,\s*Total count:\s*(\d+)")


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def sha256_of(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_server_reachable(policy: PolicySpec) -> str | None:
    try:
        with socket.create_connection((policy.host, policy.port), timeout=3):
            pass
    except OSError as e:
        return (
            f"Cannot reach policy server at {policy.host}:{policy.port} ({e}). "
            f"Start it first -- see docs/eval_reproduction.md for the launch command."
        )
    return None


def run_cell(
    policy_name: str,
    task_name: str,
    *,
    runs_dir: Path,
    date: str,
    dry_run: bool,
    force: bool,
    task_horizon_steps: int | None,
    max_episodes: int | None,
    num_workers: int | None,
    checkpoint_path: str | None = None,
    benchmark_dir_override: Path | None = None,
) -> bool:
    task = TASKS[task_name]
    policy = POLICIES[policy_name]
    # PolicySpec.checkpoint_path is the *static* default for the released checkpoint. When a
    # server is deliberately loaded with something else -- a fine-tune ladder rung, a base
    # checkpoint probe -- provenance must record what was actually served, not the default.
    # Without this the ladder's gate cells all claim to have evaluated pi05_droid_jointpos.
    served_checkpoint = checkpoint_path if checkpoint_path is not None else policy.checkpoint_path

    cell_dir = runs_dir / policy_name / task_name / date
    provenance_path = cell_dir / "provenance.json"
    results_csv = cell_dir / "results.csv"

    # A subsampled draw (scripts/benchmarks/subsample_benchmark.py) is the SAME task run
    # over fewer episodes, not a new one, so it keeps the task's name and its leaderboard
    # row and is separated by --date instead. Naming it `Close-v1-smoke50` would have put
    # it outside TASKS and therefore outside compare_to_leaderboard.py and
    # category_mix_check.py, which are exactly the tools a small cell needs most.
    benchmark_dir = benchmark_dir_override or resolve_benchmark_dir(task)

    # --date is what separates two draws of one task, and nothing used to enforce it. A cell
    # is identified by (policy, task, date) but its number is only meaningful together with
    # the episode list it came from, so a second --benchmark-dir under the same --date is
    # either a skip that reports the wrong n or a --force that destroys a cell that took
    # tens of minutes. benchmark_sha256 is already recorded for exactly this comparison;
    # read it back before doing either.
    if provenance_path.exists() and results_csv.exists():
        prior = json.loads(provenance_path.read_text())
        recorded = prior.get("benchmark_sha256")
        current = sha256_of(benchmark_dir / "benchmark.json")
        if recorded and current and recorded != current:
            print(
                f"[FAIL] {policy_name}/{task_name}: {cell_dir} already holds a cell run "
                f"against a different benchmark.\n"
                f"        recorded {recorded[:16]}  ({prior.get('benchmark_dir')})\n"
                f"        current  {current[:16]}  ({benchmark_dir})\n"
                f"        Give this draw its own --date; --force would overwrite the other one."
            )
            return False
        if not force:
            print(f"[skip] {policy_name}/{task_name}: already complete at {cell_dir}")
            return True

    if err := check_env_matches(task):
        print(f"[FAIL] {policy_name}/{task_name}: {err}")
        return False
    if err := check_server_reachable(policy):
        print(f"[FAIL] {policy_name}/{task_name}: {err}")
        return False

    eval_output_dir = cell_dir / "eval_output"

    eval_cmd = [
        sys.executable,
        "molmo_spaces/evaluation/eval_main.py",
        policy.exp_config_cls,
        "--benchmark_dir",
        str(benchmark_dir),
        "--output_dir",
        str(eval_output_dir),
        "--no_wandb",
        *task.extra_flags,
    ]
    if served_checkpoint:
        # Some policies (e.g. tiptop) have no client-side checkpoint_path field at all --
        # the server owns its own model/planner state -- so PiPolicyConfig-style configs
        # without this field would reject an undeclared-attribute assignment in
        # eval_main.py's create_eval_config(). Only pass it when the policy actually has one.
        eval_cmd += ["--checkpoint_path", served_checkpoint]
    if task_horizon_steps is not None:
        eval_cmd += ["--task_horizon_steps", str(task_horizon_steps)]
    if max_episodes is not None:
        eval_cmd += ["--max_episodes", str(max_episodes)]
    if num_workers is not None:
        eval_cmd += ["--num_workers", str(num_workers)]

    if dry_run:
        print(f"[dry-run] {policy_name}/{task_name}:\n    {' '.join(eval_cmd)}")
        return True

    cell_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell_dir / "eval_stdout.log"
    print(f"[run] {policy_name}/{task_name} -> {log_path}")
    with open(log_path, "w") as log_file:
        proc = subprocess.run(
            eval_cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True
        )
    log_text = log_path.read_text()

    if proc.returncode != 0:
        print(f"[FAIL] {policy_name}/{task_name}: eval_main.py exited {proc.returncode}, see {log_path}")
        return False

    output_dir_match = _OUTPUT_DIR_RE.search(log_text)
    if not output_dir_match:
        print(f"[FAIL] {policy_name}/{task_name}: couldn't find 'Output directory:' in {log_path}")
        return False
    raw_output_dir = Path(output_dir_match.group(1))

    episodes_match = _TOTAL_EPISODES_RE.search(log_text)
    n_episodes = int(episodes_match.group(1)) if episodes_match else None

    csv_cmd = [
        sys.executable,
        "scripts/benchmarks/eval_to_csv.py",
        str(raw_output_dir),
        policy_name,
        "--success-condition",
        "both",
        "--output-csv",
        str(results_csv),
    ]
    print(f"[run] {policy_name}/{task_name}: {' '.join(csv_cmd)}")
    csv_proc = subprocess.run(csv_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if csv_proc.returncode != 0:
        print(f"[FAIL] {policy_name}/{task_name}: eval_to_csv.py failed:\n{csv_proc.stderr}")
        return False

    pinned_assets_file = os.environ.get("MLSPACES_PINNED_ASSETS_FILE")
    provenance = {
        "molmospaces_git_sha": git_sha(),
        "pinned_assets_file": pinned_assets_file,
        "pinned_assets_sha256": sha256_of(Path(pinned_assets_file)) if pinned_assets_file else None,
        "policy": policy_name,
        # None (not "") for server-managed policies like tiptop with no client-side
        # checkpoint path -- distinguishes "genuinely has none" from "forgot to set it".
        "policy_checkpoint": served_checkpoint or None,
        # True when --checkpoint_path overrode PolicySpec's default, so a reader can tell a
        # ladder/probe cell from a released-checkpoint cell without diffing paths by eye.
        "policy_checkpoint_overridden": checkpoint_path is not None,
        "task": task_name,
        "benchmark_dir": str(benchmark_dir),
        # Which draw, not just which directory: a subsample is regenerable but not
        # unique, and a cell whose number cannot be traced to an episode list is an
        # anecdote. sha256 of the benchmark.json actually evaluated.
        "benchmark_sha256": sha256_of(benchmark_dir / "benchmark.json"),
        "benchmark_dir_overridden": benchmark_dir_override is not None,
        "eval_command": " ".join(eval_cmd),
        "eval_to_csv_command": " ".join(csv_cmd),
        "success_condition": "both",
        "task_horizon_steps": task_horizon_steps,
        "max_episodes": max_episodes,
        "seed": 42,
        "n_episodes": n_episodes,
        "results_csv": str(results_csv),
        "raw_output_dir": str(raw_output_dir),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2))
    print(f"[done] {policy_name}/{task_name}: {results_csv}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=sorted(POLICIES), help="Policy to evaluate.")
    parser.add_argument("--task", choices=sorted(TASKS), help="Task to evaluate.")
    parser.add_argument("--all-policies", action="store_true", help="Run every registered policy.")
    parser.add_argument("--all-tasks", action="store_true", help="Run every task in the 9-task table.")
    parser.add_argument("--dry-run", action="store_true", help="Print the eval_main.py command without running it.")
    parser.add_argument("--force", action="store_true", help="Re-run a cell even if already complete.")
    parser.add_argument("--runs-dir", default="runs", help="Root directory for run outputs (default: runs/).")
    parser.add_argument("--date", default=datetime.date.today().strftime("%Y%m%d"), help="Date tag for the run directory (default: today).")
    parser.add_argument("--task_horizon_steps", type=int, default=None, help="Passed through to eval_main.py.")
    parser.add_argument("--max_episodes", type=int, default=None, help="Passed through to eval_main.py, useful for smoke tests.")
    parser.add_argument("--num_workers", type=int, default=None, help="Passed through to eval_main.py.")
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Override the policy's default checkpoint path. Use when the server was started "
        "with a non-default checkpoint (e.g. a fine-tune ladder rung) so provenance records "
        "what was actually served rather than PolicySpec's static default.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Evaluate this benchmark directory instead of the task's own. For a "
        "subsampled draw from scripts/benchmarks/subsample_benchmark.py: the cell keeps "
        "the task name (so the leaderboard tooling still applies) and is separated by "
        "--date. Recorded in provenance with the benchmark.json hash.",
    )
    args = parser.parse_args()

    policy_names = sorted(POLICIES) if args.all_policies else ([args.policy] if args.policy else None)
    task_names = sorted(TASKS) if args.all_tasks else ([args.task] if args.task else None)
    if not policy_names:
        parser.error("Pass --policy <name> or --all-policies.")
    if not task_names:
        parser.error("Pass --task <name> or --all-tasks.")

    runs_dir = Path(args.runs_dir)
    results = []
    for policy_name in policy_names:
        for task_name in task_names:
            ok = run_cell(
                policy_name,
                task_name,
                runs_dir=runs_dir,
                date=args.date,
                dry_run=args.dry_run,
                force=args.force,
                task_horizon_steps=args.task_horizon_steps,
                max_episodes=args.max_episodes,
                num_workers=args.num_workers,
                checkpoint_path=args.checkpoint_path,
                benchmark_dir_override=args.benchmark_dir,
            )
            results.append((policy_name, task_name, ok))

    failures = [(p, t) for p, t, ok in results if not ok]
    print(f"\n{len(results) - len(failures)}/{len(results)} cells succeeded.")
    if failures:
        print("Failed cells:")
        for p, t in failures:
            print(f"  {p}/{t}")
        sys.exit(1)


if __name__ == "__main__":
    main()
