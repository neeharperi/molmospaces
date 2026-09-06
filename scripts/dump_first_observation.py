#!/usr/bin/env python
"""Observation-sanity check: dump the first observation's camera images for one benchmark.

    python scripts/dump_first_observation.py --task Pick-v2-RandCam

Run once per benchmark (9 total, not per policy) to catch EGL device misconfiguration and
filament-vs-classic renderer mismatches before blaming a policy for a garbage-input run --
per plans/BENCHMARK.md's Validation section. Writes runs/_debug/<task>/<camera>.png.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import TASKS, check_env_matches

from molmo_spaces.evaluation import run_evaluation
from molmo_spaces.evaluation.configs.evaluation_configs import DummyBenchmarkEvalConfig
from molmo_spaces.policy.dummy_policy import DummyPolicy


class CapturingDummyPolicy(DummyPolicy):
    """A no-op policy that writes each camera image to disk on its first call only."""

    def __init__(self, config, output_dir: Path) -> None:
        super().__init__(config)
        self.output_dir = output_dir
        self._captured = False

    def get_action(self, observation):
        if not self._captured:
            self._captured = True
            self.output_dir.mkdir(parents=True, exist_ok=True)
            obs = observation[0] if isinstance(observation, (list, tuple)) else observation
            saved = []
            for key, value in obs.items():
                if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
                    Image.fromarray(value).save(self.output_dir / f"{key}.png")
                    saved.append(key)
            print(f"  saved cameras: {saved} -> {self.output_dir}")
        return super().get_action(observation)


def dump_for_task(task_name: str, runs_dir: Path) -> None:
    task = TASKS[task_name]
    assets_dir = os.environ.get("MLSPACES_ASSETS_DIR")
    if not assets_dir:
        raise RuntimeError("MLSPACES_ASSETS_DIR is not set.")
    benchmark_dir = Path(assets_dir) / "benchmarks" / task.path
    output_dir = runs_dir / "_debug" / task_name

    print(f"[{task_name}] renderer={task.renderer} benchmark_dir={benchmark_dir}")
    exp_config = DummyBenchmarkEvalConfig()
    policy = CapturingDummyPolicy(exp_config, output_dir)

    camera_names_override = None
    for i, flag in enumerate(task.extra_flags):
        if flag == "--camera_names":
            camera_names_override = list(task.extra_flags[i + 1 :])
            break

    run_evaluation(
        eval_config_cls=DummyBenchmarkEvalConfig,
        benchmark_dir=benchmark_dir,
        task_horizon_steps=1,
        num_workers=1,
        preloaded_policy=policy,
        episode_idx=0,
        camera_names_override=camera_names_override,
        output_dir=str(output_dir / "eval_output"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=sorted(TASKS), help="Task to dump. Omit with --all-tasks.")
    parser.add_argument("--all-tasks", action="store_true", help="Dump all 9 tasks (must be run once per renderer env).")
    parser.add_argument("--runs-dir", default="runs", help="Root directory for run outputs (default: runs/).")
    args = parser.parse_args()

    task_names = sorted(TASKS) if args.all_tasks else ([args.task] if args.task else None)
    if not task_names:
        parser.error("Pass --task <name> or --all-tasks.")

    runs_dir = Path(args.runs_dir)
    for task_name in task_names:
        if err := check_env_matches(TASKS[task_name]):
            print(f"[{task_name}] SKIPPED: {err}")
            continue
        try:
            dump_for_task(task_name, runs_dir)
        except Exception as e:
            print(f"[{task_name}] FAILED: {e}")


if __name__ == "__main__":
    main()
