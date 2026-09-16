#!/usr/bin/env python3
"""How much memory the pipeline holds per episode, and whether it is data or the allocator.

    python scripts/probe_episode_retention.py --benchmark <dir> --mode prepare
    python scripts/probe_episode_retention.py --benchmark <dir> --mode accumulate

Builds real episodes, steps them to the horizon with null actions -- the worst case for the
observation cache, since a rollout that never succeeds never stops early -- and retains them
the two ways pipeline.py could. No policy server and no GPU inference, so an episode costs
~10 s rather than ~2 minutes.

  prepare      what pipeline.py does now: prepare_episode_for_saving per episode, which
               writes that episode's videos and pops the camera sensors
  accumulate   what it did before: the raw history, camera frames and all, appended to a
               per-house list and prepared only at the end of the house

`accumulate` is kept as the control. It is the shape of the defect, so a future change that
drifts back toward its numbers is a regression, and this is what says so.

**Two columns, because they answer different questions.** `d rss` is what the host sees, and
it includes pages the allocator has not returned. `data MB` walks the retained list for the
real buffer sizes of its arrays and tensors -- that is what is genuinely still held, and it
is the number a fix has to move. Measured on Open-v1 (58-step horizon, 624x352, two cameras):

  | mode       | retained data | RSS growth  |
  |------------|---------------|-------------|
  | accumulate | 74 MB/episode | ~120 MB/ep  |
  | prepare    |  4 MB/episode |  ~40 MB/ep  |

An 18x reduction in what is held. Both modes show RSS growing faster than retained data;
that gap is allocator behaviour rather than retention, which is why the two columns exist --
chasing the RSS figure alone would have sent someone looking for a leak that is not there.

Why it mattered: bench-v1 puts ~33 episodes in a house and eval runs one work item per house,
so at 74 MB/episode a worker reached ~2.4 GB of held frames, and 20 workers of that is what
stopped full-coverage Open-v1 runs partway through. See docs/eval_reproduction.md.
"""

from __future__ import annotations

import argparse
import gc
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def current_rss_mb() -> float:
    """maxrss never falls, so read the live value for a retention measurement."""
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576.0


def build_episode(benchmark_dir: pathlib.Path, index: int):
    from molmo_spaces.configs.policy_configs import DummyPolicyConfig
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
    from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

    episodes = load_all_episodes(benchmark_dir)
    spec = episodes[index % len(episodes)]
    config = JsonBenchmarkEvalConfig(
        robot_config=FrankaRobotConfig(), policy_config=DummyPolicyConfig()
    )
    sampler = JsonEvalTaskSampler(config, spec)
    sampler.reset()
    task = sampler.sample_task()
    task.reset()
    return sampler, task


def held_mb(obj, seen=None) -> float:
    """Bytes actually retained by what we put on the list.

    Walks tensors and arrays for their real buffer sizes, which is the only part that
    matters at this scale -- python object overhead is noise against a camera frame. Used to
    separate "we are still holding data" from "the allocator has not returned pages", which
    look identical in RSS and want completely different fixes.
    """
    import numpy as np

    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return 0.0
    seen.add(id(obj))
    nbytes = getattr(obj, "nbytes", None)
    if nbytes is not None and not isinstance(obj, (str, bytes)):
        return float(nbytes) / 1048576.0
    if hasattr(obj, "element_size") and hasattr(obj, "numel"):  # torch.Tensor
        return obj.element_size() * obj.numel() / 1048576.0
    if isinstance(obj, dict):
        return sum(held_mb(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(held_mb(v, seen) for v in obj)
    if isinstance(obj, (bytes, bytearray, str)):
        return len(obj) / 1048576.0
    return 0.0


def release(sampler, task) -> None:
    for obj in (task, sampler):
        closer = getattr(obj, "close", None)
        if closer is None:
            continue
        try:
            closer()
        except Exception:
            pass
    gc.collect()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=pathlib.Path, required=True)
    ap.add_argument("--mode", choices=("accumulate", "prepare"), required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=58, help="the Open-v1 horizon")
    ap.add_argument(
        "--save-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/probe_episode_retention"),
        help="where the per-episode videos go; they are a side effect, not the measurement",
    )
    args = ap.parse_args()

    from molmo_spaces.utils.save_utils import prepare_episode_for_saving

    args.save_dir.mkdir(parents=True, exist_ok=True)
    house_raw_histories: list = []

    print(f"mode={args.mode}  episodes={args.episodes}  steps={args.steps}")
    print(
        f"{'episode':>7} {'steps':>6} {'rss MB':>9} {'d rss':>8} {'held':>6} "
        f"{'data MB':>8}  {'s':>5}"
    )
    previous = current_rss_mb()
    deltas: list[float] = []
    for episode in range(args.episodes):
        t0 = time.time()
        sampler, task = build_episode(args.benchmark, episode)
        stepped = 0
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = task.step(dict())
            stepped += 1
            if bool(terminated[0]) or bool(truncated[0]):
                break

        history = task.get_history()
        if args.mode == "accumulate":
            # Exactly pipeline.py: the raw history, camera frames and all, goes on the list.
            house_raw_histories.append({"history": history, "sensor_suite": task.sensor_suite})
        else:
            # The same call pipeline.py makes, at the end of each episode instead of the
            # house. Writes the videos and pops the camera sensors.
            prepared = prepare_episode_for_saving(
                history,
                task.sensor_suite,
                fps=15.0,
                save_dir=str(args.save_dir),
                episode_idx=episode,
            )
            if prepared is not None:
                house_raw_histories.append(prepared)
            del history

        release(sampler, task)
        now = current_rss_mb()
        deltas.append(now - previous)
        retained = held_mb(house_raw_histories)
        print(
            f"{episode:>7} {stepped:>6} {now:>9.0f} {now - previous:>+8.0f} "
            f"{len(house_raw_histories):>6} {retained:>8.0f}  {time.time() - t0:>5.1f}"
        )
        previous = now

    # Episode 0 pays for the scene, the renderer and the imports, so averaging it in
    # overstates retention by several hundred MB. The marginal cost is what accumulates
    # across a house, so that is the number reported.
    marginal = deltas[1:]
    if marginal:
        mean = sum(marginal) / len(marginal)
        print(
            f"\n{args.mode}: marginal retention {mean:+.0f} MB/episode "
            f"(episodes 1-{len(deltas) - 1}, range {min(marginal):+.0f} to {max(marginal):+.0f}); "
            f"episode 0 was {deltas[0]:+.0f} MB including one-time setup"
        )
        print(f"{args.mode}: a 33-episode bench-v1 house projects to {mean * 33 / 1024:.1f} GB per worker")
        retained = held_mb(house_raw_histories)
        print(
            f"{args.mode}: actually holding {retained:.0f} MB of array data at the end "
            f"({retained / max(len(house_raw_histories), 1):.0f} MB/episode); the rest of the "
            f"RSS growth is not retained data"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
