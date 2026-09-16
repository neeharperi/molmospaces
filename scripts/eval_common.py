"""Shared task/policy tables and stats helpers for the DROID-reproduction eval harness.

Imported by scripts/eval.py, scripts/check_provenance.py, scripts/compare_to_leaderboard.py,
and scripts/dump_first_observation.py so the 9-task benchmark table (and its known,
deliberate path collisions) is defined exactly once. See plans/BENCHMARK.md for the full
9-task table this mirrors, and docs/eval_reproduction.md for the corrections applied on top
of it.
"""

from __future__ import annotations

import os
import pathlib
import sys
from dataclasses import dataclass, field

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from molmo_spaces.molmo_spaces_constants import MODELS_DIR  # noqa: E402


@dataclass(frozen=True)
class TaskSpec:
    renderer: str  # "classic" | "filament"
    path: str  # relative to $MLSPACES_ASSETS_DIR/benchmarks/
    extra_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PolicySpec:
    exp_config_cls: str  # "module.path:ClassName", passed straight to eval_main.py
    checkpoint_path: str
    host: str
    port: int


# Renderer -> the conda env that has that renderer's mujoco extra installed.
# scripts/eval.py hard-fails (rather than auto-switching) if $CONDA_DEFAULT_ENV doesn't match,
# per BENCHMARK.md's confirmed mujoco/mujoco-filament conflict (pyproject.toml gives both
# extras the same import name `mujoco` from different sources).
#
# eval_main.py's own `--use-filament` CLI flag is NOT what selects the renderer -- it's
# parsed but never read anywhere in eval_main.py or run_evaluation(). The actual renderer is
# `HAS_FILAMENT = getattr(mujoco, "mjRENDERER", "classic") == "filament"`
# (molmo_spaces/env/env.py:35), which is fixed at import time by whichever `mujoco` package
# is installed in the active env. So which conda env you activate IS the renderer choice;
# this table's `extra_flags` therefore only need to carry flags that have a real effect
# (e.g. --camera_names), not --use-filament.
RENDERER_TO_CONDA_ENV = {
    "classic": "mlspaces-classic",
    "filament": "mlspaces-filament",
}

TASKS: dict[str, TaskSpec] = {
    "Open-v1": TaskSpec(
        renderer="classic",
        path="molmospaces-bench-v1/ithor/FrankaOpenDataGenConfig/FrankaOpenDataGenConfig_20260123_json_benchmark",
    ),
    "Close-v1": TaskSpec(
        renderer="classic",
        path="molmospaces-bench-v1/ithor/FrankaCloseDataGenConfig/FrankaCloseDataGenConfig_20260123_json_benchmark",
    ),
    "Pick-v1.5": TaskSpec(
        renderer="classic",
        path="molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231",
    ),
    "Pick-v2-classic": TaskSpec(
        renderer="classic",
        path="molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
    ),
    "Pick-v2-filament": TaskSpec(
        renderer="filament",
        path="molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
    ),
    "Pick-v2-RandCam": TaskSpec(
        renderer="filament",
        path="molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark",
        extra_flags=(
            "--camera_names",
            "randomized_zed2_analogue_1",
            "wrist_camera_zed_mini",
        ),
    ),
    "PnP-v2": TaskSpec(
        renderer="filament",
        path="molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark",
    ),
    "PnP-NextTo-v2": TaskSpec(
        renderer="filament",
        path="molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark",
    ),
    "PnP-Color-v2": TaskSpec(
        renderer="filament",
        path="molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark",
    ),
}

# Group A: 2 of the 4 MolmoSpaces-Combined bench-v1 tasks (report per-task only, no average).
GROUP_A = ("Open-v1", "Close-v1")

# Group B: the complete MolmoBot-Combined bench-v2 set (7 tasks) -- comparable to the
# leaderboard's "MolmoBot Combined" column as a single pooled aggregate.
GROUP_B = (
    "Pick-v1.5",
    "Pick-v2-classic",
    "Pick-v2-filament",
    "Pick-v2-RandCam",
    "PnP-v2",
    "PnP-NextTo-v2",
    "PnP-Color-v2",
)

assert set(GROUP_A) | set(GROUP_B) == set(TASKS), "GROUP_A/GROUP_B must partition TASKS"

# Three tasks deliberately share one benchmark JSON (FrankaPickHardBench_20260206) and differ
# only by flags. Any *other* pair of tasks sharing a (path, extra_flags) tuple is a bug --
# assert it at import time rather than trusting convention (per BENCHMARK.md's explicit
# warning that this collision would otherwise silently overwrite results across tasks).
_EXPECTED_SHARED_PATH_GROUP = frozenset({"Pick-v2-classic", "Pick-v2-filament", "Pick-v2-RandCam"})


def _assert_task_table_integrity() -> None:
    by_key: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for name, spec in TASKS.items():
        by_key.setdefault((spec.path, spec.extra_flags), []).append(name)
    for key, names in by_key.items():
        if len(names) > 1 and not set(names) <= _EXPECTED_SHARED_PATH_GROUP:
            raise AssertionError(
                f"Unexpected task-path collision {key}: {names}. "
                f"Only {sorted(_EXPECTED_SHARED_PATH_GROUP)} are allowed to share a benchmark "
                f"path/flags combination; any other collision means two tasks would silently "
                f"overwrite each other's run directory."
            )
    # Output directories are keyed by task name, so distinctness is trivial -- assert it
    # anyway so a future refactor that changes the key can't silently break resumability.
    assert len(set(TASKS)) == len(TASKS), "TASKS keys must be unique (used as output-dir keys)"


_assert_task_table_integrity()


POLICIES: dict[str, PolicySpec] = {
    "pi05_droid": PolicySpec(
        exp_config_cls="molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig",
        checkpoint_path=str(MODELS_DIR / "openpi/checkpoints/pi05_droid_jointpos"),
        host="localhost",
        port=8080,
    ),
    "pi0_droid": PolicySpec(
        exp_config_cls="molmo_spaces.evaluation.configs.evaluation_configs:Pi0PolicyEvalConfig",
        checkpoint_path=str(MODELS_DIR / "openpi/checkpoints/pi0_droid_jointpos"),
        host="localhost",
        port=8081,
    ),
    "molmoact2_droid": PolicySpec(
        exp_config_cls="molmo_spaces.evaluation.configs.evaluation_configs:MolmoAct2PolicyEvalConfig",
        checkpoint_path="allenai/MolmoAct2-DROID",
        host="localhost",
        # 8102, not upstream's 8000. droid's OWN openpi server (this repo sits inside
        # droid/third_party/) serves pi0.5 to the rig on :8000, and its molmoact2 server
        # takes :8101 -- so 8000 would collide outright and 8101 would stop the rig's
        # server and this harness's from running at the same time. Same reasoning as
        # tiptop's 18765 below.
        port=8102,
    ),
    "tiptop": PolicySpec(
        exp_config_cls="molmo_spaces.evaluation.configs.evaluation_configs:TiptopEvalConfig",
        checkpoint_path="",  # TiPToP's server owns its own model/planner state, no client-side checkpoint path
        host="localhost",
        # 18765, not upstream's default 8765: this is a SHARED host and another user's process
        # already listens on 8765. Our server could never bind ("address already in use", 21
        # supervisor restarts), while the client happily connected to THEIR server and failed
        # the websocket handshake -- surfacing as "[SSL: WRONG_VERSION_NUMBER]" from the
        # client's wss:// fallback, which names neither the port nor the real problem.
        port=18765,
    ),
    "dreamzero": PolicySpec(
        exp_config_cls="molmo_spaces.evaluation.configs.evaluation_configs:DreamZeroPolicyEvalConfig",
        checkpoint_path=str(MODELS_DIR / "dreamzero/checkpoints/DreamZero-DROID"),
        host="localhost",
        # must track DreamZeroPolicyConfig.remote_config, or eval.py probes the wrong
        # port during fan-out and hard-fails a lane that is actually healthy.
        port=int(os.environ.get("DREAMZERO_PORT", "5000")),
    ),
}

# The three policies BENCHMARK.md's acceptance criteria are load-bearing for (27 required
# cells = 3 * 9, plus the 3 Group B aggregates). DreamZero (added after BENCHMARK.md was
# written) is evaluated and reported, but a miss is informational only.
REPRODUCTION_POLICIES = ("pi05_droid", "molmoact2_droid", "tiptop")

# The leaderboard's pooled-aggregate row name for the Group B (MolmoBot Combined) comparison.
GROUP_B_LEADERBOARD_TASK_NAME = "MolmoBot Combined"


def check_env_matches(task: TaskSpec) -> str | None:
    """Return an error message if the active conda env doesn't match the task's renderer."""
    expected_env = RENDERER_TO_CONDA_ENV[task.renderer]
    actual_env = os.environ.get("CONDA_DEFAULT_ENV")
    if actual_env != expected_env:
        return (
            f"Task requires the '{task.renderer}' renderer, which needs conda env "
            f"'{expected_env}', but the active env is '{actual_env}'. Run:\n"
            f"    conda activate {expected_env}"
        )
    return None


def wilson_interval(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial proportion, as a fraction in [0, 1].

    This is the acceptance-check interval (BENCHMARK.md's Validation section), distinct from
    eval_to_csv.py's own per-run Bayesian Beta(1,1) credible interval, which is informational.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return ((center - margin) / denom, (center + margin) / denom)


# A leading underscore on a runs/ directory -- at either the policy or the date level -- marks
# it as not-a-campaign-result: handshakes, A/B arms, debug runs, quarantined cells. The rule
# has to be applied at BOTH levels and by EVERY consumer. `_` is 0x5F, which sorts after the
# digits, so a 60-episode A/B arm in `_ab_C_dt100/` beat the real `20260828_full/` cell to
# `sorted(...)[-1]` and was published as cosmos_edge's Pick-v1.5 verdict -- 26.7% (n=60) in
# place of 33.8% (n=1000). Diagnostic runs must be inert to the comparison, not merely
# distinguishable by a human reading directory names.
# Substrings that mark a whole POLICY directory as quarantined, checked separately from the
# leading-underscore rule above (which applies to any single path component).
QUARANTINE_MARKERS = ("_INVALID", "_STALE", "_superseded")


def is_campaign_dir(path) -> bool:
    """True if ``path`` is a real campaign result directory rather than a diagnostic one."""
    return not path.name.startswith("_")


def latest_results_csv(runs_dir, policy: str, task: str):
    """Newest real results.csv for a cell, ignoring underscore-prefixed date directories."""
    cell_dir = runs_dir / policy / task
    if not cell_dir.exists():
        return None
    candidates = sorted(c for c in cell_dir.glob("*/results.csv") if is_campaign_dir(c.parent))
    return candidates[-1] if candidates else None


def read_overall(results_csv):
    """The OVERALL row of an eval_to_csv.py results.csv, as a pandas Series.

    Read by column name, never by position -- eval_to_csv.py's column list is a de-facto
    public API here and positional access silently reports a different metric when it grows.
    """
    import pandas as pd

    df = pd.read_csv(results_csv, comment="#")
    overall = df[df["category"] == "OVERALL"]
    if overall.empty:
        raise ValueError(f"{results_csv} has no OVERALL row")
    return overall.iloc[0]
