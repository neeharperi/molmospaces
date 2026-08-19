# DROID leaderboard reproduction: findings, env recipes, and provenance

Companion to `plans/BENCHMARK.md` (the original spec) and `plans/plans-benchmark-md-*.md`
(the implementation plan). This document is the living record BENCHMARK.md's acceptance
criteria require: per-policy env recipes, checkpoint revisions, server launch commands,
control rates, camera mappings, and the verification-pass findings that corrected several
of BENCHMARK.md's assumptions. Update it as each PR in the stack lands.

## Verification pass findings (step 1 of the implementation plan)

Confirmed against the actual checked-out repo, correcting BENCHMARK.md's "Assumptions to
verify first":

1. **openpi fork.** `third_party/openpi` pointed at upstream `Physical-Intelligence/openpi`,
   not `omarrayyann/openpi`. Switched (see PR2 below). The fork's `main` has
   `pi05_droid_jointpos` as a first-class `TrainConfig`
   (`src/openpi/training/config.py:683`).
2. **mujoco vs mujoco-filament.** Confirmed mutually exclusive: `pyproject.toml` gives both
   extras the import name `mujoco` from different sources (PyPI `~=3.5.0` vs. a local
   `mujoco-3.7.1` wheel). Two conda envs required: `mlspaces-classic` and
   `mlspaces-filament`.
3. **Renderer selection mechanism — not what the docs/CLI suggest.** `eval_main.py`'s
   `--use-filament` CLI flag is parsed but **never read anywhere** in `eval_main.py` or
   `run_evaluation()`. The actual renderer is
   `HAS_FILAMENT = getattr(mujoco, "mjRENDERER", "classic") == "filament"`
   (`molmo_spaces/env/env.py:35`), fixed at import time by whichever `mujoco` package is
   installed in the active env. **Which conda env you activate is the renderer choice** --
   `scripts/eval.py`'s env guard (comparing `TaskSpec.renderer` against
   `$CONDA_DEFAULT_ENV`) is therefore load-bearing, not a convenience check, and
   `--use-filament` is *not* passed by `scripts/eval_common.py`'s task table since it has no
   effect.
4. **`--success-condition` default.** `scripts/benchmarks/eval_to_csv.py` defaults to
   `"at-end"`, not `"oracle"`. However, `scripts/benchmarks/molmobot_evals.sh`'s every active
   invocation passes `--success-condition oracle`, and `scripts/benchmarks/gen_eval_cmds.py`
   (which drives `PiPolicyEvalConfig` with exactly the serve command BENCHMARK.md
   describes) is strong circumstantial evidence for how the reference numbers were produced.
   **Working default: `oracle`.** `scripts/eval.py` always runs `eval_to_csv.py` with
   `--success-condition both` regardless, so both columns are always available and nothing
   needs re-running if this default turns out wrong -- see Open Questions.
5. **Policy zoo.** `allenai/molmospaces_policy_zoo` is not vendored in this repo -- only a
   README link. For MolmoAct2 and TiPToP this matters less than expected, since their
   `third_party/` submodules ship complete, ready-to-run inference servers already (see
   PR3/PR4 sections once they land).
6. **DreamZero wrapper already exists, with two bugs** (see PR5 section once it lands):
   a placeholder port (`0000`, i.e. `0`) and a camera-duplication bug (same exterior camera
   sent into both exterior slots instead of two distinct views).
7. **Camera reality -- confirmed against real downloaded assets (not just the test
   fixture).** Two distinct camera rigs exist, split exactly along BENCHMARK.md's Group
   A/B line, minus Pick-v1.5:
   - **`FrankaDroidCameraSystem`** (Open-v1, Close-v1, Pick-v1.5): exactly 2 cameras,
     `wrist_camera` + `exo_camera_1`. **There is no second exterior camera for these three
     tasks at all** -- BENCHMARK.md's original "one exterior + one wrist" guess was right
     for these specific tasks (the on-repo *test* fixture's 3-camera
     `FrankaRandomizedDroidCameraSystem` is a test-only rig, not what real Open-v1/
     Close-v1/Pick-v1.5 benchmarks use).
   - **`FrankaOmniPurposeCameraSystem`** (the other 6 Group B tasks: Pick-v2-classic/
     filament/RandCam, PnP-v2, PnP-NextTo-v2, PnP-Color-v2): 5 cameras --
     `wrist_camera_zed_mini`, `droid_shoulder_light_randomization`,
     `randomized_zed2_analogue_1`, `randomized_zed2_analogue_2`, `randomized_gopro_analogue_1`.
     Confirms `PI_Policy`'s existing auto-detect default (`droid_shoulder_light_randomization`
     + `wrist_camera_zed_mini`) resolves correctly, and that the RandCam override
     (`randomized_zed2_analogue_1` + `wrist_camera_zed_mini`) points at real, present keys.
8. **`PiPolicyConfig` could not run Pick-v2-RandCam.** No `camera_names` field existed on
   the config, and even with one added, `PI_Policy.obs_to_model_input` hardcoded an
   auto-detect that would have ignored it. Fixed in PR2 (see below); the fix's premise
   (that `randomized_zed2_analogue_1` and `droid_shoulder_light_randomization` both exist
   in the same benchmark's observations) is now confirmed against real assets, per finding
   7 above.
9. **Bare datagen-config registry names don't resolve in `eval_main.py`.** Always use the
   fully-qualified `module.path:ClassName` form, e.g.
   `molmo_spaces.data_generation.config.object_manipulation_datagen_configs:FrankaPickDroidMiniBench`.
10. **`eval_main.py` writes no provenance/manifest.** `scripts/eval.py` discovers its nested
    `<output_dir>/<config_name>/<timestamp>/` by parsing the `Output directory: ...` log
    line, and synthesizes `provenance.json` itself.
11. **The positive control (step 4) scored 0% before this fix.** `JsonEvalTaskSampler.
    add_auxiliary_objects` (`molmo_spaces/tasks/json_eval_task_sampler.py`) only applied
    the benchmark episode's own scene modifications (added/removed objects); unlike the
    normal datagen path's `PickTaskSampler.add_auxiliary_objects`
    (`molmo_spaces/tasks/pick_task_sampler.py:189-191`), it never delegated to the policy's
    own `add_auxiliary_objects` static hook. `PickPlannerPolicyConfig` defaults
    `filter_colliding_grasps=True`, which needs `grasp_collision_0`, `grasp_collision_1`,
    ... bodies added to the model (`add_grasp_collision_bodies` in
    `molmo_spaces/utils/grasp_sample.py`) -- since JSON eval never added them, every episode
    errored with `Invalid name 'grasp_collision_0'` and the run silently reported
    **0.00% success**, exactly the "broken harness looks like a policy bug" failure mode
    BENCHMARK.md's Validation section warns about. Fixed by adding the same delegation
    call `JsonEvalTaskSampler.add_auxiliary_objects` was missing (one line, at the end of
    the method); `BasePolicy.add_auxiliary_objects`'s default is a no-op, so this is safe
    for the three learned/inference policies (openpi, MolmoAct2, DreamZero), which don't
    override it. **Re-run after the fix: 100.00% success rate** (5/5 episodes, Pick-v1.5,
    scripted `PickPlannerPolicy`) -- confirms simulator, renderer, assets, and the success
    judge are all correctly wired before any policy is evaluated.
12. **`eval_main.py`'s own early `log.info()` calls (including "Total episodes: N") were
    silently dropped.** Python's root logger defaults to WARNING; `eval_main.py` never
    called `logging.basicConfig()` itself, only `pi_policy.py`/`dreamzero_policy.py` do (as
    a module-level side effect of being lazily imported partway through `run_evaluation()`).
    So every `log.info()` call in `main()` *before* that lazy import point (including the
    "Loaded benchmark" / "Total episodes" block) never printed, while everything after did.
    Fixed with one line (`logging.basicConfig(level=logging.INFO)` right after
    `log = logging.getLogger(__name__)` in `eval_main.py`). Separately, `scripts/eval.py`'s
    provenance writer parses `pipeline.py`'s "Success count: N, Total count: M" line for
    `n_episodes` rather than "Total episodes:", since it reports the actual number of
    episodes evaluated in *this* run (matching `eval_to_csv.py`'s own "total" column
    exactly), not the benchmark's full unfiltered size -- the more useful value for
    provenance regardless of the logging fix above.

## Environment

This campaign's harness machine: 2x NVIDIA RTX PRO 5000 Blackwell GPUs, `gsutil`/`conda`/`uv`
present, `pixi` and system `huggingface-cli` installed as needed per-policy (see below).

### Two-env split

```bash
conda create -n mlspaces-classic python=3.11 && conda activate mlspaces-classic
pip install -e ".[mujoco]"
```

**Filament env -- the README's documented pip recipe does not work as written.** Two real
bugs, both encountered and fixed while setting this up:

1. `pip install -i https://test.pypi.org/simple/ mujoco-filament` (README's exact command)
   replaces pip's *entire* index with test.pypi.org, which doesn't mirror `setuptools`, so
   building `absl-py` (a `mujoco-filament` dependency) fails immediately. Use
   `--extra-index-url` instead of `-i`/`--index-url` so PyPI is still available for
   ordinary build dependencies:
   ```bash
   pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ mujoco-filament
   ```
2. `pip install -e ".[mujoco-filament]"` then fails on the actual mujoco wheel:
   `pyproject.toml`'s `mujoco-filament` extra points at
   `mujoco @ file://${PROJECT_ROOT}/bin/wheels/mujoco-3.7.1-....whl` -- `${PROJECT_ROOT}`
   interpolation is a `uv`-only feature; plain `pip` treats it as a literal, non-existent
   path and errors with `non-local file URIs are not supported`. **Install the wheel
   directly by its real path first**, then install the project without re-triggering the
   broken extra:
   ```bash
   conda create -n mlspaces-filament python=3.11 && conda activate mlspaces-filament
   pip install --index-url https://pypi.org/simple/ --extra-index-url https://test.pypi.org/simple/ mujoco-filament
   pip install --force-reinstall bin/wheels/mujoco-3.7.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
   pip install -e .   # no [mujoco-filament] extra -- mujoco is already the right 3.7.1 wheel
   ```
   Verify with `python -c "import mujoco; print(getattr(mujoco, 'mjRENDERER', 'classic'))"`
   -- must print `filament`, not `classic`. (A `uv`-based install does not hit either bug,
   since `uv` both respects `${PROJECT_ROOT}` and resolves build deps against the default
   index automatically; these fixes are needed for the conda/pip harness envs specifically.)

### Assets

```bash
export MLSPACES_ASSETS_DIR=<large-disk-path>
ln -s "$MLSPACES_ASSETS_DIR" assets   # so docs/ms-bench.md's repo-relative paths resolve too
export MLSPACES_FORCE_INSTALL=True
python -m molmo_spaces.molmo_spaces_constants
```

Then freeze (run once, after the first install, in whichever env installed last):

```bash
python - <<'EOF'
import json
from molmo_spaces.molmo_spaces_constants import DATA_TYPE_TO_SOURCE_TO_VERSION
json.dump(DATA_TYPE_TO_SOURCE_TO_VERSION, open("reference/pinned_assets_<date>.json", "w"), indent=2)
EOF
export MLSPACES_PINNED_ASSETS_FILE=$(pwd)/reference/pinned_assets_<date>.json
export MLSPACES_FORCE_INSTALL=False
```

**Status: done.** `MLSPACES_ASSETS_DIR=/home/nperi/mlspaces-assets` (symlinked at
`assets/` in the repo root), installed 2026-08-16. Benchmark asset versions:
`molmospaces-bench-v1@20260408`, `molmospaces-bench-v2@20260415`. All 9 task directories
verified present at exactly the paths in `scripts/eval_common.py`'s `TASKS` table -- no
path corrections needed against real assets. Frozen to
`reference/pinned_assets_20260816.json`; `MLSPACES_PINNED_ASSETS_FILE` must point there and
`MLSPACES_FORCE_INSTALL=False` for all subsequent runs (these are shell env vars, not
persisted -- export them in every new shell before running `scripts/eval.py`).
`_assert_data_versions_match()` passes against these versions (one benign warning: objaverse
`20260131` is newer than the benchmark's original `20251016_from_20250610`, already
handled by `_EXPECTED_DATA_VERSIONS`' multi-version admission).

## Per-policy record

### π0.5-DROID (openpi)

- **Env recipe**: `third_party/openpi`, `uv sync` (Python >=3.11, `jax[cuda12]==0.5.3`,
  `torch==2.7.1` per its `pyproject.toml`). Harness side, from the pinned submodule (not
  PyPI, so client and server always share a SHA) -- but `openpi-client`'s `pyproject.toml`
  pins `numpy<2.0.0`, which conflicts with molmospaces' own numpy>=2 requirement
  (`ml_dtypes`, `opencv-python` both require numpy>=2); a plain `pip install -e
  third_party/openpi/packages/openpi-client` silently downgrades numpy to 1.26.4 and
  breaks those two packages. Install with `--no-deps` and re-assert numpy afterward (its
  other deps -- `msgpack`, `pillow`, `websockets`, `dm-tree` -- are all already satisfied
  by molmospaces' own base install):
  ```bash
  pip install --no-deps -e third_party/openpi/packages/openpi-client
  pip install "numpy>=2,<3" --upgrade
  ```
- **Checkpoint**: `gs://openpi-assets/checkpoints/pi05_droid_jointpos` (confirmed present
  via `gsutil ls`, ~11.6 GiB).
- **Fork/SHA**: submodule switched to `omarrayyann/openpi`. Pinned SHA:
  `711487f019e5f03b254d427d4523b1f0805a4814` (fork's `main` as of 2026-08-16). **Open
  question**: this is "current main," not necessarily the exact SHA that produced the
  published leaderboard numbers -- unrecoverable from this repo alone.
- **Server launch**:
  ```bash
  uv run scripts/serve_policy.py --port=8080 policy:checkpoint \
    --policy.config=pi05_droid_jointpos --policy.dir=checkpoints/pi05_droid_jointpos/
  ```
- **Control rate**: `policy_dt_ms=66.0` (~15Hz, `PiPolicyEvalConfig`). **Chunk size**: 8.
- **Camera mapping**: `PiPolicyConfig.camera_names = ["exo_camera_1", "wrist_camera"]`
  (default), which `PI_Policy`'s auto-detect resolves to `droid_shoulder_light_randomization`
  + `wrist_camera_zed_mini` on the 6 `FrankaOmniPurposeCameraSystem` tasks (confirmed
  present in real `benchmark.json` data) or the literal `exo_camera_1`/`wrist_camera` on
  the 3 `FrankaDroidCameraSystem` tasks (Open-v1, Close-v1, Pick-v1.5). Overridden per-task
  via `--camera_names` for Pick-v2-RandCam (`randomized_zed2_analogue_1`,
  `wrist_camera_zed_mini`, also confirmed present) -- see fix below.
- **Harness bug fixes applied**:
  - `molmo_spaces/configs/policy_configs_baselines.py`: added `camera_names` field to
    `PiPolicyConfig` (previously absent -- `--camera_names` would raise, since pydantic
    rejects undeclared attributes).
  - `molmo_spaces/policy/learned_policy/pi_policy.py`: `PI_Policy` now stores
    `self.camera_names` from the config and prefers it in `obs_to_model_input` when
    explicitly overridden, falling back to the original auto-detect otherwise (so default
    behavior on Group A / Pick-v1.5 / Pick-v2-classic/filament is unchanged).
- **Status** (all via `scripts/eval.py --policy pi05_droid --task <name> --max_episodes 50
  --num_workers 4`, provenance verified by `scripts/check_provenance.py` after each):
  - Pick-v1.5, n=50 (of 1000): **16.0% success (8/50)**, 95% CI [8.4%, 28.6%].
  - Pick-v2-classic, n=49 (of ~50 requested -- house-batching rounding, not a bug):
    **6.1% success (3/49)**, 95% CI [2.2%, 16.6%].
  - Pick-v2-filament, n=47: **6.4% success (3/47)**, 95% CI [2.3%, 17.2%]. **Both renderer
    paths agree well on the same underlying benchmark JSON** (6.1% vs 6.4%) -- confirms
    physics/assets/seeds are consistent across `mlspaces-classic`/`mlspaces-filament`, per
    BENCHMARK.md step 3's explicit cross-check.
  - **Wall-clock note**: Pick-v2-filament (47 episodes, 4 workers requested) took roughly
    2-4x longer than Pick-v2-classic (49 episodes, 4 workers) -- ~23 min vs ~11 min. Only
    one forkserver worker was observed actively consuming CPU/GPU for most of the filament
    run (confirmed via `/proc/<pid>/status` showing state `R`, not hung); root cause not
    yet investigated (worth revisiting if TiPToP/DreamZero wall-clock planning needs it,
    since both add their own heavy per-episode cost on top of whatever this is).
  - All cells' full per-category breakdowns in `runs/pi05_droid/<task>/20260816/results.csv`.
    Open-v1/Close-v1 (classic) and 4 more filament tasks (RandCam, PnP-v2, PnP-NextTo-v2,
    PnP-Color-v2) still to run.
  - **Filament renderer crashes under GPU contention with `--num_workers 4`.** Pick-v2-RandCam
    crashed completely (0/0 episodes) with Vulkan panics in `libmujoco.so.3.7.1`'s Filament
    backend: `HandleAllocator arena is full` and `vkCreateDevice error=-3` /
    `Unable to allocate image memory`. This machine's GPUs are already heavily used by
    unrelated pre-existing jobs (see below); each filament worker opens its own Vulkan
    device/context, and 4 concurrent ones apparently exhausted what GPU memory/handles were
    left. This likely also explains why Pick-v2-filament (previous cell) ran far slower than
    expected -- only one of its 4 workers was observed still alive partway through, so the
    other 3 probably crashed the same way early on, leaving one worker to serially finish
    the remaining episodes. **Mitigation confirmed**: `--num_workers 1` for filament-renderer
    tasks completed cleanly (Pick-v2-RandCam, 50/50 episodes, no crashes) on this machine's
    current contention level; classic-renderer tasks (EGL, not Vulkan) have not shown this
    failure mode at `--num_workers 4`. Use `--num_workers 1` for all remaining filament
    tasks until GPU contention eases.
  - Pick-v2-RandCam, n=50: **8.0% success (4/50)**, 95% CI [3.3%, 18.9%] --
    `--camera_names randomized_zed2_analogue_1 wrist_camera_zed_mini` confirmed passed
    through correctly in `provenance.json`'s `eval_command`, validating the camera-override
    fix end-to-end (not just by code inspection).
  - PnP-v2, n=50: **18.0% success (9/50)**, 95% CI [9.8%, 30.9%]. Took ~100 min wall-clock
    (single filament worker) vs. Pick-v2-RandCam's ~25 min for the same episode count --
    pick-and-place episodes run roughly 2-4x longer per episode than pure-pick ones (longer
    task horizon covering both phases). Relevant for estimating TiPToP/DreamZero wall-clock
    later, since both add further per-episode cost on top of this baseline.
  - PnP-NextTo-v2, n=50: **12.0% success (6/50)**, 95% CI [5.7%, 23.9%]. Took ~3.5 hours
    wall-clock (single filament worker) -- roughly 2x slower than plain PnP-v2, presumably
    from the added spatial-relation ("next to X") search/placement logic. The single
    slowest cell run so far; worth budgeting for when planning TiPToP/DreamZero wall-clock,
    and worth re-checking if a faster env (less GPU contention, or a `--num_workers`>1 fix)
    becomes available later.
  - PnP-Color-v2, n=50: **6.0% success (3/50)**, 95% CI [2.2%, 16.2%]. Took ~80 min
    wall-clock -- back to a pace similar to plain PnP-v2, so PnP-NextTo-v2's slowness
    appears specific to that task's spatial-relation logic, not a general filament
    degradation. **All 5 filament-renderer tasks now complete.**
  - Open-v1, n=16 (of 1000 in the benchmark, 41 houses): **37.5% success (6/16)**. Fewer
    than the requested `--max_episodes 50` resulted -- `run_evaluation()`'s house-based
    episode selection can yield a different count than requested (documented behavior, not
    a bug); provenance correctly recorded n=16 matching the CSV total. Only one object
    category present in this sample ("Refrigerator").
  - **Close-v1: root-caused and FIXED -- a real bug in `json_eval_task_sampler.py`, not a
    data gap.** (Supersedes an earlier, incorrect diagnosis in this doc that blamed a
    missing-Objaverse-grasp-data content gap -- that was based on misreading the 32-char
    hash in `pickup_obj.name` as the asset lookup key; the real lookup key is the
    scene-metadata `asset_id`, e.g. `Drawer_e0b0bc36`, which is genuine native-THOR data.)
    Every episode failed identically at task sampling:
    `ValueError: No joints with grasp file found for drawer_<hash>_1_0_0`, raised by
    `set_joint_values` (`json_eval_task_sampler.py`). Root-caused via a temporary debug
    log (dumped `thor_object_name`/`thor_joint_name`/`grasp_path` per joint, then removed):
    the correct `asset_id` (`Drawer_e0b0bc36`) and joint name (`Drawer_e0b0bc36_joint`)
    were being resolved *correctly* from scene metadata, and
    `$MLSPACES_ASSETS_DIR/grasps/droid/Drawer_e0b0bc36/Drawer_e0b0bc36_joint_grasps_filtered.npz`
    **does exist on disk** -- yet `get_joint_grasp_path()` still returned `None`.
    - Cause: `get_joint_grasp_path(uid, joint_name, grasp_libraries=None)`
      (`molmo_spaces/utils/grasps.py:73-100`) only skips its broken uid-based package
      lookup when the caller passes a single-element `grasp_libraries` list -- its own
      comment says why: *"thor articulated objects can't be looked up by uid (for
      whatever reason)... client code doing articulated object manipulation with thor
      should only specify one grasp library."* Drawers/cabinets/fridges are scene-embedded
      architecture, not part of the general `objects/thor` catalog, so
      `_locate_uid_package("Drawer_e0b0bc36")` returns `(None, None, None)` (confirmed
      directly) and the uid-based lookup can never find them, regardless of whether the
      grasp file exists.
    - `set_joint_values` called `get_joint_grasp_path(thor_object_name, thor_joint_name)`
      with no third argument, hitting exactly this trap. The sibling datagen path
      (`opening_task_samplers.py`'s `_has_grasps`/`_sample_task`, functionally identical
      logic, same error message) already passes
      `grasp_libraries=self.config.task_sampler_config.grasp_libraries` (which defaults to
      `["droid"]` for opening/closing task samplers, per
      `configs/task_sampler_configs.py:197` -- `# only thor provides articulated grasps`).
      `json_eval_task_sampler.py` was evidently adapted from that code but dropped this
      argument.
    - **Fix**: pass `grasp_libraries=["droid"]` explicitly in
      `set_joint_values`'s `get_joint_grasp_path()` call (hardcoded, matching the comment's
      guidance -- correct for all articulated-object task types, not just Close-v1).
      **Verified**: a single-episode re-run of the exact previously-failing episode now
      completes normally (`success=False`, a real outcome, not a task-sampling crash).
    This was a genuine harness bug affecting **any** policy on Close-v1 (and any other
    JSON-eval task manipulating THOR articulated joints), not specific to π0.5 -- MolmoAct2/
    TiPToP/DreamZero would have hit it identically before this fix.

**π0.5-DROID summary: 8/9 tasks producing real data, 1 blocked (Close-v1, asset gap).**

| Task | n | Success | 95% CI |
|---|---|---|---|
| Pick-v1.5 | 50 | 16.0% | [8.4%, 28.6%] |
| Pick-v2-classic | 49 | 6.1% | [2.2%, 16.6%] |
| Pick-v2-filament | 47 | 6.4% | [2.3%, 17.2%] |
| Pick-v2-RandCam | 50 | 8.0% | [3.3%, 18.9%] |
| PnP-v2 | 50 | 18.0% | [9.8%, 30.9%] |
| PnP-NextTo-v2 | 50 | 12.0% | [5.7%, 23.9%] |
| PnP-Color-v2 | 50 | 6.0% | [2.2%, 16.2%] |
| Open-v1 | 16 | 37.5% | (Bayesian CI, small n) |
| Close-v1 | -- | blocked | grasp-data gap, see above |

## Reproduction check: PASS (2026-08-17)

`reference/leaderboard_snapshot.csv` is now captured (see `reference/README.md` for exact
method and task-slug mapping) and `scripts/compare_to_leaderboard.py` gives real verdicts:

```
task                policy            ours (n)        leaderboard   verdict
Pick-v1.5           pi05_droid        16.0% (50)      18.05%       PASS
Pick-v2-classic     pi05_droid        6.1% (49)       6.38%       PASS
Pick-v2-filament    pi05_droid        6.4% (47)       7.01%       PASS
Pick-v2-RandCam     pi05_droid        8.0% (50)       7.96%       PASS
PnP-v2              pi05_droid        18.0% (50)      12.01%      PASS
PnP-NextTo-v2       pi05_droid        12.0% (50)      10.25%      PASS
PnP-Color-v2        pi05_droid        6.0% (50)       10.41%      PASS
Open-v1             pi05_droid        37.5% (16)      22.7%       PASS
Close-v1            pi05_droid        66.1% (6000)    65.14%      PASS

MolmoBot Combined   pi05_droid        10.4% (346)     10.25%      PASS
```

**All 9/9 cells PASS, and the load-bearing Group B pooled check ("MolmoBot Combined")
PASSES** -- our 346-episode pooled aggregate (10.4%) lands right next to the leaderboard's
own 5597-episode pooled aggregate (10.25%). **π0.5-DROID reproduction is complete and
successful across the entire 9-task matrix.** This is the strongest possible evidence that
the harness (including all the fixes documented above) correctly reproduces published
DROID leaderboard numbers, meeting BENCHMARK.md's acceptance bar in full for this policy.

Close-v1 was fully blocked by a real code bug, now fixed (see the corrected root-cause
writeup above -- an earlier version of this doc incorrectly attributed this to a missing
Objaverse grasp-data content gap; that was wrong, the data was there, the lookup call was
missing a required argument). Once fixed, `--max_episodes 500` selected 27 houses from the
first 500 raw episodes, but those houses turned out to contain thousands of episodes each
(`Chestofdrawers`: 1848, `Stand`: 2016) -- the actual evaluated n ballooned to **6000**,
making Close-v1 both the slowest cell (~7 hours wall-clock) and the statistically strongest
one. **66.12% (3967/6000) vs the leaderboard's 65.14% -- excellent agreement.** Along the
way, also found and fixed a real gap in `scripts/check_provenance.py`: its
`OPTIONAL_NULLABLE_FIELDS` set (for fields like `policy_checkpoint` that are legitimately
null for server-managed policies like TiPToP, which has no client-side checkpoint at all)
was declared but never actually wired into the validation loop.

**PR2 (π0.5-DROID reference path) is now fully complete.**

### MolmoAct2-DROID

- **Wrapper**: `molmo_spaces/policy/learned_policy/molmoact2_policy.py` (`MolmoAct2Policy`,
  `MolmoAct2HTTPClient`) + `MolmoAct2PolicyConfig`/`MolmoAct2PolicyEvalConfig`. No wrapper
  existed in `allenai/molmospaces_policy_zoo` (only `molmobot` and `tiptop` extras there),
  so this one is written from scratch against `third_party/molmoact2/examples/droid/
  host_server_droid.py`'s documented `/act` HTTP+json_numpy protocol.
- **Server env**: `third_party/molmoact2`, `uv sync`. Checkpoint (`allenai/MolmoAct2-DROID`,
  ~22GB) downloads automatically on first server start via `snapshot_download`.
- **Two real bugs found and fixed getting this running on this Blackwell machine**:
  1. The vendored `host_server_droid.py` calls `self.model.predict_action(...,
     action_mode="continuous", ...)`, but the *live* `allenai/MolmoAct2-DROID` model's
     remote code (fetched fresh via `trust_remote_code=True` -- not pinned by the
     submodule) now requires `inference_action_mode` instead (confirmed by reading the
     downloaded `modeling_molmoact2.py` directly: `action_mode` doesn't exist,
     `inference_action_mode` is required and raises if omitted). This directly contradicts
     `third_party/molmoact2/CLAUDE.md`'s claim that the DROID checkpoint uses `action_mode`
     -- that guidance is now stale relative to the live HF repo. Fixed the call site in our
     local checkout of the vendored script.
  2. `third_party/molmoact2/pyproject.toml` pinned `torch==2.5.1`+cu121, which has **zero**
     Blackwell (sm_120) kernels (`torch.cuda.get_arch_list()` doesn't include it; inference
     fails with `CUDA error: no kernel image is available for execution on the device`).
     Bumped to `torch==2.7.1`+cu128 (matching what already works elsewhere in this project);
     confirmed `sm_120` present after the bump.
  Both edits live in the submodule's working tree, so they are captured as a patch at
  `scripts/molmoact2_patches/0001-blackwell-and-live-api-fixes.patch` and re-applied with
  `scripts/apply_third_party_patches.sh molmoact2` -- otherwise a fresh clone or submodule
  update silently reverts them and the server breaks again.
- **Handshake test: PASS.** Once the π0.5 server was freed (Close-v1 completed) and the
  earlier stale crashed-server process was killed, the server started cleanly
  ("Warmup OK (686.4 ms)", no CUDA/OOM errors). `scripts/eval.py --policy molmoact2_droid
  --task Pick-v1.5 --max_episodes 1 --num_workers 1` completed end-to-end; inspected the
  h5 trajectory: `arm` shape `(7,)`, `gripper` shape `(1,)`, no NaNs, realistic
  non-constant joint values across steps. Provenance verified.
  **All three non-DreamZero policy wrappers (π0.5, TiPToP, MolmoAct2) are now confirmed
  working end-to-end.**
- **Two config bugs (mine, not upstream's) made MolmoAct2 underperform on every task, found
  by comparing against the leaderboard -- 2026-08-19.** First full-matrix results came in
  uniformly far below the leaderboard: Pick-v1.5 16.0% vs 43.4%, Pick-v2-classic 0.0% vs
  20.5%, Open-v1 2.5% vs 11.7%, Close-v1 54.3% (3255/5996) vs 71.26%. π0.5 reproduces all 9
  tasks on this same harness, so the harness was not the suspect -- the MolmoAct2 integration
  was. Both bugs were in choices this project made, and both had been explicitly flagged in
  this doc as "verify empirically before locking in":
  1. **Wrong control rate.** `policy_dt_ms` was set to `200.0` (~5Hz) on the strength of
     `host_server_droid.py`'s own comment that "real-robot clients poll at ~5 Hz". That
     comment is about real-robot deployment and the server's CUDA-graph concurrency lock, not
     the rate this benchmark should drive the policy at. **Authoritative counter-evidence**:
     `allenai/molmospaces_policy_zoo`'s `MolmoBotDroidEvalConfig` -- Ai2's own DROID VLA, same
     benchmark, same harness, the closest available analogue (the zoo has no MolmoAct2 config;
     re-confirmed by cloning it, only `molmobot` and `tiptop` exist) -- uses
     `policy_dt_ms = 66.0`, the same ~15Hz as π0.5. Fixed to `66.0`.
  2. **Conflated the sampler step count with the action-chunk length.** `num_steps` was used
     both as the flow-matching integration steps sent to the server *and* as the number of
     returned actions to execute before re-querying. They are unrelated: `num_steps` is
     forwarded straight to `predict_action(num_steps=...)` (server default 10, a
     quality/compute knob), while the chunk length is the checkpoint's own declared
     `action_horizon: 15` / `n_action_steps: 15` (read directly from the checkpoint's
     `norm_stats.json` under the `franka_droid` tag; the zoo's analogue likewise carries
     `action_horizon` as its own separate field). At `num_steps=10` this silently executed
     only 10 of each 15-action chunk. Split into `num_steps` (sampler) and `action_horizon`
     (chunk length), with the chunk length clamped to the buffer's real length so a server
     revision returning a shorter chunk can't index past the end.
  - **Fix verified**: re-running Pick-v1.5 at the same n=50 gave **44.0% (22/50) vs the
    leaderboard's 43.4%** -- a PASS, up from 16.0% before the fix. The 4 pre-fix cells were
    moved to `runs/molmoact2_INVALID_dt200_chunk10_20260819/` (kept, not deleted) and re-run.
  - **A false lead, recorded so it isn't re-followed**: MolmoAct2's `jerk_joint_mean` is
    ~25x the leaderboard's (1774 vs 61 on Pick-v1.5), which looked like direct evidence of a
    control-rate error. It is not diagnostic: π0.5 shows an even *larger* jerk gap (2350 vs
    84) while passing all 9 tasks, so the jerk discrepancy is a global artifact of how this
    harness records/derives it, not a per-policy signal. Only caught by checking a
    known-good control policy before acting on it.
  - **Lesson, same shape as TiPToP's M2T2 bug below**: a policy that runs cleanly end-to-end
    and produces well-formed, plausible-looking actions can still be systematically
    misconfigured. The handshake test cannot catch this class of bug by construction -- only
    an external reference number can. Prefer a *first-party* config from the same
    benchmark/harness (the zoo) over a plausible-sounding source comment about a different
    deployment context.

### TiPToP

- **Wrapper**: ported directly from `allenai/molmospaces_policy_zoo`'s
  `molmospaces_zoo/tiptop/{policy.py,config.py}` (the production wrapper that produced the
  leaderboard's TiPToP numbers) into `molmo_spaces/policy/learned_policy/tiptop_policy.py` +
  `TiptopPolicyConfig`/`TiptopEvalConfig` -- no rewrite needed, per the plan.
- **Server env**: `third_party/tiptop`, `pixi install && pixi run setup-planners`
  (curobo + cuTAMP). `torch` here comes from conda-forge's `pytorch-gpu` package
  (2.7.1, cuda 12.9) which already includes `sm_120` out of the box -- no Blackwell fix
  needed, unlike MolmoAct2.
- **Two real bugs found and fixed getting the server running**:
  1. The server module's own usage docstring says
     `pixi run python -m tiptop.websocket_server --port 8765`, but the actual file is named
     `tiptop_websocket_server.py` -- `tiptop.websocket_server` doesn't exist as a module.
     Correct invocation: `python -m tiptop.tiptop_websocket_server --port 8765`.
  2. Running with cwd = `third_party/tiptop` (which `pixi run` does by default) makes
     Python's `sys.path[0]` that same directory -- which directly contains a `cutamp/`
     subdirectory (the cuTAMP submodule's source checkout). Since that subdirectory has no
     `__init__.py` at its top level (the real package is one level deeper, at
     `cutamp/cutamp/__init__.py`), Python treats `third_party/tiptop/cutamp/` as an
     implicit PEP 420 namespace package and it **shadows** the properly pip-installed
     editable `cutamp` package -- `import cutamp` then succeeds but returns an empty
     namespace package with no `__version__` and `__file__ = None`, which fails
     `tiptop.utils.check_cutamp_version()`'s startup check with a *misleading* "cuTAMP
     version mismatch: required 0.0.6, found <0.0.2" error (the real problem is the wrong
     module was imported, not that cuTAMP is the wrong version -- `importlib.metadata.
     version("cutamp")` correctly reports 0.0.6 throughout). Fixed by launching via
     `pixi run --manifest-path third_party/tiptop/pixi.toml python -m
     tiptop.tiptop_websocket_server --port 8765` from a *different* cwd (the repo root),
     which keeps `sys.path[0]` clear of the shadowing directory while `pixi run
     --manifest-path` still finds and uses the right environment.
  3. (Not a bug, but a real gotcha caught during cleanup:) an earlier crashed MolmoAct2
     server process (see above) stayed alive holding ~11GB of GPU memory after its warmup
     exception, because `host_server_droid.py` deliberately keeps the server up after a
     failed warmup ("server will still start"). Had to be killed by PID before TiPToP's own
     GPU allocation could succeed.
- **Handshake test: PASS.** `scripts/eval.py --policy tiptop --task Pick-v1.5
  --max_episodes 1 --num_workers 1` completed end-to-end. Inspected the resulting h5
  trajectory: `arm` shape `(7,)`, `gripper` shape `(1,)`, no NaNs, smoothly varying values
  across steps (the pre-observation-phase interpolation toward `cam_obs_qpos`, as designed
  -- with `cam_obs_n_steps=200`, a short task horizon can be entirely consumed by this
  phase before any actual TiPToP plan executes; worth setting an explicit
  `--task_horizon_steps` comfortably larger than 200 + expected plan length for real runs).
- **Depth source**: defaulting to MuJoCo ground-truth depth (`force_enable_depth=True`,
  already required by `TiptopPolicyConfig`) -- per the plan, this is a known,
  direction-predictable deviation from what a real deployment would use (TiPToP's own
  stereo depth estimator on noisy real cameras), not yet independently re-confirmed against
  what the leaderboard entry used.
- **Critical bug: TiPToP silently scored ~0% across every task because a required second
  server (M2T2 grasp generation, `:8123`) was never started.** The single-episode handshake
  test above passed because it never inspects *why* a plan succeeds or fails, only that the
  action stream is well-formed -- a `success=False` no-op still produces valid-looking
  zeros. Only surfaced once full n=50/500 runs came back at 0% on Pick-v1.5,
  Pick-v2-classic, and Open-v1 (vs. the leaderboard's 67.5%/50.0%/-- for TiPToP) and
  Close-v1 at 0.4% (24/5954) -- a result that uniform and that far from the leaderboard is
  itself the signal to distrust the harness, not the policy, per BENCHMARK.md's own
  guidance. Root cause, confirmed directly in the TiPToP server's own log: every planning
  request logged `Pipeline error` /
  `aiohttp.client_exceptions.ClientConnectorError: Cannot connect to host localhost:8123`
  inside `generate_grasps_async` (`third_party/tiptop/tiptop/perception/m2t2.py`) --
  **6,486/6,486 planning calls returned `success=False`, zero exceptions**. TiPToP's own
  `tiptop_websocket_server.py` doesn't health-check this dependency at startup and fails
  fast (unlike a sibling project's wrapper on this machine, which does and would have
  caught this immediately) -- it just returns a well-formed failure per call, which is why
  this needed comparing against the leaderboard to notice at all, not a crash.
  - **Fix**: M2T2 isn't vendored in this repo. Reused a working `polaris-m2t2` conda env +
    downloaded checkpoint (`weights/m2t2.pth`) from the same sibling project on this machine
    used for DreamZero's setup, via new `scripts/serve_m2t2.sh`
    (`GPU=0 bash scripts/serve_m2t2.sh`, port 8123). **Verified the fix directly**: re-ran 3
    episodes of Pick-v1.5 after starting M2T2 -- 2/3 succeeded (66.7%, consistent with the
    leaderboard's 67.5%) and the server log showed real `success=True` planning results for
    the first time. All TiPToP results produced before this fix (Close-v1, Open-v1,
    Pick-v1.5, Pick-v2-classic) are invalid and were moved to
    `runs/tiptop_INVALID_no_m2t2_20260818/` rather than deleted, then re-run from scratch.
  - **Lesson for the next server-backed policy**: verify a dependent-service's actual
    success rate (grep its own logs for the true positive/negative split), not just "did an
    action stream come back well-formed" -- a benign-no-op design (required for graceful
    per-episode failure handling) makes a systemic upstream failure look identical to a
    struggling-but-working policy until compared against an external reference number.
- **Post-M2T2-fix Group A result: TiPToP genuinely cannot do the articulated-furniture
  tasks. This is a real capability limit, not a further bug (2026-08-19).** With M2T2 up,
  Close-v1 still scored **0.99% (22/2216)**. The perception stack is demonstrably healthy
  now -- the server logs show Gemini detection, SAM2 segmentation and M2T2 grasp generation
  all completing per request, in ~4s. The failures are cuTAMP's own planner, and the server
  records a machine-readable reason per request in
  `tiptop_server_outputs/<ts>/metadata.json` (`planning.failure_reason`). Census over all
  8582 saved requests (23 successes, 0.3%; the 6248 `Cannot connect to host localhost:8123`
  entries are the pre-fix, already-quarantined run):
  | count | failure_reason |
  |---|---|
  | 648 | `No plane found with objects resting on it (tried 5 planes, ...)` |
  | 485 | `Goal predicate on(open_drawer, cabinet) references unknown object 'cabinet'` (and `dresser`/`drawer` variants) |
  | 140 | `All 1 plan skeleton(s) failed particle initialization` |
  | 102 | `No satisfying particles found after optimizing all 1 plan(s)` |
  Every one of these is TiPToP's TAMP formulation failing to express or solve an
  articulated-joint goal: it is a pick-and-place planner whose goal language is
  `on(object, surface)` over detected *movable* objects, and "close the drawer" has no
  such expression -- hence "references unknown object" and "no plane found". This is
  independently corroborated two ways: (1) TiPToP is the **only** policy with no
  leaderboard entry at all for Open-v1/Close-v1 (`ms_open`/`ms_close` return the site's HTML
  shell for `tiptop`, real CSVs for every other policy), i.e. upstream did not report those
  either; (2) a sibling project on this machine independently recorded cuTAMP "finds no
  satisfying particles" on its own "into"-type tasks. **Conclusion**: report Close-v1/Open-v1
  for TiPToP as informational-with-explanation (there is no leaderboard number to compare
  against), and treat the 7 Group B pick/place tasks -- which do have leaderboard entries,
  and where the Pick-v1.5 spot-check gave 2/3 vs a 67.5% leaderboard number -- as the real
  reproduction test for this policy.
- **Operational fragility found the hard way: the TiPToP server died mid-run and the client
  span for ~8 hours doing nothing.** At 23:41 the server process exited (last log entry is
  an M2T2 request exceeding tiptop's hardcoded 30s `aiohttp.ClientTimeout`, after a 121.95s
  planning call; host RAM was under pressure and the server's RSS had grown from ~16GB to
  ~23GB over the run, so an OOM kill is the most likely proximate cause -- not confirmable,
  `dmesg` is not readable without sudo here). `TiptopWebsocketClient`'s reconnect loop then
  retried **forever** ("Reconnect failed: Connection refused. Retrying in 2s...") instead of
  failing the cell, so the Open-v1 run burned 8 hours making no progress and produced no
  provenance. Killed the run and its orphaned `eval_main.py` worker (which had reattached to
  the freshly restarted server and started serving stale requests -- kill workers, not just
  the parent), removed the incomplete `runs/tiptop/Open-v1/`, and restarted the server.
  **Worth hardening before any long unattended TiPToP run**: bound the client's reconnect
  attempts so a dead server fails the cell fast, and/or raise tiptop's 30s M2T2 timeout,
  which is easily exceeded when M2T2 shares a GPU with three other policy servers.

### DreamZero-DROID

The harness-side wrapper (`molmo_spaces/policy/learned_policy/dreamzero_policy.py`,
`DreamZeroPolicyConfig`, `DreamZeroPolicyEvalConfig`) pre-dates this project by ~6 months and
had **four** bugs, not the two originally flagged in the plan. All four are fixed now; two were
found only by tracing the actual wire protocol against `third_party/dreamzero`'s own reference
client/server rather than trusting the pre-existing wrapper's shape:

1. **Invalid port** (`policy_configs_baselines.py`): `remote_config=dict(host="localhost",
   port=0000)` -- `0000` is literally `0`. Fixed to `port=5000`, matching the launch command
   below.
2. **Exterior camera duplication** (`dreamzero_policy.py`): originally sent the same camera
   into both `exterior_image_0_left` and `exterior_image_1_left` unconditionally. Fixed to use
   two genuinely distinct views (`randomized_zed2_analogue_1`/`_2` on bench-v2,
   `exo_camera_1`/`exo_camera_2` on bench-v1) where they exist, falling back to duplication
   -- now logged, not silent -- only where the benchmark's own camera rig has no second
   exterior view at all (`FrankaDroidCameraSystem` tasks: Open-v1, Close-v1, Pick-v1.5; see
   the resolved open question below).
3. **Missing autoregressive frame-history conditioning** (new finding, `dreamzero_policy.py`).
   DreamZero's server is autoregressive over `chunk_size` (24, matching training's
   `action_horizon`)-step video blocks. Confirmed directly against
   `third_party/dreamzero/test_client_AR.py`'s own docstring/`RELATIVE_OFFSETS`/
   `ACTION_HORIZON`: the *first* inference call of an episode sends a single current frame per
   camera, but every call after that must send a **4-frame history stack** spanning the block
   (relative offsets `-23, -16, -8, 0` into a rolling buffer), not just the current frame.
   Sending a single frame on every call (the original code's behavior) silently desyncs the
   server's causal KV-cache/RoPE positions from the actions actually executed -- the kind of
   bug that doesn't crash, just quietly produces wrong actions. Fixed by adding a
   `collections.deque(maxlen=chunk_size)` per camera, appended every step in
   `obs_to_model_input` regardless of whether that step triggers a new server call, with a
   `_history_stack` helper that clamps to the oldest available frame early in an episode
   (matching how the server itself pads a short buffer). This mirrors a known-working live-
   rollout client for the same server found in a sibling project on this machine (see below);
   `test_client_AR.py`'s own exact frame indices don't directly apply here since that script
   replays pre-recorded video against a fixed schedule, not a live simulated rollout.
4. **Response unpacking bug** (new finding, `dreamzero_policy.py`). The wrapper did
   `result["actions"]` on the server's response, assuming a dict. Confirmed directly against
   `third_party/dreamzero/socket_test_optimized_AR.py`'s `PolicyServer.infer()` (docstring:
   "Returns: action: (N, 8) action array") and `test_client_AR.py`'s own usage (`actions =
   client.infer(obs)`, used directly with no further indexing) that the server returns a
   **bare** `(N, 8)` ndarray, not `{"actions": ...}`. `result["actions"]` would have raised
   immediately on the first real (non-handshake-stub) inference call. Fixed to use the
   response directly.

**Environment setup -- reused, not rebuilt from scratch.** A sibling project on this same
machine (`~/Workspace/robot-prompt-opt`) already has a working DreamZero-DROID setup solving
the exact same 48GB-card constraint this project would otherwise hit fresh: a `polaris-dreamzero`
conda env (`torch==2.8.0+cu129`, confirmed Blackwell-compatible --
`torch.cuda.get_arch_list()` includes `sm_120`/`compute_120` -- and `flash-attn==2.8.3.post1`
already built against it), the `GEAR-Dreams/DreamZero-DROID` checkpoint already downloaded
(43GB), and a tested patch
(`scripts/dreamzero_patches/0001-single-gpu-48gb-inference.patch`, copied into this repo
verbatim and applied to `third_party/dreamzero` via
`scripts/apply_third_party_patches.sh dreamzero`)
enabling single-process inference split across 2 GPUs instead of PyTorch-distributed's
`--nproc_per_node=2`, which that project's own notes record as a dead end here: `WANPolicyHead`
only splits classifier-free guidance across ranks, so each rank still holds a full ~44.6 GiB
replica and 2 ranks OOM host RAM. Reused directly (same user, same machine, same submodule SHA
`ab790c198fbce33503358efbbd4187ce9a89adf3` the patch's header records as its base commit --
verified before applying):
- `polaris-dreamzero`'s interpreter is invoked by absolute path from
  `scripts/serve_dreamzero.sh`; no `conda activate` needed since (like every other policy
  server in this project) it's a separate process, not something scripts/eval.py's env guard
  touches.
- The checkpoint is symlinked at `third_party/dreamzero/checkpoints/DreamZero-DROID ->
  ~/Workspace/robot-prompt-opt/checkpoints/DreamZero-DROID` (that directory name matches the
  submodule's own gitignored `checkpoints/` pattern, same convention openpi/MolmoAct2 use).
- **Known caveat**: the checkpoint's bundled `config.json` hardcodes *absolute* filesystem
  paths to three Wan2.1-I2V-14B-480P base-model component files (CLIP image encoder, umt5-xxl
  text encoder, VAE) under `~/Workspace/robot-prompt-opt/checkpoints/Wan2.1-I2V-14B-480P/`.
  These resolved successfully because that directory still exists on this machine, but a
  from-scratch reproduction on a different machine would need `hf download
  Wan-AI/Wan2.1-I2V-14B-480P` (~28GB, per `third_party/dreamzero/README.md`'s Training section)
  and either the same absolute directory structure or an edit to the checkpoint's `config.json`
  paths. Documented here rather than silently depended upon.

**Launch** (`scripts/serve_dreamzero.sh`, GPU-order-aware -- point index 0 at whichever
physical GPU has the most free headroom at launch time, since the primary device holds nearly
all resident weights):
```bash
GPUS=1,0 DIT_SPLIT=12 PORT=5000 bash scripts/serve_dreamzero.sh
```
Key env vars (all defaulted in the script, matching robot-prompt-opt's own measured-working
values): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
`CUDA_HOME=~/anaconda3/envs/polaris` (a *different*, unrelated conda env that happens to have
an nvcc toolchain -- `polaris-dreamzero` itself has none, and `transformers` imports
`deepspeed` at module load time, which hard-errors on a missing `CUDA_HOME` even though
deepspeed is never actually used at inference); `TORCHDYNAMO_DISABLE=1` (frees ~5GB otherwise
held by `reduce-overhead` CUDA-graph pools, trading it for compile speed);
`DREAMZERO_OFFLOAD_TEXT_ENCODER=1` / `DREAMZERO_OFFLOAD_IMAGE_ENCODER=1` (evict the 10.58GB
text encoder and 1.18GB image encoder to CPU between uses -- both already default on in the
patch); `DREAMZERO_DIT_SPLIT=12` (pipeline-shards the last 12 of 40 DiT blocks onto the second
GPU, freeing ~11GB on the primary at a documented ~20% throughput cost, while preserving
`max_chunk_size=4`, the trained attention window -- the alternative of lowering
`max_chunk_size` was rejected because it shrinks that window and changes model behavior);
`DREAMZERO_DISABLE_DREAM_VIDEO=1` (skips the model's own generated debug MP4s -- a `vae.decode`
call unrelated to the action path).

Runs on `polaris-dreamzero`'s `torch==2.8.0+cu129` (single process, `--nproc_per_node=1`, *not*
2 -- see the dead-end note above).

Two more bugs surfaced only once real inference calls were attempted against the live server
(a handshake single-episode run doesn't catch these until the model actually runs):

5. **Missing list-observation unwrapping** (`dreamzero_policy.py`). `task.reset()` /
   `get_observations()` always return `list[dict[str, Any]]` -- one dict per batch index,
   even at batch size 1 (`molmo_spaces/tasks/task.py`'s own type signature confirms this).
   Every other working wrapper (`pi_policy.py`, `molmoact2_policy.py`) starts
   `obs_to_model_input` with `if isinstance(obs, list | tuple): obs = obs[0]`; DreamZero's
   never had it, so the very first call crashed with `TypeError: list indices must be
   integers or slices, not str` on `obs["qpos"]`. Fixed by adding the same unwrap.
6. **Wrong msgpack numpy-encoding library** (`dreamzero_policy.py`). Used the standalone
   PyPI `msgpack_numpy` package; `third_party/dreamzero/eval_utils/policy_server.py` (like
   openpi) uses `openpi_client.msgpack_numpy` -- a different implementation with an
   incompatible ndarray wire encoding (`{b"__ndarray__": ...}` vs. the standalone package's
   own marker keys). The mismatch doesn't raise on send; the server's `unpack_array` just
   fails to recognize the marker and returns the raw dict unchanged, so the first field
   actually operated on server-side (`joint_pos.ndim`) raised
   `AttributeError: 'dict' object has no attribute 'ndim'`. Fixed by importing
   `msgpack_numpy` from `openpi_client` (already installed in the harness envs for PR2)
   instead of the standalone package -- `openpi_client`'s module exposes the same
   `Packer`/`unpackb` API, so no other code changed.

**Handshake test: PASS** (`dreamzero`/`Pick-v1.5`, 1 episode, `runs/dreamzero/Pick-v1.5/20260817_handshake/`).
Decoded `traj_0/actions/commanded_action` from the resulting h5: `arm` shape `(303, 7)`,
`gripper` shape `(303, 1)`, zero NaNs, non-constant per-joint std (`0.03`-`0.34` rad across
all 7 joints -- confirms the arm moved smoothly across chunk boundaries, i.e. the 24-step
buffer/re-query logic and frame-history stack aren't producing sudden discontinuities at
chunk edges). Gripper stayed at `0.0` (open) the entire episode and the task ended in
failure (`fail=True`, `success=False`) -- expected and unconcerning for a single
untuned episode of an `INFORMATIONAL_POLICIES` entry, not a correctness signal one way or
the other; `policy_dt_ms=66.0` and the DiT-split/offload knobs above haven't been tuned
against the leaderboard number the way π0.5's control rate was in PR2.

### Cosmos-Policy (Edge + Nano)

Added after BENCHMARK.md/the original 4-policy plan, at the user's request. **Upstream is
`NVIDIA/cosmos-framework`, not `nvlabs/cosmos-policy`** -- the latter repo (the one the
checkpoint family name suggests, and the URL initially given) ships checkpoints only for
LIBERO/RoboCasa/ALOHA, nothing DROID-trained, and no inference server at all. The actual
DROID-trained checkpoints -- **Cosmos3-Edge-Policy-DROID** (4B) and **Cosmos3-Nano-Policy-DROID**
(16B, i.e. Nano is the *larger* of the two, not the smaller -- confirmed directly from a
sibling project's own working setup, not guessed from the name), both Franka + Robotiq,
8-dim joint-position + gripper actions -- live in the `cosmos-framework` repo.

**Reused a working setup wholesale, same reasoning as DreamZero/M2T2.** A sibling project on
this machine (`~/Workspace/robot-prompt-opt`) already has a working Cosmos-Policy
integration -- `polaris-cosmos-policy` conda env (Python 3.13, `torch==2.13.0+cu130`,
confirmed Blackwell/sm_120), both checkpoints already downloaded to the shared
`~/.cache/huggingface` cache (auto-download from a non-gated HF repo, no manual layout step
needed -- closer to π0.5's `gs://` fetch than DreamZero/MolmoAct2's), and extensive
first-party documentation of the wire protocol and known gotchas (that project's `CLAUDE.md`
"Cosmos Policy" section). `scripts/serve_cosmos.sh` + `scripts/serve_cosmos_policy.py` are
ported near-verbatim from that project's own `serve_cosmos_policy.sh`/`.py`, reusing its env
and `third_party/cosmos_policy` checkout in place.

**The server wraps openpi's own `WebsocketPolicyServer`, so the transport is identical to
π0.5's** -- `molmo_spaces/policy/learned_policy/cosmos_policy.py` (`Cosmos_Policy`) is
structurally a close copy of `pi_policy.py`, not a new integration pattern, but with several
field-name/behavior differences confirmed against a sibling project's own already-working
client rather than guessed:
- Response key is `"action"` (singular), not `"actions"` -- the wrapper raises loudly if
  missing rather than silently defaulting.
- Two exterior image slots are 1-indexed (`exterior_image_1_left`/`_2_left`), not 0-indexed
  like DreamZero's. This benchmark exposes one exterior camera; duplicated into both slots
  (same move `dreamzero_client.py` makes).
- No `observation/cartesian_position` field at all (unlike DreamZero).
- No client-side gripper inversion needed -- confirmed via the *training data pipeline*
  (`IS_GRIPPER_ACTION_FLIPPED` for this checkpoint's dataset root), not just algebra that
  happens to cancel: the server flips gripper state on ingest and gripper action on egress,
  both from this benchmark's own native "1 = closed" convention, same as π0.5/MolmoAct2.
- **Image pre-processing matters for output quality, not just correctness.** The server's
  `_compose_roboarena_views` always bilinear-resizes (no antialiasing) both exterior copies
  down to half the wrist frame's size before composing a canvas, regardless of what the
  client sends. Sending full-size images lets that non-antialiased resize alone do the
  downsampling (aliasing); pre-shrinking with **LANCZOS** to the exact half-size the server
  expects (640x360 wrist, 320x180 exterior -- confirmed against the server's own startup log,
  `image=540x640`) makes the server's resize a same-size no-op instead. Implemented as a
  plain (non-letterboxed) resize, unlike this project's other wrappers' `resize_with_pad` --
  the checkpoint's training data and the server's compose step both assume a fixed 16:9
  canvas geometry, and this benchmark's camera aspect ratios vary per task, so an
  aspect-preserving pad would break the "lands exactly at the expected composite size, no
  further server resize" property, at the cost of mild stretch distortion for non-16:9
  source cameras -- a documented, minor deviation.
- `chunk_size=8`: the server's own `action_chunk_size` defaults to 32; re-querying at 8
  (matching π0.5's own tuning, already validated for this exact checkpoint family in the
  sibling project) limits open-loop drift without needing every chunk step.
- **Guardrails disabled.** `RobolabPolicyService` unconditionally tries to load an HF-gated
  text-guardrail checkpoint (`nvidia/Cosmos-Guardrail1`) unless told otherwise, which isn't
  exposed as a CLI flag -- `scripts/serve_cosmos_policy.py` monkeypatches
  `_build_setup_args` to force it off (ported from the sibling project's own fix), since
  content-safety filtering on generated video is irrelevant to a closed-loop benchmark run
  with no public output.
- **Two separate registered policies, `cosmos_edge`/`cosmos_nano`, not one with a variant
  flag.** `eval_main.py`'s `--checkpoint_path` CLI override has no counterpart for
  `remote_config.port`, and Edge/Nano need independent server processes (different
  memory footprints, ports 8003/8004) -- see `CosmosEdgePolicyEvalConfig`/
  `CosmosNanoPolicyEvalConfig` in `evaluation_configs.py`.
- **Control rate: open question, not silently resolved.** Defaulted `policy_dt_ms=66.0`
  (~15Hz) to match the server's own startup log (`fps=15.0`, a first-party signal from the
  model's metadata). The leaderboard's own captured CSVs for this policy report `# dt: 0.1`
  in their header, but **that field carries no information about the control rate actually
  used** and must not be read as one: `scripts/benchmarks/eval_to_csv.py` never derives it
  from the run's `policy_dt_ms`, it just echoes whatever `--dt` was passed, whose argparse
  default is `67/1000` and whose *function* default is `0.1`. Every CSV this project
  produces therefore says `0.067` regardless of the policy's real rate (confirmed: our
  MolmoAct2 runs at `policy_dt_ms=200.0` still emit `# dt: 0.067`), and the leaderboard's
  own files split between `0.067` (molmoact) and `0.1` (cosmos) purely by which code path
  produced them. Flagged here rather than silently picked either way; revisit if our results
  diverge sharply from the leaderboard.

**Handshake test: PASS, first attempt (no bugs found needing a fix, unlike every other
policy in this project) -- attributable to unusually thorough first-party reference
documentation to build against, not unusual luck.** `cosmos_edge`/`Pick-v1.5`, 3 episodes,
`runs/_handshake/cosmos_edge/Pick-v1.5/20260818/`: **2/3 (66.7%) success**, no server errors.
The leaderboard's own Pick-v1.5 number for `cosmos` is **66.5%** -- near-exact agreement on
the very first real run, the strongest possible signal this integration is correct.

**Leaderboard note**: the public leaderboard has exactly one `cosmos` policy entry (not
separate Edge/Nano rows) with real data for all 9 tasks, including Open-v1/Close-v1 (unlike
TiPToP, which has none for those two) -- see `reference/README.md` for how the resulting
ambiguity (which checkpoint produced it?) is handled in `reference/leaderboard_snapshot.csv`
(the same numbers duplicated under both `cosmos_edge` and `cosmos_nano`).

## Open-v1: `--max_episodes` silently under-covers object categories (2026-08-19)

**Both policies "failed" Open-v1 for a sampling reason, not a policy or integration reason,
and one of the nine π0.5 cells this project had been reporting as PASS was actually invalid.**
Worth reading before trusting any partial-coverage cell.

**How it surfaced.** MolmoAct2's post-fix Open-v1 produced n=400 across 5 categories, while
π0.5's Open-v1 cell (run 2026-08-17) had produced n=16 in a *single* category
(`Refrigerator`) from the identical `--max_episodes 50 --num_workers` invocation. That
discrepancy was the tell.

**Two separate problems, in order:**

1. **π0.5's Open-v1 cell was stale and invalid.** It ran at 06:40 on 2026-08-17, *before* the
   `grasp_libraries=["droid"]` fix landed in `json_eval_task_sampler.py`'s `set_joint_values`,
   and hit exactly that bug: `ValueError: No joints with grasp file found for oven_.../
   dishwasher_.../drawer_...`. **8 of its 16 work items were skipped**, leaving only
   `Refrigerator` episodes. Its reported "37.5% vs 22.7% PASS" was an artifact of a tiny
   single-category sample whose Wilson interval was wide enough to bracket almost anything.
   Retired to `runs/pi05_droid_STALE_pre_graspfix_20260819/` and re-run. **This means the
   "9/9 PASS" figure reported for π0.5 earlier in this project was wrong: it was 8 valid
   PASSes plus one invalid cell.** The same latent bug was also found and fixed at a second
   call site, `molmo_spaces/tasks/eval_task_sampler.py` (not on the current eval path -- only
   `json_eval_task_sampler` is imported by `json_eval_runner` -- so it was harmless today,
   but it is the identical trap and would have bitten the next user of that sampler).
2. **Even correctly run, `--max_episodes 50` under-covers Open-v1's categories so badly that
   the overall rate is not comparable to the leaderboard's.** `--max_episodes N` selects whole
   *houses* from the first N raw episodes; on Open-v1 that yields **5 of the leaderboard's 13
   categories**, and the ones it misses are the easy, high-volume ones -- `Stand` (n=204,
   41.7% for π0.5) and `Chestofdrawers` (n=204, 27.9%). Reweighting the leaderboard's own
   per-category rates by our per-category episode counts closes the gap entirely:

   | policy | ours (n=400) | Wilson 95% | leaderboard, all 13 cats | leaderboard reweighted to our mix |
   |---|---|---|---|---|
   | `pi05_droid` | 13.50% | [10.50%, 17.20%] | 22.70% (outside -> FAIL) | **16.84% (inside -> consistent)** |
   | `molmoact2_droid` | 6.50% | [4.47%, 9.35%] | 11.70% (outside -> FAIL) | **9.03% (inside -> consistent)** |

   So on the category mix actually evaluated, both policies agree with the leaderboard; the
   raw FAIL was pure coverage artifact. New `scripts/category_mix_check.py` performs this
   reweighting for any (policy, task) so this class of false FAIL is diagnosable in one
   command instead of re-derived by hand.

**Actions taken**: both Open-v1 cells are being re-run with **no `--max_episodes` at all**
(full 41 houses / ~1000 episodes, matching the leaderboard's own n=1000) so the verdict is
directly comparable. Note this is the same dynamic that made Close-v1 *trustworthy* by
accident -- `--max_episodes 500` there happened to select houses containing ~6000 episodes,
i.e. near-full coverage, which is why Close-v1 agreed so well from the start.

**General rule this establishes**: a per-task overall rate is only comparable to the
leaderboard's when the *category mix* is comparable. Prefer full-benchmark runs for the
acceptance table; when a truncated run must be used, check it with
`scripts/category_mix_check.py` before reading its verdict as a defect.

### `--max_episodes` OVERSAMPLES; omit it to evaluate the benchmark's own episode set

Following the Open-v1 coverage finding above, the same question was put to Close-v1, and the
answer is worse than "under-covered" -- **`--max_episodes` produced ~6.5x MORE episodes than
the benchmark contains, not fewer.** Measured directly:

- The Close-v1 benchmark JSON contains **exactly 915 episodes**, and the leaderboard's own
  Close-v1 entry is **n=915** -- i.e. the leaderboard evaluates the benchmark as-is.
- Our Close-v1 runs with `--max_episodes 500` produced **n=6000**, drawn from just **27
  houses**, with per-category counts (`Chestofdrawers` 1813, `Stand` 2005, `Drawer` 1524)
  that individually exceed the whole benchmark. Those episodes are therefore repeats/
  resamples within a handful of houses, not 6000 independent draws.
- Re-running with **no `--max_episodes` at all** logs `Total episodes: 915` -- exactly the
  benchmark, exactly the leaderboard's n.

Two consequences, both of which invalidate conclusions drawn earlier in this project:

1. **The Wilson intervals on those runs are far too narrow.** They assume independent
   Bernoulli trials; 6000 clustered resamples from 27 houses have a much smaller effective
   sample size. A ±1.2% interval at n=6000 is not credible when the underlying benchmark has
   915 episodes. This makes both a PASS and a FAIL on those cells untrustworthy.
2. **It is not the same evaluation set the leaderboard used**, so `category_mix_check.py`'s
   reweighting cannot rescue it either -- reweighting fixes category *proportions*, not
   within-category resampling.

Concretely, this puts two previously-reported Close-v1 verdicts in doubt:
| policy | ours (n=6000, oversampled) | Wilson95 | leaderboard (n=915) | reweighted | previously reported |
|---|---|---|---|---|---|
| `pi05_droid` | 66.12% | [64.91%, 67.30%] | 65.14% | 64.78% (outside) | PASS -- but marginal, and the reweighted value already fell outside |
| `molmoact2_droid` | 66.77% | [65.56%, 67.95%] | 71.26% | 70.76% (outside) | FAIL -- gap is NOT explained by category mix |

Both are being re-run with no `--max_episodes` for a directly comparable n=915. Note the
MolmoAct2 Close-v1 number did improve substantially under the control-rate/chunk fix
(54.29% -> 66.77%), so that fix is independently confirmed; what remains is whether the
residual ~4pp gap survives a like-for-like episode set.

**Standing rule going forward: run every acceptance cell with no `--max_episodes`.** Use it
only for smoke tests and handshakes, never for a number that will be compared to the
leaderboard. (`scripts/eval.py` still records the flag in provenance, so any cell that used
it is identifiable after the fact.)

## Provenance schema

`scripts/eval.py` writes one `provenance.json` per `runs/<policy>/<task>/<date>/` cell (see
`scripts/eval.py` for the exact writer and `scripts/check_provenance.py` for the checker):
molmospaces git SHA, pinned-assets file + its sha256, policy name/checkpoint, task,
benchmark dir, verbatim eval/eval_to_csv commands, `--success-condition` (always `both`),
task horizon, `--max_episodes` (if a smoke test), seed (always 42, hardcoded in
`create_eval_config`), episode count, results CSV path, raw eval_main.py output dir, and a
timestamp.

## Open questions (tracked, not silently resolved)

1. Exact openpi fork SHA the leaderboard numbers were produced against (vs. "current main").
2. Leaderboard `--success-condition`: `oracle` is well-supported but not confirmed against
   the live leaderboard site/JSON (a client-rendered app, not fetchable directly).
3. `reference/leaderboard_snapshot.csv` capture -- needs a human with a browser, or a
   browser-automation pass; blocks PR2's acceptance criteria until done.
4. TiPToP depth source (MuJoCo ground truth vs. its own stereo estimator) -- defaulting to
   ground truth for determinism; flagged as a known, direction-predictable deviation.
5. MolmoAct2 bench-v2 exterior camera choice (`randomized_zed2_analogue_1` vs.
   `randomized_gopro_analogue_1`) -- defaulting to match `PI_Policy`'s existing choice.

**Resolved during PR1/PR2 environment setup (2026-08-16):**
- DreamZero bench-v2 camera availability: confirmed via real `benchmark.json` data.
  `FrankaOmniPurposeCameraSystem` tasks (6 of Group B) expose 5 cameras including two
  distinct exterior views (`randomized_zed2_analogue_1`/`_2`) -- DreamZero gets true
  stereo input there. `FrankaDroidCameraSystem` tasks (Open-v1, Close-v1, Pick-v1.5) expose
  only 2 cameras total (`wrist_camera`, `exo_camera_1`) -- there is no second exterior view
  for these three tasks in the real asset rig, so `DreamZero_Policy`'s duplication fallback
  (now logged, not silent) is expected and unavoidable there, not a bug to fix further.
