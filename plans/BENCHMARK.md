# Stand up MolmoSpaces and reproduce DROID leaderboard numbers across six policies

## Goal
Get `molmospaces` installed and running locally, then evaluate six DROID-embodiment policy
configurations from five vendored repos — `third_party/openpi` (π0.5-DROID),
`third_party/tiptop`, `third_party/dreamzero`, `third_party/molmoact2`, and
`third_party/cosmos` (Cosmos Policy **Edge** and **Nano**) — across 9 MolmoSpaces-Bench
tasks. Success means the numbers we produce for TiPToP, MolmoAct2-DROID, and π0.5-DROID
are statistically consistent with the public leaderboard, so that any *new* policy we
later evaluate in this harness can be trusted and compared. DreamZero and the two Cosmos
variants are additional data points; they are compared against the leaderboard where an
entry exists but are not part of the pass/fail reproduction bar (see *Open questions*).

Reproduction target: our per-task 95% credible/confidence interval must overlap the
leaderboard point estimate. Exact per-episode agreement is not expected.

Environment parity: every environment this repo builds — harness and per-policy — is
sourced from `robot-prompt-opt` rather than specified independently, so that a policy
behaves identically in both projects and results transfer without re-validation. Parity is
a first-class deliverable here, not a nice-to-have: the whole point of reproducing known
leaderboard numbers is to license comparisons, and a comparison across two repos with
silently divergent CUDA or transformers pins isn't licensed by anything.

## The 9 tasks

Group A — two MolmoSpaces Combined articulated tasks (bench-v1, **classic renderer**):

| # | Leaderboard name | `--benchmark_dir` (relative to assets root) |
|---|---|---|
| 1 | Open-v1 | `molmospaces-bench-v1/ithor/FrankaOpenDataGenConfig/FrankaOpenDataGenConfig_20260123_json_benchmark` |
| 2 | Close-v1 | `molmospaces-bench-v1/ithor/FrankaCloseDataGenConfig/FrankaCloseDataGenConfig_20260123_json_benchmark` |

Group B — the **complete** MolmoBot Combined set (bench-v2):

| # | Leaderboard name | Renderer | `--benchmark_dir` + flags |
|---|---|---|---|
| 3 | Pick-v1.5 (MSProc) | classic | `molmospaces-bench-v2/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231` |
| 4 | Pick-v2-classic | classic | `molmospaces-bench-v2/procthor-objaverse/FrankaPickHardBench/FrankaPickHardBench_20260206_json_benchmark` |
| 5 | Pick-v2-filament | filament | same dir as #4, plus `--use-filament` |
| 6 | Pick-v2-RandCam | filament | same dir as #4, plus `--use-filament --camera_names randomized_zed2_analogue_1 wrist_camera_zed_mini` |
| 7 | Pick & Place-v2 | filament | `molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceHardBench/FrankaPickandPlaceHardBench_20260206_json_benchmark` + `--use-filament` |
| 8 | PnP-NextTo-v2 | filament | `molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceNextToHardBench/FrankaPickandPlaceNextToHardBench_20260305_json_benchmark` + `--use-filament` |
| 9 | PnP-Color-v2 | filament | `molmospaces-bench-v2/procthor-objaverse/FrankaPickandPlaceColorHardBench/FrankaPickandPlaceColorHardBench_20260304_json_benchmark` + `--use-filament` |

Two consequences of this exact split, both of which shape the runner:

- **The renderer is not a property of the benchmark set.** Four tasks (#1–#4) run classic,
  five (#5–#9) run filament, and the split cuts through the middle of bench-v2. Whatever
  falls out of assumption 3 below about `mujoco` vs `mujoco-filament` coexistence now
  affects both halves of the campaign, not a tail of it.
- **Three tasks share one benchmark JSON.** Pick-v2-classic, Pick-v2-filament, and
  Pick-v2-RandCam all point at `FrankaPickHardBench_20260206_json_benchmark` and differ
  only by flags; separately, Pick-v1.5 reuses the `FrankaPickDroidMiniBench_..._20251231`
  JSON name that also exists under bench-v1 as Pick-v1.1. So the runner must key run
  directories and CSV rows on **task name**, never on benchmark path — otherwise three
  results silently overwrite each other and Pick-v1.5 gets confused with a bench-v1 task.
  Assert distinct output paths in the runner rather than trusting convention.

Because Group B is the complete MolmoBot Combined set, its 7-task average is directly
comparable to the leaderboard's *MolmoBot Combined* column — a stronger check than any
single cell. Group A is only two of the four MolmoSpaces Combined tasks, so compare those
per task and do **not** compute a "MolmoSpaces Combined" average from them.

Excluded: Pick-v1 and Pick & Place-v1 from MolmoSpaces Combined. Our overall average is
therefore not the leaderboard's *All Combined* number; don't report it as such.

Paths above come from the published docs
([ms-bench](https://allenai.github.io/molmospaces/ms-bench/),
[mb-bench](https://allenai.github.io/molmospaces/mb-bench/)); the mb-bench page was last
updated 2026-04-23, so treat the Group B paths as *assumptions to verify* against
`docs/mb-bench.md` in the checked-out repo. Note the mb-bench docs write these paths as
`$MLSPACES_ASSETS_DIR/benchmarks/...` while ms-bench uses a repo-relative
`assets/benchmarks/...` — resolve both through `MLSPACES_ASSETS_DIR` in the runner.

## Acceptance criteria
- [ ] A single command (`scripts/eval.py`) runs any (policy, task)
      cell and is resumable — re-running skips cells with completed output.
- [ ] All 6 policy configurations × 9 tasks (54 cells) produce a per-episode results CSV
      via `scripts/benchmarks/eval_to_csv.py`, with zero episodes in an `error`/crashed state.
- [ ] For TiPToP, MolmoAct2-DROID, and π0.5-DROID on each of the 9 tasks: our 95%
      interval on the success rate overlaps the recorded leaderboard value. Failures are
      allowed only if accompanied by a written diagnosis (see *Validation*).
- [ ] Our 7-task Group B average for each of those three policies overlaps the
      leaderboard's *MolmoBot Combined* value, and the rank order of the three matches.
      This is the load-bearing check — a whole-set aggregate over ~7×N episodes has a much
      tighter interval than any single cell.
- [ ] `reference/leaderboard_snapshot.csv` is committed, with the date pulled and the
      embodiment filter used, so future runs compare against a frozen target.
- [ ] Every environment is defined by an artifact shared with `robot-prompt-opt` (lockfile,
      `environment.yml`, `pixi.lock`, or image digest — whichever that repo already uses),
      not by a hand-written spec in this repo. No environment is created with a bare
      `pip install <pkg>` that isn't reflected in a shared lock.
- [ ] `scripts/check_env_parity.py` exits zero: for each shared env, the resolved package
      set here matches `robot-prompt-opt`'s. Any intentional divergence is listed in
      `docs/env_parity.md` with a reason, not silently tolerated.
- [ ] At least one (policy, task) cell run in both repos' environments produces identical
      per-episode outcomes given identical seeds. Matching package lists don't prove
      matching behaviour; this does.
- [ ] `docs/eval_reproduction.md` records, per policy: env recipe, checkpoint revision
      (HF SHA or GCS path), server launch command, control rate, and camera mapping.
- [ ] Every run's provenance is captured: molmospaces git SHA, asset versions,
      `--success-condition`, task horizon, seeds.

## Out of scope
- Training or fine-tuning any policy. Inference only — no Cosmos post-training,
  distillation, or DCP checkpoint conversion, and no Cosmos3-Super variant.
- `Pick-v1` and `Pick & Place-v1`; navigation tasks; MolmoBot's own policies.
- Real-robot / RoboArena evaluation.
- Submitting results to the public leaderboard.
- Isaac and ManiSkill backends — benchmarking is MuJoCo-only.
- Optimizing throughput beyond what's needed to finish in reasonable wall-clock.

## Assumptions to verify first
Do these before writing code; several of them reshape later steps.

0. **Assumed:** `robot-prompt-opt` is checked out somewhere reachable (sibling directory,
   submodule, or a known remote) and already defines environments for at least some of
   these policies — DreamZero in particular, given that project's subject matter.
   **Check:** locate it; read its env tooling (`uv.lock` / `pyproject.toml`,
   `environment.yml`, `pixi.lock`, `Dockerfile`, `requirements*.txt`); list which of
   {molmospaces harness, openpi, molmoact2, tiptop, dreamzero, cosmos} it already covers,
   and whether envs are per-policy or one shared env. **This determines the parity
   mechanism** (see *Approach*), so do it before creating a single environment — retrofitting
   parity after the fact means rebuilding everything. **If it isn't reachable:** stop and
   ask, rather than building independent envs that will have to be torn down.
1. **Assumed:** `third_party/{openpi,tiptop,dreamzero,molmoact2,cosmos}` already exist as
   submodules or vendored checkouts. **Check:** `cat .gitmodules`, `git submodule status`,
   `ls third_party/`. **If wrong:** add them (see *Upstreams* below) and record pinned SHAs.
2. **Assumed:** `third_party/openpi` is the `omarrayyann/openpi` fork, which carries the
   `*_jointpos` DROID configs the MolmoSpaces docs call for — not upstream
   `Physical-Intelligence/openpi`. **Check:** `git -C third_party/openpi remote -v` and
   grep the config registry for `pi05_droid_jointpos`. **If wrong:** switch remotes; the
   base-frame/EE-action upstream configs will not match the leaderboard entry.
3. **Assumed:** `mujoco` and `mujoco-filament` extras are mutually exclusive installs
   (filament ships as a separate distribution from test.pypi). **Check:** try
   `pip install -e ".[mujoco,mujoco-filament]"` in a scratch env; inspect `pyproject.toml`.
   **If they conflict:** build two harness envs — `mlspaces-classic` for Open-v1,
   Close-v1, Pick-v1.5, Pick-v2-classic and `mlspaces-filament` for the five remaining
   filament tasks — and make the runner select the env from the task table, not from the
   benchmark path. Plan for this outcome; it splits the campaign 4/5 rather than being a
   minor tail case. Also verify the two envs produce *identical* results on a shared
   benchmark: Pick-v2-classic and Pick-v2-filament use the same JSON, so any difference
   beyond the renderer (physics, asset versions, seeds) would be a bug, not a finding.
4. **Assumed:** the leaderboard's reported metric is the *oracle* success condition
   (`eval_to_csv.py --success-condition oracle`), not `both`. **Check:** the leaderboard
   doc linked from the evaluation guide, and `scripts/benchmarks/eval_to_csv.py --help`.
   **If wrong:** emit both columns and compare against the right one — this single flag
   can move numbers by several points.
5. **Assumed:** existing wrappers for some of these policies already live in
   [`allenai/molmospaces_policy_zoo`](https://github.com/allenai/molmospaces_policy_zoo)
   (extras-per-policy layout, e.g. `pip install -e .[molmobot]`). **Check:** list
   `molmospaces_zoo/` and the `pyproject.toml` extras. **If a wrapper exists, use it** —
   the leaderboard numbers were likely produced with it, and rewriting it is the fastest
   way to fail reproduction.
6. **Assumed:** benchmark episodes expose one exterior + one wrist camera
   (`exo_camera_1`, `wrist_camera`) in bench-v1 and zed-named cameras in bench-v2.
   **Check:** read one `benchmark.json` per benchmark and list the `cameras[].name`
   values. **If confirmed:** DreamZero (trained on three DROID views:
   `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left`) needs an explicit
   mapping decision, recorded in the docs. The Cosmos DROID recipe has a `concat_view` knob
   and a `use_state` flag, so it carries the same question — resolve both the same way and
   write the mapping down per policy.
7. **Assumed:** `MLSPACES_FORCE_INSTALL` defaults to `True`, i.e. assets get overwritten
   on every resource-manager instantiation. **Check:** `molmo_spaces_constants.py`.
   **If confirmed:** export `MLSPACES_FORCE_INSTALL=False` after the first install and
   write a `MLSPACES_PINNED_ASSETS_FILE` so asset versions are frozen across the campaign.
8. **Assumed:** `third_party/cosmos` is `NVIDIA/cosmos-framework`, not `NVIDIA/cosmos`.
   These are two different repos and the distinction decides whether anything works: the
   policy server we need (`cosmos_framework.scripts.action_policy_server_robolab`) and the
   DROID action-policy recipes live in **cosmos-framework**; `NVIDIA/cosmos` is the model
   platform and cookbook. **Check:** `git -C third_party/cosmos remote -v`; confirm
   `python -c "import cosmos_framework"` resolves. **If wrong:** vendor cosmos-framework
   too, or repoint. Note the repo bundles AgentSkills under `.claude/skills/` covering
   install, checkpoint download, inference, and error triage — read those before
   hand-rolling any setup, they are maintained alongside the code.
9. **Assumed:** the Nano policy checkpoint is published as `nvidia/Cosmos3-Nano-Policy-DROID`
   (the Cosmos 3 technical report names it) and Edge as `nvidia/Cosmos3-Edge-Policy-DROID`
   (confirmed in NVIDIA's Edge release post). **Check:** resolve both on HF and record the
   revision SHAs. **If the Nano policy checkpoint isn't public**, that variant becomes
   blocked, not buildable — post-training it ourselves is explicitly out of scope. Raise it
   rather than substituting the base `Cosmos3-Nano` checkpoint, which is not a policy.

Upstreams, for reference: `allenai/molmospaces`, `omarrayyann/openpi`,
`tiptop-robot/tiptop`, `dreamzero0/dreamzero`, `allenai/molmoact2`,
`NVIDIA/cosmos-framework` (and `NVIDIA/cosmos` for model cards).

## Approach

**One harness environment, five isolated policy servers.** MolmoSpaces' own evaluation
guide already models this: `eval_main.py` drives MuJoCo in the `mlspaces` env while the
policy runs behind a socket server in its own environment. That is the only workable shape
here, because the stacks are irreconcilable — openpi is JAX under `uv`, DreamZero is torch
≥2.8 / CUDA 12.9 / flash-attn launched through `torch.distributed.run`, MolmoAct2 rides the
HF/LeRobot transformers stack, TiPToP is a `pixi` project built on cuTAMP, and
cosmos-framework wants CUDA 13.0 with `uv sync` groups and Python 3.13. Do not attempt a
merged environment; the resolver fight will eat days and any "solution" silently changes
versions relative to whatever produced the leaderboard entry.

Each policy therefore gets: (a) its own env under `envs/<policy>/` or its native tool
(`uv` / `conda` / `pixi`), (b) a documented server launch command, (c) a thin
`InferencePolicy` subclass + `BasePolicyConfig` + `JsonBenchmarkEvalConfig` living in the
harness env that speaks to it over the socket. The
[external-repo integration recipe](https://allenai.github.io/molmospaces/evaluation_guide/)
in the docs is the template; `PiPolicyEvalConfig` in
`molmo_spaces/evaluation/configs/evaluation_configs.py` is the worked example to copy.

**Environments come from `robot-prompt-opt`; this repo consumes, it does not specify.**
The isolation story above says *how many* environments exist and what talks to what; it
says nothing about which versions go in them, and that is where parity lands. The
mechanism depends on what verification item 0 finds, in descending order of preference:

1. **Shared definitions, single source.** If both repos can point at the same lock
   artifacts — a git submodule holding `envs/`, or this repo referencing
   `robot-prompt-opt`'s lockfiles by path — drift becomes impossible rather than merely
   detectable. Prefer this even if it means a small refactor over there.
2. **Copied locks plus a drift check.** Vendor the lockfiles with their upstream commit
   recorded, and have `scripts/check_env_parity.py` re-resolve and diff on CI. Cheaper to
   set up, but parity now depends on someone reading a red check.
3. **Container images by digest.** If that project already builds images, reuse them by
   digest, never by tag — `:latest` is not a version.

Whatever the layer, the rule is the same: no environment in this repo is created by a bare
`pip install`. If a package is needed that `robot-prompt-opt` doesn't have, it gets added
*there* first and flows back, so the two never diverge by accident.

The molmospaces harness env is the one place parity may not be available yet — that
project may have no MuJoCo/filament environment at all. If so, define it here in the same
format that repo uses, and land it there too, so parity holds from the start rather than
being retrofitted later.

**Cosmos is the cheap one, and it should be built second.** NVIDIA's action policy server
speaks the OpenPI WebSocket protocol — the same protocol π0.5 uses — so the harness-side
client written for openpi should be reusable for both Cosmos variants with nothing changed
but the port, the policy name, and possibly the prompt format. Edge and Nano differ only in
checkpoint and resource footprint, not in interface, so they share one env and one wrapper
parameterised by checkpoint. If a new client class turns out to be necessary, that is a
signal the protocol assumption is wrong — stop and check rather than forking the client.

Rejected: running everything in-process with dependency pinning per policy via
`pip install --force-reinstall` between runs. Fast to write, impossible to trust.

## Implementation steps

1. **Verification pass.** Work through *Assumptions to verify first* and write findings
   into `docs/eval_reproduction.md`. Stop and report if #3 or #5 come back differently
   than assumed — both change the step order below.

2. **Capture the reference numbers.** The leaderboard is a JS app, so per-task values
   need exporting by hand or from whatever JSON endpoint the page fetches. Commit
   `reference/leaderboard_snapshot.csv` with columns
   `task,policy,success_rate,n_episodes,embodiment,metric,snapshot_date`. Filter to the
   DROID embodiment. Without `n_episodes` the interval math in the comparison step can't
   be done — if the leaderboard doesn't expose it, fall back to the episode count in each
   benchmark JSON and note the substitution. Include entries for Cosmos and DreamZero if
   they exist; a third-party write-up puts Cosmos3-Nano-Policy-DROID near the top of the
   All-Combined column in mid-2026, so an entry probably does. Rough sanity anchors from
   those write-ups (All-Combined averages, mid-2026): DreamZero ≈40.7, MolmoAct2 ≈38.6,
   Cosmos3-Nano ≈39.0, with π0.5 below all three. If our snapshot disagrees wildly with
   that shape, the filter is wrong.

3. **Harness environment, parity-sourced.** Build it from `robot-prompt-opt`'s definition
   if one exists; otherwise create it in that repo's format and land it there too. Target:
   conda `mlspaces`, Python 3.11, `pip install -e ".[mujoco]"` in the molmospaces checkout;
   add `.[mujoco-filament]` (second env if verification item 3 says they conflict). Set
   `MLSPACES_ASSETS_DIR` to a large disk. Install benchmark assets with
   `python -m molmo_spaces.molmo_spaces_constants`, then pin them. Also
   `pip install openpi_client` here — the OpenPI client library is needed on the harness
   side, not the server side. Record the resolved package set as the parity baseline before
   running anything.

4. **Positive control before any policy.** Run one scripted/TAMP datagen config from
   `molmo_spaces/data_generation/config/object_manipulation_datagen_configs.py` (e.g. the
   Franka pick config) through `eval_main.py` on 5 episodes of Pick-v1.5. This proves
   simulator, renderer, assets, and the success judge are all wired before a policy can
   be blamed. A broken harness scores every policy at 0% and looks like a policy bug.

5. **π0.5-DROID via openpi.** Fetch `gs://openpi-assets/checkpoints/pi05_droid_jointpos`,
   serve with `uv run scripts/serve_policy.py --port=8080 policy:checkpoint
   --policy.config=pi05_droid_jointpos --policy.dir=checkpoints/pi05_droid_jointpos/`,
   confirm the port matches `PiPolicyConfig` in
   `molmo_spaces.configs.policy_configs_baselines`, then run Pick-v1.5 end to end — it is
   in scope and it is the benchmark the evaluation guide walks through, so any deviation
   is ours rather than the docs'. This is the reference path (the only policy with a
   config already shipped in the repo) and everything downstream copies its shape. Do not
   proceed to other policies until π0.5 lands inside its leaderboard interval on
   Pick-v1.5, then on Pick-v2-classic and Pick-v2-filament — that pair is the cheapest way
   to confirm both renderer paths work before the other three policies multiply the cost.

6. **Cosmos Policy Edge and Nano.** One env from `third_party/cosmos` (cosmos-framework,
   `uv sync` with the cu130 group; consult the repo's bundled AgentSkills first). Serve
   Edge with `python -m cosmos_framework.scripts.action_policy_server_robolab
   --checkpoint_path nvidia/Cosmos3-Edge-Policy-DROID --port 8001
   --format-prompt-as-json True`, then Nano with its checkpoint on the same script. Since
   this is the OpenPI protocol, validate by pointing the *existing* openpi client at port
   8001 and running Pick-v1.5 — if that works unmodified, both variants are essentially
   free and only the run matrix grows. Record for each variant: BF16 memory footprint
   (Edge ≈9 GB, Nano ≈32 GB at 16B), action space (`joint_pos` per the DROID recipe),
   whether `use_state` is expected, and the prompt format flag. Run them back to back so
   the only difference between the two result sets is the checkpoint.

7. **MolmoAct2-DROID.** Use `allenai/MolmoAct2-DROID`. Check the policy zoo first; if
   absent, wrap the LeRobot inference path in a small socket server mirroring the openpi
   protocol so the harness-side client is shared. Record the action chunk size and control
   rate (`policy_dt_ms`) — a wrong control rate is the most common cause of a policy that
   "works" but scores near zero.

8. **TiPToP.** Install per `tiptop-robot.readthedocs.io` (pixi). This one is structurally
   different: a TAMP system needing camera intrinsics/extrinsics, a URDF, and depth. The
   benchmark JSON supplies camera pose, offset, quaternion, and FOV per episode, so
   calibration is derivable rather than measured. Decide and document whether depth comes
   from MuJoCo ground truth or from TiPToP's own depth estimator on rendered RGB — the
   leaderboard entry used one of them and the choice materially changes grasping success.
   Its planning latency also means the per-episode wall clock differs from the VLAs.

9. **DreamZero-DROID.** Last, because it's the expensive one. `conda` py3.11, torch cu129,
   `flash-attn` built with `MAX_JOBS=8`, checkpoint `GEAR-Dreams/DreamZero-DROID` (14B).
   Server: `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone
   --nproc_per_node=2 socket_test_optimized_AR.py --port 5000 --enable-dit-cache
   --model-path <ckpt>`. Verify with `test_client_AR.py` before touching the harness. Two
   GPUs is the documented minimum and we have at most two, so MuJoCo rendering shares a
   GPU with inference — set `MUJOCO_EGL_DEVICE_ID` explicitly (it does not always track
   `CUDA_VISIBLE_DEVICES`) and watch for OOM. Budget ~3s per inference on H100-class
   hardware after warmup, more on anything older.

10. **Matrix runner.** `scripts/eval.py` taking `--policy` and `--task`, resolving
    benchmark dir, renderer flag, camera overrides, and harness env from a single table
    keyed by task name. Writes to `runs/<policy>/<task>/<date>/`, skips completed cells, and
    refuses to start if the expected policy server isn't reachable. Two guards worth writing
    as asserts rather than comments: output paths must be unique across the three tasks that
    share `FrankaPickHardBench`, and a task marked `filament` must never be dispatched into
    the classic env. Run one server at a time — GPU memory won't hold two.

11. **Aggregation and comparison.** Per cell, `scripts/benchmarks/eval_to_csv.py
    <run_dir> <policy_name> --success-condition <oracle|both> --output-csv ...`. Then a
    `scripts/compare_to_leaderboard.py` that joins against
    `reference/leaderboard_snapshot.csv` and emits a PASS/FAIL table.

## Validation

**Handshake test, per policy (before any full run).** One episode, `--idx 0`. Assert:
the client receives a dict with `arm` shape `(7,)` and `gripper` shape `(1,)`; no NaNs;
and actions are *not* constant across ≥10 consecutive steps. A frozen or all-zeros action
stream is the standard silent failure of a mis-wired policy server and otherwise only
shows up as a 0% score twelve hours later.

**Observation sanity, per benchmark (all 9).** Dump the first observation's camera images
to `runs/_debug/<task>/<camera>.png` and view them. They must be non-black, correctly
oriented, and show a Franka in an indoor scene. This catches EGL device misconfiguration
and filament-vs-classic renderer mismatch, both of which produce plausible-looking runs
with garbage inputs. Do this once per benchmark, not per policy.

**Positive control.** The scripted planner from step 4 should score high (well above the
VLA range) on Pick-v1.5. If it doesn't, the harness is broken, not the policies.

**The reproduction check itself.** `scripts/compare_to_leaderboard.py` computes a Wilson
95% interval on our success rate for each (policy, task) cell — `n` is the episode count
in the benchmark JSON — and reports PASS when that interval contains the leaderboard point
estimate. Expected output shape:

```
task               policy            ours (n)        leaderboard   verdict
Pick-v1.5          pi05_droid        34.0% (50)      31.2%         PASS
Pick-v2-classic    pi05_droid        18.0% (50)      16.4%         PASS
Pick-v2-filament   molmoact2_droid   12.0% (50)      27.5%         FAIL
...
MolmoBot Combined  pi05_droid        24.1% (350)     23.6%         PASS
```

Exit non-zero if any of the 27 required cells (3 policies × 9 tasks) fails, or if any of
the 3 Group B aggregates misses. Note upfront
that with ~50 episodes a 95% interval is roughly ±13 points near 50% — so a PASS is weak
evidence per cell, and the *pattern* across 9 tasks is what actually matters. A single
FAIL is noise; three FAILs in the same direction for one policy is a wiring bug (most
likely camera mapping, control rate, or action space).

**Environment parity, two levels.** First, `scripts/check_env_parity.py` compares the
resolved package set of each env against `robot-prompt-opt`'s corresponding lock and exits
non-zero on any difference not listed in `docs/env_parity.md`. Run it in CI, not just once
by hand — parity decays silently. Second, and more convincing: pick one cheap cell (π0.5 on
Pick-v1.5, or whichever policy both repos already run) and execute it in both repos'
environments with the same seeds. Per-episode success flags should match exactly. If the
package sets agree but the outcomes don't, something outside the lock is leaking in —
driver version, `MUJOCO_GL` backend, asset version, an env var — and that's worth finding
now rather than mid-campaign.

**Provenance smoke test.** `python scripts/check_provenance.py runs/` asserts every run
directory contains the molmospaces SHA, asset pin file hash, policy checkpoint id, and the
exact eval command. A run without provenance is not reproducible and should not count.

## Final review
Before opening the PR, run a fresh code-review agent over the complete diff with no memory
of writing it. Point it specifically at: (a) whether any hardcoded benchmark path drifted
from `docs/ms-bench.md` / `docs/mb-bench.md` in the repo; (b) whether each policy wrapper
implements `prepare_model`, `reset`, and `get_action` and actually calls `reset()` between
episodes — a missing reset leaks action-chunk state across episodes and quietly depresses
scores; (c) camera-name and action-space handling per policy, since that's where a
copy-paste between wrappers does the most damage; (d) whether the two Cosmos variants
really differ only by checkpoint, with no accidental divergence in prompt format or
control rate; (e) anything from *Out of scope* that leaked in, especially training or
post-training code; (f) `--success-condition` consistency between the runner and the
comparison script; (g) any `pip install`, `conda install`, or `uv add` in a script or doc
that bypasses the shared lock — those are parity leaks and they're easy to miss in review
because each one looks locally reasonable. Fix findings, then re-run the comparison.

## Risks and rollback
- **Parity and reproduction can pull in opposite directions.** If `robot-prompt-opt` pins a
  transformers or torch version that differs from what produced a leaderboard entry, one of
  the two goals has to bend. Default: parity wins, and a reproduction miss traceable to a
  shared pin is reported as a finding about the pin — then fixed in `robot-prompt-opt` so
  both repos move together. What must not happen is a quiet local override that makes this
  repo's numbers look right while breaking the parity the numbers were meant to license.
  Any such divergence goes in `docs/env_parity.md` before it goes in the code.
- **cosmos-framework wants CUDA 13.0 while cuRobo wants 12.8 and DreamZero 12.9.** Per-env
  CUDA *runtimes* coexist fine, but there is one **driver** on the node and it must be new
  enough for cu130. Check `nvidia-smi` driver version before installing anything. If it's
  too old, either upgrade the driver (verifying the other three stacks still run) or run
  cosmos-framework in the NGC container it targets. Discovering this after four envs are
  built is the expensive ordering mistake here — check at step 1.
- **Cosmos Nano at 16B may not co-reside with MuJoCo rendering** on a single GPU alongside
  its own KV cache. Edge (~9 GB) is safe. Mitigation: give Nano the second GPU and render
  on the first; fall back to CPU rendering if needed.
- **Filament and classic renderers can't coexist in one env** — likely, and now costly:
  the task set splits 4 classic / 5 filament, so a late discovery means re-running roughly
  half the matrix in a second env. Mitigation: settle it in step 1, two harness envs,
  task→env mapping in the runner table.
- **DreamZero doesn't fit in 2 GPUs alongside MuJoCo rendering** — plausible on <80GB
  cards. Mitigation: render on CPU (`osmesa`) for DreamZero runs only, accepting the
  slowdown; or descope DreamZero to a subset of tasks and say so explicitly. Do not
  silently reduce its episode count — that breaks comparability.
- **Leaderboard entries predate the current molmospaces version** (v0.2.0 shipped
  2026-06-12; benchmark JSONs are dated Dec 2025–Mar 2026). Numbers may not be
  reproducible at HEAD at all. Mitigation: if a policy misses consistently, re-run one
  task at the molmospaces SHA closest to the leaderboard snapshot date before concluding
  our integration is wrong.
- **Three-camera policies on two-camera episodes** (DreamZero, possibly MolmoAct2-DROID).
  Mitigation: decide the mapping once, document it, and note it in any reported number —
  duplicating an exterior view into the second slot is a real deviation, not a detail.
- **Cost of getting it wrong is wall-clock, not data.** Everything here is read-only
  inference into fresh run directories; rollback is `rm -rf runs/`. No migrations, no
  one-way doors.

## Open questions for the implementer
- **Are DreamZero and Cosmos also expected to match their leaderboard entries?** Both were
  at or near #1 on MolmoSpaces at some point, so reference numbers likely exist. Default:
  run them, compare them, but report a miss as informational rather than failing the
  acceptance criteria. If Cosmos-Edge has no entry, it's a pure new data point.
- **Which repo is authoritative when a pin has to change?** Default read of "use the same
  environments as robot-prompt-opt": that repo is the source of truth, and changes flow
  from there to here. Confirm before landing a change in the other direction.
- **Does parity extend to the harness env, or only the policy envs?** Default: everything,
  including molmospaces itself, since that's where the MuJoCo/renderer versions live and
  those move results.
- **Metric definition** — oracle vs. both. Default to `oracle` and emit both columns.
- **Episode count per task**: use whatever the benchmark JSON ships; do not subsample.
  If wall-clock forces a subset for DreamZero only, use the first N by index (stable
  ordering) and record N per cell.
- **Where do run artifacts live?** Default `runs/` in-repo, gitignored, with only CSVs
  and the comparison table committed.

## PR shape
A stack, merged in order. Each leaves the tree working.

1. **Parity mechanism + harness + provenance** — the shared env definitions and
   `check_env_parity.py` land first, because every later PR builds an environment and each
   one built the wrong way has to be rebuilt. Plus env docs, asset pinning, matrix runner
   skeleton, positive control, observation-sanity script. No policies.
2. **openpi / π0.5-DROID** — the reference path, all 9 tasks, plus
   `compare_to_leaderboard.py` and the committed leaderboard snapshot.
3. **Cosmos Policy Edge + Nano** — one env, one wrapper, two checkpoints, 18 cells. Comes
   early precisely because it reuses PR 2's client; if it doesn't, that's worth learning
   before the harder integrations.
4. **MolmoAct2-DROID** — wrapper + results.
5. **TiPToP** — wrapper + results (largest integration surface; depth/calibration
   decisions documented here).
6. **DreamZero** — wrapper + results, and the final combined comparison table in
   `docs/eval_reproduction.md`.

Splitting this way means PR 2 alone answers the real question — *does this harness
reproduce a known number?* — and if the answer is no, PRs 3–6 aren't wasted work yet.