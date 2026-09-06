# Environment parity

> ## STATUS ON THIS MACHINE (2026-08-28): peer parity is UNVERIFIABLE, and that is a real
> ## reduction in a deliverable BENCHMARK.md declared pass/fail. Read this before the rest.
>
> Everything below this box was written for a host that had `~/Workspace/robot-prompt-opt`
> checked out beside this repo. **That peer does not exist on this machine.** There is nothing
> for `scripts/check_env_parity.py` to diff against and nothing for
> `scripts/check_cross_repo_cell.py` to run in, so the tier-2 "copied recipe plus a drift
> check" mechanism described below is reduced to "copied recipe". Both scripts are kept --
> they document the intent and would work again beside a peer -- but they are inert here.
> This is recorded rather than quietly dropped, because BENCHMARK.md lists
> `check_env_parity.py exits zero` as an acceptance criterion and it now cannot be met.
>
> **What still holds.** The recipe was always the real artifact: the peer's `uv.lock`,
> `pixi.lock` and Dockerfiles were inert there too (see "Mechanism" below), and its
> environments were built by the same hand-translated `conda create` + `pip install` sequence
> `scripts/setup_envs.sh` carries. Every pin in that script is unchanged, so the environments
> here resolve from the same recipe the recorded numbers came from.
>
> **What replaces the peer diff.** `scripts/setup_envs.sh --check` -- a *functional*
> verification, and a stronger one than a package-list diff for the failure modes that
> actually cost this campaign results: it asserts the CUDA arch is present in the torch build,
> the exact pins are installed, the source-built CUDA extensions import, and the checkpoints
> declare what the wrappers assume. It does not assert similarity to anything.
>
> **Two intentional divergences from the recorded environments, both forced by hardware:**
>
> | # | divergence | why |
> |---|---|---|
> | A | `TORCH_CUDA_ARCH_LIST=9.0`, assertions require `sm_90` (was `12.0` / `sm_120`) | This host is 4x H100 NVL (Hopper). The recorded campaign ran on 2x RTX PRO 5000 (Blackwell). Affects only the compile target for cuRobo, cuTAMP and pointnet2_ops -- all three build CUDA kernels from source and neither arch is a default. No version pin changed. |
> | B | `DREAMZERO_DIT_SPLIT=0` rather than `12` | The split exists solely to fit a 48 GB card; the patch's own default is 0. At 95 GB the model is resident on one GPU, which also drops the ~20% pipeline-shard throughput penalty documented in `scripts/dreamzero_patches/0001-single-gpu-48gb-inference.patch`. A behaviour change versus the recorded run, so it is named here rather than left implicit. |
>
> Divergence #1 below (the `openpi` fork/checkpoint) still stands and is unrelated to hardware.

BENCHMARK.md makes this a pass/fail deliverable, not a nice-to-have: the point of reproducing
known leaderboard numbers is to license comparisons, and a comparison across two repos with
silently divergent CUDA or `transformers` pins isn't licensed by anything. This document is the
register of what is shared, what differs, and why.

- **Built by**: `scripts/setup_envs.sh` (this repo) and
  `~/Workspace/robot-prompt-opt/scripts/setup_envs.sh` (the peer, ABSENT here -- see the box above).
- **Enforced by**: `scripts/check_env_parity.py`, which diffs the resolved distribution set of
  each paired environment and exits non-zero on any difference not listed in the
  `parity-allow` block at the bottom of this file. **Inert on this machine.**
- **Verified working by**: `scripts/setup_envs.sh --check`, which is a different question --
  it asserts each env *functions* (CUDA arch, exact pins, source-built extensions import,
  checkpoints declare what the wrappers assume), not that it matches the peer.
  **This is the operative gate here.**

## Mechanism: copied recipe plus a drift check

BENCHMARK.md lists three parity tiers in descending order of preference: shared definitions,
copied locks plus a drift check, or container digests. This campaign lands on the second, and
the reason is a fact about the peer repo rather than a preference.

`robot-prompt-opt`'s `uv.lock`, `pixi.lock` and `Dockerfile`s are inert. Nothing consumes
them; there is no `.venv/` at its root. Its environments are built by a 595-line
`scripts/setup_envs.sh` that hand-translates upstream `uv`/`pixi` configs into `conda create` +
`pip install`, with `scripts/constraints/*.txt` supplying the few pins that need to hold
globally. So "share the lock artifacts" has nothing to point at -- the recipe *is* the
artifact, and this repo mirrors it.

**Separate environments, same recipe.** Each project keeps its own conda envs (`polaris-*`
there, `mlspaces-*` here) rather than sharing one set in place. That way either project can be
upgraded without silently changing the other's results mid-campaign, and `check_env_parity.py`
turns "did they stay the same?" into a question with a machine-checkable answer instead of an
assumption. Before this work, three of this repo's policy servers ran *inside* the peer's envs
by absolute path, which is a stronger coupling than it looks: a `pip install` over there would
have changed numbers here with no signal at all.

### What tier 2 costs

Worth stating plainly, because both costs were paid during the first build and both are
inherent to copying a recipe rather than a lock:

1. **Transitive dependencies are not pinned.** The recipe pins direct dependencies; everything
   underneath resolves to whatever is newest on PyPI on build day. Two correct builds of the
   same recipe a day apart differed on `charset-normalizer`, `idna`, `uvicorn`, `pygments`,
   `python-dotenv`, and for DreamZero on 20 packages including `numba`, `wandb` and
   `nvidia-modelopt`. `scripts/check_env_parity.py --sync-to-peer` exists to reconcile exactly
   this class of difference; it refuses to touch packages present on only one side, which are
   structural and need a decision.
2. **A recipe can stop working when the resolver changes.** The Cosmos env's ~76-package bulk
   install works on the peer's older pip and aborts on this one with `resolution-too-deep`,
   installing nothing. It is now split into ten role-grouped chunks here -- same package set,
   ten small solves instead of one intractable one. A lockfile would never have hit this,
   because it does not ask the resolver anything.

Neither is a reason to abandon the tier; they are reasons the drift check has to be run rather
than assumed.

**One rule.** No environment in this repo is created by a bare `pip install` outside
`setup_envs.sh`. If a package is needed that `robot-prompt-opt` doesn't have, it gets added
there first and flows back, so the two never diverge by accident.

## The environments

| env | peer | python | torch / CUDA | status |
|---|---|---|---|---|
| `mlspaces-classic` | *(none)* | 3.11 | — (MuJoCo 3.5, classic renderer) | no counterpart |
| `mlspaces-filament` | *(none)* | 3.11 | — (MuJoCo 3.7.1, filament renderer) | no counterpart |
| `openpi` (uv venv) | `polaris-openpi` | 3.11 | jax 0.5.3 / cu12 | **intentional divergence** |
| `mlspaces-molmoact2` | `polaris-molmoact2` | 3.11 | 2.8.0+cu129 | mirrored |
| `mlspaces-m2t2` | `polaris-m2t2` | 3.10 | 2.8.0+cu129 | mirrored |
| `mlspaces-tiptop` | `polaris-tiptop` | 3.12 | 2.8.0+cu129 | mirrored |
| `mlspaces-dreamzero` | `polaris-dreamzero` | 3.11 | 2.8.0+cu129 | mirrored |
| `mlspaces-cosmos-policy` | `polaris-cosmos-policy` | 3.13 | 2.13.0+cu130 | mirrored |

### Pinned upstream checkouts

`robot-prompt-opt` floats five of its `third_party/` clones on branch HEAD -- its `sync_clone`
does `git pull --ff-only` when no tag is given. That is incompatible with parity, and it is not
hypothetical: `NVIDIA/cosmos-framework`'s `main` has already moved to `84bd8828` since that
repo's env was built from `c14617c2`. `scripts/setup_envs.sh` here pins every checkout to an
explicit SHA, recorded below, and these are the SHAs the peer's envs were built from
(2026-08-19).

| checkout | remote | SHA |
|---|---|---|
| `third_party/openpi` | `omarrayyann/openpi` | `711487f019e5f03b254d427d4523b1f0805a4814` |
| `third_party/molmoact2` | `allenai/molmoact2` | `5aac8f8a1180d79757ce500f819a02217079811c` |
| `third_party/dreamzero` | `dreamzero0/dreamzero` | `ab790c198fbce33503358efbbd4187ce9a89adf3` |
| `third_party/tiptop` | `tiptop-robot/tiptop` (v0.3.0) | `d8f5afdaa94a7432220c3042f9f80be5ab45aae8` |
| `third_party/tiptop/curobo` | `williamshen-nz/curobo` (fork) | `b5fad1df2a3ac4d3e33e369918b7d62d0e59ebd1` |
| `third_party/tiptop/cutamp` | `tiptop-robot/cuTAMP` (v0.0.6) | `e206ab817599406abd709e8ba19f445889bd641c` |
| `third_party/m2t2` | `williamshen-nz/m2t2-private` | `401d3f65ba4cecadebd8c7113aa347c1a051b684` |
| `third_party/cosmos` | `NVIDIA/cosmos-framework` | `c14617c2bc93dacbf69674fb964eec93182933d9` |

All but one are git submodules, so the SHA is recorded in git itself rather than only in a
shell variable. **`third_party/m2t2` is the exception and is a plain gitignored clone on
purpose**: it lives in a private fork, and as a submodule `git submodule update --init` would
hard-fail for everyone who clones this repo, including people who never run TiPToP. As a clone
created on demand by `scripts/setup_envs.sh mlspaces-m2t2`, only those who actually build that
env need access. Its SHA is pinned by that script and recorded above.

Note `third_party/cosmos` is **cosmos-framework**, not `NVIDIA/cosmos` and not
`nvlabs/cosmos-policy`. The DROID-trained checkpoints and the
`cosmos_framework.scripts.action_policy_server_robolab` server live only in cosmos-framework;
the repo the checkpoint family name suggests ships neither.

## Divergences, with reasons

### 1. `openpi` — different fork and different checkpoint

The peer's `polaris-openpi` carries upstream `Physical-Intelligence/openpi` and serves the
`pi05_droid_jointpos_polaris` config from
`gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris`. This repo carries the
`omarrayyann/openpi` fork and serves `pi05_droid_jointpos` from
`gs://openpi-assets/checkpoints/pi05_droid_jointpos`.

This is **not** a version pin that could be reconciled. They are different checkpoints trained
for different evaluation suites, and the MolmoSpaces leaderboard entry this campaign has to
reproduce was produced with the latter. Adopting the peer's would make the π0.5 reproduction
fail by construction, which would defeat the purpose parity is meant to serve.

BENCHMARK.md's stated default is that parity wins and a reproduction miss traceable to a shared
pin is reported as a finding about the pin. That default is aimed at *pins*; it does not
sensibly extend to swapping the model under test. Recorded here as an accepted divergence
rather than resolved either way silently, per the same section's requirement that any such
divergence "goes in `docs/env_parity.md` before it goes in the code".

It also stays a `uv` venv rather than becoming a conda env, because `uv sync` is what the fork
supports and what the MolmoSpaces evaluation guide documents.

### 2. `mlspaces-classic` / `mlspaces-filament` — no counterpart exists

`robot-prompt-opt` evaluates in Isaac Sim against 2DGS Gaussian-splat reconstructions. It has
no MuJoCo, no filament renderer, and no molmospaces dependency anywhere. There is nothing to
mirror, so these two are defined here, in that repo's format, and are the artifact that would
flow the other way if it ever grows a MuJoCo path.

They are two envs rather than one because the renderer is not a runtime switch:
`molmo_spaces/env/env.py:35` reads `getattr(mujoco, "mjRENDERER", "classic")` at import, and
`pyproject.toml` gives the `mujoco` and `mujoco-filament` extras the same import name from
different sources. Which env is active *is* the renderer choice, which is why
`scripts/eval_common.py`'s env guard is load-bearing rather than a convenience check.

### 3. `tiptop` version string — our local patch, surfaced by setuptools-scm

`pip list` shows `tiptop 0.3.0.post1.dev0+gd8f5afdaa.d20260819` here against `0.3.0` in the
peer. This is not a different tiptop: both are editable installs of the same checkout at the
same SHA (`d8f5afda`, tag v0.3.0). setuptools-scm derives the version from git state, and this
repo applies `scripts/tiptop_patches/0001-raise-m2t2-async-timeout.patch` to that working tree,
so the tree is dirty and the version gains a `.post1.dev0+g<sha>.d<date>` suffix.

The patch itself is the real divergence and is deliberate: tiptop's async M2T2 client caps at
30s while its own synchronous path uses 500s, and under GPU contention here M2T2 calls have
exceeded 120s. A timeout there does not raise -- it becomes a well-formed `success=False` plan,
i.e. silent score depression rather than a visible failure. The peer's own TiPToP wrapper
health-checks M2T2 differently and has not hit this.

`check_env_parity.py` never syncs editable installs, for a related reason worth stating: their
version strings describe a source tree, not a package index, and `pip install tiptop==0.3.0`
would replace a local editable checkout with a PyPI wheel -- silently changing what the server
runs. The checker refuses and labels them instead.

### 4. MolmoAct2's torch build — resolved by converging, not by exception

Worth recording because it *was* a divergence and no longer is. Upstream `molmoact2`'s
`pyproject.toml` pins `torch==2.5.1`+cu121, which has no `sm_120` kernels, so every CUDA op on
this Blackwell host fails with "no kernel image is available for execution on the device". This
repo previously worked around that with `torch 2.7.1+cu128`; the peer uses `2.8.0+cu129`.
`scripts/setup_envs.sh` now installs `2.8.0+cu129` to match. The local patch
(`scripts/molmoact2_patches/0001-blackwell-and-live-api-fixes.patch`) is still applied, but for
its other half: the live `allenai/MolmoAct2-DROID` remote code renamed `action_mode` to
`inference_action_mode`, and the vendored `host_server_droid.py` still calls the old name.

**Consequence to watch:** MolmoAct2's earlier spot-checks were taken on the 2.7.1+cu128 build
and do not carry over. Re-handshake and re-spot-check before trusting its matrix cells.

## Behavioural parity, not just package parity

`check_env_parity.py` answers "do these two environments contain the same packages?".
BENCHMARK.md:96-98 asks a harder question -- do they *behave* the same -- and requires one cell
run in both to produce identical per-episode outcomes. `scripts/check_cross_repo_cell.py` is
that check.

It was run (Cosmos-Edge, Pick-v1.5, 3 episodes, seed 42) and the result is worth reading before
relying on it: one of three episodes differed across environments, **and the same environment
restarted produced the identical difference on the identical episode**. All three fresh servers
seed `numpy.default_rng(0)`, so the seed sequences were byte-identical; the residual is
GPU-level nondeterminism (bf16 reductions, attention kernels, cuDNN autotuning), which is not
reproducible across processes.

So the two environments are indistinguishable at this resolution -- but "identical per-episode
outcomes" is not a property this policy has even against itself, and cannot serve as a
pass/fail gate. **Always run the same-environment control alongside the cross-environment
comparison**, and read the cross-env difference only relative to it. Full write-up in
`docs/eval_reproduction.md`.

## Machine-readable allowlist

`scripts/check_env_parity.py` parses the block below. `env <name>` exempts an environment from
comparison entirely; `pkg <env> <package> <reason>` exempts one distribution. Everything else
that differs is a failure. Add an entry only with the sentence that justifies it -- that
sentence is the whole point of the file.

```parity-allow
# Different fork and different checkpoint, not a version difference -- see "Divergences" #1.
env  openpi

# polaris-m2t2 carries a vestigial editable install of m2t2 itself (m2t2 1.0.0 in pip list)
# whose finder resolves to nothing: `import m2t2` there gives __file__ = None. It is a leftover
# from before that project switched to the m2t2_repo.pth approach, which is what actually makes
# the package importable in both envs (m2t2/ ships no __init__.py, so `pip install .` builds an
# empty wheel -- documented in that repo's own setup_envs.sh comment). Reproducing a broken
# empty wheel here would make the envs more similar and both worse; the real import path is
# identical. Not replicated.
pkg  mlspaces-m2t2  m2t2  Vestigial empty editable install in the peer; real import is via m2t2_repo.pth in both.

# Both envs install `huggingface_hub[cli]` and both resolve huggingface_hub to the SAME version
# (0.36.2), but the peer lacks five of that extra's dependencies. The cause is pip resolution
# order, not a recipe difference: over there huggingface_hub was already present as a
# transitive dependency when `[cli]` was requested, so pip treated the requirement as satisfied
# and never pulled the extra's own deps; here `[cli]` resolved from scratch. Both envs have a
# working `hf` entrypoint. These five are interactive-prompt and alternate-HTTP libraries used
# only by the CLI, never on the inference path. Uninstalling deps of an extra we deliberately
# asked for, to match an env that is missing them by accident, would make both worse.
pkg  mlspaces-dreamzero  inquirerpy  huggingface_hub[cli] extra dep; peer missed it via pip resolution order. CLI-only.
pkg  mlspaces-dreamzero  pfzy        huggingface_hub[cli] extra dep; peer missed it via pip resolution order. CLI-only.
pkg  mlspaces-dreamzero  truststore  huggingface_hub[cli] extra dep; peer missed it via pip resolution order. CLI-only.
pkg  mlspaces-dreamzero  httpx2      huggingface_hub[cli] extra dep; peer missed it via pip resolution order. CLI-only.
pkg  mlspaces-dreamzero  httpcore2   huggingface_hub[cli] extra dep; peer missed it via pip resolution order. CLI-only.

# Same checkout, same SHA, both editable; the suffix is setuptools-scm reporting that this
# repo's M2T2-timeout patch makes the working tree dirty. See "Divergences" #3.
pkg  mlspaces-tiptop  tiptop  Editable install of the same SHA; version suffix is our applied patch, per setuptools-scm.
```
