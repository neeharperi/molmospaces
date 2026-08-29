#!/usr/bin/env bash
# Build the conda environments this repo's DROID-leaderboard reproduction needs.
#
#   mlspaces-classic        eval harness, classic MuJoCo renderer  (Open-v1, Close-v1,
#                           Pick-v1.5, Pick-v2-classic)
#   mlspaces-filament       eval harness, filament renderer        (the other 5 tasks)
#   mlspaces-molmoact2      MolmoAct2-DROID policy server (:8000)
#   mlspaces-cosmos-policy  Cosmos3-{Edge,Nano}-Policy-DROID server (:8003/:8004)
#   mlspaces-tiptop         TiPToP planning server (:8765), PyTorch + cuRobo + cuTAMP
#   mlspaces-m2t2           M2T2 grasp server (:8123), hard dependency of mlspaces-tiptop
#   mlspaces-dreamzero      DreamZero-DROID policy server (:5000)
#   openpi                  pi0.5-DROID server (:8080) -- NOT a conda env and NOT at parity;
#                           built by `uv sync` in third_party/openpi. Check-only here.
#
# They cannot be merged: the projects carry mutually exclusive exact pins on torch,
# transformers, numpy and python itself. The policy servers are separate processes anyway,
# reached over a socket -- see BENCHMARK.md's "One harness environment, five isolated policy
# servers".
#
# PARITY. Every policy env here is a deliberate mirror of the corresponding `polaris-*` env
# built by ~/Workspace/robot-prompt-opt/scripts/setup_envs.sh: same python version, same torch
# and CUDA build, same pins, same source builds, in the same order. This repo keeps its OWN
# copies rather than sharing that repo's envs in place, so the two projects can be upgraded
# independently, but they are meant to resolve identically -- scripts/check_env_parity.py is
# what enforces that, and docs/env_parity.md is where every intentional difference is written
# down. If you need a package that robot-prompt-opt doesn't have, add it THERE first and let
# it flow back here, or the two diverge silently.
#
# One structural difference from the robot-prompt-opt original, and it is deliberate: every
# third-party checkout below is pinned to an explicit SHA. That script floats five of its
# clones on branch HEAD, which is why cosmos-framework's `main` has already moved past the
# commit its env was built from. Floating clones and parity are incompatible.
#
# Usage:
#   scripts/setup_envs.sh                        # create/repair all of them
#   scripts/setup_envs.sh --check                # verify only, create nothing
#   scripts/setup_envs.sh mlspaces-tiptop        # just one env (repeatable)
#
# Building an env is only step one. Because this is a recipe and not a lockfile, transitive
# dependencies resolve to whatever is newest on build day, so a fresh build will differ from
# the peer on a handful of packages even when it is perfectly correct. Follow every build with:
#
#   python scripts/check_env_parity.py --sync-to-peer --env <name>
#
# which pins those back to the peer's exact versions, then re-run `--check` here to confirm the
# env still works. Occasionally a version downgrade orphans a package the newer version pulled
# in (moviepy -> proglog was the real case); check_env_parity.py reports those as one-sided
# differences for a human to remove or document, rather than guessing.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

CONDA="${CONDA:-$HOME/anaconda3/bin/conda}"
ENVS_DIR="$($CONDA info --base)/envs"
PY_VER=3.11

# --- pinned upstreams -------------------------------------------------------------------
# These SHAs are the ones robot-prompt-opt's envs were built from (recorded 2026-08-19).
# Bumping one is an env change: re-run the matching --check and note it in docs/env_parity.md.
MOLMOACT2_SHA="5aac8f8a1180d79757ce500f819a02217079811c"   # third_party/molmoact2 submodule
DREAMZERO_SHA="ab790c198fbce33503358efbbd4187ce9a89adf3"   # third_party/dreamzero submodule
TIPTOP_SHA="d8f5afdaa94a7432220c3042f9f80be5ab45aae8"      # third_party/tiptop submodule (v0.3.0)
CUROBO_SHA="b5fad1df2a3ac4d3e33e369918b7d62d0e59ebd1"
CUTAMP_SHA="e206ab817599406abd709e8ba19f445889bd641c"      # == tag v0.0.6
M2T2_SHA="401d3f65ba4cecadebd8c7113aa347c1a051b684"
COSMOS_SHA="c14617c2bc93dacbf69674fb964eec93182933d9"

CUROBO_REPO="https://github.com/williamshen-nz/curobo.git"   # fork; upstream cuRobo will not do
CUTAMP_REPO="https://github.com/tiptop-robot/cuTAMP.git"
M2T2_REPO="https://github.com/williamshen-nz/m2t2-private.git"
# The DROID-trained checkpoints and their server live in cosmos-framework, NOT in
# nvlabs/cosmos-policy (which the checkpoint family name suggests and which ships no
# DROID-trained weights at all). See docs/eval_reproduction.md's Cosmos section.
COSMOS_REPO="https://github.com/NVIDIA/cosmos-framework.git"

CUTAMP_VER="0.0.6"      # tiptop/utils.py asserts this exact version at import
TORCH_CU129="https://download.pytorch.org/whl/cu129"
TORCH_CU130="https://download.pytorch.org/whl/cu130"
FLASH_ATTN_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
MOLMOACT2_REPO_ID="allenai/MolmoAct2-DROID"
# Unlike robot-prompt-opt, which keeps a repo-local checkpoints/hf-cache, this repo uses the
# machine's default HF cache: MolmoAct2-DROID, both Cosmos policies and DreamZero-DROID are
# already resident there, and a second copy is 20-45 GB each for no benefit. The weights are
# not part of the environment, so this is not a parity difference.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"

# GPU architecture. cuRobo, cuTAMP and M2T2's pointnet2_ops all build CUDA kernels from
# source and none of them targets the local arch by default.
#
# THIS HOST IS HOPPER, NOT BLACKWELL. The campaign this script was written for ran on 2x RTX
# PRO 5000 Blackwell (sm_120); this machine is 4x H100 NVL (sm_90). Everything else in the
# recipe -- the cu129/cu130 wheel indices, the flash-attn build, the exact pins -- is
# arch-agnostic and stays as it was, because the pins are what makes results comparable
# across the two runs. Only the compile target and the corresponding assertions change.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
# The arch string the check_env() assertions below require torch to have been built with.
export EXPECTED_SM="${EXPECTED_SM:-sm_90}"

ALL_TARGETS=(mlspaces-classic mlspaces-filament mlspaces-molmoact2 mlspaces-m2t2
             mlspaces-tiptop mlspaces-dreamzero mlspaces-cosmos-policy openpi)

CHECK_ONLY=0
TARGETS=()
for a in "$@"; do
    case "$a" in
        --check) CHECK_ONLY=1 ;;
        mlspaces-classic|mlspaces-filament|mlspaces-molmoact2|mlspaces-m2t2|mlspaces-tiptop|\
mlspaces-dreamzero|mlspaces-cosmos-policy|openpi) TARGETS+=("$a") ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${ALL_TARGETS[@]}")

# deepspeed (pulled in transitively by transformers) hard-errors at import without CUDA_HOME,
# even though inference never uses it; its compatibility probe only needs nvcc to EXIST -- it
# never compiles anything here. Resolved dynamically rather than hardcoded, because the only
# envs that carry a CUDA toolkit (m2t2, tiptop) are built by this same script: naming one
# statically makes the DreamZero check pass or fail depending on target ORDER, which is how
# `setup_envs.sh mlspaces-dreamzero` on a fresh machine ends up failing on a missing nvcc that
# has nothing to do with DreamZero.
resolve_cuda_home() {
    local e
    for e in mlspaces-tiptop mlspaces-m2t2; do
        if [ -x "$ENVS_DIR/$e/bin/nvcc" ]; then echo "$ENVS_DIR/$e"; return; fi
    done
    # Fall back to anything already on PATH; deepspeed only needs the directory to resolve.
    command -v nvcc >/dev/null 2>&1 && dirname "$(dirname "$(command -v nvcc)")"
}
export CUDA_HOME="${CUDA_HOME:-$(resolve_cuda_home)}"
if [ -z "$CUDA_HOME" ]; then
    echo "note: no CUDA toolkit found yet; build mlspaces-m2t2 or mlspaces-tiptop first if" >&2
    echo "      the mlspaces-dreamzero check fails importing transformers/deepspeed." >&2
fi

py()   { echo "$ENVS_DIR/$1/bin/python"; }
pip_() { echo "$ENVS_DIR/$1/bin/pip"; }
have_env() { [ -x "$(py "$1")" ]; }

ensure_env() {  # $1 = env name, $2 = python version (defaults to $PY_VER)
    local ver="${2:-$PY_VER}"
    if have_env "$1"; then echo "  env $1 exists"; else
        echo "  creating env $1 (python $ver)"
        "$CONDA" create -n "$1" "python=$ver" -y
    fi
}

# Rebuilding an env out from under a running evaluation corrupts both. The reproduction matrix
# runs for days, so this is a live hazard rather than a theoretical one.
# $1 = interpreter prefix path (a conda env dir, or third_party/openpi/.venv)
refuse_if_prefix_busy() {
    local prefix="$1" label="$2"
    if pgrep -f "^$prefix/bin/python" >/dev/null 2>&1; then
        echo "  REFUSING to rebuild $label: a process is using it right now:" >&2
        pgrep -af "^$prefix/bin/python" | head -3 >&2
        echo "  A reproduction matrix runs for days; rebuilding under it corrupts both." >&2
        return 1
    fi
}

refuse_if_busy() {  # $1 = conda env name
    refuse_if_prefix_busy "$ENVS_DIR/$1" "$1"
}

# Build environment for the CUDA extensions compiled from source. Headers land in
# targets/<triple>/include, which torch's build does not put on the include path, so nvcc
# cannot find cuda_runtime_api.h; libraries are already mirrored into lib/.
cuda_env() {  # $1 = env prefix
    printf '%s\n' "CUDA_HOME=$1" "PATH=$1/bin:$PATH" \
        "CPATH=$1/targets/x86_64-linux/include:${CPATH:-}"
}

# Clone (or fetch) and hard-pin to a SHA. Unlike robot-prompt-opt's sync_clone this never
# floats a branch -- see the header.
sync_clone() {  # $1 = url, $2 = destination, $3 = sha
    local url="$1" dest="$2" sha="$3"
    if [ -d "$dest/.git" ]; then
        git -C "$dest" rev-parse --verify --quiet "$sha^{commit}" >/dev/null 2>&1 \
            || git -C "$dest" fetch --quiet --tags origin
    else
        echo "  cloning $url -> $dest"
        git clone --quiet "$url" "$dest"
    fi
    if [ "$(git -C "$dest" rev-parse HEAD)" != "$sha" ]; then
        echo "  pinning $dest -> $sha"
        git -C "$dest" checkout --quiet "$sha"
    fi
}

# Submodules are already pinned by the parent repo; just assert we are where we think we are.
assert_submodule_sha() {  # $1 = path, $2 = sha
    local got; got="$(git -C "$1" rev-parse HEAD)"
    if [ "$got" != "$2" ]; then
        echo "  WARNING: $1 is at $got, docs/env_parity.md records $2" >&2
    fi
}

# ---------------------------------------------------------------- harness envs
# The renderer is NOT chosen by a CLI flag. molmo_spaces/env/env.py:35 fixes it at import time
# from whichever `mujoco` distribution is installed, so these two envs ARE the renderer choice
# and pyproject.toml gives both extras the same import name from different sources. Hence two
# envs rather than one with a switch.
_harness_common() {  # $1 = env name
    local PIP; PIP="$(pip_ "$1")"
    # openpi-client is needed on the HARNESS side (the eval client speaks openpi's msgpack
    # websocket to pi0.5, Cosmos and DreamZero). Its pyproject pins numpy<2.0.0, which
    # conflicts with molmospaces' own numpy>=2 requirement -- a plain install silently
    # downgrades numpy and breaks ml_dtypes and opencv. Its other deps (msgpack, pillow,
    # websockets, dm-tree) are already satisfied by the base install, so --no-deps is safe.
    "$PIP" install --no-deps -e third_party/openpi/packages/openpi-client
    "$PIP" install -U "numpy>=2,<3"
    # json-numpy is needed on the HARNESS side too, not just in mlspaces-molmoact2. MolmoAct2
    # is the one policy that speaks HTTP rather than the msgpack websocket, and
    # molmoact2_policy.py's MolmoAct2HTTPClient imports json_numpy to encode the request. It
    # was missing from this function, and the failure is quiet in the worst way: the client's
    # connect retries swallow the ImportError as a connection failure ("Connection attempt 1
    # failed: No module named 'json_numpy'"), the episode is skipped, and the cell reports
    # "Success count: 0, Total count: 0" -- a full matrix would have looked like MolmoAct2
    # scoring zero everywhere rather than like a missing dependency.
    "$PIP" install json-numpy
}

setup_classic() {
    refuse_if_busy mlspaces-classic || return 1
    ensure_env mlspaces-classic
    local PIP; PIP="$(pip_ mlspaces-classic)"
    # No PIP_CONSTRAINT here: common.txt's setuptools<81 exists for the source-built CUDA
    # extensions (pointnet2_ops, cuRobo, cuTAMP) and for isaaclab, none of which are in a
    # harness env. Applying it anyway would downgrade setuptools in a live evaluation env for
    # no reason -- a behaviour change mid-campaign, which is exactly what parity is meant to
    # prevent.
    "$PIP" install -e ".[mujoco]"
    _harness_common mlspaces-classic
}

setup_filament() {
    refuse_if_busy mlspaces-filament || return 1
    ensure_env mlspaces-filament
    local PIP; PIP="$(pip_ mlspaces-filament)"
    # Two real bugs in the README's documented recipe, both hit while first building this env:
    #
    # 1. `pip install -i https://test.pypi.org/simple/ mujoco-filament` replaces pip's ENTIRE
    #    index with test.pypi.org, which does not mirror setuptools, so building absl-py (a
    #    mujoco-filament dependency) fails immediately. --extra-index-url keeps PyPI available
    #    for ordinary build dependencies.
    "$PIP" install --index-url https://pypi.org/simple/ \
        --extra-index-url https://test.pypi.org/simple/ mujoco-filament
    # 2. `pip install -e ".[mujoco-filament]"` then fails on the mujoco wheel itself:
    #    pyproject.toml spells it `mujoco @ file://${PROJECT_ROOT}/bin/wheels/...`, and
    #    ${PROJECT_ROOT} interpolation is a uv-only feature -- plain pip treats it as a literal
    #    path and errors with "non-local file URIs are not supported". Install the wheel by its
    #    real path, then install the project WITHOUT the extra so it isn't re-triggered.
    "$PIP" install --force-reinstall \
        "$REPO/bin/wheels/mujoco-3.7.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    "$PIP" install -e .   # no PIP_CONSTRAINT -- see setup_classic
    _harness_common mlspaces-filament
}

# ---------------------------------------------------------------- mlspaces-molmoact2
setup_molmoact2() {
    refuse_if_busy mlspaces-molmoact2 || return 1
    ensure_env mlspaces-molmoact2
    local PIP; PIP="$(pip_ mlspaces-molmoact2)"
    assert_submodule_sha "$REPO/third_party/molmoact2" "$MOLMOACT2_SHA"

    "$PIP" install "setuptools<81" wheel
    # Upstream's pyproject pins torch 2.5.1 from the cu121 index, and cu121 has no sm_120, so
    # every CUDA op here fails with "no kernel image is available". Same cu129 build
    # robot-prompt-opt uses. NOTE this differs from the torch 2.7.1+cu128 this repo's env
    # previously carried -- converging on 2.8.0+cu129 is the point of the parity work; see
    # docs/env_parity.md.
    "$PIP" install --index-url "$TORCH_CU129" "torch==2.8.0" "torchvision==0.23.0"
    # MolmoAct2 ships its modelling code in the checkpoint (config.json auto_map, loaded with
    # trust_remote_code), so third_party/molmoact2 supplies only the server script. The
    # transformers range is upstream's own.
    "$PIP" install "transformers>=4.57,<4.58" accelerate safetensors einops pillow sentencepiece protobuf
    "$PIP" install "huggingface_hub[cli]" hf-transfer
    # The server is FastAPI and json_numpy is its wire format. Deliberately NOT installing
    # openpi-client here: this policy speaks HTTP, and staying off openpi-client frees this env
    # from its numpy<2 pin.
    "$PIP" install fastapi "uvicorn[standard]" json-numpy
    # Our local fixes to the vendored server script (the live checkpoint renamed action_mode ->
    # inference_action_mode) live as a patch, since a submodule update would silently revert them.
    bash scripts/apply_third_party_patches.sh molmoact2
}

# ---------------------------------------------------------------- mlspaces-m2t2
# M2T2 predicts 6-DOF grasps from a point cloud. TiPToP posts to it for every plan; without it
# every planning call returns a well-formed success=False and the policy silently scores ~0%
# (docs/eval_reproduction.md's TiPToP section). A hard dependency, not an accelerator.
setup_m2t2() {
    refuse_if_busy mlspaces-m2t2 || return 1
    ensure_env mlspaces-m2t2 3.10
    local PIP; PIP="$(pip_ mlspaces-m2t2)"
    sync_clone "$M2T2_REPO" "$REPO/third_party/m2t2" "$M2T2_SHA"

    "$PIP" install "setuptools<81" wheel
    "$PIP" install --index-url "$TORCH_CU129" "torch==2.8.0" "torchvision==0.23.0"
    # nvcc matching that torch build; without it the extension below picks up whatever CUDA is
    # on PATH and torch aborts on the version mismatch. git-lfs is for the weights fetch at the
    # end of this function: it is NOT installed system-wide on this host, and robot-prompt-opt
    # gets it from its `polaris` env's bin, which this repo has no counterpart to -- so without
    # installing it here the clone below silently leaves 133-byte pointer files. Neither is a
    # pip package, so neither appears in the parity diff.
    "$CONDA" install -n mlspaces-m2t2 -c nvidia -c conda-forge cuda-toolkit=12.9 git-lfs -y
    "$PIP" install -r "$REPO/third_party/m2t2/requirements.txt"
    local CUDAENV; mapfile -t CUDAENV < <(cuda_env "$ENVS_DIR/mlspaces-m2t2")
    env "${CUDAENV[@]}" "$PIP" install --no-build-isolation "$REPO/third_party/m2t2/pointnet2_ops"
    # m2t2/ ships no __init__.py, so find_packages() returns nothing and `pip install .` builds
    # an empty wheel. Upstream never noticed -- their server is always launched from the repo
    # root, where the directory is importable anyway. Put it on the path explicitly instead.
    "$(py mlspaces-m2t2)" - "$REPO/third_party/m2t2" <<'EOF'
import pathlib, site, sys
pathlib.Path(site.getsitepackages()[0], "m2t2_repo.pth").write_text(sys.argv[1] + "\n")
EOF
    # The weights are git-lfs objects. A plain clone leaves ~130-byte pointer files, which
    # surface much later as an opaque "invalid load key, 'v'" from torch.load inside the server.
    local W="$REPO/third_party/m2t2/weights"
    if [ ! -f "$W/m2t2.pth" ] || [ "$(stat -c%s "$W/m2t2.pth")" -lt 1000000 ]; then
        echo "  fetching M2T2 weights"
        [ -d "$W/.git" ] || git clone --quiet https://huggingface.co/wentao-yuan/m2t2 "$W"
        # git-lfs from this env's bin, not the system PATH (it isn't installed there).
        PATH="$ENVS_DIR/mlspaces-m2t2/bin:$PATH" git -C "$W" lfs install --local
        PATH="$ENVS_DIR/mlspaces-m2t2/bin:$PATH" git -C "$W" lfs pull
    fi
    if [ "$(stat -c%s "$W/m2t2.pth" 2>/dev/null || echo 0)" -lt 1000000 ]; then
        echo "  M2T2 weights are still a git-lfs pointer; is git-lfs installed?" >&2
        return 1
    fi
}

# ---------------------------------------------------------------- mlspaces-tiptop
# Upstream builds this with pixi (python 3.12, conda-forge CUDA). robot-prompt-opt translates
# that to conda+pip, and this mirrors the translation rather than the pixi original -- parity
# with that env is what licenses comparing TiPToP numbers across the two projects.
setup_tiptop() {
    refuse_if_busy mlspaces-tiptop || return 1
    ensure_env mlspaces-tiptop 3.12
    local PIP; PIP="$(pip_ mlspaces-tiptop)"
    local TT="$REPO/third_party/tiptop"
    assert_submodule_sha "$TT" "$TIPTOP_SHA"

    "$PIP" install "setuptools<81" wheel ninja
    "$CONDA" install -n mlspaces-tiptop -c nvidia cuda-toolkit=12.9 -y
    "$PIP" install --index-url "$TORCH_CU129" "torch==2.8.0" "torchvision==0.23.0"

    local CUDAENV; mapfile -t CUDAENV < <(cuda_env "$ENVS_DIR/mlspaces-tiptop")

    # -- cuRobo (fork) -- collision-aware motion planning. 5-20 minutes of CUDA kernels.
    sync_clone "$CUROBO_REPO" "$TT/curobo" "$CUROBO_SHA"
    "$PIP" install pybind11 networkx numpy-quaternion pyyaml "setuptools_scm>=6.2" trimesh tqdm \
                   importlib_resources scikit-image "warp-lang>=0.9,<1.13" "yourdfpy>=0.0.60" \
                   pycollada "open3d>=0.19.0" roma
    ( cd "$TT/curobo" && env "${CUDAENV[@]}" "$PIP" install -e . --no-build-isolation --no-deps )

    # -- cuTAMP -- the TAMP solver, at the exact SHA of the tag tiptop/utils.py asserts. It also
    # carries the Franka URDF, cuRobo YAML and gripper collision spheres that tiptop itself does
    # not provide.
    sync_clone "$CUTAMP_REPO" "$TT/cutamp" "$CUTAMP_SHA"
    "$PIP" install einops "h5py>=3.15.1,<4" "msgpack-numpy>=0.4.8" "websockets>=12.0"
    ( cd "$TT/cutamp" && env "${CUDAENV[@]}" "$PIP" install -e . --no-build-isolation --no-deps )

    # -- tiptop itself, --no-deps because its dependency list also pulls the ZED/RealSense camera
    # stack; the pieces that matter in sim are listed explicitly. bamboo-franka-client is here
    # despite there being no robot: tiptop/utils.py imports it at module scope and nearly
    # everything imports tiptop.utils. numpy is left unpinned (rerun-sdk 0.27.3 needs numpy>=2,
    # and nothing here talks to openpi-client, whose numpy<2 pin constrains the other envs).
    "$PIP" install aiohttp "bamboo-franka-client>=0.1.1" dill "google-genai>=0.6.0" jaxtyping \
                   matplotlib omegaconf "opencv-python>=4.12,<4.13" \
                   "rerun-sdk==0.27.3" ruamel.yaml scipy "supervision>=0.17.0" tyro
    # Segmentation, in-process on the GPU. Spelled SAM-2 to match the project metadata;
    # "sam2 @ ..." is rejected as a name mismatch.
    "$PIP" install "SAM-2 @ git+https://github.com/facebookresearch/segment-anything-2.git"
    "$PIP" install --no-deps -e "$TT"
    # Deliberately skipped: pyzed (needs the ZED SDK) and FoundationStereo. Depth comes straight
    # from MuJoCo, so no stereo camera or depth server is involved.
}

# ---------------------------------------------------------------- mlspaces-dreamzero
setup_dreamzero() {
    refuse_if_busy mlspaces-dreamzero || return 1
    ensure_env mlspaces-dreamzero
    local PIP; PIP="$(pip_ mlspaces-dreamzero)"
    assert_submodule_sha "$REPO/third_party/dreamzero" "$DREAMZERO_SHA"

    export PIP_CONSTRAINT="$REPO/scripts/constraints/common.txt"
    "$PIP" install "setuptools<81" wheel
    "$PIP" install -e third_party/dreamzero --extra-index-url "$TORCH_CU129"
    # wan2_1_attention.py falls back Transformer Engine -> FA2, and TE is GB200-only, so
    # flash-attn is required rather than optional. The prebuilt wheel avoids a long source build.
    "$PIP" install "$FLASH_ATTN_WHL"
    # transformers imports deepspeed's integration even though inference never uses it. 0.19.x
    # causes a circular import through transformers.modeling_utils; 0.16.5 needs CUDA_HOME.
    "$PIP" install "deepspeed==0.16.5"
    "$PIP" install "huggingface_hub[cli]"
    unset PIP_CONSTRAINT
    # The single-GPU/48GB inference patch. Without it WANPolicyHead only splits classifier-free
    # guidance across ranks, so each rank holds a full ~44.6 GiB replica and 2 ranks OOM host RAM.
    bash scripts/apply_third_party_patches.sh dreamzero
}

# ---------------------------------------------------------------- mlspaces-cosmos-policy
setup_cosmos_policy() {
    refuse_if_busy mlspaces-cosmos-policy || return 1
    ensure_env mlspaces-cosmos-policy 3.13
    local PIP; PIP="$(pip_ mlspaces-cosmos-policy)"
    # third_party/cosmos is a submodule, so a recursive clone already provides it; sync_clone
    # covers the non-recursive case and re-asserts the SHA either way.
    sync_clone "$COSMOS_REPO" "$REPO/third_party/cosmos" "$COSMOS_SHA"

    "$PIP" install "setuptools<81" wheel
    # cosmos-framework's own cu130-torch213 uv group pins exactly this torch/CUDA build.
    "$PIP" install --index-url "$TORCH_CU130" "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" \
        "torchcodec==0.14.0+cu130"
    # natten has no PyPI wheel; upstream resolves it through a custom uv index declared in
    # cosmos-framework/pyproject.toml, which is a plain PEP 503 index and works with
    # --extra-index-url the same way.
    "$PIP" install --extra-index-url https://nvidia-cosmos.github.io/cosmos-dependencies/v1.5.0 \
        "natten==0.21.6+cu130.torch213"
    # Deliberately skipped: flash-attn / flash-attn-3-nv (upstream's own cu130-torch213 group
    # leaves both commented out -- no wheel exists for this torch/CUDA ABI) and
    # transformer-engine / torchao (GB200-only training accelerators, in the "-train" group only,
    # and irrelevant to inference-only serving).
    "$PIP" install accelerate av cattrs "diffusers>=0.39.0" einops hydra-core imageio-ffmpeg \
        imageio loguru msgpack nvidia-cudnn-frontend nvidia-ml-py obstore omegaconf pydantic \
        requests scipy termcolor "transformers>=4.57.1,<5.0.0" tyro websockets

    # The policy-server group: a WebSocket server wrapping OpenPI's own WebsocketPolicyServer --
    # which is why the harness-side client written for pi0.5 is reused unchanged for Cosmos.
    # --no-deps, then its real deps by hand, deliberately NOT respecting openpi-client's
    # numpy<2.0.0 pin: letting the resolver satisfy it forces dm-tree back to 0.1.8, the last
    # release before it shipped a cp313 wheel, and building 0.1.8 from source fails regardless of
    # cmake version (its vendored ~2022 abseil-cpp does not compile against gcc 13). numpy>=2
    # lets pip resolve dm-tree 0.1.10 and sidesteps the whole toolchain issue.
    "$PIP" install --no-deps openpi-server openpi-client
    "$PIP" install "filelock>=3.27.0" "dm-tree>=0.1.8" "msgpack>=1.0.5" "pillow>=9.0.0" \
        "websockets>=11.0"
    "$PIP" install --no-deps -e "$REPO/third_party/cosmos"

    # Upstream's quickstart is `uv sync --all-extras --group=cu130-torch213-train
    # --group=policy-server` -- and --all-extras is not optional convenience:
    # cosmos_framework/utils/config.py imports cosmos_framework.trainer at module scope (inside a
    # dataclass body), pulling in the entire training stack just to import the Config class the
    # DROID server's argument parsing depends on. Nearly all of these are declared only in
    # pyproject.toml's "train" extras, never in the base dependencies -- an upstream metadata gap.
    # megatron-core, lerobot and torchtitan are left out: they are uv git sources pinned to
    # specific revisions, not on PyPI, and nothing in the DROID inference path reaches them.
    # aioboto3 must be pinned: unpinned alongside multi-storage-client's boto3<2,>=1.36, pip's
    # resolver backtracks it to pre-2019 releases trying to jointly satisfy ~80 packages and never
    # recovers (ResolutionImpossible).
    # Installed in CHUNKS rather than one command, and this is not cosmetic. As a single
    # ~76-package solve, modern pip aborts with `resolution-too-deep` ("the dependency graph is
    # too complex for pip to solve efficiently") and installs NOTHING -- the peer built this env
    # on an older pip that merely backtracked slowly. Splitting it means several small solves
    # instead of one intractable joint one; the resulting package set is the same, and
    # `scripts/check_env_parity.py --sync-to-peer` reconciles any version that resolved
    # differently. Chunks are grouped by role so a failure points somewhere specific.
    # The two pins are load-bearing, not preferences:
    #   * aioboto3 unpinned alongside multi-storage-client's boto3<2,>=1.36 sends the resolver
    #     back to pre-2019 releases (whose own boto3 pins bottom out under 1.9.50) and it never
    #     recovers -- ResolutionImpossible.
    #   * multi-storage-client is pinned to the version upstream declares.
    local COSMOS_CHUNKS=(
        "iopath webdataset einx fvcore futureproof arrgh flopth"
        "multi-storage-client[boto3,google-cloud-storage,fsspec,observability-otel,vault]==0.44.0"
        "aioboto3>=15.0.0 aiofiles aiohttp blobfile boto3 botocore s3fs rsa"
        "pandas polars fastparquet zarr tensorstore h5py lz4 semver pytz packaging parse datasets more-itertools"
        "better-profanity nltk sentencepiece tiktoken ftfy qwen-vl-utils openai"
        "fastapi httpx flask gradio ray[serve] python-memcached psycopg2-binary pydispatcher"
        "kornia timm peft open-clip-torch lpips torch-fidelity torch-optimizer retinaface-py"
        "matplotlib mediapy moviepy imagecodecs scikit-image opencv-contrib-python pillow>=12.2.0"
        "librosa soundfile trimesh plyfile pygltflib polyscope xatlas glfw pyopengl"
        "slangtorch pycocotools py3nvml wandb typeguard pytest ninja pyyaml"
    )
    local chunk
    for chunk in "${COSMOS_CHUNKS[@]}"; do
        # shellcheck disable=SC2086
        "$PIP" install $chunk
    done
    # The bulk install drags huggingface-hub/transformers forward via gradio/datasets; re-pin
    # transformers to what cosmos-framework itself declares, last, so it sticks.
    "$PIP" install "transformers>=4.57.1,<5.0.0"
}

# ---------------------------------------------------------------- openpi (divergent)
setup_openpi() {
    # `uv sync` rewrites third_party/openpi/.venv in place. The pi0.5 server runs FROM that
    # venv and the reproduction matrix drives it for days, so guard this the same way the conda
    # envs are guarded -- resyncing under a live server is how a multi-day campaign turns into a
    # confusing mid-run failure.
    refuse_if_prefix_busy "$REPO/third_party/openpi/.venv" openpi || return 1
    # NOT at parity with polaris-openpi, and deliberately so -- see docs/env_parity.md. That env
    # carries upstream Physical-Intelligence/openpi with the pi05_droid_jointpos_polaris config;
    # this one carries the omarrayyann fork with pi05_droid_jointpos, which is the config the
    # MolmoSpaces leaderboard entry was produced with. Adopting the other would not be a version
    # difference, it would be a different checkpoint, and the reproduction would fail by
    # construction. Built by uv, not conda, because that is what the fork supports.
    ( cd "$REPO/third_party/openpi" && uv sync )
}

# ---------------------------------------------------------------- verification
check_env() {  # $1 = env name
    local name="$1" PY rc=0
    if [ "$name" = "openpi" ]; then
        PY="$REPO/third_party/openpi/.venv/bin/python"
        [ -x "$PY" ] || { echo "  MISSING: third_party/openpi/.venv (run: uv sync there)"; return 1; }
        "$PY" - <<'EOF' || rc=1
import jax, openpi.training.config as c
print(f"  jax {jax.__version__} devices={jax.devices()}")
names = {x.name for x in c._CONFIGS}
assert "pi05_droid_jointpos" in names, "pi05_droid_jointpos not registered (wrong openpi fork?)"
print("  openpi config pi05_droid_jointpos OK")
EOF
        [ $rc -eq 0 ] && echo "  $name OK (intentional divergence)" || echo "  $name FAILED"
        return $rc
    fi
    PY="$(py "$name")"
    if ! have_env "$name"; then echo "  MISSING: env $name"; return 1; fi
    case "$name" in
      mlspaces-classic|mlspaces-filament)
        WANT_RENDERER="${name#mlspaces-}" "$PY" - <<'EOF' || rc=1
import os, sys, mujoco
ok = True
want = os.environ["WANT_RENDERER"]
got = getattr(mujoco, "mjRENDERER", "classic")
print(f"  mujoco {mujoco.__version__} renderer={got}")
if got != want:
    print(f"  FAIL: env is named for the {want!r} renderer but mujoco reports {got!r}. "
          f"molmo_spaces/env/env.py:35 fixes the renderer at import from this value, so the "
          f"env name and the installed mujoco MUST agree or every task runs the wrong renderer.")
    ok = False
import molmo_spaces, numpy  # noqa: F401
print(f"  molmo_spaces importable, numpy {numpy.__version__}")
if int(numpy.__version__.split(".")[0]) < 2:
    print("  FAIL: numpy was downgraded below 2 -- openpi-client installed without --no-deps?")
    ok = False
import openpi_client  # noqa: F401
print("  openpi_client OK (harness-side msgpack websocket client)")
import json_numpy  # noqa: F401  MolmoAct2's client speaks HTTP+json_numpy, not websocket
print("  json_numpy OK (harness-side MolmoAct2 HTTP client)")
sys.exit(0 if ok else 1)
EOF
        ;;
      mlspaces-molmoact2)
        MOLMOACT2_REPO_ID="$MOLMOACT2_REPO_ID" "$PY" - <<'EOF' || rc=1
import json, os, sys, torch, transformers
print(f"  torch {torch.__version__} avail={torch.cuda.is_available()}  transformers {transformers.__version__}")
ok = True
# A check that passes on an env whose torch cannot see a GPU is worse than no check: the
# mlspaces-cosmos-policy env did exactly that, printing avail=False and still reporting OK,
# and the real failure surfaced much later as "CUDA is required for OmniMoTModel inference".
# Driver/runtime mismatches (cu130 torch on an r570 driver) look fine at import and fail at
# first allocation, so allocate.
if not torch.cuda.is_available():
    print("  FAIL: torch cannot see any GPU (driver/runtime mismatch?)"); ok = False
else:
    try:
        torch.zeros(8, device="cuda")
    except Exception as e:
        print(f"  FAIL: CUDA allocation failed: {type(e).__name__}: {str(e)[:160]}"); ok = False
if not transformers.__version__.startswith("4.57"):
    print("  FAIL: MolmoAct2's remote code wants transformers 4.57.x"); ok = False
_sm = os.environ["EXPECTED_SM"]
if _sm not in " ".join(torch.cuda.get_arch_list()):
    print(f"  FAIL: {_sm} missing from torch arch_list"); ok = False
import fastapi, json_numpy, uvicorn  # noqa: F401
from huggingface_hub import snapshot_download
repo_id = os.environ["MOLMOACT2_REPO_ID"]
try:
    snap = snapshot_download(repo_id=repo_id, local_files_only=True)
except Exception as e:
    print(f"  FAIL: checkpoint missing ({type(e).__name__}); fetch it with: hf download {repo_id}")
    sys.exit(1)
cfg = json.load(open(os.path.join(snap, "config.json")))
if cfg.get("model_type") != "molmoact2":
    print(f"  FAIL: model_type is {cfg.get('model_type')!r}, expected 'molmoact2'"); ok = False
# The two facts the harness-side action handling depends on.
tag = json.load(open(os.path.join(snap, "norm_stats.json")))["metadata_by_tag"].get("franka_droid")
if tag is None:
    print("  FAIL: norm_stats.json has no 'franka_droid' tag"); ok = False
else:
    if tag.get("control_mode") != "absolute joint pose":
        print(f"  FAIL: control_mode is {tag.get('control_mode')!r}"); ok = False
    if tag.get("action_horizon") != 15:
        print(f"  FAIL: action_horizon is {tag.get('action_horizon')}, not 15 "
              f"(MolmoAct2PolicyConfig.action_horizon must match)"); ok = False
    if ok:
        print(f"  MolmoAct2 checkpoint OK ({tag['control_mode']}, horizon {tag['action_horizon']})")
sys.exit(0 if ok else 1)
EOF
        ;;
      mlspaces-m2t2)
        M2T2_DIR="$REPO/third_party/m2t2" "$PY" - <<'EOF' || rc=1
import os, sys, torch
print(f"  torch {torch.__version__} avail={torch.cuda.is_available()}")
ok = True
# A check that passes on an env whose torch cannot see a GPU is worse than no check: the
# mlspaces-cosmos-policy env did exactly that, printing avail=False and still reporting OK,
# and the real failure surfaced much later as "CUDA is required for OmniMoTModel inference".
# Driver/runtime mismatches (cu130 torch on an r570 driver) look fine at import and fail at
# first allocation, so allocate.
if not torch.cuda.is_available():
    print("  FAIL: torch cannot see any GPU (driver/runtime mismatch?)"); ok = False
else:
    try:
        torch.zeros(8, device="cuda")
    except Exception as e:
        print(f"  FAIL: CUDA allocation failed: {type(e).__name__}: {str(e)[:160]}"); ok = False
_sm = os.environ["EXPECTED_SM"]
if _sm not in " ".join(torch.cuda.get_arch_list()):
    print(f"  FAIL: {_sm} missing from torch arch_list"); ok = False
try:
    import pointnet2_ops._ext  # noqa: F401  the CUDA extension, not just the python package
    print("  pointnet2_ops CUDA extension OK")
except Exception as e:
    print(f"  FAIL: pointnet2_ops: {type(e).__name__}: {e}"); ok = False
try:
    from m2t2.m2t2 import M2T2  # noqa: F401
    print("  m2t2 OK")
except Exception as e:
    print(f"  FAIL: m2t2 import: {type(e).__name__}: {e}"); ok = False
w = os.path.join(os.environ["M2T2_DIR"], "weights", "m2t2.pth")
if not os.path.isfile(w) or os.path.getsize(w) < 1_000_000:
    print(f"  FAIL: {w} missing or still a git-lfs pointer"); ok = False
sys.exit(0 if ok else 1)
EOF
        ;;
      mlspaces-tiptop)
        CUTAMP_VER="$CUTAMP_VER" "$PY" - <<'EOF' || rc=1
import os, sys, torch
from importlib.metadata import version, PackageNotFoundError
print(f"  torch {torch.__version__} avail={torch.cuda.is_available()}")
ok = True
# A check that passes on an env whose torch cannot see a GPU is worse than no check: the
# mlspaces-cosmos-policy env did exactly that, printing avail=False and still reporting OK,
# and the real failure surfaced much later as "CUDA is required for OmniMoTModel inference".
# Driver/runtime mismatches (cu130 torch on an r570 driver) look fine at import and fail at
# first allocation, so allocate.
if not torch.cuda.is_available():
    print("  FAIL: torch cannot see any GPU (driver/runtime mismatch?)"); ok = False
else:
    try:
        torch.zeros(8, device="cuda")
    except Exception as e:
        print(f"  FAIL: CUDA allocation failed: {type(e).__name__}: {str(e)[:160]}"); ok = False
_sm = os.environ["EXPECTED_SM"]
if _sm not in " ".join(torch.cuda.get_arch_list()):
    print(f"  FAIL: {_sm} missing from torch arch_list"); ok = False
try:
    import curobo  # noqa: F401
    from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401
    print("  curobo OK")
except Exception as e:
    print(f"  FAIL: curobo: {type(e).__name__}: {e}"); ok = False
want = os.environ["CUTAMP_VER"]
try:
    got = version("cutamp")
    print(f"  cutamp {got}")
    if got != want:
        print(f"  FAIL: tiptop asserts cutamp=={want}"); ok = False
except PackageNotFoundError:
    print("  FAIL: cutamp not installed"); ok = False
try:
    # Also proves bamboo-franka-client is importable: tiptop.utils imports it at module scope.
    from tiptop.tiptop_websocket_server import _run_server  # noqa: F401
    print("  tiptop server importable")
except Exception as e:
    print(f"  FAIL: tiptop import: {type(e).__name__}: {e}"); ok = False
if not os.environ.get("GOOGLE_API_KEY"):
    print("  FAIL: GOOGLE_API_KEY unset; TiPToP has no offline path for Gemini"); ok = False
sys.exit(0 if ok else 1)
EOF
        ;;
      mlspaces-dreamzero)
        "$PY" - <<'EOF' || rc=1
import inspect, os, sys, torch, flash_attn, transformers  # noqa: F401
print(f"  torch {torch.__version__} avail={torch.cuda.is_available()}  "
      f"flash_attn {flash_attn.__version__}  transformers {transformers.__version__}")
ok = True
# See the identical assertion in the other env checks: a driver/runtime mismatch imports
# cleanly and only fails at first allocation, so allocate rather than trust is_available().
if not torch.cuda.is_available():
    print("  FAIL: torch cannot see any GPU (driver/runtime mismatch?)"); ok = False
else:
    try:
        torch.zeros(8, device="cuda")
    except Exception as e:
        print(f"  FAIL: CUDA allocation failed: {type(e).__name__}: {str(e)[:160]}"); ok = False
_sm = os.environ["EXPECTED_SM"]
if _sm not in " ".join(torch.cuda.get_arch_list()):
    print(f"  FAIL: {_sm} missing from torch arch_list"); ok = False
import groot.vla.model.dreamzero.base_vla as bv
if "compute_device" not in inspect.getsource(bv.VLA.prepare_input):
    print("  FAIL: dreamzero single-GPU patch NOT applied "
          "(run scripts/apply_third_party_patches.sh dreamzero)"); ok = False
else:
    print("  dreamzero single-GPU patch applied")
sys.exit(0 if ok else 1)
EOF
        ;;
      mlspaces-cosmos-policy)
        # Checked WITH the CUDA 13 forward-compat libs on the path, because that is how
        # scripts/serve_cosmos.sh actually runs it. This env's torch is cu130 and needs an
        # r580 driver; this host has r570, so without these libs torch imports fine and then
        # cannot see a GPU. Verifying a configuration the server does not use would be
        # theatre in either direction -- a false FAIL here, or a false PASS if the assertion
        # were dropped instead.
        LD_LIBRARY_PATH="${CUDA_COMPAT_DIR:-$HOME/cuda-compat-13/usr/local/cuda-13.0/compat}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$PY" - <<'EOF' || rc=1
import os, sys, torch
print(f"  torch {torch.__version__} avail={torch.cuda.is_available()}")
ok = True
# A check that passes on an env whose torch cannot see a GPU is worse than no check: the
# mlspaces-cosmos-policy env did exactly that, printing avail=False and still reporting OK,
# and the real failure surfaced much later as "CUDA is required for OmniMoTModel inference".
# Driver/runtime mismatches (cu130 torch on an r570 driver) look fine at import and fail at
# first allocation, so allocate.
if not torch.cuda.is_available():
    print("  FAIL: torch cannot see any GPU (driver/runtime mismatch?)"); ok = False
else:
    try:
        torch.zeros(8, device="cuda")
    except Exception as e:
        print(f"  FAIL: CUDA allocation failed: {type(e).__name__}: {str(e)[:160]}"); ok = False
if "cu130" not in torch.__version__:
    print("  FAIL: torch is not the cu130 build"); ok = False
_sm = os.environ["EXPECTED_SM"]
if _sm not in " ".join(torch.cuda.get_arch_list()):
    print(f"  FAIL: {_sm} missing from torch arch_list"); ok = False
for mod, label in (("natten", "natten"),
                   ("openpi_server.websocket_policy_server", "openpi_server"),
                   ("cosmos_framework.scripts.action_policy_server_robolab",
                    "action_policy_server_robolab")):
    try:
        __import__(mod); print(f"  {label} OK")
    except Exception as e:
        print(f"  FAIL: {label}: {type(e).__name__}: {e}"); ok = False
sys.exit(0 if ok else 1)
EOF
        ;;
    esac
    [ $rc -eq 0 ] && echo "  $name OK" || echo "  $name FAILED"
    return $rc
}

# ---------------------------------------------------------------- main
FAIL=0
for t in "${TARGETS[@]}"; do
    echo "=== $t ==="
    if [ "$CHECK_ONLY" -eq 0 ]; then
        case "$t" in
            mlspaces-classic)       setup_classic ;;
            mlspaces-filament)      setup_filament ;;
            mlspaces-molmoact2)     setup_molmoact2 ;;
            mlspaces-m2t2)          setup_m2t2 ;;
            mlspaces-tiptop)        setup_tiptop ;;
            mlspaces-dreamzero)     setup_dreamzero ;;
            mlspaces-cosmos-policy) setup_cosmos_policy ;;
            openpi)                 setup_openpi ;;
        esac
    fi
    check_env "$t" || FAIL=1
done

if [ $FAIL -ne 0 ]; then
    echo; echo "one or more environments failed verification"; exit 1
fi
echo; echo "all environments OK"
