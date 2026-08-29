#!/usr/bin/env bash
# Serve Cosmos3-{Edge,Nano}-Policy-DROID for the eval harness (default port 8003, matching
# CosmosPolicyConfig.remote_config). Upstream is NVIDIA/cosmos-framework, not nvlabs/cosmos-policy
# -- see docs/eval_reproduction.md's Cosmos-Policy section for why. cosmos_framework is vendored
# at third_party/cosmos (pinned SHA in docs/env_parity.md) and served from this repo's own
# `mlspaces-cosmos-policy` env, a mirror of robot-prompt-opt's `polaris-cosmos-policy`. It used
# to run inside that sibling env in place, which coupled this repo's results to a `pip install`
# over there; see docs/env_parity.md for why that was separated.
#
# Default checkpoint is Edge (4B, ~8-10 GiB, faster). Switch to Nano (16B, ~32 GiB+) with:
#   CKPT=nvidia/Cosmos3-Nano-Policy-DROID PORT=8004 bash scripts/serve_cosmos.sh
# Both checkpoints auto-download from a non-gated HF repo on first launch (already cached on
# this machine under ~/.cache/huggingface from the sibling project's own setup) -- no manual
# checkpoints/ layout step, closer to pi0.5's gs:// auto-fetch than DreamZero/MolmoAct2's.
# The HF cache is shared between projects deliberately: weights are not part of an environment,
# and a second 20-45 GB copy per checkpoint buys nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8003}"
GPU="${GPU:-0}"
CKPT="${CKPT:-nvidia/Cosmos3-Edge-Policy-DROID}"

# Edge's documented invocation adds these two flags; Nano's does not. Keyed off $CKPT rather
# than a separate "clear this when switching" env var -- "${VAR:-default}" treats an
# explicitly empty override the same as unset.
if [ -z "${COSMOS_EXTRA_ARGS+set}" ]; then
    if [ "$CKPT" = "nvidia/Cosmos3-Edge-Policy-DROID" ]; then
        COSMOS_EXTRA_ARGS="--format-prompt-as-json True --guidance-interval 960 1001"
    else
        COSMOS_EXTRA_ARGS=""
    fi
fi

# CUDA FORWARD COMPATIBILITY, and this env is the only one that needs it.
#
# cosmos-framework pins torch 2.13.0+cu130, i.e. CUDA 13.0, which requires an r580 driver.
# This host runs 570.207 (CUDA 12.8), so torch loads fine and then reports
# `torch.cuda.is_available() == False`, and the server dies with the far-downstream message
# "RuntimeError: CUDA is required for OmniMoTModel inference in this repo."
#
# NVIDIA's supported answer for exactly this case -- newer CUDA runtime, older driver, on a
# datacenter GPU -- is the cuda-compat package, which ships a forward-compatible libcuda.so
# that talks to the older kernel module. H100 is a supported (Hopper) part. Unpacked without
# root by scripts/install_nvidia_gl.sh.
#
# Scoped to THIS server rather than exported globally on purpose: every other env here runs
# cu128/cu129 torch against the stock driver and works, and putting a forward-compat libcuda
# ahead of them on the library path is a change none of them need.
#
# The alternative -- downgrading this env to a cu128 torch -- would break the pinned
# natten==0.21.6+cu130.torch213 wheel and would silently change the environment the
# leaderboard comparison is supposed to license.
_COMPAT="${CUDA_COMPAT_DIR:-$HOME/cuda-compat-13/usr/local/cuda-13.0/compat}"
if [ -d "$_COMPAT" ]; then
    export LD_LIBRARY_PATH="$_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    echo "warning: no CUDA 13 compat libs at $_COMPAT; torch will not see the GPUs." >&2
    echo "         run scripts/install_nvidia_gl.sh (no root required)" >&2
fi

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPU" HF_TOKEN="${HF_TOKEN:-}" \
  "${COSMOS_PYTHON:-${MLSPACES_ENVS:-$HOME/anaconda3/envs}/mlspaces-cosmos-policy/bin/python}" \
  scripts/serve_cosmos_policy.py \
  --checkpoint-path "$CKPT" --port "$PORT" $COSMOS_EXTRA_ARGS
