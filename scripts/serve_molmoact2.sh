#!/usr/bin/env bash
# Serve MolmoAct2-DROID (:8000) for the eval harness.
#
#   GPU=3 PORT=8000 bash scripts/serve_molmoact2.sh
#
# Launched by hand in the reference campaign; scripted here so the lane assignment is explicit
# and reproducible, matching serve_cosmos.sh's GPU=/PORT= convention.
#
# Runs from this repo's own mlspaces-molmoact2 env (torch 2.8.0+cu129), NOT from
# the molmoact2 checkout's uv venv: the submodule's pyproject pins torch 2.5.1+cu121, and the
# env recipe in scripts/setup_envs.sh is what the campaign's results are pinned to.
#
# The vendored host_server_droid.py needs scripts/molmoact2_patches/0001-*.patch applied --
# the live allenai/MolmoAct2-DROID remote code (fetched fresh via trust_remote_code, so NOT
# pinned by the submodule) renamed action_mode -> inference_action_mode, and the unpatched
# call raises. That rename landed upstream in allenai/molmoact2, so a current checkout has it.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "${BASH_SOURCE[0]:-$0}")/lib/models_dir.sh"
# 8102, not 8000: droid's own openpi server holds :8000 and its molmoact2 server holds
# :8101 (this repo lives in droid/third_party/). See scripts/eval_common.py.
PORT="${PORT:-8102}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="$GPU" HF_TOKEN="${HF_TOKEN:-}" \
  "${MOLMOACT2_PYTHON:-${MLSPACES_ENVS:-$HOME/anaconda3/envs}/mlspaces-molmoact2/bin/python}" \
  "$MLSPACES_MODELS_DIR/molmoact2/examples/droid/host_server_droid.py" \
  --host 0.0.0.0 --port "$PORT" "$@"
