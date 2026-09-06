#!/usr/bin/env bash
# Serve M2T2 grasp generation (:8123), which TiPToP requires -- it is not optional.
# tiptop_websocket_server.py posts a voxel-downsampled point cloud to /predict and gets back
# 6-DOF grasp candidates per object; without this server up, every planning call fails at the
# perception step (ConnectionRefusedError) and silently returns success=False, not a crash --
# see docs/eval_reproduction.md's TiPToP section for how this was found.
#
# M2T2 is vendored at third_party/m2t2 (pinned SHA in docs/env_parity.md) and served from this
# repo's own `mlspaces-m2t2` env, a mirror of robot-prompt-opt's `polaris-m2t2`. Both are built
# by scripts/setup_envs.sh mlspaces-m2t2, which also git-lfs-pulls the weights.
set -euo pipefail
PORT="${PORT:-8123}"
GPU="${GPU:-0}"
M2T2_DIR="${M2T2_DIR:-$PWD/third_party/m2t2}"
CKPT="${CKPT:-$M2T2_DIR/weights/m2t2.pth}"

[ -f "$CKPT" ] || {
  echo "M2T2 weights not found at $CKPT" >&2
  echo "Fetch them with: bash scripts/setup_envs.sh mlspaces-m2t2" >&2
  exit 1
}

# M2T2 predates torch 2.6, which flipped torch.load's weights_only default to True. Its
# checkpoint carries a config object alongside the tensors, so the strict loader rejects it.
# Safe to restore the old behavior: the checkpoint is from the authors' own HuggingFace repo.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "$M2T2_DIR"
CUDA_VISIBLE_DEVICES="$GPU" \
  "${M2T2_PYTHON:-${MLSPACES_ENVS:-$HOME/anaconda3/envs}/mlspaces-m2t2/bin/python}" m2t2_server.py \
  --port "$PORT" --checkpoint "$CKPT" --config "$M2T2_DIR/config.yaml" "$@"
