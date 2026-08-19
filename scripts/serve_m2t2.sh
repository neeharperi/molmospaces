#!/usr/bin/env bash
# Serve M2T2 grasp generation (:8123), which TiPToP requires -- it is not optional.
# tiptop_websocket_server.py posts a voxel-downsampled point cloud to /predict and gets back
# 6-DOF grasp candidates per object; without this server up, every planning call fails at the
# perception step (ConnectionRefusedError) and silently returns success=False, not a crash --
# see docs/eval_reproduction.md's TiPToP section for how this was found.
#
# M2T2 itself isn't vendored in this repo; reused in place from a sibling project on this
# machine (same user, same machine, same reasoning as DreamZero's checkpoint/env reuse).
set -euo pipefail
PORT="${PORT:-8123}"
GPU="${GPU:-0}"
M2T2_DIR="${M2T2_DIR:-$HOME/Workspace/robot-prompt-opt/third_party/m2t2}"
CKPT="${CKPT:-$M2T2_DIR/weights/m2t2.pth}"

[ -f "$CKPT" ] || {
  echo "M2T2 weights not found at $CKPT" >&2
  echo "Fetch them with: git clone https://huggingface.co/wentao-yuan/m2t2 $M2T2_DIR/weights" >&2
  exit 1
}

# M2T2 predates torch 2.6, which flipped torch.load's weights_only default to True. Its
# checkpoint carries a config object alongside the tensors, so the strict loader rejects it.
# Safe to restore the old behavior: the checkpoint is from the authors' own HuggingFace repo.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "$M2T2_DIR"
CUDA_VISIBLE_DEVICES="$GPU" \
  "${M2T2_PYTHON:-$HOME/anaconda3/envs/polaris-m2t2/bin/python}" m2t2_server.py \
  --port "$PORT" --checkpoint "$CKPT" --config "$M2T2_DIR/config.yaml" "$@"
