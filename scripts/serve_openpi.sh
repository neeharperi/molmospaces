#!/usr/bin/env bash
# Serve an openpi DROID checkpoint (pi0.5 on :8080, pi0 on :8081) for the eval harness.
#
#   PORT=8080 GPU=3 CONFIG=pi05_droid_jointpos bash scripts/serve_openpi.sh
#   PORT=8081 GPU=3 CONFIG=pi0_droid_jointpos  bash scripts/serve_openpi.sh
#
# The reference campaign launched this by hand and never committed a script, which was fine
# when openpi had a GPU to itself. It does not here: the campaign runs seven policy servers
# across four cards, so two things that used to be implicit have to be explicit.
#
# 1. CUDA_VISIBLE_DEVICES. Without it openpi takes GPU0 regardless of the lane assignment.
#
# 2. JAX PREALLOCATION -- the one that actually bites. jax[cuda12] grabs 75% of the visible
#    device at startup by default: ~71 GiB of a 95 GiB H100 NVL, for a model whose weights are
#    ~12 GiB. Any server sharing that card then fails to allocate, and the error surfaces over
#    there (MolmoAct2 OOMing at warmup) rather than here, which makes it look like the other
#    policy's bug. Disabling preallocation and capping the pool keeps the footprint near the
#    real working set. This is also why pi0 and pi0.5 can share one card at all.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8080}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-pi05_droid_jointpos}"
CKPT_DIR="${CKPT_DIR:-checkpoints/$CONFIG}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.20}"

[ -d "third_party/openpi/$CKPT_DIR" ] || {
  echo "checkpoint not found: third_party/openpi/$CKPT_DIR" >&2
  echo "fetch it with: gsutil -m cp -r gs://openpi-assets/checkpoints/$CONFIG third_party/openpi/checkpoints/" >&2
  exit 1
}

cd third_party/openpi
CUDA_VISIBLE_DEVICES="$GPU" uv run scripts/serve_policy.py --port="$PORT" policy:checkpoint \
  --policy.config="$CONFIG" --policy.dir="$CKPT_DIR/"
