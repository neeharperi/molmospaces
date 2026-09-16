#!/usr/bin/env bash
# Serve DreamZero-DROID (14B) for the eval harness. A single 48 GB card cannot hold this model
# (see the checkout's own single-GPU commits, 36372a6..655db67); this pipeline-
# shards a tail of the DiT onto a second GPU instead of torch.distributed's --nproc_per_node=2,
# which is a dead end here (each rank holds a full ~44.6 GiB replica and OOMs host RAM). See
# docs/eval_reproduction.md's DreamZero section for the measured footprint and GPU assignment.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "${BASH_SOURCE[0]:-$0}")/lib/models_dir.sh"
PORT="${PORT:-5000}"
GPUS="${GPUS:-1,0}"  # order matters: index 0 (primary, full resident weights) gets the listed
                      # GPU that has the most headroom free at launch time.
# 12, not 0: this host has two 48 GB cards, and the model does not fit one. Stock it peaks
# near 47.5 GiB against ~46.2 usable and OOMs on the THIRD inference of an episode -- the
# first two succeed, which is what makes it a confusing failure rather than an obvious one.
# Moving the last 12 of 40 DiT blocks to the second GPU frees ~11 GiB on the primary, at a
# documented ~20% throughput cost. On a 95 GB card set DIT_SPLIT=0 and pay neither.
DIT_SPLIT="${DIT_SPLIT:-12}"
MODEL="${MODEL:-$MLSPACES_MODELS_DIR/dreamzero/checkpoints/DreamZero-DROID}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# deepspeed is unused at inference but transformers imports it, and it hard-errors unless nvcc
# exists for its compatibility probe. mlspaces-dreamzero itself has no CUDA toolkit installed,
# so this borrows nvcc from mlspaces-tiptop, which does (it needs one to build cuRobo/cuTAMP).
# Any env with a toolchain works; nothing here actually runs deepspeed.
_envs="${MLSPACES_ENVS:-$HOME/anaconda3/envs}"
if [ -z "${CUDA_HOME:-}" ]; then
    for _e in mlspaces-tiptop mlspaces-m2t2; do
        [ -x "$_envs/$_e/bin/nvcc" ] && { CUDA_HOME="$_envs/$_e"; break; }
    done
fi
export CUDA_HOME
# The reduce-overhead compiles of the image encoder and VAE hold ~5 GB in CUDA-graph private
# pools; disabled here to keep the memory instead of the ~compile speedup.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
# Encoder offloading ON, because this host is the 48 GB-card case these were written for.
# Evicting the 10.58 GB text encoder and the 1.18 GB image encoder to CPU between uses is what
# lets the model fit at all here. Purely a memory/speed trade: the weights and the arithmetic
# are identical whether they live on the GPU or get copied there per call.
#
# Turn both OFF on a big card. On a 95 GB H100 with the DiT unsharded the server sits at
# ~33 GiB, the eviction buys nothing, and the reload per inference is pure cost -- measured
# at 1531 image-encoder reloads in a single Open-v1 cell.
export DREAMZERO_OFFLOAD_TEXT_ENCODER="${DREAMZERO_OFFLOAD_TEXT_ENCODER:-1}"
export DREAMZERO_OFFLOAD_IMAGE_ENCODER="${DREAMZERO_OFFLOAD_IMAGE_ENCODER:-1}"
export DREAMZERO_DIT_SPLIT="$DIT_SPLIT"
export DREAMZERO_DISABLE_DREAM_VIDEO="${DREAMZERO_DISABLE_DREAM_VIDEO:-1}"

CUDA_VISIBLE_DEVICES="$GPUS" \
  "${DREAMZERO_PYTHON:-${MLSPACES_ENVS:-$HOME/anaconda3/envs}/mlspaces-dreamzero/bin/python}" -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  "$MLSPACES_MODELS_DIR/dreamzero/socket_test_optimized_AR.py" \
  --port "$PORT" --enable-dit-cache --model-path "$MODEL"
