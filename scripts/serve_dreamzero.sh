#!/usr/bin/env bash
# Serve DreamZero-DROID (14B) for the eval harness. A single 48 GB card cannot hold this model
# (see scripts/dreamzero_patches/0001-single-gpu-48gb-inference.patch's header); this pipeline-
# shards a tail of the DiT onto a second GPU instead of torch.distributed's --nproc_per_node=2,
# which is a dead end here (each rank holds a full ~44.6 GiB replica and OOMs host RAM). See
# docs/eval_reproduction.md's DreamZero section for the measured footprint and GPU assignment.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-5000}"
GPUS="${GPUS:-1,0}"  # order matters: index 0 (primary, full resident weights) gets the listed
                      # GPU that has the most headroom free at launch time.
DIT_SPLIT="${DIT_SPLIT:-12}"
MODEL="${MODEL:-$PWD/third_party/dreamzero/checkpoints/DreamZero-DROID}"

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
export DREAMZERO_OFFLOAD_TEXT_ENCODER="${DREAMZERO_OFFLOAD_TEXT_ENCODER:-1}"
export DREAMZERO_OFFLOAD_IMAGE_ENCODER="${DREAMZERO_OFFLOAD_IMAGE_ENCODER:-1}"
export DREAMZERO_DIT_SPLIT="$DIT_SPLIT"
export DREAMZERO_DISABLE_DREAM_VIDEO="${DREAMZERO_DISABLE_DREAM_VIDEO:-1}"

CUDA_VISIBLE_DEVICES="$GPUS" \
  "${DREAMZERO_PYTHON:-${MLSPACES_ENVS:-$HOME/anaconda3/envs}/mlspaces-dreamzero/bin/python}" -m torch.distributed.run \
  --standalone --nproc_per_node=1 \
  third_party/dreamzero/socket_test_optimized_AR.py \
  --port "$PORT" --enable-dit-cache --model-path "$MODEL"
