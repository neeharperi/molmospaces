#!/usr/bin/env bash
# Launch a base->DROID openpi fine-tune on one NVLink-bonded GPU pair.
#
#   MODEL=pi05 bash scripts/train_openpi_droid.sh           # GPUs 0,1
#   MODEL=pi0  bash scripts/train_openpi_droid.sh           # GPUs 2,3
#   MODEL=pi05 SMOKE=1 bash scripts/train_openpi_droid.sh   # 1k-step smoke
#   MODEL=pi0  SMOKE=1 bash scripts/train_openpi_droid.sh   # 1k-step smoke (pi0)
#   MODEL=pi05 ORACLE=1 bash scripts/train_openpi_droid.sh  # DEAD control, see below
#   MODEL=pi05 RESUME=1 bash scripts/train_openpi_droid.sh  # resume in place
#   MODEL=pi05 MEMFRAC=0.9 bash scripts/train_openpi_droid.sh  # no gate headroom
#
# MEMFRAC (default 0.75) is the JAX preallocation fraction. It is deliberately below the
# 0.9 the first pilot used, so ~23 GB/GPU stays free for the ladder gate's inference server
# and renderer. Raise it only for a run nothing needs to evaluate mid-flight.
#
# GPU pairing is not cosmetic. `nvidia-smi topo -m` on this host reports NV12 between
# GPUs 0-1 and between 2-3, and SYS (cross-NUMA PCIe) between the pairs -- with CPU
# affinity split 0-55,112-167 / 56-111,168-223 to match. fsdp_devices=2 confined to one
# pair keeps every parameter all-gather on the 318 GB/s NVLink bridge; a 4-way mesh
# would cross NUMA on every step. Running the two models as two concurrent 2-GPU jobs
# therefore finishes both in the wall-clock of one, with no cross-pair traffic.
#
# This uses .venv-train, NOT .venv. The serving venv must stay byte-stable: the archived
# campaign results in runs/ were produced by it, and scripts/serve_openpi.sh still uses
# it. `uv sync --group rlds` into a live serving venv can also swap .so files under a
# running server process.
set -euo pipefail
cd "$(dirname "$0")/../third_party/openpi"

MODEL="${MODEL:-pi05}"
SMOKE="${SMOKE:-0}"
ORACLE="${ORACLE:-0}"
RESUME="${RESUME:-0}"
EXP="${EXP:-base2droid_$(date +%Y%m%d)}"

case "$MODEL" in
  pi05) GPUS="${GPUS:-0,1}"; CFG="pi05_droid_jointpos_from_base" ;;
  pi0)  GPUS="${GPUS:-2,3}"; CFG="pi0_droid_jointpos_from_base"  ;;
  *) echo "MODEL must be pi0 or pi05" >&2; exit 1 ;;
esac
if [ "$ORACLE" = "1" ]; then
  # DEAD -- retained only so old invocations do not silently do something else.
  # The premise (a run initialised from the already-DROID checkpoint should show a much
  # lower step-0 loss than one from base) was measured and is FALSE: the reference scores
  # 3.21 on real DROID against base's 0.041 and random init's 2.05, while still scoring
  # 23.3% in sim. Real-DROID loss and sim success measure different things, so this is not
  # a valid gate. Gate on sim success instead (scripts/gate_openpi_ckpt.sh).
  echo "WARNING: ORACLE=1 is a dead control; its premise was tested and is false." >&2
  CFG="pi05_droid_jointpos_oracle_smoke"; EXP="oracle_$(date +%Y%m%d_%H%M%S)"
elif [ "$SMOKE" = "1" ]; then
  CFG="${CFG}_smoke"; EXP="smoke_$(date +%Y%m%d_%H%M%S)"
fi

# 0.75, not 0.9. At 0.9 JAX preallocates ~86 GB of each 93.6 GiB card and leaves ~8.5 GB --
# not enough for the ~9 GB openpi inference server the ladder gate needs, so gating an
# in-flight run would mean stopping it (and paying ~24 min of time-to-first-batch on resume).
# 0.75 leaves ~23 GB/GPU, enough for the gate server plus a classic renderer, at no cost to
# training as long as it does not OOM -- which SMOKE=1 is there to check before the real launch.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEMFRAC:-0.75}"
export CUDA_VISIBLE_DEVICES="$GPUS"
# droid_rlds_dataset.py already calls tf.config.set_visible_devices([], "GPU"), but that
# runs after TF has initialised its CUDA context; this stops it a step earlier.
export TF_FORCE_GPU_ALLOW_GROWTH=true
# No wandb credentials exist on this host (no ~/.netrc, no WANDB_API_KEY), and
# train.py's init_wandb would otherwise block on an interactive login.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-/data/neehar/openpi_runs/wandb}"
mkdir -p "$WANDB_DIR"

FLAGS=(--exp-name="$EXP")
[ "$RESUME" = "1" ] && FLAGS+=(--resume)

# Log to a stable in-repo path. The first pilot logged into a Claude session's scratchpad
# under /tmp, which is per-session and disappears with it -- the loss history of a 10-day run
# should not be that easy to lose.
LOG_DIR="/data/neehar/Workspace/molmospaces/runs/_train"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${CFG}__${EXP}.log"

echo "config=$CFG exp=$EXP gpus=$GPUS memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION venv=.venv-train"
echo "log=$LOG"
exec > >(tee -a "$LOG") 2>&1
exec env UV_PROJECT_ENVIRONMENT=.venv-train \
  /data/neehar/.local/bin/uv run --group rlds scripts/train.py "$CFG" "${FLAGS[@]}"
