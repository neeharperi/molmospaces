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
source "$(dirname "${BASH_SOURCE[0]:-$0}")/lib/models_dir.sh"
cd "$MLSPACES_MODELS_DIR/openpi"

# Where the data and the run artefacts live. Nothing host-specific is baked into openpi's
# configs -- every one of these is a tyro CLI flag, assembled into FLAGS below -- so this
# block is the whole of what has to change on a new machine.
OPENPI_DATA_ROOT="${OPENPI_DATA_ROOT:-/data/nperi/openpi}"
DROID_RLDS_DIR="${DROID_RLDS_DIR:-$OPENPI_DATA_ROOT/rlds}"     # parent of droid/1.0.1
OPENPI_ASSETS_DIR="${OPENPI_ASSETS_DIR:-$OPENPI_DATA_ROOT/assets}"
OPENPI_CKPT_DIR="${OPENPI_CKPT_DIR:-$OPENPI_DATA_ROOT/checkpoints}"
UV="${UV:-$(command -v uv)}"
[ -x "$UV" ] || { echo "uv not found on PATH; set UV=/path/to/uv" >&2; exit 1; }

# Fail on the environment before doing anything, and in the order a human would check it.
# openpi's configs keep upstream's <path_to_droid_rlds_dataset> sentinel, so this script is
# the only thing that knows where the data lives -- a wrong value here should say so now
# rather than surface inside TFDS half a minute later.
[ -d "$DROID_RLDS_DIR/droid" ] || {
  echo "DROID_RLDS_DIR=$DROID_RLDS_DIR does not contain a droid/ directory." >&2
  echo "It must be the PARENT of droid/1.0.1. That tree is ~1.7 TB; fetch it with" >&2
  echo "  gsutil -m cp -r gs://gresearch/robotics/droid/1.0.1 \"$DROID_RLDS_DIR/droid/\"" >&2
  exit 1
}
# $OPENPI_DATA_ROOT is typically on a volume the user does not own yet. Say which command
# fixes it rather than letting mkdir's bare "Permission denied" be the whole message.
if ! mkdir -p "$OPENPI_DATA_ROOT" 2>/dev/null; then
  echo "cannot create OPENPI_DATA_ROOT=$OPENPI_DATA_ROOT" >&2
  echo "  sudo mkdir -p $OPENPI_DATA_ROOT && sudo chown \"$(id -un):$(id -gn)\" $OPENPI_DATA_ROOT" >&2
  echo "or set OPENPI_DATA_ROOT somewhere writable." >&2
  exit 1
fi

MODEL="${MODEL:-pi05}"
SMOKE="${SMOKE:-0}"
ORACLE="${ORACLE:-0}"
RESUME="${RESUME:-0}"
EXP="${EXP:-base2droid_$(date +%Y%m%d)}"

# One GPU list, not a per-model pair. An earlier revision put pi05 on 0,1 and pi0 on 2,3 so
# both could train at once; this host has two cards, so they cannot, and MODEL now selects
# only the config.
case "$MODEL" in
  pi05) CFG="pi05_droid_jointpos_from_base" ;;
  pi0)  CFG="pi0_droid_jointpos_from_base"  ;;
  *) echo "MODEL must be pi0 or pi05" >&2; exit 1 ;;
esac
GPUS="${GPUS:-0,1}"

if [ "$ORACLE" = "1" ]; then
  # Refuses rather than running something else. The premise -- that a run initialised from
  # the already-DROID checkpoint should show a much lower step-0 loss than one from base --
  # was measured and is FALSE: the reference scores 3.21 on real DROID against base's 0.041
  # and random init's 2.05, while still scoring 23.3% in sim. Real-DROID loss and sim success
  # measure different things, so it was never a valid gate. Gate on sim success instead:
  # scripts/gate_openpi_ckpt.sh.
  echo "ORACLE=1 is gone: its premise was tested and is false. Use gate_openpi_ckpt.sh." >&2
  exit 2
fi
if [ "$SMOKE" = "1" ]; then
  # No separate _smoke configs. They existed to hold a step count and a save interval, which
  # are both CLI flags -- and a config that only differs by those is a config that drifts
  # from the one it is meant to be smoke-testing.
  #
  # On a 2x48GB host this is not just a pre-flight, it is very likely the only thing that
  # runs: the full recipe is bs128 for 200k steps, and openpi has no gradient accumulation
  # with which to trade batch size for memory at a fixed effective batch. Establish the real
  # footprint here before planning a full run.
  SMOKE_FLAGS=(--num-train-steps="${SMOKE_STEPS:-1000}" --save-interval=500 --no-wandb-enabled)
  EXP="smoke_$(date +%Y%m%d_%H%M%S)"
else
  SMOKE_FLAGS=()
fi

# 0.95. The 0.75 this used to carry was sized for 93.6 GiB H100s, where it left ~23 GB/GPU
# for the ladder gate's inference server so a run could be evaluated mid-flight. On a 48 GB
# card 0.75 is ~35.8 GB, which does not even cover the optimizer state, so that design is not
# available here: train, then gate, rather than both at once.
# One number for both cards, which is worth knowing before raising it: on this host GPU 1
# also carries Xorg, gnome-shell and a browser -- about 900 MB -- so 0.95 asks for 46.5 GB of
# a card that has 48.0 free. Measured 2026-09-16: a bs2 run at 0.95 still died inside the
# first train_step, needing 8.5 GB beyond budget, and a bs8 run at 0.85 needed 8.9 GB -- a 4x
# batch cut moving the shortfall by 5%, which says the failing allocation is the optimizer
# state rather than activations. See docs/eval_reproduction.md: no batch size makes this fit
# two 48 GB cards.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEMFRAC:-0.95}"
export CUDA_VISIBLE_DEVICES="$GPUS"
# droid_rlds_dataset.py already calls tf.config.set_visible_devices([], "GPU"), but that
# runs after TF has initialised its CUDA context; this stops it a step earlier.
export TF_FORCE_GPU_ALLOW_GROWTH=true
# No wandb credentials exist on this host (no ~/.netrc, no WANDB_API_KEY), and
# train.py's init_wandb would otherwise block on an interactive login.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$OPENPI_DATA_ROOT/wandb}"
mkdir -p "$WANDB_DIR"

FLAGS=(--exp-name="$EXP")
FLAGS+=(--data.rlds-data-dir="$DROID_RLDS_DIR")
FLAGS+=(--assets-base-dir="$OPENPI_ASSETS_DIR")
FLAGS+=(--checkpoint-base-dir="$OPENPI_CKPT_DIR")
[ -n "${BATCH_SIZE:-}" ] && FLAGS+=(--batch-size="$BATCH_SIZE")
[ "$RESUME" = "1" ] && FLAGS+=(--resume)
FLAGS+=("${SMOKE_FLAGS[@]}")
# Anything else, verbatim. Needed for a run against a dataset that is not the full release:
# the `*_from_base` configs default filter_dict_path to the published
# droid_sample_ranges_v1_0_1.json, whose keys are the real 1.0.1 episodes, so a 100-episode
# sample would be filtered to nothing -- and every knob openpi has is a tyro flag, so a
# pass-through is the whole fix.
#
#   EXTRA_FLAGS="--data.datasets.0.filter-dict-path=None" MODEL=pi05 SMOKE=1 ...
#
# Deliberately word-split rather than an array: this is a shell launcher meant to be typed.
# shellcheck disable=SC2206
[ -n "${EXTRA_FLAGS:-}" ] && FLAGS+=(${EXTRA_FLAGS})

# Somewhere durable, not a scratch directory. The first pilot logged into a per-session
# /tmp path that disappeared with the session -- the loss history of a ten-day run should
# not be that easy to lose.
LOG_DIR="${LOG_DIR:-$OPENPI_DATA_ROOT/logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${CFG}__${EXP}.log"

echo "config=$CFG exp=$EXP gpus=$GPUS memfrac=$XLA_PYTHON_CLIENT_MEM_FRACTION venv=.venv-train"
echo "rlds=$DROID_RLDS_DIR assets=$OPENPI_ASSETS_DIR ckpt=$OPENPI_CKPT_DIR"
echo "log=$LOG"
exec > >(tee -a "$LOG") 2>&1
exec env UV_PROJECT_ENVIRONMENT=.venv-train \
  "$UV" run --group rlds scripts/train.py "$CFG" "${FLAGS[@]}"
