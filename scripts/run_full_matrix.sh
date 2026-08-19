#!/usr/bin/env bash
# Run one policy's full 9-task matrix at FULL COVERAGE (no --max_episodes), which is the only
# setting comparable to the leaderboard -- see docs/eval_reproduction.md's
# "--max_episodes OVERSAMPLES" section for why truncated runs are not.
#
#   scripts/run_full_matrix.sh <policy> [classic|filament|all]
#
# Resumable: scripts/eval.py skips any cell that already has provenance.json + results.csv for
# the same --date, so re-invoking after an interruption picks up where it left off. Each cell
# is run independently, so one failing task does not abort the rest of the matrix.
#
# Renderer phases are separate because the two mujoco builds cannot coexist in one env
# (pyproject gives both extras the same import name); classic tasks run in mlspaces-classic,
# filament tasks in mlspaces-filament. Filament uses a single worker: 4 concurrent Vulkan
# contexts exhausted GPU handles on this machine (HandleAllocator arena is full).
set -uo pipefail
cd "$(dirname "$0")/.."

POLICY="${1:?usage: run_full_matrix.sh <policy> [classic|filament|all]}"
PHASE="${2:-all}"
DATE_TAG="${DATE_TAG:-20260819_full}"
CLASSIC_WORKERS="${CLASSIC_WORKERS:-4}"
FILAMENT_WORKERS="${FILAMENT_WORKERS:-1}"
# EGL device index -> physical GPU is REVERSED from nvidia-smi's ordering on this host
# (verified via DRM render-node PCI IDs): MUJOCO_EGL_DEVICE_ID=1 renders on nvidia-smi GPU0.
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}"

CLASSIC_TASKS=(Close-v1 Open-v1 Pick-v1.5 Pick-v2-classic)
FILAMENT_TASKS=(Pick-v2-filament Pick-v2-RandCam PnP-v2 PnP-NextTo-v2 PnP-Color-v2)

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL=False
export MLSPACES_PINNED_ASSETS_FILE="$PWD/reference/pinned_assets_20260816.json"

run_phase() {
    local env_name="$1" workers="$2"; shift 2
    conda activate "$env_name"
    for task in "$@"; do
        echo "=== [$(date +%H:%M:%S)] $POLICY / $task  (env=$env_name, workers=$workers, FULL coverage) ==="
        python scripts/eval.py --policy "$POLICY" --task "$task" \
            --num_workers "$workers" --date "$DATE_TAG"
        echo "    -> exit $?"
    done
    conda deactivate
}

[ "$PHASE" = "classic" ] || [ "$PHASE" = "all" ] && run_phase mlspaces-classic "$CLASSIC_WORKERS" "${CLASSIC_TASKS[@]}"
[ "$PHASE" = "filament" ] || [ "$PHASE" = "all" ] && run_phase mlspaces-filament "$FILAMENT_WORKERS" "${FILAMENT_TASKS[@]}"

echo "=== [$(date +%H:%M:%S)] $POLICY matrix phase='$PHASE' complete ==="
