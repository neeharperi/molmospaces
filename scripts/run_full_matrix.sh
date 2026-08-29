#!/usr/bin/env bash
# Run one policy's full 9-task matrix at FULL COVERAGE (no --max_episodes), which is the only
# setting comparable to the leaderboard -- see docs/eval_reproduction.md's
# "--max_episodes OVERSAMPLES" section for why truncated runs are not.
#
#   scripts/run_full_matrix.sh <policy> [classic|filament|all] [all|group-a|group-b]
#
# The third argument selects which tasks to run, defaulting to `all` (9 tasks). `group-b` is
# the complete MolmoBot Combined set (the 7 bench-v2 tasks) -- the load-bearing pooled
# comparison against the leaderboard; `group-a` is the 2 bench-v1 tasks. Task membership and
# each task's renderer come from scripts/eval_common.py's TASKS/GROUP_A/GROUP_B, not from a
# second copy of the split maintained here, so the two can't drift.
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

POLICY="${1:?usage: run_full_matrix.sh <policy> [classic|filament|all] [all|group-a|group-b]}"
PHASE="${2:-all}"
GROUP="${3:-all}"
case "$GROUP" in all|group-a|group-b) ;; *) echo "bad group '$GROUP' (want all|group-a|group-b)" >&2; exit 2 ;; esac
DATE_TAG="${DATE_TAG:-20260819_full}"
CLASSIC_WORKERS="${CLASSIC_WORKERS:-4}"
FILAMENT_WORKERS="${FILAMENT_WORKERS:-1}"
# LANE AWARENESS. The campaign runs one copy of this script per policy, concurrently, each
# rendering on its own GPU. LANE_GPU is the nvidia-smi index this lane owns (logging and
# numactl only); MUJOCO_EGL_DEVICE_ID is the EGL device index that actually renders there.
#
# Both are REQUIRED, not defaulted, and that is the point. This used to default to 1, with a
# comment that the EGL index was reversed from nvidia-smi's ordering -- true of the 2-GPU
# Blackwell host the campaign started on, and not a fact that travels. A stale default on a
# 4-GPU host puts every lane on one card: nothing errors, no log says anything, the campaign
# just runs N times slower under N times the contention. A missing required variable fails at
# second zero instead. Run scripts/probe_egl_mapping.py for this host's table.
LANE_GPU="${LANE_GPU:?set LANE_GPU to the nvidia-smi GPU index this lane owns}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:?set from scripts/probe_egl_mapping.py for LANE_GPU=$LANE_GPU}"

# 7 lanes x 4 classic workers = 28 MuJoCo processes on 224 CPUs. Uncapped, each one's BLAS
# spawns a thread per core and they thrash rather than share.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"

# Task lists are derived from scripts/eval_common.py so the renderer split and the
# group membership are defined exactly once (eval_common has no third-party imports, so
# any python3 can read it).
tasks_for() {  # $1 = classic|filament
    python3 - "$1" "$GROUP" <<'EOF'
import sys
sys.path.insert(0, "scripts")
from eval_common import TASKS, GROUP_A, GROUP_B
renderer, group = sys.argv[1], sys.argv[2]
selected = {"all": tuple(TASKS), "group-a": GROUP_A, "group-b": GROUP_B}[group]
print(" ".join(t for t in selected if TASKS[t].renderer == renderer))
EOF
}

read -r -a CLASSIC_TASKS <<< "$(tasks_for classic)"
read -r -a FILAMENT_TASKS <<< "$(tasks_for filament)"

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL=False
export MLSPACES_PINNED_ASSETS_FILE="$PWD/reference/pinned_assets_20260816.json"

run_phase() {
    local env_name="$1" workers="$2"; shift 2
    # group-a has no filament tasks (and group-b none of bench-v1's), so a phase can be empty.
    [ "$#" -gt 0 ] || { echo "=== no $GROUP tasks for env=$env_name, skipping phase ==="; return 0; }
    conda activate "$env_name"
    for task in "$@"; do
        echo "=== [$(date +%H:%M:%S)] $POLICY / $task  (env=$env_name, workers=$workers, gpu=$LANE_GPU, egl=$MUJOCO_EGL_DEVICE_ID, FULL coverage) ==="
        python scripts/eval.py --policy "$POLICY" --task "$task" \
            --num_workers "$workers" --date "$DATE_TAG"
        rc=$?
        echo "    -> exit $rc"
        # Collected rather than fatal: one failing cell must not abort the rest of the lane,
        # but a lane that quietly failed 4 of 9 cells should not print an unqualified
        # "complete" either.
        [ $rc -eq 0 ] || FAILED_CELLS="${FAILED_CELLS:-} $task"
    done
    conda deactivate
}

[ "$PHASE" = "classic" ] || [ "$PHASE" = "all" ] && run_phase mlspaces-classic "$CLASSIC_WORKERS" "${CLASSIC_TASKS[@]}"
[ "$PHASE" = "filament" ] || [ "$PHASE" = "all" ] && run_phase mlspaces-filament "$FILAMENT_WORKERS" "${FILAMENT_TASKS[@]}"

echo "=== [$(date +%H:%M:%S)] $POLICY matrix phase='$PHASE' group='$GROUP' complete; failed cells:${FAILED_CELLS:- none} ==="
