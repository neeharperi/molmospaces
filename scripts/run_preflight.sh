#!/usr/bin/env bash
# Run the pre-flight cells: two full-coverage cells per policy, chosen by
# scripts/preflight_plan.py, then compare them against the leaderboard.
#
#   DATE_TAG=20260828_full bash scripts/run_preflight.sh classic
#   DATE_TAG=20260828_full bash scripts/run_preflight.sh filament
#
# Split by renderer because the two need different conda envs and different worker counts,
# and because the classic half can start before the filament worker-count question is settled.
#
# FULL COVERAGE, no --max_episodes -- these numbers are compared to the leaderboard, and
# --max_episodes both under-covers categories and oversamples episodes (see
# docs/eval_reproduction.md). Lanes run concurrently, one per policy, each rendering on its
# own GPU per scripts/launch_campaign.sh's LANES table.
set -uo pipefail
cd "$(dirname "$0")/.."
PHASE="${1:?usage: run_preflight.sh <classic|filament>}"
DATE_TAG="${DATE_TAG:?set DATE_TAG}"
CLASSIC_WORKERS="${CLASSIC_WORKERS:-4}"
FILAMENT_WORKERS="${FILAMENT_WORKERS:-4}"

# Per-policy worker override. The worker count must not exceed what the policy's SERVER can
# actually serve concurrently, or the extra workers do not add throughput -- they add
# handshake contention, and lost episodes.
#
# dreamzero: its server runs --nproc_per_node=1, i.e. exactly one inference at a time. Four
# eval workers against it produced 660 handshake timeouts and, after the client's 5-attempt
# give-up, 5 SKIPPED episodes in a single cell. Skipped episodes silently shrink a cell that
# is supposed to be the benchmark's full episode set, which is precisely what makes a number
# non-comparable to the leaderboard. One worker per inference slot.
declare -A POLICY_WORKERS=( [dreamzero]=1 )
workers_for() {  # $1 = policy, $2 = default
    echo "${POLICY_WORKERS[$1]:-$2}"
}
EGL_MAP="runs/_egl_mapping.txt"
egl_for() { awk -v g="$1" '$1 !~ /^#/ && $1 == g {print $2}' "$EGL_MAP"; }

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
. "$PWD/scripts/nvidia_gl_env.sh"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL=False
export MLSPACES_PINNED_ASSETS_FILE="$PWD/reference/pinned_assets_20260816.json"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# policy:gpu:classic_task[,...]:filament_task[,...]  -- from scripts/preflight_plan.py.
# pi0 and dreamzero have leaderboard entries ONLY for Open-v1/Close-v1, so their pre-flight is
# those two (both classic) and they have no filament cell. That gap is real and is reported by
# preflight_plan.py rather than hidden.
CELLS=(
  "pi05_droid:3:Pick-v2-classic:PnP-NextTo-v2"
  "pi0_droid:3:Open-v1,Close-v1:"
  "molmoact2_droid:2:Pick-v2-classic:PnP-NextTo-v2"
  "tiptop:2:Pick-v2-classic:PnP-NextTo-v2"
  "dreamzero:0:Open-v1,Close-v1:"
  "cosmos_edge:1:Pick-v2-classic:PnP-NextTo-v2"
  "cosmos_nano:1:Pick-v2-classic:PnP-NextTo-v2"
)

mkdir -p runs/_lanes
for spec in "${CELLS[@]}"; do
    IFS=':' read -r policy gpu classic filament <<< "$spec"
    tasks="$classic"; env_name=mlspaces-classic; workers="$(workers_for "$policy" "$CLASSIC_WORKERS")"
    if [ "$PHASE" = "filament" ]; then
        tasks="$filament"; env_name=mlspaces-filament; workers="$(workers_for "$policy" "$FILAMENT_WORKERS")"
    fi
    [ -n "$tasks" ] || { echo "  $policy: no $PHASE pre-flight cell (no leaderboard entry)"; continue; }
    egl="$(egl_for "$gpu")"
    log="runs/_lanes/preflight_${PHASE}_${policy}_${DATE_TAG}.log"
    (
        conda activate "$env_name"
        export MUJOCO_EGL_DEVICE_ID="$egl"
        IFS=','; for t in $tasks; do unset IFS
            echo "=== [$(date +%H:%M:%S)] $policy / $t (env=$env_name, workers=$workers, gpu=$gpu) ==="
            python scripts/eval.py --policy "$policy" --task "$t" \
                --num_workers "$workers" --date "$DATE_TAG"
            echo "    -> exit $?"
        done
    ) > "$log" 2>&1 &
    echo "  launched $policy [$PHASE] tasks=$tasks gpu=$gpu egl=$egl -> $log"
    sleep 20
done
wait
echo "=== $PHASE pre-flight complete ==="
