#!/usr/bin/env bash
# Evaluate one rung of a base->DROID fine-tune ladder against the sim benchmark.
#
#   POLICY=pi05_droid STEP=10000 bash scripts/gate_openpi_ckpt.sh
#   POLICY=pi0_droid  STEP=10000 EXP=base2droid_20260912 bash scripts/gate_openpi_ckpt.sh
#   POLICY=pi05_droid STEP=500 MAX_EPISODES=300 bash scripts/gate_openpi_ckpt.sh   # quick look
#
# WHY A SIM GATE AND NOT THE LOSS. Training loss cannot gate this run. It is averaged over a
# 32-dim action vector of which 24 dims are zero padding and 7 are joints; the gripper is one
# dimension, ~3% of the signal. pi05_base posts loss 0.041 -- better than the reference
# checkpoint's 3.21 -- while scoring 6.0% against the reference's 23.3% on Pick-v1.5, because it
# almost never closes the gripper (2/30 episodes vs 24/30). Aggregate action-prediction loss is
# structurally incapable of measuring what this benchmark measures.
# See docs/eval_reproduction.md:3581-3748.
#
# The cell is written under a LEADING-UNDERSCORE date tag on purpose. Ladder rungs are not
# campaign results: eval_common.is_campaign_dir() skips underscore-prefixed dirs, which keeps
# them out of compare_to_leaderboard.py and check_provenance.py. Without the underscore a rung
# would supersede the archived released-checkpoint cell in the verdict table, because
# latest_results_csv() takes the latest date dir.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "${BASH_SOURCE[0]:-$0}")/lib/models_dir.sh"

POLICY="${POLICY:?set POLICY=pi05_droid or pi0_droid}"
STEP="${STEP:?set STEP=<checkpoint step>}"
EXP="${EXP:-base2droid_20260912}"
TASK="${TASK:-Pick-v1.5}"
SERVE_GPU="${SERVE_GPU:-3}"
EGL_GPU="${EGL_GPU:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"

python3 -c "
import sys; sys.path.insert(0, 'scripts')
from eval_common import TASKS
sys.exit(0 if '$TASK' in TASKS else 1)
" || { echo "unknown TASK '$TASK' -- see eval_common.TASKS" >&2; exit 1; }

case "$POLICY" in
  pi05_droid) CFG=pi05_droid_jointpos_from_base; SERVE_CFG=pi05_droid_jointpos_polaris; PORT=8080 ;;
  pi0_droid)  CFG=pi0_droid_jointpos_from_base;  SERVE_CFG=pi0_droid_jointpos_polaris;  PORT=8081 ;;
  *) echo "POLICY must be pi05_droid or pi0_droid" >&2; exit 1 ;;
esac

CKPT_SRC="/data/neehar/openpi_runs/checkpoints/$CFG/$EXP/$STEP"
TAG="${EXP}_${STEP}"
LINK_DIR="$MLSPACES_MODELS_DIR/openpi/checkpoints/$TAG"
DATE_TAG="_gate_${TAG}"

[ -d "$CKPT_SRC/params" ] || { echo "no checkpoint at $CKPT_SRC/params" >&2; exit 1; }

# Serve from a symlink pair rather than pointing --policy.dir at the training output directly:
# serve_policy.py wants <dir>/params and <dir>/assets side by side, and this is the layout the
# existing pilot_500 / pilot_2000 links already use.
mkdir -p "$LINK_DIR"
ln -sfn "$CKPT_SRC/params" "$LINK_DIR/params"
if [ -d "$CKPT_SRC/assets" ]; then
  ln -sfn "$CKPT_SRC/assets" "$LINK_DIR/assets"
else
  # Older rungs may not carry assets; fall back to the reference norm stats, which is what the
  # *_from_base configs train against anyway (base ships joint-VELOCITY stats, 1.63x larger).
  ln -sfn "$MLSPACES_MODELS_DIR/openpi/checkpoints/${SERVE_CFG%_polaris}/assets" "$LINK_DIR/assets"
fi

echo "=== gating $POLICY $EXP step $STEP on $TASK ==="
echo "checkpoint: $CKPT_SRC"
echo "serving   : GPU $SERVE_GPU port $PORT config $SERVE_CFG"
echo "rendering : EGL device $EGL_GPU"

SERVER_LOG="runs/_servers/gate_${TAG}_${POLICY}.log"
mkdir -p runs/_servers

# Wait for the port to be FREE before starting ours. Without this, a gate run back-to-back with
# a previous one races its dying server: the readiness poll below connects to the OLD listener on
# its first try, declares "up", and then eval.py -- running seconds later, after the old server
# has actually exited and while ours is still loading (~50s) -- gets ECONNREFUSED and the cell is
# lost. Observed exactly once, on pi05_droid/Open-v1, whose wait printed "up" with zero dots.
echo -n "waiting for :$PORT to be free "
for _ in $(seq 1 60); do
  python3 -c "import socket,sys;s=socket.create_connection(('localhost',$PORT),1);s.close()" 2>/dev/null || { echo "free"; break; }
  sleep 5; echo -n "."
done
python3 -c "import socket;s=socket.create_connection(('localhost',$PORT),1);s.close()" 2>/dev/null && {
  echo; echo "port $PORT still occupied by another process; refusing to start" >&2; exit 1; }

setsid env PORT="$PORT" GPU="$SERVE_GPU" CONFIG="$SERVE_CFG" CKPT_DIR="checkpoints/$TAG" \
  bash scripts/serve_openpi.sh </dev/null >"$SERVER_LOG" 2>&1 &
SERVER_PGID=$!
# shellcheck disable=SC2317
cleanup() { kill -TERM -"$SERVER_PGID" 2>/dev/null || kill -TERM "$SERVER_PGID" 2>/dev/null || true; }
trap cleanup EXIT

echo -n "waiting for :$PORT "
for _ in $(seq 1 90); do
  if python3 -c "import socket,sys;s=socket.create_connection(('localhost',$PORT),2);s.close()" 2>/dev/null; then
    echo "up"; break
  fi
  sleep 10; echo -n "."
done
python3 -c "import socket;s=socket.create_connection(('localhost',$PORT),2);s.close()" 2>/dev/null || {
  echo; echo "server never came up; see $SERVER_LOG" >&2; tail -20 "$SERVER_LOG" >&2; exit 1; }

export MUJOCO_EGL_DEVICE_ID="$EGL_GPU"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL=False
export MLSPACES_PINNED_ASSETS_FILE="$PWD/reference/pinned_assets_20260816.json"

EVAL_ARGS=(--policy "$POLICY" --task "$TASK" --date "$DATE_TAG" --num_workers "$NUM_WORKERS"
           --checkpoint_path "$MLSPACES_MODELS_DIR/openpi/checkpoints/$TAG" --force)
# MAX_EPISODES is for a quick look only. It selects whole HOUSES, so it under-covers categories
# and can oversample; never read a capped rung as a leaderboard-comparable number.
[ -n "${MAX_EPISODES:-}" ] && EVAL_ARGS+=(--max_episodes "$MAX_EPISODES")

# The conda env IS the renderer choice -- env.py fixes HAS_FILAMENT at import from the mujoco
# build, and eval_main.py's --use-filament flag is parsed but never read. Hardcoding the classic
# env would silently render the five filament tasks with the wrong backend, so resolve it from
# the task table the same way run_full_matrix.sh does.
# One python call, two values. A nested $(python3 -c "...") inside a double-quoted echo ends
# the outer quote at the inner one -- that split "renderer" and ran `er` as a command.
read -r RENDERER CONDA_ENV <<<"$(python3 -c "
import sys; sys.path.insert(0, 'scripts')
from eval_common import TASKS, RENDERER_TO_CONDA_ENV
r = TASKS['$TASK'].renderer
print(r, RENDERER_TO_CONDA_ENV[r])
")"
[ -n "$CONDA_ENV" ] || { echo "could not resolve conda env for TASK=$TASK" >&2; exit 1; }
echo "renderer  : $RENDERER -> conda env $CONDA_ENV"

# Re-check right before handing off. eval.py does its own reachability probe and fails the cell
# on ECONNREFUSED, so a server that died between readiness and here costs the whole cell.
python3 -c "import socket;s=socket.create_connection(('localhost',$PORT),3);s.close()" 2>/dev/null || {
  echo "server on :$PORT went away before eval started; see $SERVER_LOG" >&2; tail -20 "$SERVER_LOG" >&2; exit 1; }

source "$HOME/anaconda3/etc/profile.d/conda.sh"
# conda run, not the interpreter by path: eval.py hard-fails when CONDA_DEFAULT_ENV=base.
conda run -n "$CONDA_ENV" --no-capture-output python scripts/eval.py "${EVAL_ARGS[@]}"

CELL="runs/$POLICY/$TASK/$DATE_TAG"
echo
echo "=== rung $STEP result ==="
python3 - "$CELL/results.csv" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(l for l in open(sys.argv[1]) if not l.startswith("#"))]
o = next((r for r in rows if r["category"] == "OVERALL"), None)
if o:
    print(f"  oracle success : {o['oracle_rate_pct']}%  ({o['oracle_successes']}/{o['total']})  "
          f"95% CI [{o['oracle_ci_95_low_pct']}, {o['oracle_ci_95_high_pct']}]")
PY
conda run -n "$CONDA_ENV" --no-capture-output python scripts/grasp_rate.py "$CELL"
echo
echo "targets on Pick-v1.5: leaderboard 18.05% | reference checkpoint 23.3% | pi05_base 6.0%"
echo "grasp-episode rate  : reference 80.0% | pi05_base 6.7%   (leading indicator; climbs first)"
