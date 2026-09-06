#!/usr/bin/env bash
# Isolate chunk_size from dt in the refproto result.
#
# arm refproto (chunk 32, dt 100, horizon 500 -> 50s) scored 72.33% against the campaign's
# 38.10% and the leaderboard's 66.50%. It moved three things at once. This arm moves ONE:
#
#     chunk 32, dt 66 (default), horizon 500 -> 33s
#
#   ~72%  => chunk_size alone explains it. Re-run the Cosmos cells with chunk 32 and nothing
#            else, which is the recipe-backed change: action_policy_droid_nano.py:195 sets
#            chunk_length=32 and line 232 pins the tokenizer's encode_exact_durations=[33] to
#            match, and chunk_size=8 was borrowed from pi0.5 by its own comment.
#   ~38%  => chunk alone is not it; duration or dt carries the effect, and arm `timeonly`
#            (chunk 8, dt 66, horizon 750 -> 50s) discriminates those.
#
# This is now the decisive arm: the reason for testing dt 100 was retracted once the "# dt"
# header turned out to be eval_to_csv.py's jerk-normalisation constant rather than a control
# rate, so dt has no independent justification left. chunk does.
#
# Runs on GPU1 (~50 GB free, 39% util, and NOT the filament card) alongside the three-arm
# script's `timeonly` on GPU2, because timeonly is ~7.5h out and this arm is ~1.5h -- chunk 32
# means 4x fewer server round-trips per episode.
set -uo pipefail
cd "$(dirname "$0")/.."
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
PORT="${PORT:-8008}" GPU="${GPU:-1}" N=300 TASK=Pick-v1.5 ARM=chunkonly
. scripts/nvidia_gl_env.sh 2>/dev/null || true
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL=False
export MLSPACES_PINNED_ASSETS_FILE="$PWD/reference/pinned_assets_20260816.json"
EGL=$(awk -v g=$GPU '$1!~/^#/ && $1==g{print $2}' runs/_egl_mapping.txt)
slog="runs/_servers/cosmos_chunkonly_ab.log"
clog="runs/cosmos_nano/$TASK/_ab_proto_${ARM}/eval_stdout.log"
rm -f "$clog"

echo "=== [$(date +%H:%M:%S)] arm $ARM (chunk=32 dt=66ms horizon=500 => 33s sim) gpu=$GPU port=$PORT ==="

resolved=$(COSMOS_CHUNK_SIZE=32 conda run -n mlspaces-classic --no-capture-output python -c \
  "from molmo_spaces.evaluation.configs.evaluation_configs import CosmosNanoPolicyEvalConfig as C; c=C(); print(f'__V__{c.policy_config.chunk_size}/{c.policy_dt_ms}__')" 2>/dev/null \
  | grep -o '__V__[0-9./]*__' | tail -1 | sed 's/__V__//; s/__//')
if [ "$resolved" != "32/66.0" ]; then
  echo "  !! ABORTED: expected chunk/dt 32/66.0, resolved '$resolved'"; exit 1
fi
echo "  gate 0 ok: chunk/dt = $resolved"

CKPT=nvidia/Cosmos3-Nano-Policy-DROID GPU=$GPU PORT=$PORT \
  nohup bash scripts/serve_cosmos.sh >>"$slog" 2>&1 &
for _ in $(seq 1 120); do
  (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3<&-; break; }; sleep 10
done
(exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "  !! never bound :$PORT"; exit 1; }
exec 3<&-; sleep 5

COSMOS_PORT=$PORT COSMOS_CHUNK_SIZE=32 LANE_GPU=$GPU MUJOCO_EGL_DEVICE_ID=$EGL \
  conda run -n mlspaces-classic --no-capture-output \
    python scripts/eval.py --policy cosmos_nano --task "$TASK" \
    --num_workers 4 --max_episodes $N --task_horizon_steps 500 \
    --date "_ab_proto_${ARM}" >/dev/null 2>&1 &
evalpid=$!

ok=""
for _ in $(seq 1 90); do
  grep -q "Cosmos model at localhost:$PORT" "$clog" 2>/dev/null && { ok=port; break; }
  grep -qE "Cosmos model at localhost:(8003|8004|8007)" "$clog" 2>/dev/null && { ok=wrong; break; }
  kill -0 $evalpid 2>/dev/null || break
  sleep 10
done
if [ "$ok" != port ]; then
  echo "  !! ABORTED: client not on :$PORT (saw $(grep -o 'localhost:[0-9]*' "$clog" 2>/dev/null | tail -1))"
  kill $evalpid 2>/dev/null; pkill -f "_ab_proto_${ARM}" 2>/dev/null
  pkill -f "serve_cosmos_policy.py .*--port $PORT" 2>/dev/null; exit 1
fi
echo "  gate 1 ok: client on :$PORT. running..."

wait $evalpid
pkill -f "serve_cosmos_policy.py .*--port $PORT" 2>/dev/null
f="runs/cosmos_nano/$TASK/_ab_proto_${ARM}/results.csv"
echo "=== chunk-only arm complete (campaign 38.10%, refproto 72.33%, leaderboard 66.50%) ==="
[ -f "$f" ] && grep OVERALL "$f" | awk -F, '{printf "  chunkonly  %s/%s = %s%%\n",$8,$4,$9}'
