#!/usr/bin/env bash
# Does chunk_size matter AT THE BENCHMARK'S OWN EPISODE BUDGET?
#
# The previous three arms are all void: they passed --task_horizon_steps, which overrides the
# benchmark's per-episode length. Pick-v1.5 specifies task_horizon_sec=20; the arms ran 33-50s,
# up to 2.5x that, so their gains were bought with time the benchmark does not allow. See
# runs/cosmos_nano/Pick-v1.5/_INVALID_horizon_override_20260906/README.md.
#
# The premise behind that flag was also wrong. eval_main.py:295 determine_task_horizon() reads
# the benchmark's task_horizon_sec and converts to steps via policy_dt_ms, so episode DURATION
# is fixed and step count is derived -- the harness is dt-invariant by construction and there
# was never a dt-induced confound to correct.
#
# So: no override here. Both arms get the benchmark's 20s. At dt 66ms that is ~303 steps,
# which is ~38 rows at chunk 8 and ~10 rows at chunk 32 -- same wall of simulated time, different
# open-loop granularity. That is the real question, and the only one the recipe motivates:
# action_policy_droid_nano.py:195 sets chunk_length=32 with the tokenizer's
# encode_exact_durations=[33] pinned to match, while CosmosPolicyConfig.chunk_size=8 was
# borrowed from pi0.5 by its own comment.
#
# n=300/arm against a ~38% base rate and this harness's measured ~5pp arm-to-arm noise floor.
set -uo pipefail
cd "$(dirname "$0")/.."
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
PORT="${PORT:-8007}" GPU="${GPU:-2}" N=300 TASK=Pick-v1.5
. scripts/nvidia_gl_env.sh 2>/dev/null || true
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL=False
export MLSPACES_PINNED_ASSETS_FILE="$PWD/reference/pinned_assets_20260816.json"
EGL=$(awk -v g=$GPU '$1!~/^#/ && $1==g{print $2}' runs/_egl_mapping.txt)

run_arm() {  # $1 = label   $2 = chunk
  local arm="$1" chunk="$2"
  local slog="runs/_servers/cosmos_chunkvalid_${arm}.log"
  local clog="runs/cosmos_nano/$TASK/_ab_cv_${arm}/eval_stdout.log"
  echo "=== [$(date +%H:%M:%S)] arm $arm (chunk=$chunk, dt=66ms, benchmark horizon 20s) ==="
  rm -f "$clog"

  local resolved
  resolved=$(COSMOS_CHUNK_SIZE="$chunk" conda run -n mlspaces-classic --no-capture-output python -c \
    "from molmo_spaces.evaluation.configs.evaluation_configs import CosmosNanoPolicyEvalConfig as C; c=C(); print(f'__V__{c.policy_config.chunk_size}/{c.policy_dt_ms}__')" 2>/dev/null \
    | grep -o '__V__[0-9./]*__' | tail -1 | sed 's/__V__//; s/__//')
  [ "$resolved" = "$chunk/66.0" ] || { echo "  !! ABORT: expected $chunk/66.0, got '$resolved'"; return 1; }
  echo "  gate 0 ok: chunk/dt = $resolved"

  CKPT=nvidia/Cosmos3-Nano-Policy-DROID GPU=$GPU PORT=$PORT \
    nohup bash scripts/serve_cosmos.sh >>"$slog" 2>&1 &
  for _ in $(seq 1 120); do
    (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3<&-; break; }; sleep 10
  done
  (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "  !! never bound :$PORT"; return 1; }
  exec 3<&-; sleep 5

  # NOTE: deliberately NO --task_horizon_steps. The benchmark's 20s must stand.
  COSMOS_PORT=$PORT COSMOS_CHUNK_SIZE=$chunk LANE_GPU=$GPU MUJOCO_EGL_DEVICE_ID=$EGL \
    conda run -n mlspaces-classic --no-capture-output \
      python scripts/eval.py --policy cosmos_nano --task "$TASK" \
      --num_workers 4 --max_episodes $N --date "_ab_cv_$arm" >/dev/null 2>&1 &
  local evalpid=$!

  local ok=""
  for _ in $(seq 1 90); do
    grep -q "Cosmos model at localhost:$PORT" "$clog" 2>/dev/null && { ok=port; break; }
    grep -qE "Cosmos model at localhost:(8003|8004|8008)" "$clog" 2>/dev/null && { ok=wrong; break; }
    kill -0 $evalpid 2>/dev/null || break
    sleep 10
  done
  [ "$ok" = port ] || {
    echo "  !! ABORT: client not on :$PORT (saw $(grep -o 'localhost:[0-9]*' "$clog" 2>/dev/null | tail -1))"
    kill $evalpid 2>/dev/null; pkill -f "_ab_cv_$arm" 2>/dev/null
    pkill -f "serve_cosmos_policy.py .*--port $PORT" 2>/dev/null; return 1; }
  echo "  gate 1 ok: client on :$PORT"

  # gate 2 -- the benchmark horizon must NOT have been overridden
  local hz
  hz=$(grep -o "task_horizon override: [0-9]*" "$clog" 2>/dev/null | tail -1)
  [ -z "$hz" ] || { echo "  !! ABORT: a horizon override leaked in ($hz)"; kill $evalpid 2>/dev/null; return 1; }
  echo "  gate 2 ok: no horizon override; benchmark's 20s stands. running..."

  wait $evalpid
  pkill -f "serve_cosmos_policy.py .*--port $PORT" 2>/dev/null
  sleep 25
}

run_arm chunk8  8
run_arm chunk32 32

echo "=== chunk A/B at the benchmark budget (campaign 38.10%, leaderboard 66.50%) ==="
for a in chunk8 chunk32; do
  f="runs/cosmos_nano/$TASK/_ab_cv_$a/results.csv"
  [ -f "$f" ] && grep OVERALL "$f" | awk -v A="$a" -F, '{printf "  %-9s %4s/%-5s = %6s%%\n",A,$8,$4,$9}'
done
