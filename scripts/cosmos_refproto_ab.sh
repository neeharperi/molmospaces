#!/usr/bin/env bash
# A/B Cosmos-Nano's client-side action chunk size on Pick-v1.5.
#
# WHY THIS, AND WHY NOT THE JSON PROMPT.
#
# The decisive new evidence is a cross-model one. Edge (4B) and Nano (16B) score within a few
# points of each other on every task they share, at n~1000:
#
#     task              edge(4B)   nano(16B)   leaderboard(nano)
#     Open-v1             8.3%       6.9%          32.0%
#     Close-v1           55.5%      53.4%          79.1%
#     Pick-v1.5          33.8%      38.1%          66.5%
#     Pick-v2-classic     8.2%      11.2%          32.3%
#
# A 4x parameter gap producing no separation says the bottleneck is downstream of model
# quality and shared by both -- not a property of either checkpoint. The harness itself is
# not the problem: it reproduces pi05 (9/9), molmoact2 (9/9), pi0 (2/2), dreamzero (2/2) and
# tiptop (2/2) against their leaderboard entries. So the fault is Cosmos-specific AND shared.
#
# That same table retires the JSON-prompt hypothesis as a natural experiment already run at
# full n: serve_cosmos.sh passes --format-prompt-as-json True for Edge only, so Edge receives
# JSON and Nano receives plain text -- and they score the same. NVIDIA's own model card for
# Cosmos3-Nano-Policy-DROID independently documents plain "language instructions" with no such
# flag. Two independent lines against it; not worth 14h of GPU.
#
# chunk_size is the strongest remaining shared-and-Cosmos-specific candidate:
#   * it is OUR deviation from the vendor default. The server generates 32 actions per call
#     (chunk=32 in its own startup banner); CosmosPolicyConfig consumes only the first 8 and
#     re-queries. Nothing in NVIDIA's documented invocation asks for 8.
#   * it applies identically to Edge and Nano -- the shared-bottleneck signature.
#   * it is the only arm ever tested that moved the number UP: 41.67% vs 33.33% baseline.
#
# POWER. That +8.3pp came from n=60 (z=0.94) -- underpowered, not disproven. And two
# supposedly-identical baseline replicates at n=300 landed 6.67% and 11.33% (z=2.00), so this
# harness's arm-to-arm noise floor is ~5pp, not the ~1.5pp nominal. Hence n=300/arm here at a
# ~38% base rate (SE ~2.8pp), against a 28pp gap to close. Adequate; a null at this n would be
# real information rather than another underpowered shrug.
set -uo pipefail
cd "$(dirname "$0")/.."
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
PORT="${PORT:-8007}" GPU="${GPU:-2}" N=300 TASK=Pick-v1.5
. scripts/nvidia_gl_env.sh 2>/dev/null || true
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL=False
export MLSPACES_PINNED_ASSETS_FILE="$PWD/reference/pinned_assets_20260816.json"
EGL=$(awk -v g=$GPU '$1!~/^#/ && $1==g{print $2}' runs/_egl_mapping.txt)

run_arm() {  # $1 = label   $2 = chunk   $3 = policy dt ms   $4 = task horizon steps
  local arm="$1" chunk="$2" dtms="$3" horizon="$4"
  local slog="runs/_servers/cosmos_proto_ab_${arm}.log"
  local clog="runs/cosmos_nano/$TASK/_ab_proto_${arm}/eval_stdout.log"
  echo "=== [$(date +%H:%M:%S)] arm $arm  (chunk=$chunk dt=${dtms}ms horizon=${horizon} => $(awk -v h=$horizon -v d=$dtms 'BEGIN{printf "%.0f", h*d/1000}')s sim) ==="
  rm -f "$clog"

  # GATE 0 -- the override must actually resolve. Cheap, and catches a shadowed env var
  # before it costs GPU time; CosmosNanoPolicyEvalConfig shadowed COSMOS_PORT exactly that
  # way on 2026-09-05 and the resulting A/B measured nothing for 4.7h.
  local resolved
  # Importing the config prints the asset-pinning banner to stdout, so tag the value and
  # grep for the tag rather than trusting the whole stream.
  resolved=$(COSMOS_CHUNK_SIZE="$chunk" conda run -n mlspaces-classic --no-capture-output python -c \
    "from molmo_spaces.configs.policy_configs_baselines import CosmosPolicyConfig as C; print(f'__CHUNK__{C().chunk_size}__')" 2>/dev/null \
    | grep -o '__CHUNK__[0-9]*__' | tail -1 | sed 's/__CHUNK__//; s/__//')
  local resolved_dt
  resolved_dt=$(COSMOS_DT_MS="$dtms" conda run -n mlspaces-classic --no-capture-output python -c \
    "from molmo_spaces.evaluation.configs.evaluation_configs import CosmosNanoPolicyEvalConfig as C; print(f'__DT__{C().policy_dt_ms}__')" 2>/dev/null \
    | grep -o '__DT__[0-9.]*__' | tail -1 | sed 's/__DT__//; s/__//')
  if [ "${resolved_dt%.0}" != "${dtms%.0}" ]; then
    echo "  !! arm $arm ABORTED: COSMOS_DT_MS=$dtms resolved to '$resolved_dt'"; return 1
  fi
  if [ "$resolved" != "$chunk" ]; then
    echo "  !! arm $arm ABORTED: COSMOS_CHUNK_SIZE=$chunk resolved to '$resolved'"; return 1
  fi
  echo "  gate 0 ok: chunk_size=$resolved dt=${resolved_dt}ms"

  CKPT=nvidia/Cosmos3-Nano-Policy-DROID GPU=$GPU PORT=$PORT \
    nohup bash scripts/serve_cosmos.sh >>"$slog" 2>&1 &
  for _ in $(seq 1 120); do
    (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3<&-; break; }; sleep 10
  done
  (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "  !! never bound :$PORT"; return 1; }
  exec 3<&-
  sleep 5

  COSMOS_PORT=$PORT COSMOS_CHUNK_SIZE=$chunk COSMOS_DT_MS=$dtms LANE_GPU=$GPU MUJOCO_EGL_DEVICE_ID=$EGL \
    conda run -n mlspaces-classic --no-capture-output \
      python scripts/eval.py --policy cosmos_nano --task "$TASK" \
      --num_workers 4 --max_episodes $N --task_horizon_steps $horizon \
      --date "_ab_proto_$arm" >/dev/null 2>&1 &
  local evalpid=$!

  # GATE 1 -- the client must talk to THIS server, not the campaign one on :8004.
  local ok=""
  for _ in $(seq 1 90); do
    grep -q "Cosmos model at localhost:$PORT" "$clog" 2>/dev/null && { ok=port; break; }
    grep -qE "Cosmos model at localhost:(8003|8004)" "$clog" 2>/dev/null && { ok=wrong; break; }
    kill -0 $evalpid 2>/dev/null || break
    sleep 10
  done
  if [ "$ok" != port ]; then
    echo "  !! arm $arm ABORTED: client not on :$PORT (saw $(grep -o 'localhost:[0-9]*' "$clog" 2>/dev/null | tail -1))"
    kill $evalpid 2>/dev/null; pkill -f "_ab_proto_$arm" 2>/dev/null
    pkill -f "serve_cosmos_policy.py .*--port $PORT" 2>/dev/null; return 1
  fi
  echo "  gate 1 ok: client on :$PORT. running..."

  wait $evalpid
  pkill -f "serve_cosmos_policy.py .*--port $PORT" 2>/dev/null
  sleep 25
}

# The task horizon is 500 POLICY STEPS (JsonBenchmarkEvalConfig.task_horizon), so dt sets how
# much simulated time an episode gets: 500 x 66ms = 33s, but the reference row's dt of 0.1
# gives 500 x 100ms = 50s. Changing dt alone therefore confounds control granularity with a
# 1.5x larger time budget. Three arms decompose it:
#
#   A current   dt 66,  chunk 8,  500 steps -> 33s   what the campaign has run
#   B refproto  dt 100, chunk 32, 500 steps -> 50s   the reference protocol, as specified
#   C timeonly  dt 66,  chunk 8,  750 steps -> 50s   B's time budget, A's control
#
#   C ~ B > A  => the deficit is episode TIME, not control granularity
#   B > C ~ A  => it is the chunk/rate, and the extra time is incidental
#   all equal  => neither; the reference row itself is the remaining explanation
#
# This also predicts the geometry signature: a truncated horizon cuts precision grasps on
# narrow objects first, while fast top-grasps of bowls and boxes finish inside 33s either way.
# And it explains why pi05 and molmoact pass -- their reference dt is 0.067, so their 33s
# matches; pi05's Group A reference is 0.1, but Open/Close finish well inside 33s so the
# truncation never binds there.
run_arm current  8  66  500
run_arm refproto 32 100 500
run_arm timeonly 8  66  750

echo "=== nano chunk-size A/B complete (campaign 38.1%, leaderboard 66.5%) ==="
for a in current refproto timeonly; do
  f="runs/cosmos_nano/$TASK/_ab_proto_$a/results.csv"
  [ -f "$f" ] && grep OVERALL "$f" | awk -v A="$a" -F, '{printf "  %-10s %4s/%-5s = %6s%%\n",A,$8,$4,$9}'
done
