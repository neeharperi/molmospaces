#!/usr/bin/env bash
# A/B the Cosmos diffusion sampling budget on Pick-v2-classic.
#
# The whole campaign has run the server's DEFAULT --num-steps 4. That is an aggressive
# speed/quality tradeoff for a diffusion policy (UniPC is built for few steps, but four is
# the floor), and nothing establishes that the published leaderboard row used it. Every
# other Cosmos knob tested so far -- control rate, chunk size, camera pairing -- came back
# null, and this is the one that directly governs action QUALITY rather than framing.
#
# Arms restart the server per setting because --num-steps is a server-side flag.
# --max_episodes is legitimate here: A/B arms, not leaderboard cells.
set -uo pipefail
cd "$(dirname "$0")/.."
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
PORT=8007 GPU=2 N=300 TASK=Pick-v2-classic
. scripts/nvidia_gl_env.sh 2>/dev/null || true
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL=False
export MLSPACES_PINNED_ASSETS_FILE="$PWD/reference/pinned_assets_20260816.json"
EGL=$(awk -v g=$GPU '$1!~/^#/ && $1==g{print $2}' runs/_egl_mapping.txt)

# Wait for the camera A/B to release GPU2 before adding load.
while ss -ltn 2>/dev/null | grep -qE ':8006\s'; do
  echo "  waiting for camera A/B (:8006) to finish..."; sleep 300
done

run_arm() {  # $1 = steps
  local steps="$1" arm="steps${1}"
  echo "=== [$(date +%H:%M:%S)] arm $arm ==="
  # Preserve Edge's documented flags; only add --num-steps.
  COSMOS_EXTRA_ARGS="--format-prompt-as-json True --guidance-interval 960 1001 --num-steps $steps" \
    GPU=$GPU PORT=$PORT nohup bash scripts/serve_cosmos.sh \
      >> "runs/_servers/cosmos_steps_ab_server_${arm}.log" 2>&1 &
  local srv=$!
  for _ in $(seq 1 90); do
    (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3<&-; break; }
    sleep 10
  done
  if ! (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
    echo "  !! server never bound :$PORT for $arm"; return 1
  fi
  exec 3<&-
  grep -o "Using sampler:[^]]*" "runs/_servers/cosmos_steps_ab_server_${arm}.log" | tail -1 | sed 's/^/  confirmed /'

  COSMOS_PORT=$PORT LANE_GPU=$GPU MUJOCO_EGL_DEVICE_ID=$EGL \
    conda run -n mlspaces-classic --no-capture-output \
      python scripts/eval.py --policy cosmos_edge --task "$TASK" \
      --num_workers 4 --max_episodes $N --date "_ab_steps_$arm" 2>&1 | tail -4

  # tear the server down so the next arm's setting actually takes effect
  pkill -f "serve_cosmos_policy.py .*--port $PORT" 2>/dev/null
  sleep 20
}

run_arm 4
run_arm 16
run_arm 32

echo "=== steps A/B complete ==="
for s in 4 16 32; do
  f="runs/cosmos_edge/$TASK/_ab_steps_steps$s/results.csv"
  [ -f "$f" ] && grep OVERALL "$f" | awk -v S="$s" -F, '{printf "  num_steps=%-3s %4s/%-5s = %6s%%\n",S,$8,$4,$9}'
done
