#!/usr/bin/env bash
# A/B the Cosmos second-exterior camera on Pick-v2-classic.
#
# The checkpoint's own prompt asks for "two horizontally concatenated third-person
# perspective views of the scene from opposite sides, WITH THE ROBOT VISIBLE".
# We have never given it that. History:
#   - zed2_1 + zed2_2  -> 5.2%  (retracted; zed2_2 often contains no robot at all)
#   - droid_shoulder duplicated into both slots -> 8.2%  (current default)
# Untried: droid_shoulder paired with another camera that DOES show the robot.
# runs/_debug/Pick-v2-classic/ has two such: randomized_gopro_analogue_1 and
# randomized_zed2_analogue_1.
#
# Leaderboard for this cell is 32.3%, so there is ~24pp of headroom to detect.
# --max_episodes is legitimate here: these are A/B arms, not leaderboard cells.
set -uo pipefail
cd "$(dirname "$0")/.."
ENVS="${MLSPACES_ENVS:-$HOME/anaconda3/envs}"
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
PORT=8006 GPU=2 N=300 TASK=Pick-v2-classic
. scripts/nvidia_gl_env.sh 2>/dev/null || true
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL=False
export MLSPACES_PINNED_ASSETS_FILE="$PWD/reference/pinned_assets_20260816.json"

echo "=== starting dedicated Cosmos Edge server :$PORT on GPU$GPU ==="
( GPU=$GPU PORT=$PORT nohup bash scripts/serve_cosmos.sh \
    >> runs/_servers/cosmos_camera_ab_server.log 2>&1 & )
for _ in $(seq 1 90); do
  (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3<&-; break; }
  sleep 10
done
(exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "server never bound :$PORT"; exit 1; }
exec 3<&-
echo "server up"

run_arm() {  # $1 = arm label   $2 = COSMOS_EXTERIOR_2 value ("" = duplicate)
  local arm="$1" exo2="$2"
  echo "=== [$(date +%H:%M:%S)] arm $arm  (exterior_2='${exo2:-<duplicate>}') ==="
  # eval.py hard-fails unless CONDA_DEFAULT_ENV names the renderer's env, so the env must be
  # ACTIVATED -- calling "$ENVS/mlspaces-classic/bin/python" directly leaves it as 'base'.
  COSMOS_PORT=$PORT COSMOS_EXTERIOR_2="$exo2" \
  LANE_GPU=$GPU MUJOCO_EGL_DEVICE_ID=$(awk -v g=$GPU '$1!~/^#/ && $1==g{print $2}' runs/_egl_mapping.txt) \
    conda run -n mlspaces-classic --no-capture-output \
      python scripts/eval.py \
      --policy cosmos_edge --task "$TASK" --num_workers 4 \
      --max_episodes $N --date "_ab_cam_$arm" 2>&1 | tail -4
}

run_arm A_duplicate ""
run_arm B_gopro     randomized_gopro_analogue_1
run_arm C_zed2_1    randomized_zed2_analogue_1

echo "=== camera A/B complete ==="
for a in A_duplicate B_gopro C_zed2_1; do
  f="runs/cosmos_edge/$TASK/_ab_cam_$a/results.csv"
  [ -f "$f" ] && printf "  %-14s %s\n" "$a" "$(grep OVERALL "$f" | awk -F, '{printf "%s/%s = %s%%", $8, $4, $9}')"
done
