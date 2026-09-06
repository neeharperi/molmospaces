#!/usr/bin/env bash
# A/B Cosmos-Nano's prompt format on Pick-v1.5.
#
# WHY. The leaderboard's `cosmos` row is Cosmos3-Nano-Policy (confirmed from the leaderboard
# UI), so Nano is the checkpoint our remaining failures are measured against. The only DROID
# policy training recipe vendored in this repo -- action_policy_droid_nano.py:219, i.e. the
# NANO recipe -- sets format_prompt_as_json=True. But scripts/serve_cosmos.sh passes
# --format-prompt-as-json only when CKPT is Edge; Nano is served with plain-text prompts.
# Verified in runs/_servers/cosmos_nano_20260828_full.log: Nano receives
#   prompt='pick up the brown animal vessel. This video contains concatenated views...'
# while Edge receives '{"cinematography": {...}}'.
#
# That is a train/serve format mismatch on the exact checkpoint the reference used.
#
# POWER. n=300/arm is adequate HERE, unlike the earlier Pick-v2-classic A/Bs. Those chased
# ~4pp effects at an ~8% base rate, where two identical-config draws differed by 4.7pp. This
# is a ~38% base rate (SE ~2.8pp at n=300) looking for a gap of 28pp (38.1 -> 66.5). If the
# prompt format is the cause, it cannot hide at this n.
set -uo pipefail
cd "$(dirname "$0")/.."
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
PORT="${PORT:-8007}" GPU="${GPU:-2}" N=300 TASK=Pick-v1.5
. scripts/nvidia_gl_env.sh 2>/dev/null || true
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL=False
export MLSPACES_PINNED_ASSETS_FILE="$PWD/reference/pinned_assets_20260816.json"
EGL=$(awk -v g=$GPU '$1!~/^#/ && $1==g{print $2}' runs/_egl_mapping.txt)

run_arm() {  # $1 = label   $2 = extra server args
  local arm="$1" extra="$2" slog="runs/_servers/cosmos_nano_json_ab_${1}.log"
  echo "=== [$(date +%H:%M:%S)] arm $arm  (COSMOS_EXTRA_ARGS='$extra') ==="
  COSMOS_EXTRA_ARGS="$extra" CKPT=nvidia/Cosmos3-Nano-Policy-DROID \
    GPU=$GPU PORT=$PORT nohup bash scripts/serve_cosmos.sh >>"$slog" 2>&1 &
  for _ in $(seq 1 120); do
    (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null && { exec 3<&-; break; }; sleep 10
  done
  (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null || { echo "  !! never bound :$PORT"; return 1; }
  exec 3<&-
  sleep 5
  local elog="runs/_servers/cosmos_nano_json_ab_${arm}_client.log"
  # eval.py keeps its detailed log in the CELL directory and only prints summary lines to
  # stdout, so the connection line the port gate looks for never reaches $elog. Grep the
  # cell log instead; $elog is kept only for the final tail.
  local clog="runs/cosmos_nano/$TASK/_ab_nanojson_${arm}/eval_stdout.log"
  rm -f "$clog"
  COSMOS_PORT=$PORT LANE_GPU=$GPU MUJOCO_EGL_DEVICE_ID=$EGL \
    conda run -n mlspaces-classic --no-capture-output \
      python scripts/eval.py --policy cosmos_nano --task "$TASK" \
      --num_workers 4 --max_episodes $N --date "_ab_nanojson_$arm" >"$elog" 2>&1 &
  local evalpid=$!

  # GATE 1 -- the client must connect to THIS server, not the campaign one.
  # CosmosNanoPolicyEvalConfig used to hardcode port=8004, silently shadowing COSMOS_PORT;
  # both arms then ran against the plaintext campaign server, the arm-server log stayed
  # empty, and the post-hoc `grep prompt=` below printed nothing without failing. Two
  # identical arms would have come back as a clean null. Gate early and loudly instead.
  local ok=""
  for _ in $(seq 1 90); do
    grep -q "Cosmos model at localhost:$PORT" "$clog" 2>/dev/null && { ok=port; break; }
    grep -qE "Cosmos model at localhost:(8003|8004)" "$clog" 2>/dev/null && { ok=wrong; break; }
    kill -0 $evalpid 2>/dev/null || break
    sleep 10
  done
  if [ "$ok" != port ]; then
    echo "  !! arm $arm ABORTED: client not on :$PORT (saw: $(grep -o 'localhost:[0-9]*' "$clog" 2>/dev/null | tail -1))"
    kill $evalpid 2>/dev/null; pkill -f "_ab_nanojson_$arm" 2>/dev/null
    pkill -f "serve_cosmos_policy.py .*--port $PORT" 2>/dev/null
    return 1
  fi

  # GATE 2 -- the prompt form must actually be what this arm claims. '{' => JSON.
  local ch=""
  for _ in $(seq 1 60); do
    ch=$(grep -o "prompt='." "$slog" 2>/dev/null | tail -1 | sed "s/.*prompt='//")
    [ -n "$ch" ] && break
    kill -0 $evalpid 2>/dev/null || break
    sleep 10
  done
  local want_json=no; [ "$arm" = B_json ] && want_json=yes
  local got_json=no; [ "$ch" = "{" ] && got_json=yes
  if [ -z "$ch" ] || [ "$got_json" != "$want_json" ]; then
    echo "  !! arm $arm ABORTED: prompt form mismatch (want json=$want_json, first char='$ch')"
    kill $evalpid 2>/dev/null; pkill -f "_ab_nanojson_$arm" 2>/dev/null
    pkill -f "serve_cosmos_policy.py .*--port $PORT" 2>/dev/null
    return 1
  fi
  echo "  gates passed: client on :$PORT, prompt json=$got_json. running..."

  wait $evalpid
  tail -4 "$elog"
  echo -n "  prompt form seen: "
  grep -o "prompt='.\{0,30\}" "$slog" | tail -1
  pkill -f "serve_cosmos_policy.py .*--port $PORT" 2>/dev/null
  sleep 25
}

run_arm A_plaintext ""
run_arm B_json      "--format-prompt-as-json True"

echo "=== nano prompt-format A/B complete (leaderboard 66.5%) ==="
for a in A_plaintext B_json; do
  f="runs/cosmos_nano/$TASK/_ab_nanojson_$a/results.csv"
  [ -f "$f" ] && grep OVERALL "$f" | awk -v A="$a" -F, '{printf "  %-12s %4s/%-5s = %6s%%\n",A,$8,$4,$9}'
done
