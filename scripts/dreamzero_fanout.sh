#!/usr/bin/env bash
# Fan DreamZero out onto GPUs as the other lanes finish.
#
#   DATE_TAG=20260828_full bash scripts/dreamzero_fanout.sh watch     # poll + launch (the real mode)
#   DATE_TAG=20260828_full bash scripts/dreamzero_fanout.sh plan      # print what it would do, launch nothing
#
# WHY THIS EXISTS. DreamZero sets the campaign's completion date: ~75 h per cell against
# ~11 h for Cosmos Edge, and 8 cells left. At one instance that is ~25 days. Every other
# lane finishes inside 10. So the campaign is DreamZero, and the only lever on it is
# running several cells at once.
#
# WHY IT IS N SERVERS AND NOT N WORKERS. The obvious lever -- raise the lane from
# --num_workers 1 to 4 like every other lane -- silently corrupts results. DreamZero's
# server (third_party/dreamzero/socket_test_optimized_AR.py) is autoregressive and keeps a
# rolling frame history keyed to ONE _current_session_id held on the policy object, not per
# connection. N workers interleave N session ids through it and every alternating request
# trips the session-change branch and wipes the history both episodes depend on. Actions
# stay well-formed throughout, so nothing fails loudly. See docs/eval_reproduction.md.
#
# So: one server per worker, each on its own GPU and port, each owning one session.
#
# GPU OWNERSHIP. A GPU is claimable once every lane assigned to it in launch_campaign.sh's
# LANES table has exited. We do not preempt: all four cards currently sit at 85-100%, and
# splitting a busy card measured 0.94x earlier in this campaign. The win only exists on an
# idle card, so we wait for one rather than manufacturing contention.
set -uo pipefail
cd "$(dirname "$0")/.."
DATE_TAG="${DATE_TAG:?set DATE_TAG, e.g. 20260828_full}"
MODE="${1:-plan}"
POLL_SECS="${POLL_SECS:-600}"
ENVS="${MLSPACES_ENVS:-$HOME/anaconda3/envs}"
source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
# The harness env the lanes need. launch_campaign.sh exports these for the normal lanes and
# this script inherits nothing from it, so without them every fan-out lane dies immediately
# on "MLSPACES_ASSETS_DIR is not set." -- while still holding its claim.
. "$PWD/scripts/nvidia_gl_env.sh" 2>/dev/null || true
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"
export MLSPACES_FORCE_INSTALL="${MLSPACES_FORCE_INSTALL:-False}"
export MLSPACES_PINNED_ASSETS_FILE="${MLSPACES_PINNED_ASSETS_FILE:-$PWD/reference/pinned_assets_20260816.json}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
CLAIMS="runs/_lanes/dz_claims"
mkdir -p "$CLAIMS" runs/_lanes runs/_servers

# GPU -> the lanes that must finish before that GPU is free (from launch_campaign.sh).
# GPU 0 is DreamZero's own card and is never claimable here.
gpu_owners() { case "$1" in 1) echo "cosmos_nano cosmos_edge";; 2) echo "molmoact2_droid tiptop";; 3) echo "pi05_droid pi0_droid";; *) echo "";; esac; }

lane_running() {  # $1 = policy. Bracket guard so this never matches its own pipeline.
    ps -eo args 2>/dev/null | grep -q "[e]val\.py .*--policy $1\b"
}

gpu_free() {
    local g="$1" p
    for p in $(gpu_owners "$g"); do lane_running "$p" && return 1; done
    return 0
}

egl_for() { awk -v g="$1" '$1 !~ /^#/ && $1 == g {print $2}' runs/_egl_mapping.txt; }

# DreamZero cells still needing a result, in the lane's own task order.
pending_cells() {  # exits non-zero if the query itself fails, so callers can tell it apart
    "$ENVS/mlspaces-classic/bin/python" - "$DATE_TAG" <<'PY'
import sys, pathlib
sys.path.insert(0, "scripts")
from eval_common import TASKS
date_tag = sys.argv[1]
for t in TASKS:
    if not pathlib.Path(f"runs/dreamzero/{t}/{date_tag}/results.csv").exists():
        print(f"{t}\t{TASKS[t].renderer}")
PY
}

claim() { mkdir "$CLAIMS/$1" 2>/dev/null; }   # mkdir is atomic: first caller wins

# A cell is claimable only if nobody is already producing it. Two guards, because a
# second writer in one cell dir mixes trajectories from both attempts into one
# results.csv -- silently, and only detectable afterwards by duplicate house ids.
#   (1) the main lane is running it right now (its in-flight cell has no results.csv
#       yet, so pending_cells lists it and would happily hand it to a free GPU);
#   (2) a previous attempt left h5 files behind without a results.csv, i.e. an
#       orphaned partial. eval.py only resumes on provenance.json + results.csv
#       together, so it would re-run into the same eval_output/ and mix.
in_flight() {  # $1 = task
    ps -eo args 2>/dev/null | grep -q "[e]val\.py .*--policy dreamzero .*--task $1\b"
}
has_orphan_partial() {  # $1 = task
    local d="runs/dreamzero/$1/$DATE_TAG"
    [ -f "$d/results.csv" ] && return 1
    [ -n "$(find "$d" -name '*.h5' -print -quit 2>/dev/null)" ]
}

launch() {  # $1 = task  $2 = renderer  $3 = gpu
    local task="$1" renderer="$2" gpu="$3"
    local port=$((5000 + gpu)) egl; egl="$(egl_for "$gpu")"
    [ -n "$egl" ] || { echo "  !! no EGL id for GPU $gpu; run scripts/probe_egl_mapping.py" >&2; return 1; }
    local slog="runs/_servers/dreamzero_fanout_gpu${gpu}.log"
    local llog="runs/_lanes/dz_fanout_gpu${gpu}_${DATE_TAG}.log"

    # If the port is already held, a previous server is still up (or a stale one is). Do not
    # start a second: the readiness probe below only checks that SOMETHING holds the port, so
    # a leftover server made every subsequent launch look instantly successful.
    if (exec 3<>/dev/tcp/127.0.0.1/"$port") 2>/dev/null; then
        exec 3<&-
        # A server is already listening. Since the pid-file guard above proved no fan-out lane
        # is running on this card, that server is idle -- reuse it rather than stacking a
        # second one (which would OOM) or releasing the claim (which left the card idle).
        echo "  [$(date +%H:%M:%S)] GPU$gpu: reusing existing server on :$port for $task ($renderer)"
    else
        echo "  [$(date +%H:%M:%S)] GPU$gpu: starting server on :$port for $task ($renderer)"
        ( GPUS="$gpu" DIT_SPLIT=0 PORT="$port" nohup bash scripts/serve_dreamzero.sh >>"$slog" 2>&1 & )
    fi

    for _ in $(seq 1 120); do   # serve_dreamzero warms for several minutes; 20 min ceiling
        (exec 3<>/dev/tcp/127.0.0.1/"$port") 2>/dev/null && { exec 3<&-; break; }
        sleep 10
    done
    if ! (exec 3<>/dev/tcp/127.0.0.1/"$port") 2>/dev/null; then
        echo "  !! GPU$gpu: server never bound :$port -- releasing claim on $task" >&2
        rmdir "$CLAIMS/$task" 2>/dev/null; return 1
    fi
    exec 3<&-

    local env_name=mlspaces-classic
    [ "$renderer" = "filament" ] && env_name=mlspaces-filament
    echo "  [$(date +%H:%M:%S)] GPU$gpu: server up, launching $task in $env_name" | tee -a "$llog"
    # eval.py hard-fails unless CONDA_DEFAULT_ENV names the renderer's env. Addressing the
    # interpreter by path leaves it as 'base', so every lane died instantly with
    # "active env is 'base'" while this script cheerfully claimed the next cell.
    ( DREAMZERO_PORT="$port" LANE_GPU="$gpu" MUJOCO_EGL_DEVICE_ID="$egl" \
      nohup conda run -n "$env_name" --no-capture-output \
        python scripts/eval.py \
        --policy dreamzero --task "$task" --num_workers 1 --date "$DATE_TAG" \
        >>"$llog" 2>&1 & )
    local lane_pid=$!
    echo "$lane_pid" > "runs/_lanes/dz_fanout_gpu${gpu}.pid"
    # A lane that dies in the first 60s did not start; release the claim so the cell is not
    # stranded. Previously a startup failure held its claim forever and the card idled.
    sleep 60
    if ! kill -0 "$lane_pid" 2>/dev/null && [ ! -e "runs/dreamzero/$task/$DATE_TAG" ]; then
        echo "  !! GPU$gpu: $task lane died at startup; releasing claim. See $llog" >&2
        rmdir "$CLAIMS/$task" 2>/dev/null
        rm -f "runs/_lanes/dz_fanout_gpu${gpu}.pid"
    fi
}

echo "dreamzero fan-out: mode=$MODE date=$DATE_TAG"
while :; do
    # Distinguish "no cells left" from "the query failed". Conflating them makes a
    # transient error look like successful completion and silently ends the fan-out.
    if ! cells_raw="$(pending_cells)"; then
        echo "  !! could not enumerate pending cells; retrying in ${POLL_SECS}s" >&2
        sleep "$POLL_SECS"; continue
    fi
    mapfile -t cells < <(printf '%s\n' "$cells_raw" | sed '/^$/d')
    if [ "${#cells[@]}" -eq 0 ]; then echo "all DreamZero cells have results; nothing left to fan out"; exit 0; fi

    for gpu in 1 2 3; do
        gpu_free "$gpu" || { [ "$MODE" = plan ] && echo "  GPU$gpu busy (waiting on:$(gpu_owners $gpu))"; continue; }
        # Already running a fan-out lane on this card? Track it by pid file. The previous
        # check pgrep'd for "dz_fanout_gpu<N>", which only ever appeared in the redirect
        # target -- never in argv -- so it matched nothing, the guard never fired, and this
        # loop claimed one more cell on every poll until it had claimed them all.
        pidf="runs/_lanes/dz_fanout_gpu${gpu}.pid"
        if [ -f "$pidf" ] && kill -0 "$(cat "$pidf" 2>/dev/null)" 2>/dev/null; then
            [ "$MODE" = plan ] && echo "  GPU$gpu already running a fan-out lane (pid $(cat "$pidf"))"
            continue
        fi
        for spec in "${cells[@]}"; do
            task="${spec%%$'\t'*}"; renderer="${spec##*$'\t'}"
            [ -d "$CLAIMS/$task" ] && continue
            if in_flight "$task"; then
                [ "$MODE" = plan ] && echo "  skip $task: the main lane is running it now"
                continue
            fi
            if has_orphan_partial "$task"; then
                echo "  !! skip $task: h5 files present with no results.csv (orphaned partial)." >&2
                echo "     Clear runs/dreamzero/$task/$DATE_TAG/eval_output/ before letting this cell run." >&2
                continue
            fi
            if [ "$MODE" = plan ]; then echo "  GPU$gpu FREE -> would claim $task ($renderer) on :$((5000+gpu))"; break; fi
            claim "$task" || continue
            launch "$task" "$renderer" "$gpu" || true
            break
        done
    done

    [ "$MODE" = plan ] && exit 0
    sleep "$POLL_SECS"
done
