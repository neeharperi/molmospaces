#!/usr/bin/env bash
# Serve TiPToP's planner (:18765) for the eval harness, with its perception behind it.
#
#   GPU=1 bash scripts/serve_tiptop.sh
#   GPU=1 PORT=18765 M2T2=skip bash scripts/serve_tiptop.sh
#   GPU=1 STEREO=skip bash scripts/serve_tiptop.sh
#
# The script the rest of this repo already assumed existed: scripts/sim.sh answers
# a missing server with "bash scripts/serve_tiptop.sh" and there was no such file,
# and launch_campaign.sh carried the invocation inline. GPU=/PORT= convention as
# serve_openpi.sh and serve_molmoact2.sh.
#
# Three things here are not obvious, and all three have cost a run.
#
# 1. **PORT is 18765, not TiPToP's own default 8765.** This is a shared host and
#    another user's process holds 8765: our server could never bind (21 supervisor
#    restarts, all "address already in use") while the client connected happily to
#    THEIR server and failed the websocket handshake -- which surfaced as
#    "[SSL: WRONG_VERSION_NUMBER]" from the client's own wss:// fallback, naming
#    neither the port nor the collision. droid's rig server wants 8765 as well, and
#    both harnesses have to be able to run at once. See eval_common.py's POLICIES.
#
# 2. **The cwd has to be this repo, not the tiptop checkout.** `python -m` puts the
#    cwd on sys.path[0], and third_party/tiptop holds a cutamp/ directory whose top
#    level has no __init__.py -- so it shadows the properly installed editable
#    cutamp as a PEP 420 namespace package. `import cutamp` then succeeds and
#    returns an empty module, and startup dies with "cuTAMP version mismatch:
#    required 0.0.6, found <0.0.2", which is a lie about the version and says
#    nothing about the shadowing. `cd` to the repo root is the whole fix.
#
# 3. **M2T2 (:8123) and FoundationStereo (:1234) both fail quietly.** With either
#    down a plan comes back a well-formed success=False and a cell scores ~0% --
#    which is how a whole campaign of TiPToP results was lost (6,486 of 6,486
#    planning calls failed; see docs/eval_reproduction.md). So both are started
#    here rather than remembered, unless M2T2=skip / STEREO=skip says something
#    else already owns one -- which is what launch_campaign.sh does for M2T2,
#    since it runs that in a lane of its own.
#
#    Depth is the one that used to be left out. The planner only sends a stereo
#    pair to :1234 when the rig gives it one, so without that server a run reads
#    as a success that quietly planned on the ZED's own depth instead.
#
#    Started through **tiptop's** launcher, not this repo's scripts/serve_m2t2.sh.
#    That one serves a private clone at $MLSPACES_MODELS_DIR/m2t2 out of the
#    parity-pinned mlspaces-m2t2 env, created on demand by setup_envs.sh and absent
#    on this machine; tiptop vendors both under third_party/ with their own pixi
#    environments and their weights already pulled. Its serve_perception.sh takes
#    the service names, so starting one that is already up is a no-op that says so.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "${BASH_SOURCE[0]:-$0}")/lib/models_dir.sh"
PORT="${PORT:-18765}"
GPU="${GPU:-0}"
M2T2="${M2T2:-auto}"
M2T2_PORT="${M2T2_PORT:-8123}"
STEREO="${STEREO:-auto}"
STEREO_PORT="${STEREO_PORT:-1234}"

# Gemini grounds the instruction into the atoms cuTAMP plans over, per request, so
# an absent key is a run that fails at the first episode rather than at startup.
: "${GOOGLE_API_KEY:?set GOOGLE_API_KEY -- Gemini grounds the instruction, and every request needs it}"

TIPTOP_DIR="$MLSPACES_MODELS_DIR/tiptop"
[ -d "$TIPTOP_DIR" ] || {
  echo "tiptop checkout not found: $TIPTOP_DIR" >&2
  echo "MLSPACES_MODELS_DIR is $MLSPACES_MODELS_DIR -- set it if the checkouts live elsewhere." >&2
  exit 1
}

TIPTOP_PY="${TIPTOP_PYTHON:-${MLSPACES_ENVS:-$HOME/anaconda3/envs}/mlspaces-tiptop/bin/python}"
[ -x "$TIPTOP_PY" ] || {
  echo "no interpreter at $TIPTOP_PY" >&2
  echo "build it with: bash scripts/setup_envs.sh mlspaces-tiptop" >&2
  exit 1
}

PERCEPTION=()
if [ "$STEREO" != "skip" ]; then PERCEPTION+=(stereo); fi
if [ "$M2T2" != "skip" ]; then PERCEPTION+=(m2t2); fi
if [ "${#PERCEPTION[@]}" -gt 0 ]; then
  echo "starting tiptop's vendored perception: ${PERCEPTION[*]}"
  STEREO_PORT="$STEREO_PORT" M2T2_PORT="$M2T2_PORT" \
    "$TIPTOP_DIR/examples/droid/serve_perception.sh" --gpu "$GPU" --no-follow "${PERCEPTION[@]}"
fi

# The env's bin on PATH, which calling its python by absolute path does not do and
# `conda activate` would have. cuRobo needs it, and the way it needs it is worth
# writing down, because the error names neither cause:
#
#   third_party/tiptop/curobo is ONE checkout installed editable into TWO
#   environments -- this one at torch 2.8.0+cu129, and tiptop's own pixi env at
#   torch 2.7.1 -- and an editable install leaves its compiled extensions IN the
#   source tree. So whichever env built last owns curobolib/*.so, and the other
#   gets `undefined symbol: _ZN3c104cuda9SetDeviceEa` on import. cuRobo handles
#   that by falling back to a JIT build, which is the right answer: torch caches
#   those under ~/.cache/torch_extensions keyed by torch version, so the two envs
#   stop fighting over one build directory. But the fallback shells out to `ninja`,
#   and without this line the ninja sitting in this very env is not on PATH -- so
#   the failure surfaces as "Ninja is required to load C++ extensions (pip install
#   ninja to get it)" about a ninja that is already installed.
#
# First start therefore spends a few minutes compiling those kernels. After that
# the cache is warm and startup is the cuRobo solver warmup alone.
#
# Three variables, the same three scripts/setup_envs.sh::cuda_env() exports when it
# builds these extensions ahead of time, and for the same reason it gives: the
# nvidia channel's cuda-toolkit puts its headers in targets/<triple>/include, which
# torch's build does not put on the include path -- so the C++ half of the build
# fails on `fatal error: cuda_runtime_api.h: No such file or directory` while the
# nvcc half, which knows its own include dir, compiles fine.
TIPTOP_PREFIX="$(dirname "$(dirname "$TIPTOP_PY")")"
export CUDA_HOME="${CUDA_HOME:-$TIPTOP_PREFIX}"
export PATH="$TIPTOP_PREFIX/bin:$PATH"
export CPATH="$TIPTOP_PREFIX/targets/x86_64-linux/include:${CPATH:-}"

# And the arch to compile them for, explicitly. Left unset, torch builds for every
# arch on every visible card -- so what lands in the cache depends on which GPU
# CUDA_VISIBLE_DEVICES happened to expose, and a build made while one card was
# masked is reused later when it is not. 12.0 is sm_120, these RTX PRO 5000
# Blackwells, the same value setup_envs.sh and tiptop's own Dockerfile pin.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"

echo "serving tiptop on :$PORT, GPU $GPU, from $(pwd)"
CUDA_VISIBLE_DEVICES="$GPU" exec "$TIPTOP_PY" \
  -m tiptop.tiptop_websocket_server --port "$PORT" "$@"
