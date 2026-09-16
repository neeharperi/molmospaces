#!/usr/bin/env bash
# Run scripts/probe_policy_payload.py in both arms of an A/B and print the verdict.
#
#   scripts/probe_payload_ab.sh                 # HEAD~1 vs the working tree
#   scripts/probe_payload_ab.sh upstream/main   # the whole fork vs the working tree
#
# Exists because the recipe has two steps that are easy to get subtly wrong, and getting
# either wrong yields a false PASS rather than an error:
#
#   * the baseline arm must run from a NEUTRAL cwd. `molmo_spaces` resolves off sys.path[0]
#     before PYTHONPATH, so launching from this checkout's root silently measures this
#     checkout twice. (The editable install's _EditableFinder is appended after PathFinder
#     and is not the hazard -- see docs/eval_reproduction.md.)
#   * the baseline arm must run the WORKING TREE's probe against the baseline's package.
#     The instrument has to be held fixed while the code under test changes.
#
# --compare re-checks the first of those itself, and VOIDs when both arms name the same
# molmo_spaces file. The wrapper is what keeps anyone from having to remember the rest.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
BASE="${1:-HEAD~1}"
OUT="${OUT:-$(mktemp -d)}"

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${MLSPACES_ENV:-mlspaces-classic}" || exit 1

# The baseline tree's own molmospaces_constants.py requires this, and driving it by hand
# does not set it -- the refusal names MLSPACES_MODELS_DIR but reads as a missing checkout.
export MLSPACES_MODELS_DIR="${MLSPACES_MODELS_DIR:-$(cd "$REPO/.." && pwd)}"

WT="$OUT/baseline"
git worktree add --detach "$WT" "$BASE" >/dev/null 2>&1 || {
    echo "could not create a worktree at $BASE" >&2; exit 1
}
trap 'git worktree remove --force "$WT" >/dev/null 2>&1' EXIT

echo "=== B: working tree"
python scripts/probe_policy_payload.py --json "$OUT/b.json" > "$OUT/b.log" 2>&1
echo "    exit $? -> $OUT/b.log"

echo "=== A: $BASE"
( cd / && PYTHONPATH="$WT" python "$REPO/scripts/probe_policy_payload.py" --json "$OUT/a.json" ) \
    > "$OUT/a.log" 2>&1
echo "    exit $? -> $OUT/a.log"

echo
python scripts/probe_policy_payload.py --compare "$OUT/a.json" "$OUT/b.json"
