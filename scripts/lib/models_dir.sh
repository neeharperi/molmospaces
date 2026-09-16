# Where the policy checkouts live -- openpi, molmoact2, tiptop, dreamzero. Source me:
#
#     source "$(dirname "$0")/lib/models_dir.sh"
#
# and then use "$MLSPACES_MODELS_DIR/openpi" rather than "third_party/openpi".
#
# They are NOT vendored in this repository. Each is an independent checkout with its own
# remote, its own upstream and its own environment, sitting beside this one -- so the
# default below is the parent of this repository, which is exactly where they are, and
# nothing needs setting in the normal layout.
#
# Resolved from BASH_SOURCE and never from the caller's cwd, so a script works the same
# whether it was invoked from the repository root or from anywhere else. That is the same
# idiom droid's own scripts/lib/env.sh uses, for the same reason.
#
# The Python half is MODELS_DIR in molmo_spaces/molmo_spaces_constants.py. The two compute
# their defaults independently -- from scripts/lib/ here, from molmo_spaces/__file__ there
# -- and land on the same directory for an editable checkout. Exporting
# MLSPACES_MODELS_DIR once (launch_campaign.sh, run_full_matrix.sh) moves both in lockstep,
# which is the point of them sharing one variable name.
MLSPACES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MLSPACES_MODELS_DIR="${MLSPACES_MODELS_DIR:-$(cd "$MLSPACES_ROOT/.." && pwd)}"
export MLSPACES_ROOT MLSPACES_MODELS_DIR
