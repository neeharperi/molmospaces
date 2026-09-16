#!/usr/bin/env bash
# Apply this repo's local fixes to a third_party submodule. Idempotent: an already-applied
# patch is detected and skipped, so this is safe to re-run after updating a checkout.
#
#   scripts/apply_third_party_patches.sh dreamzero    # 48GB-card single-GPU inference
#   scripts/apply_third_party_patches.sh              # everything still carried here
#
# WINDING DOWN. The model checkouts are no longer submodules of this repository -- each is
# an independent checkout we control, in droid/third_party/ -- so a fix belongs as a commit
# in the repo it fixes, where it can be reviewed, bisected and rebased onto upstream rather
# than replayed after every checkout. Two are left here pending that migration; the rest
# have gone:
#
#   openpi     landed as commits in the openpi checkout, and smaller than this patch was:
#              the DualCheckpointManager --resume fix had no target (that class does not
#              exist outside the fork the patch was cut against), the jointpos serving
#              configs turned out to be registered by openpi itself in
#              src/openpi/training/misc/polaris_config.py, and the host paths became CLI
#              flags in scripts/train_openpi_droid.sh rather than dataclass defaults.
#              What remained: the pbar.write loss-logging fix, the base-vs-DROID norm
#              stats fix, and the two *_from_base configs.
#   molmoact2  both hunks landed UPSTREAM (allenai/molmoact2), and further: the
#              action_mode -> inference_action_mode rename, and torch, which upstream took
#              to 2.11.0/cu128 rather than the 2.7.1 this patch asked for.
#   tiptop     the Gemini model id landed in the droid checkout (and better -- hoisted into
#              a GEMINI_MODEL_ID constant that compute_gripper_mask.py imports); the M2T2
#              async timeout is now a commit there too.
#
# See the per-patch header for the rationale and the base commit it was generated against,
# and docs/eval_reproduction.md for how each was found.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "${BASH_SOURCE[0]:-$0}")/lib/models_dir.sh"

apply_for() {
    local name="$1"
    local submodule="${MLSPACES_MODELS_DIR:?source scripts/lib/models_dir.sh first}/$name"
    local patch_dir="scripts/${name}_patches"

    [ -d "$patch_dir" ] || { echo "no patch dir $patch_dir; nothing to do for $name"; return 0; }
    [ -d "$submodule/.git" ] || [ -f "$submodule/.git" ] || {
        echo "error: $submodule is not a git checkout" >&2
        return 1
    }

    echo "== $name =="
    for p in "$patch_dir"/*.patch; do
        [ -e "$p" ] || continue
        local base; base=$(basename "$p")
        if git -C "$submodule" apply --reverse --check "$MLSPACES_ROOT/$p" 2>/dev/null; then
            echo "  already applied: $base"
        elif git -C "$submodule" apply --check "$MLSPACES_ROOT/$p" 2>/dev/null; then
            git -C "$submodule" apply "$MLSPACES_ROOT/$p"
            echo "  applied:         $base"
        else
            echo "  FAILED (conflicts or wrong base): $base" >&2
            echo "    the patch header records the base commit it was generated against;" >&2
            echo "    $submodule is at $(git -C "$submodule" rev-parse --short HEAD)" >&2
            return 1
        fi
    done
}

if [ $# -gt 0 ]; then
    for name in "$@"; do apply_for "$name"; done
else
    for patch_dir in scripts/*_patches; do
        [ -d "$patch_dir" ] || continue
        name=$(basename "$patch_dir"); name="${name%_patches}"
        apply_for "$name"
    done
fi
echo "done."
