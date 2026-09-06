#!/usr/bin/env bash
# Apply this repo's local fixes to a third_party submodule. Idempotent: an already-applied
# patch is detected and skipped, so this is safe to re-run after a submodule update.
#
#   scripts/apply_third_party_patches.sh dreamzero    # 48GB-card single-GPU inference
#   scripts/apply_third_party_patches.sh molmoact2    # Blackwell + live-checkpoint API fix
#   scripts/apply_third_party_patches.sh tiptop       # raise the M2T2 async client timeout
#   scripts/apply_third_party_patches.sh              # all submodules that have patches
#
# These patches must be re-applied after any fresh clone or submodule checkout -- the fixes
# live here rather than in the submodules because we don't control those upstreams. See the
# per-patch header for the rationale and the base commit it was generated against, and
# docs/eval_reproduction.md for how each was found.
set -euo pipefail
cd "$(dirname "$0")/.."

apply_for() {
    local name="$1"
    local submodule="third_party/$name"
    local patch_dir="scripts/${name}_patches"

    [ -d "$patch_dir" ] || { echo "no patch dir $patch_dir; nothing to do for $name"; return 0; }
    [ -d "$submodule/.git" ] || [ -f "$submodule/.git" ] || {
        echo "error: $submodule is not a git checkout; run 'git submodule update --init' first" >&2
        return 1
    }

    echo "== $name =="
    for p in "$patch_dir"/*.patch; do
        [ -e "$p" ] || continue
        local base; base=$(basename "$p")
        if git -C "$submodule" apply --reverse --check "../../$p" 2>/dev/null; then
            echo "  already applied: $base"
        elif git -C "$submodule" apply --check "../../$p" 2>/dev/null; then
            git -C "$submodule" apply "../../$p"
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
