# pinned_assets_20260917.json

`pinned_assets_20260816.json` with **one** value changed: `robots/g1` 20260802 -> 20260815.
Everything else -- every Franka asset, every scene, every object, every benchmark -- is
byte-identical, so a cell run against this pin is comparable to one run against the 0816 pin.

## Why it exists

On 2026-09-17 the `robots/g1` symlinks under `$MLSPACES_ASSETS_DIR` were repointed from
20260802 to 20260815, and after that every `molmo_spaces` import died before doing anything:

    ValueError: Version mismatch for robots/g1: installed=20260815, requested=20260802.
                Set force_install=True to overwrite.

The cause was `scripts/sim.sh --scenes`, which ran `sim_server.py --list` **without**
exporting `MLSPACES_PINNED_ASSETS_FILE`. `--list` short-circuits before the EGL probe but
after importing `molmo_spaces`, and `molmo_spaces/env/arena/randomization/texture.py:10`
calls `get_resource_manager()` at import time -- so an unpinned manager installed the newest
version of everything it could see. Merely *listing the scene configs* upgraded the assets
out from under every other harness on the machine. That gap in `sim.sh` is fixed.

## Why bump the pin rather than restore the tree

Restoring means `MLSPACES_FORCE_INSTALL=True`, which rewrites symlinks across the whole
shared asset tree to fix one entry. `robots/g1` is a Unitree humanoid: it is referenced by
**zero** episodes in either benchmark this campaign evaluates (checked, both draws are
Franka), and `_assert_data_versions_match()` warns rather than fails on a version it does not
use. Bumping one unused line in a derived pin changes nothing that any Franka rollout loads,
and touches no shared state at all.

## To restore 20260802 instead

Both versions are in `~/.cache/molmo-spaces-resources/robots/g1/`, so this needs no download:

    MLSPACES_PINNED_ASSETS_FILE=<...>/reference/pinned_assets_20260816.json \
    MLSPACES_FORCE_INSTALL=True \
    python -c "from molmo_spaces.molmo_spaces_constants import get_resource_manager; get_resource_manager()"

Then delete this file and drop the `MLSPACES_PINNED_ASSETS_FILE` export.
