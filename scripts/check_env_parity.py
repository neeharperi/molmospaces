"""Verify this repo's environments match their robot-prompt-opt counterparts.

    python scripts/check_env_parity.py                    # check every paired env
    python scripts/check_env_parity.py --env mlspaces-tiptop
    python scripts/check_env_parity.py --show-allowed     # print the parsed allowlist
    python scripts/check_env_parity.py --sync-to-peer --env mlspaces-m2t2   # mutating; see below

BENCHMARK.md makes environment parity a pass/fail deliverable: a comparison between two repos
with silently divergent CUDA or transformers pins isn't licensed by anything. This script is
the enforcement. For each paired environment it resolves the installed distribution set from
both interpreters and diffs them, exiting non-zero on any difference that is not written down
in docs/env_parity.md.

The allowlist deliberately lives in docs/env_parity.md rather than here, because BENCHMARK.md
asks for divergences to be recorded with a *reason* in that file -- keeping the machine's copy
somewhere else is how the two drift apart. This module parses the ```parity-allow``` block out
of the doc, so adding an exception means writing the sentence that justifies it.

Environments are compared, not built: scripts/setup_envs.sh is what builds them, and its
--check mode verifies each one works. This script only answers "are they the same?".

The exception is --sync-to-peer, which installs the peer's exact version of any package that
differs only by version. This exists because "copied recipe plus a drift check" has one
structural weakness a copied *lockfile* would not: the recipe pins direct dependencies, not
transitive ones, so unpinned transitives (charset-normalizer, idna, uvicorn, ...) resolve to
whatever is newest on PyPI on the day each env is built. Two correct builds of the same recipe
a day apart therefore differ, and without a way to reconcile them the parity check is red
forever and stops being read -- the exact failure mode BENCHMARK.md warns about when it notes
that a drift check means "parity now depends on someone reading a red check". --sync-to-peer
does NOT touch packages that are present on one side only; those are structural differences
that need a human decision and an entry in docs/env_parity.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Overridable so this runs on a machine where the sibling checkout lives elsewhere.
PEER_REPO = Path(os.environ.get("ROBOT_PROMPT_OPT", Path.home() / "Workspace" / "robot-prompt-opt"))
CONDA_ENVS = Path(os.environ.get("CONDA_ENVS_DIR", Path.home() / "anaconda3" / "envs"))
PARITY_DOC = REPO / "docs" / "env_parity.md"
# Syncing pinned versions pulls their own dependencies, which can move other packages; repeat
# until nothing changes. Bounded so a genuine oscillation (A pins B down, B pins A up) stops
# rather than looping forever.
SYNC_MAX_ROUNDS = 5


@dataclass(frozen=True)
class EnvSpec:
    """One environment and the robot-prompt-opt env it is supposed to mirror.

    `peer` is None for environments that have no counterpart at all, which is a different
    situation from a divergent one: there is nothing to diff, so the env is reported and
    skipped rather than failed. `interpreter` overrides the conda-prefix convention for envs
    that aren't conda envs (openpi is a uv venv).
    """

    peer: str | None
    interpreter: Path | None = None  # None -> $CONDA_ENVS/<name>/bin/python


@dataclass(frozen=True)
class Finding:
    package: str
    ours: str | None
    theirs: str | None
    message: str
    is_editable: bool = False

    @property
    def syncable(self) -> bool:
        """True when both sides have the package and only the version differs.

        A package present on one side only is deliberately NOT syncable: installing it would
        paper over a real recipe difference, and removing it could break the env. Neither are
        editable installs -- see editable().
        """
        return self.ours is not None and self.theirs is not None and not self.is_editable


ENVS: dict[str, EnvSpec] = {
    # Harness envs. robot-prompt-opt renders with Isaac Sim + a 2DGS splat rasterizer and has
    # no MuJoCo environment at all, so there is nothing to mirror; these are defined here and
    # are the artifact that would flow the other way if that project ever needs them.
    "mlspaces-classic": EnvSpec(peer=None),
    "mlspaces-filament": EnvSpec(peer=None),
    # pi0.5. A different fork AND a different checkpoint, not a version difference -- the one
    # intentional divergence in the campaign. See docs/env_parity.md.
    "openpi": EnvSpec(
        peer="polaris-openpi", interpreter=REPO / "third_party/openpi/.venv/bin/python"
    ),
    # The five mirrored policy servers.
    "mlspaces-molmoact2": EnvSpec(peer="polaris-molmoact2"),
    "mlspaces-m2t2": EnvSpec(peer="polaris-m2t2"),
    "mlspaces-tiptop": EnvSpec(peer="polaris-tiptop"),
    "mlspaces-dreamzero": EnvSpec(peer="polaris-dreamzero"),
    "mlspaces-cosmos-policy": EnvSpec(peer="polaris-cosmos-policy"),
}

# Every env scripts/setup_envs.sh knows how to build must appear above, or a new env could be
# added there and never be checked for parity -- the exact silent drift this script exists to
# prevent. Asserted at import, in the same spirit as eval_common's task-table integrity check.
_SETUP_SCRIPT = REPO / "scripts" / "setup_envs.sh"


def _assert_env_table_integrity() -> None:
    if not _SETUP_SCRIPT.exists():  # allow importing from a partial checkout
        return
    declared = set(
        re.findall(r"^ALL_TARGETS=\((.*?)\)$", _SETUP_SCRIPT.read_text(), re.S | re.M)[0].split()
    )
    missing = declared - set(ENVS)
    if missing:
        raise AssertionError(
            f"setup_envs.sh builds {sorted(missing)} but check_env_parity.py does not check "
            f"them; every buildable env needs a parity verdict, even if that verdict is "
            f"'no counterpart'."
        )


_assert_env_table_integrity()


def parse_allowlist(doc: Path) -> tuple[set[str], dict[tuple[str, str], str]]:
    """Pull the machine-readable allowlist out of docs/env_parity.md.

    Returns (envs allowed to differ wholesale, {(env, package): reason}). Lines look like:

        env  openpi                         Different fork and checkpoint, so ...
        pkg  mlspaces-tiptop  opencv-python  Pinned here because ...
    """
    if not doc.exists():
        raise SystemExit(f"{doc} is missing; it is where divergences must be recorded.")
    blocks = re.findall(r"```parity-allow\n(.*?)```", doc.read_text(), re.S)
    if not blocks:
        raise SystemExit(f"{doc} has no ```parity-allow``` block to read exceptions from.")
    whole_env: set[str] = set()
    per_pkg: dict[tuple[str, str], str] = {}
    for lineno, raw in enumerate(blocks[0].splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        kind = parts[0]
        if kind == "env" and len(parts) >= 2:
            whole_env.add(parts[1])
        elif kind == "pkg" and len(parts) == 3:
            env, rest = parts[1], parts[2].split(None, 1)
            if len(rest) != 2:
                raise SystemExit(f"{doc}:{lineno}: 'pkg' needs env, package and a reason: {line!r}")
            per_pkg[(env, rest[0].lower())] = rest[1]
        else:
            raise SystemExit(f"{doc}:{lineno}: unparseable allowlist line: {line!r}")
    return whole_env, per_pkg


def _pip_list(interpreter: Path, *extra: str) -> list[dict]:
    out = subprocess.run(
        [
            str(interpreter),
            "-m",
            "pip",
            "list",
            "--format=json",
            "--disable-pip-version-check",
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"pip list failed in {interpreter}:\n{out.stderr.strip()}")
    return json.loads(out.stdout)


def _norm(name: str) -> str:
    # PEP 503 normalisation: pip reports the name as declared, which varies by installer
    # (SAM-2 vs sam_2, opencv_python vs opencv-python) and would otherwise read as a diff.
    return re.sub(r"[-_.]+", "-", name).lower()


def installed(interpreter: Path) -> dict[str, str] | None:
    """{normalised distribution name: version} for one interpreter, or None if it's missing."""
    if not interpreter.exists():
        return None
    return {_norm(d["name"]): d["version"] for d in _pip_list(interpreter)}


def editable(interpreter: Path) -> set[str]:
    """Distributions installed in editable mode, i.e. backed by a local checkout.

    These must never be synced. Their version string comes from setuptools-scm and encodes the
    checkout's git state -- a locally patched tree reports something like
    0.3.0.post1.dev0+gd8f5afdaa.d20260819 where a clean one reports 0.3.0 -- so a version
    difference here reflects the source tree, not the package index. Worse, `pip install
    <name>==<peer version>` on one of these would replace a local editable checkout with a
    PyPI wheel and silently change what the server actually runs.
    """
    if not interpreter.exists():
        return set()
    return {_norm(d["name"]) for d in _pip_list(interpreter, "--editable")}


def interpreter_for(name: str, spec: EnvSpec) -> Path:
    return spec.interpreter or (CONDA_ENVS / name / "bin" / "python")


def compare(
    name: str, spec: EnvSpec, allowed_pkgs: dict[tuple[str, str], str]
) -> tuple[str, list[Finding]]:
    """Return (verdict, findings) for one environment."""
    ours = installed(interpreter_for(name, spec))
    if ours is None:
        return "MISSING", [
            Finding(
                "",
                None,
                None,
                f"{interpreter_for(name, spec)} does not exist (run scripts/setup_envs.sh {name})",
            )
        ]
    if spec.peer is None:
        return "NO-PEER", [
            Finding(
                "",
                None,
                None,
                f"{len(ours)} packages; robot-prompt-opt has no counterpart to compare against",
            )
        ]
    peer_python = CONDA_ENVS / spec.peer / "bin" / "python"
    theirs = installed(peer_python)
    if theirs is None:
        return "PEER-MISSING", [
            Finding("", None, None, f"peer env {spec.peer} is not built on this machine")
        ]
    editables = editable(interpreter_for(name, spec)) | editable(peer_python)

    findings = []
    for pkg in sorted(set(ours) | set(theirs)):
        if (name, pkg) in allowed_pkgs:
            continue
        here, there = ours.get(pkg), theirs.get(pkg)
        if here == there:
            continue
        is_ed = pkg in editables
        note = (
            " [editable local checkout; version reflects git state, never synced]" if is_ed else ""
        )
        if here is None:
            msg = f"{pkg}: absent here, {there} in {spec.peer}{note}"
        elif there is None:
            msg = f"{pkg}: {here} here, absent in {spec.peer}{note}"
        else:
            msg = f"{pkg}: {here} here, {there} in {spec.peer}{note}"
        findings.append(Finding(pkg, here, there, msg, is_editable=is_ed))
    return ("OK" if not findings else "DIVERGED"), findings


def sync_to_peer(name: str, spec: EnvSpec, findings: list[Finding]) -> bool:
    """Install the peer's exact version for every version-only difference. Returns True if it ran."""
    syncable = [f for f in findings if f.syncable]
    if not syncable:
        return False
    interpreter = interpreter_for(name, spec)
    pins = [f"{f.package}=={f.theirs}" for f in syncable]
    print(f"  syncing {len(pins)} package(s) to {spec.peer}: {' '.join(pins)}")
    out = subprocess.run(
        [str(interpreter), "-m", "pip", "install", "--disable-pip-version-check", *pins],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        print(f"  sync FAILED:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--env", action="append", choices=sorted(ENVS), help="check only this env (repeatable)"
    )
    ap.add_argument(
        "--show-allowed", action="store_true", help="print the parsed allowlist and exit"
    )
    ap.add_argument("--parity-doc", type=Path, default=PARITY_DOC)
    ap.add_argument(
        "--sync-to-peer",
        action="store_true",
        help="MUTATING: install the peer's exact version for every version-only difference, "
        "then re-check. Does not touch packages present on only one side.",
    )
    args = ap.parse_args()

    whole_env, per_pkg = parse_allowlist(args.parity_doc)
    if args.show_allowed:
        print(f"envs allowed to diverge wholesale: {sorted(whole_env) or '(none)'}")
        for (env, pkg), reason in sorted(per_pkg.items()):
            print(f"  {env} / {pkg}: {reason}")
        return 0

    if not PEER_REPO.exists():
        print(f"robot-prompt-opt not found at {PEER_REPO}. Set $ROBOT_PROMPT_OPT.", file=sys.stderr)
        return 1

    names = args.env or list(ENVS)
    failed = []
    for name in names:
        spec = ENVS[name]
        if name in whole_env:
            print(f"{name}: ALLOWED-DIVERGENCE (documented in {args.parity_doc.name})")
            continue
        verdict, findings = compare(name, spec, per_pkg)
        if args.sync_to_peer and verdict == "DIVERGED":
            # Iterate to a fixpoint rather than syncing once: installing a pinned version pulls
            # its own dependency graph, which can move packages that previously matched (a
            # gradio downgrade dragged huggingface-hub from 1.28 to 0.36 -- a MAJOR version, and
            # exactly the kind of difference parity exists to catch). One pass would leave those
            # behind and report a confusing partial result.
            for round_ in range(1, SYNC_MAX_ROUNDS + 1):
                print(f"{name}: DIVERGED  (vs {spec.peer})  -- syncing (round {round_})")
                if not sync_to_peer(name, spec, findings):
                    break
                verdict, findings = compare(name, spec, per_pkg)
                if verdict != "DIVERGED":
                    break
            else:
                print(f"  gave up after {SYNC_MAX_ROUNDS} rounds; remaining differences below")
        print(f"{name}: {verdict}" + (f"  (vs {spec.peer})" if spec.peer else ""))
        for f in findings:
            print(f"    {f.message}")
        # NO-PEER is informational: there is nothing to be out of parity *with*. Everything
        # else that isn't OK is a failure -- including MISSING, since an env that doesn't
        # exist can't be verified and shouldn't read as green.
        if verdict not in ("OK", "NO-PEER"):
            failed.append(name)

    print()
    if failed:
        print(f"parity FAILED for: {', '.join(failed)}")
        print(
            f"Either fix the environment, or record the difference with a reason in {args.parity_doc}."
        )
        return 1
    print("environment parity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
