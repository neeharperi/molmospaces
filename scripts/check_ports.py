#!/usr/bin/env python
"""Verify every policy port is free, or already held by one of our own servers.

    python scripts/check_ports.py            # before starting servers
    python scripts/check_ports.py --serving  # after: every port must be ours and listening

Why this is its own check. `scripts/eval.py` probes whether a port ACCEPTS a connection, which
answers "is something there", not "is it ours". On a shared host those differ, and the
difference is expensive: another user's process was listening on TiPToP's upstream default
port 8765, so our server could never bind (21 supervisor restarts, all "address already in
use") while the client connected to THEIR server and failed the websocket handshake. That
surfaced as `[SSL: WRONG_VERSION_NUMBER]` -- from the client's own wss:// fallback after the
ws:// attempt failed -- which names neither the port nor the collision.

Left unnoticed, this is the TiPToP-scores-0% failure mode again: a cell that runs to
completion against the wrong server, or fails in a way that reads as a policy problem.

Ownership is read from /proc: `ss` only shows the owning PID for processes we can see, so a
listening socket with no attributable PID is, by elimination, someone else's.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import POLICIES  # noqa: E402

# M2T2 is not a policy, so it is not in POLICIES, but TiPToP cannot plan without it.
EXTRA_PORTS = {"m2t2": 8123}


def listening_ports_with_owner() -> dict[int, str | None]:
    """port -> owning process name, or None when the owner is not ours to see."""
    out: dict[int, str | None] = {}
    try:
        txt = subprocess.check_output(["ss", "-ltnp"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return out
    for line in txt.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        local = parts[3]
        try:
            port = int(local.rsplit(":", 1)[1])
        except (ValueError, IndexError):
            continue
        owner = None
        if "users:((" in line:
            owner = line.split('users:(("', 1)[1].split('"', 1)[0]
        out[port] = owner
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serving", action="store_true",
                    help="require each port to be listening AND ours (post-startup check)")
    a = ap.parse_args()

    ports = {name: spec.port for name, spec in POLICIES.items()} | EXTRA_PORTS
    listening = listening_ports_with_owner()
    bad = []
    print()
    for name, port in sorted(ports.items(), key=lambda kv: kv[1]):
        if port not in listening:
            state, detail = ("FREE", "nothing listening")
            if a.serving:
                state, detail = "MISSING", "expected our server here, nothing is listening"
                bad.append((name, port, detail))
        elif listening[port] is None:
            state = "FOREIGN"
            detail = ("listening, but the owning PID is not visible to us -- another user's "
                      "process. Pick a different port for this policy.")
            bad.append((name, port, detail))
        else:
            state, detail = "OURS", f"listening ({listening[port]})"
        print(f"  {state:8s} {name:18s} :{port:<6d} {detail}")
    if bad:
        print(f"\n  {len(bad)} port problem(s):")
        for name, port, detail in bad:
            print(f"    {name} :{port} -- {detail}")
        print()
        return 1
    print("\n  all policy ports OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
