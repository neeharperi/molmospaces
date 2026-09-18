#!/usr/bin/env python
"""The other half: identical payloads in, identical actions out?

    conda activate mlspaces-classic
    python scripts/probe_cross_harness_action.py --host 127.0.0.1 --port 8080

`probe_cross_harness_payload.py` shows the two harnesses build the same observation for
pi0/pi0.5 -- images byte-identical, gripper identical, joints to float32 epsilon. This
sends both of those payloads to **one live server**, back to back, and diffs the action
chunk that comes back. That is the literal form of "are the actions the same across
harnesses".

Two controls make the number readable, and without them it means nothing:

  * **self**: the same payload twice. Anything the server does nondeterministically shows
    up here, and it is the floor any cross-harness difference has to clear.
  * **perturbed**: one joint moved by 1e-3 rad. If a real change produces a delta of the
    same order as the cross-harness one, the probe cannot resolve the question and says so
    rather than reporting a reassuring zero.

Talks openpi's own websocket client, so it works against `scripts/serve_policy.py` (the
:8080/:8081 cells) as well as `examples/droid/serve_droid.py`. The rig-side payload is put
in openpi's key namespace with the same rename `openpi/examples/droid/serve_droid.py:186`
does -- that rename is part of the deployment path and belongs inside the comparison.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DROID_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DROID_ROOT))


def rig_to_openpi_keys(payload: dict, prompt: str) -> dict:
    """What openpi's droid server does with a role-keyed observation, reproduced.

    serve_droid.py:186-194. Kept here rather than imported because importing it drags in
    jax; this is six lines and the fixture test asserts the shapes it produces.
    """
    images, state = payload["images"], payload["state"]
    return {
        "observation/exterior_image_1_left": np.asarray(images["external"]),
        "observation/wrist_image_left": np.asarray(images["wrist"]),
        "observation/joint_position": np.asarray(state["joint_positions"], np.float32),
        "observation/gripper_position": np.asarray([float(state["gripper_position"])], np.float32),
        "prompt": prompt,
    }


def delta(a: dict, b: dict) -> tuple[float, str]:
    aa, bb = np.asarray(a["actions"]), np.asarray(b["actions"])
    if aa.shape != bb.shape:
        return float("inf"), f"shape {aa.shape} vs {bb.shape}"
    d = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
    return float(d.max()), f"max|d|={d.max():.3e} mean|d|={d.mean():.3e} shape={aa.shape}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--policy", default="pi05_droid")
    args = ap.parse_args()

    import probe_policy_payload as ppp
    import probe_cross_harness_payload as xh

    ppp.IMG_H, ppp.IMG_W = xh.BENCH_H, xh.BENCH_W
    from molmo_spaces.evaluation.configs import evaluation_configs as E
    import droid.policy.observation as observation
    import droid.policy.protocol as protocol
    import json

    cameras = ["exo_camera_1", "wrist_camera"]
    ms_obs = ppp.build_obs(cameras)
    droid_obs = xh.build_droid_obs(ms_obs, 0.824033)
    prompt = ppp.PROMPT

    cfg = getattr(E, ppp.POLICIES[args.policy])()
    ppp._ensure_camera_config(cfg, cameras)
    pol = cfg.policy_config.policy_cls(cfg)
    pol.task = ppp._FakeTask()
    pol.reset()
    native = pol.obs_to_model_input([ms_obs])

    meta = json.loads((DROID_ROOT / "scripts/tests/fixtures/policy_specs/openpi.json").read_text())["metadata"]
    spec = protocol.parse_spec(meta)
    rig = rig_to_openpi_keys(observation.build(spec, droid_obs, prompt=prompt, step=0), prompt)

    # The payload comparison, restated here so this script stands alone.
    for key in sorted(set(native) & set(rig)):
        if key == "prompt":
            same = native[key] == rig[key]
            print(f"  payload {key:36s} {'same' if same else f'DIFFERS {native[key]!r} vs {rig[key]!r}'}")
            continue
        a, b = np.asarray(native[key]), np.asarray(rig[key])
        d = np.abs(a.astype(np.float64) - b.reshape(a.shape).astype(np.float64)).max() if a.shape == b.shape else float("inf")
        print(f"  payload {key:36s} max|d|={d:.3e}")

    from openpi_client import websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"\n  server {args.host}:{args.port}\n")

    # The prompt differs by case between harnesses (MolmoSpaces lowercases). Hold it fixed
    # here so this measures the observation path, not the known prompt divergence.
    native = dict(native, prompt=prompt)

    client.infer(native)                       # warm the compile cache; discard
    a1 = client.infer(native)
    a2 = client.infer(native)
    b1 = client.infer(rig)

    perturbed = dict(native)
    jp = np.array(native["observation/joint_position"], np.float32).copy()
    jp[0] += 1e-3
    perturbed["observation/joint_position"] = jp
    p1 = client.infer(perturbed)

    self_d, self_s = delta(a1, a2)
    cross_d, cross_s = delta(a1, b1)
    pert_d, pert_s = delta(a1, p1)

    print(f"  self      (same payload twice)     {self_s}")
    print(f"  CROSS     (simulation vs deployment) {cross_s}")
    print(f"  perturbed (joint 0 + 1e-3 rad)     {pert_s}")

    print()
    if pert_d <= max(self_d, 1e-12) * 10:
        print("  VOID: a real 1e-3 rad change moves the action no more than the server's own"
              "\n  run-to-run noise, so this server cannot resolve the question. Restart it"
              "\n  with DETERMINISTIC=1 and try again.")
        return 2
    if cross_d <= self_d:
        print("  IDENTICAL: the cross-harness difference is at or below the server's own"
              "\n  run-to-run noise, while a 1e-3 rad perturbation is clearly resolved.")
        return 0
    print(f"  DIFFERS: cross-harness {cross_d:.3e} exceeds the self-consistency floor {self_d:.3e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
