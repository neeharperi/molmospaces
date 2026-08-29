#!/usr/bin/env python
"""Map MUJOCO_EGL_DEVICE_ID -> physical GPU on THIS host.

    conda activate mlspaces-classic && python scripts/probe_egl_mapping.py

Run once per machine, before scheduling any parallel campaign, and record the result in
docs/eval_reproduction.md.

Why this exists. scripts/run_full_matrix.sh used to hardcode MUJOCO_EGL_DEVICE_ID=1 with a
comment that the EGL index was REVERSED from nvidia-smi's ordering "on this host" -- true of
the 2-GPU Blackwell machine the campaign started on, and not a fact that travels. Carrying a
stale constant onto a 4-GPU host is the quietest possible way to lose all parallelism: every
lane renders on one card, nothing errors, and the campaign just runs N times slower with N
times the contention. So the constant is now required-and-probed rather than defaulted.

The enumeration here is deliberately the SAME call the harness makes --
molmo_spaces/renderer/opengl_context.py:39 indexes straight into EGL.eglQueryDevicesEXT()
with MUJOCO_EGL_DEVICE_ID -- rather than a plausible-looking equivalent, so the mapping this
prints is the mapping the renderer will actually use.

Each EGL device is resolved to its DRM render node, then to a PCI address via
/dev/dri/by-path, then to an nvidia-smi index. Note that a host's display/BMC VGA adapter
occupies a card node without a render node, which is a plausible source of the off-by-one
that produced the "reversed" mapping on the old host.
"""

from __future__ import annotations

import glob
import os
import subprocess

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from mujoco.egl import egl_ext as EGL  # noqa: E402
from OpenGL.EGL.EXT.device_query import eglQueryDeviceStringEXT  # noqa: E402

EGL_DRM_DEVICE_FILE_EXT = 0x3233


def drm_node_to_pci() -> dict[str, str]:
    out = {}
    for link in glob.glob("/dev/dri/by-path/*-render"):
        name = os.path.basename(link)
        pci = name.removeprefix("pci-").removesuffix("-render")
        out[os.path.realpath(link)] = pci
    return out


def pci_to_smi_index() -> dict[str, int]:
    out = {}
    txt = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"], text=True
    )
    for line in txt.strip().splitlines():
        idx, bus = (s.strip() for s in line.split(","))
        out[bus.lower()[-7:]] = int(idx)  # "00000000:3A:00.0" -> "3a:00.0"
    return out


def main() -> None:
    node2pci, pci2smi = drm_node_to_pci(), pci_to_smi_index()
    devices = EGL.eglQueryDevicesEXT()
    print(f"EGL reports {len(devices)} device(s); nvidia-smi reports {len(pci2smi)} GPU(s)\n")
    print(f"  {'MUJOCO_EGL_DEVICE_ID':<22} {'DRM render node':<22} {'PCI':<14} nvidia-smi GPU")
    rows = []
    for i, dev in enumerate(devices):
        try:
            raw = eglQueryDeviceStringEXT(dev, EGL_DRM_DEVICE_FILE_EXT)
            node = os.path.realpath(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception as e:  # a software/surfaceless EGL device has no DRM node
            node = f"<none: {type(e).__name__}>"
        pci = node2pci.get(node, "?")
        smi = pci2smi.get(pci[-7:], "?")
        rows.append((i, smi))
        print(f"  {i:<22} {node:<22} {pci:<14} {smi}")

    identity = all(str(s) == str(i) for i, s in rows if s != "?")
    print()
    if identity:
        print("  Mapping is IDENTITY: MUJOCO_EGL_DEVICE_ID == nvidia-smi index.")
    else:
        print("  Mapping is NOT identity -- use the table above, not the GPU index.")
    print("\n  Lane assignment, as MUJOCO_EGL_DEVICE_ID values:")
    for i, smi in rows:
        if smi != "?":
            print(f"    to render on nvidia-smi GPU {smi}  ->  MUJOCO_EGL_DEVICE_ID={i}")


if __name__ == "__main__":
    main()
