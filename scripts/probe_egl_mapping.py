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

Each EGL device is resolved to its DRM node, then to a PCI address via /dev/dri/by-path, then
to an nvidia-smi index. Both node flavours must be handled: EGL_DRM_DEVICE_FILE_EXT returns a
PRIMARY node (/dev/dri/cardN) on this driver, not a render node (/dev/dri/renderDN), and a
lookup table built from only `*-render` symlinks silently resolves nothing.

Measured on this host (4x H100 NVL, driver 570.207), the mapping is REVERSED and has a
non-GPU entry:

    EGL 0 -> card4 -> ae:00.0 -> nvidia-smi GPU 3
    EGL 1 -> card3 -> ad:00.0 -> nvidia-smi GPU 2
    EGL 2 -> card2 -> 3b:00.0 -> nvidia-smi GPU 1
    EGL 3 -> card1 -> 3a:00.0 -> nvidia-smi GPU 0
    EGL 4 -> (no DRM node; a software/surfaceless device)

card0 is the BMC VGA adapter at 02:00.0 -- a primary node with no render node and no CUDA
device. That asymmetry is why EGL's device count (5) exceeds nvidia-smi's (4), and it is the
kind of thing that makes an eyeballed mapping wrong.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import pathlib
import subprocess
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from mujoco.egl import egl_ext as EGL  # noqa: E402
from OpenGL.EGL.EXT.device_query import eglQueryDeviceStringEXT  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EGL_DRM_DEVICE_FILE_EXT = 0x3233
_egl_lib = ctypes.CDLL(ctypes.util.find_library("EGL") or "libEGL.so.1")
_egl_lib.eglGetProcAddress.restype = ctypes.c_void_p
_egl_lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]

EGL_CUDA_DEVICE_NV = 0x323A  # EGL_NV_device_cuda: the CUDA ordinal, i.e. the nvidia-smi index


def drm_node_to_pci() -> dict[str, str]:
    """Both primary (cardN) and render (renderDN) nodes -> PCI address.

    EGL hands back whichever flavour the driver chooses; on this host it is the primary node.
    Indexing only render nodes resolves nothing and, worse, resolves nothing *silently*.
    """
    out = {}
    for suffix in ("card", "render"):
        for link in glob.glob(f"/dev/dri/by-path/*-{suffix}"):
            pci = os.path.basename(link).removeprefix("pci-").removesuffix(f"-{suffix}")
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



RENDER_SNIPPET = """
import os, time
os.environ["MUJOCO_GL"] = "egl"
import mujoco
m = mujoco.MjModel.from_xml_string(
    "<mujoco><visual><global offwidth='1280' offheight='720'/></visual>"
    "<worldbody><light pos='0 0 3'/><geom type='box' size='.3 .3 .3'/></worldbody></mujoco>")
d = mujoco.MjData(m); r = mujoco.Renderer(m, 720, 1280)
t0 = time.time()
while time.time() - t0 < 40:
    mujoco.mj_step(m, d); r.update_scene(d); r.render()
"""


def gpu_mem() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return [int(x) for x in out.split()]


def measure_mapping(n_devices: int) -> None:
    """Render on each EGL device and see which GPU's memory moves.

    Ground truth, and the reason this mode exists: under the NVIDIA EGL vendor the devices are
    not DRM devices, so the inference above resolves nothing, and the CUDA-ordinal extension
    cannot be queried before a display is initialized. Measuring sidesteps the whole question --
    it reports what the renderer actually did, not what an API says it should do.
    """
    import tempfile, time
    print("\n  measuring (each device renders for ~15s)...\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(RENDER_SNIPPET)
        script = f.name
    rows = []
    try:
        for i in range(n_devices):
            before = gpu_mem()
            env = dict(os.environ, MUJOCO_EGL_DEVICE_ID=str(i))
            proc = subprocess.Popen([sys.executable, script], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(15)
            delta = [a - b for a, b in zip(gpu_mem(), before)]
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            peak = max(delta)
            gpu = delta.index(peak) if peak > 30 else None
            rows.append((i, gpu))
            print(f"    MUJOCO_EGL_DEVICE_ID={i}  ->  "
                  + (f"nvidia-smi GPU {gpu}  (+{peak} MiB)" if gpu is not None
                     else f"INCONCLUSIVE (deltas {delta}) -- is another job churning memory?"))
            time.sleep(2)
    finally:
        os.unlink(script)
    good = [(i, g) for i, g in rows if g is not None]
    if len(good) == len(rows) and all(i == g for i, g in good):
        print("\n  MEASURED: identity. MUJOCO_EGL_DEVICE_ID == nvidia-smi index.")
    elif good:
        print("\n  MEASURED mapping (use this, not the nvidia-smi index):")
        for i, g in sorted(good, key=lambda r: r[1]):
            print(f"    to render on nvidia-smi GPU {g}  ->  MUJOCO_EGL_DEVICE_ID={i}")
    else:
        print("\n  Nothing measurable. Re-run when the GPUs are quieter.")
        return
    write_map(good)


def write_map(pairs: list[tuple[int, int]]) -> None:
    """Cache the mapping as `<gpu> <egl_device_id>` lines for the launcher to read.

    Cached because measuring costs ~60s and the answer only changes when the driver or the
    card layout does. scripts/launch_campaign.sh regenerates it if absent, so there is no
    stale-file hazard on a fresh machine -- only on one whose GPUs were physically changed,
    which is worth a manual re-run anyway.
    """
    out = REPO_ROOT / "runs" / "_egl_mapping.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# gpu egl_device_id  -- generated by scripts/probe_egl_mapping.py"]
    lines += [f"{g} {i}" for i, g in sorted(pairs, key=lambda r: r[1])]
    out.write_text("\n".join(lines) + "\n")
    print(f"\n  wrote {out}")


def main() -> None:
    measure = "--measure" in sys.argv
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

    resolved = [(i, s) for i, s in rows if s != "?"]
    print()
    if measure or not resolved:
        if not resolved:
            print("  Inference resolved nothing (expected under the NVIDIA EGL vendor, whose")
            print("  devices are not DRM devices). Falling back to measurement.")
        measure_mapping(len(devices))
        return
    if not resolved:
        print("  NOTHING RESOLVED. Do not guess a mapping from this -- every EGL device failed")
        print("  to resolve to an nvidia-smi index, so any table below would be fabricated.")
        raise SystemExit(2)
    if len(resolved) < len(pci2smi):
        print(f"  WARNING: only {len(resolved)} of {len(pci2smi)} GPUs resolved.")
    identity = all(str(s) == str(i) for i, s in resolved)
    if identity:
        print("  Mapping is IDENTITY: MUJOCO_EGL_DEVICE_ID == nvidia-smi index.")
    else:
        print("  Mapping is NOT identity -- use the table below, NOT the nvidia-smi index.")
    print("\n  Lane assignment, as MUJOCO_EGL_DEVICE_ID values:")
    for i, smi in sorted(resolved, key=lambda r: int(r[1])):
        print(f"    to render on nvidia-smi GPU {smi}  ->  MUJOCO_EGL_DEVICE_ID={i}")


if __name__ == "__main__":
    main()
