#!/usr/bin/env python
"""Map MUJOCO_EGL_DEVICE_ID -> physical GPU on THIS host.

    conda activate mlspaces-classic && python scripts/probe_egl_mapping.py
    ... --infer-only    # the DRM hint alone, no rendering, nothing written

Run once per machine, before scheduling any parallel campaign, and record the result in
docs/eval_reproduction.md. It renders on each EGL device for ~15 s, so run it when the GPUs
are quiet -- a lane churning memory next door makes the deltas unreadable and it says so.

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

Measured on the 2x RTX PRO 5000 Blackwell host (driver 580.178.04), the mapping is REVERSED
and the DRM inference gets it wrong:

    EGL 0 -> card0 -> 01:00.0 -> inferred GPU 0, MEASURED GPU 1
    EGL 1 -> card2 -> c1:00.0 -> inferred GPU 1, MEASURED GPU 0
    EGL 2 -> (EGLError; no DRM node)

That is why measurement is now the default rather than the fallback: the inference resolved
cleanly here and concluded "identity", which is the one wrong answer that looks right.

Measured on the 4x H100 NVL host (driver 570.207), the mapping is REVERSED and has a
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


# How quiet the machine has to be before a memory-delta attribution means anything, and how
# much the winning card has to move. A MuJoCo offscreen renderer at 1280x720 lands in the
# hundreds of MiB, so 200 is comfortably below one renderer and far above driver noise.
CONTROL_SECONDS = 8
QUIET_MIB = 100
MIN_PEAK_MIB = 200


def gpu_mem() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return [int(x) for x in out.split()]


def measure_mapping(n_devices: int) -> list[tuple[int, int]]:
    """Render on each EGL device and see which GPU's memory moves.

    Ground truth, and the reason this mode exists: under the NVIDIA EGL vendor the devices are
    not DRM devices, so the inference above resolves nothing, and the CUDA-ordinal extension
    cannot be queried before a display is initialized. Measuring sidesteps the whole question --
    it reports what the renderer actually did, not what an API says it should do.
    """
    import tempfile
    import time
    print("\n  measuring (each device renders for ~15s)...\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(RENDER_SNIPPET)
        script = f.name
    rows = []
    try:
        # A control window first, with nothing of ours running. This probe attributes a
        # device by which GPU's *total* memory moved, and a neighbour allocating during the
        # window is indistinguishable from our own renderer -- a lane starting on GPU 0 while
        # device 1 renders 2.7 GB on GPU 1 makes GPU 0 the peak and the wrong answer gets
        # cached. Per-process attribution is not available here: an EGL renderer holds a
        # graphics context, which nvidia-smi --query-compute-apps does not list at all.
        # So the machine being quiet is a precondition, and it is now measured rather than
        # assumed.
        before = gpu_mem()
        time.sleep(CONTROL_SECONDS)
        drift = [abs(a - b) for a, b in zip(gpu_mem(), before, strict=True)]
        print(f"    control ({CONTROL_SECONDS}s, nothing of ours running): drift {drift} MiB")
        if max(drift) > QUIET_MIB:
            print(f"\n  The GPUs are not quiet: {max(drift)} MiB moved with nothing of ours")
            print("  running, which is more than a renderer's own footprint would stand out")
            print("  against. Re-run when the other jobs are done; nothing was written.")
            return []

        for i in range(n_devices):
            before = gpu_mem()
            env = dict(os.environ, MUJOCO_EGL_DEVICE_ID=str(i))
            proc = subprocess.Popen([sys.executable, script], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(15)
            delta = [a - b for a, b in zip(gpu_mem(), before, strict=True)]
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            ranked = sorted(delta, reverse=True)
            peak = ranked[0]
            runner_up = ranked[1] if len(ranked) > 1 else 0
            # Unambiguous means big AND clear of the next card, not merely the maximum. One
            # renderer allocates on one device; two cards moving together is interference.
            decisive = peak >= MIN_PEAK_MIB and peak >= 2 * max(runner_up, 0)
            gpu = delta.index(peak) if decisive else None
            rows.append((i, gpu))
            print(f"    MUJOCO_EGL_DEVICE_ID={i}  ->  "
                  + (f"nvidia-smi GPU {gpu}  (+{peak} MiB, next card +{runner_up})" if gpu is not None
                     else f"INCONCLUSIVE (deltas {delta}) -- is another job churning memory?"))
            time.sleep(2)
    finally:
        os.unlink(script)

    good = [(i, g) for i, g in rows if g is not None]
    # Partial and non-injective maps used to be written anyway. Both are wrong in the way
    # that costs the parallelism this script exists to protect: a launcher reading a map with
    # one device missing, or with two devices pointing at one card, pins lanes to the wrong
    # GPU and nothing errors.
    if len(good) < len(rows):
        print(f"\n  Only {len(good)} of {len(rows)} device(s) measured decisively.")
        print("  A partial map would pin some lanes correctly and others silently wrong,")
        print("  so nothing was written. Re-run when the GPUs are quieter.")
        return []
    if len({g for _, g in good}) != len(good):
        print(f"\n  Two EGL devices measured onto the same GPU ({good}), which one renderer")
        print("  each cannot produce -- so the windows were contaminated. Nothing written.")
        return []

    if all(i == g for i, g in good):
        print("\n  MEASURED: identity. MUJOCO_EGL_DEVICE_ID == nvidia-smi index.")
    else:
        print("\n  MEASURED mapping (use this, not the nvidia-smi index):")
        for i, g in sorted(good, key=lambda r: r[1]):
            print(f"    to render on nvidia-smi GPU {g}  ->  MUJOCO_EGL_DEVICE_ID={i}")
    write_map(good)
    return good


def write_map(pairs: list[tuple[int, int]]) -> None:
    """Cache the mapping as `<gpu> <egl_device_id>` lines for the launcher to read.

    Cached because measuring costs ~60s and the answer only changes when the driver or the
    card layout does. scripts/launch_campaign.sh regenerates it if absent, so there is no
    stale-file hazard on a fresh machine -- only on one whose GPUs were physically changed,
    which is worth a manual re-run anyway.
    """
    out = REPO_ROOT / "runs" / "_egl_mapping.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# gpu egl_device_id  -- generated by scripts/probe_egl_mapping.py"]
    lines += [f"{g} {i}" for i, g in sorted(pairs, key=lambda r: r[1])]
    out.write_text("\n".join(lines) + "\n")
    print(f"\n  wrote {out}")


def main() -> int:
    # Measurement is the default, and the DRM inference below is only a cross-check.
    #
    # It used to be the other way round -- infer, and measure only when inference resolved
    # nothing -- and on the 2x RTX PRO 5000 host that silently produced the wrong answer.
    # There the inference resolves cleanly (EGL 0 -> card0 -> 01:00.0 -> GPU 0) and concludes
    # identity, while an allocation measurement shows EGL 0 rendering on GPU **1**: the
    # mapping is reversed. Inference resolving nothing is a visible failure that already fell
    # back to measurement; inference resolving *wrongly* is not, and it costs exactly the
    # parallelism this script exists to protect -- every lane on one card, nothing erroring.
    infer_only = "--infer-only" in sys.argv
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
    if not resolved:
        print("  Inference resolved nothing (expected under the NVIDIA EGL vendor, whose")
        print("  devices are not DRM devices).")
    elif len(resolved) < len(pci2smi):
        print(f"  Inference resolved only {len(resolved)} of {len(pci2smi)} GPUs.")
    else:
        print("  Inference suggests: " + ", ".join(f"EGL {i} -> GPU {s}" for i, s in resolved))

    if infer_only:
        print("\n  --infer-only: not measured, so not written. This is a hint, not the mapping.")
        return 0

    measured = measure_mapping(len(devices))
    if not measured:
        # Non-zero, because the caller's whole question was "what is the mapping" and there
        # is no answer. launch_campaign.sh gates on this exit status.
        print("\n  NO MAPPING. runs/_egl_mapping.txt was left as it was"
              f"{' (it exists and may be stale)' if (REPO_ROOT / 'runs' / '_egl_mapping.txt').exists() else ' (absent)'}.")
        return 1
    if resolved:
        inferred = {i: str(s) for i, s in resolved}
        disagree = [(i, g) for i, g in measured if i in inferred and inferred[i] != str(g)]
        if disagree:
            # Loudly, because the inference is the thing a reader would otherwise have
            # believed, and it is what an earlier version of this script wrote down.
            print("\n  WARNING: the DRM inference DISAGREES with what was measured:")
            for i, gpu in disagree:
                print(f"    EGL {i}: inferred GPU {inferred[i]}, measured GPU {gpu}")
            print("  The measurement is the mapping. The inference is resolving DRM nodes,")
            print("  which under this vendor do not correspond to the devices EGL hands out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
