# Source this before any MuJoCo rendering on this host.
#
#   . scripts/nvidia_gl_env.sh
#
# WHY THIS EXISTS. It was written for a host whose NVIDIA driver was a COMPUTE-ONLY install.
# A host with a full install needs only the second branch below -- the vendor pin -- and not
# the unpacked prefix; both are handled, and which one applies is decided by what is on disk
# rather than by a flag.
#
# The compute-only case: nvidia-utils-570 present (so CUDA and nvidia-smi work) but
# libnvidia-gl-570 absent, so the host has
#
#   * no libEGL_nvidia.so and no /usr/share/glvnd/egl_vendor.d/10_nvidia.json
#   * no Vulkan ICD at all (/usr/share/vulkan/icd.d is absent)
#   * no Vulkan loader (libvulkan.so.1)
#
# The only EGL vendor installed is Mesa, whose devices are DRM nodes under /dev/dri -- and
# /dev/dri/* are root:video / root:render 0660 while this user is in neither group. So MuJoCo's
# EGL path failed with "Cannot initialize a EGL device display ... does not support the
# PLATFORM_DEVICE extension", and filament (Vulkan) could not have worked at all.
#
# Docker does not solve this: nvidia-container-toolkit bind-mounts the HOST driver's userspace
# into the container, so a container on a compute-only host has no EGL/Vulkan either
# (verified with NVIDIA_DRIVER_CAPABILITIES=all).
#
# The fix needs no root. The driver's userspace libraries are just files, and they only have to
# match the running kernel module -- so libnvidia-gl-570 and libvulkan1 are unpacked with
# `dpkg -x` into a user-owned prefix and pointed at with the standard loader env vars. The
# version is pinned to the running driver ON PURPOSE: a mismatch between libnvidia-eglcore and
# the kernel module fails at context creation, not at load, which is a confusing place to
# discover it. Re-run scripts/install_nvidia_gl.sh after any host driver upgrade.
#
# Verified working: classic (EGL/OpenGL) and filament (Vulkan) both render. The EGL device
# index is REVERSED from the nvidia-smi index on the 2-card host -- measured by allocation,
# and only stable once the vendor is pinned. Ask scripts/probe_egl_mapping.py rather than
# assuming either answer.
NVIDIA_GL_PREFIX="${NVIDIA_GL_PREFIX:-$HOME/nvidia-gl}"

SYSTEM_EGL_VENDOR=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
SYSTEM_VK_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json

if [ -d "$NVIDIA_GL_PREFIX/usr/lib/x86_64-linux-gnu" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_GL_PREFIX/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_GL_PREFIX/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    export VK_ICD_FILENAMES="$NVIDIA_GL_PREFIX/usr/share/vulkan/icd.d/nvidia_icd.json"
    NVIDIA_GL_FOUND=yes
elif [ -f "$SYSTEM_EGL_VENDOR" ]; then
    # The host has a full driver install, so there is nothing to unpack -- but the vendor
    # still has to be PINNED, and that is not optional.
    #
    # glvnd enumerates every vendor it finds. With Mesa's JSON also present,
    # eglQueryDevicesEXT() returns 5 devices here rather than 2: the two cards, plus Mesa
    # entries whose DRM nodes this user cannot open (/dev/dri/* are root:video and
    # root:render 0660). Two things follow, both bad. MuJoCo indexes straight into that
    # list, so MUJOCO_EGL_DEVICE_ID stops meaning "which GPU" -- and which index is which
    # card depends on vendor load order, so a lane assignment measured once stops being
    # true. And a lane that lands on a Mesa entry fails with "failed to open
    # /dev/dri/renderD129: Permission denied" followed by an ImportError about
    # PLATFORM_DEVICE, which names neither the vendor nor the permission as the cause.
    #
    # Pinned, the list is exactly the two cards and the mapping is stable (and reversed --
    # see scripts/probe_egl_mapping.py, which measures it).
    export __EGL_VENDOR_LIBRARY_FILENAMES="$SYSTEM_EGL_VENDOR"
    [ -f "$SYSTEM_VK_ICD" ] && export VK_ICD_FILENAMES="$SYSTEM_VK_ICD"
    NVIDIA_GL_FOUND=yes
else
    echo "warning: no NVIDIA GL userspace at $NVIDIA_GL_PREFIX and none in /usr;" >&2
    echo "         run scripts/install_nvidia_gl.sh (no root required)" >&2
fi

# Set once, for whichever branch found a vendor. These used to be set inside the branches,
# and PYOPENGL_PLATFORM only inside the full-install one -- so the two supported host shapes
# handed the renderer different environments, and on the unpacked-prefix host anything going
# through PyOpenGL (scripts/probe_egl_mapping.py, which is what decides the lane mapping)
# picked its platform by guesswork while MuJoCo itself was told egl.
if [ "${NVIDIA_GL_FOUND:-no}" = "yes" ]; then
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi
unset NVIDIA_GL_FOUND
