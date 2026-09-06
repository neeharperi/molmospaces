# Source this before any MuJoCo rendering on this host.
#
#   . scripts/nvidia_gl_env.sh
#
# WHY THIS EXISTS. This machine's NVIDIA driver is a COMPUTE-ONLY install: nvidia-utils-570 is
# present (so CUDA and nvidia-smi work) but libnvidia-gl-570 is not, so the host has
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
# Verified working: classic (EGL/OpenGL) and filament (Vulkan) both render, and EGL device
# index maps IDENTICALLY to nvidia-smi index (measured by allocation, not inferred).
NVIDIA_GL_PREFIX="${NVIDIA_GL_PREFIX:-$HOME/nvidia-gl}"

if [ -d "$NVIDIA_GL_PREFIX/usr/lib/x86_64-linux-gnu" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_GL_PREFIX/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_GL_PREFIX/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    export VK_ICD_FILENAMES="$NVIDIA_GL_PREFIX/usr/share/vulkan/icd.d/nvidia_icd.json"
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
else
    echo "warning: no NVIDIA GL userspace at $NVIDIA_GL_PREFIX;" >&2
    echo "         run scripts/install_nvidia_gl.sh (no root required)" >&2
fi
