#!/usr/bin/env bash
# Install the NVIDIA EGL/Vulkan userspace into a user-owned prefix. No root required.
#
#   bash scripts/install_nvidia_gl.sh
#
# See scripts/nvidia_gl_env.sh for why this is necessary on this host. In short: the driver is
# a compute-only install, so CUDA works but there is no EGL vendor library, no Vulkan ICD and
# no Vulkan loader -- which makes MuJoCo's classic renderer fail to create a context and the
# filament renderer impossible.
#
# The extracted libraries MUST match the running kernel module. That is asserted below rather
# than assumed, because a mismatch does not fail at load time -- it fails later, at context
# creation, with an opaque EGLError.
set -euo pipefail
PREFIX="${NVIDIA_GL_PREFIX:-$HOME/nvidia-gl}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
MAJOR="${DRIVER%%.*}"
echo "running kernel driver: $DRIVER (branch $MAJOR)"

cd "$WORK"
apt-get download "libnvidia-gl-$MAJOR" libvulkan1
# Assert the GL package matches the kernel module before unpacking it anywhere.
GLDEB="$(ls libnvidia-gl-${MAJOR}_*.deb)"
case "$GLDEB" in
    *"${DRIVER}"*) echo "  libnvidia-gl matches the running driver" ;;
    *) echo "  ERROR: $GLDEB does not match running driver $DRIVER." >&2
       echo "  A mismatched libnvidia-eglcore fails at context creation, not at load." >&2
       exit 1 ;;
esac

mkdir -p "$PREFIX"
for d in *.deb; do dpkg -x "$d" "$PREFIX"; done

for f in usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 \
         usr/lib/x86_64-linux-gnu/libvulkan.so.1 \
         usr/share/glvnd/egl_vendor.d/10_nvidia.json \
         usr/share/vulkan/icd.d/nvidia_icd.json; do
    [ -e "$PREFIX/$f" ] || { echo "  ERROR: missing $f after extraction" >&2; exit 1; }
done
echo "installed to $PREFIX"
echo "now: . scripts/nvidia_gl_env.sh"
