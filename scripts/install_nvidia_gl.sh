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
# ---- CUDA 13 forward-compat, for the cosmos env only -------------------------------------
# cosmos-framework pins torch 2.13.0+cu130 (CUDA 13.0), which needs an r580 driver; this host
# runs 570.x. NVIDIA's cuda-compat package is the supported answer on datacenter GPUs: a
# forward-compatible libcuda.so that drives the older kernel module. Not in this host's apt
# sources, so it comes from NVIDIA's public CUDA repo. Only scripts/serve_cosmos.sh puts it on
# the library path -- every other env runs cu128/cu129 against the stock driver quite happily.
COMPAT_PREFIX="${CUDA_COMPAT_PREFIX:-$HOME/cuda-compat-13}"
CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
echo "fetching cuda-compat-13-0"
COMPAT_DEB="$(curl -s "$CUDA_REPO/Packages.gz" | zcat \
    | awk '/^Package: cuda-compat-13-0$/{f=1} f&&/^Filename:/{print $2; f=0}' | tail -1)"
[ -n "$COMPAT_DEB" ] || { echo "  ERROR: cuda-compat-13-0 not found in $CUDA_REPO" >&2; exit 1; }
curl -sL -o cuda-compat.deb "$CUDA_REPO/$(basename "$COMPAT_DEB")"
rm -rf "$COMPAT_PREFIX"; mkdir -p "$COMPAT_PREFIX"
dpkg -x cuda-compat.deb "$COMPAT_PREFIX"
[ -e "$COMPAT_PREFIX/usr/local/cuda-13.0/compat/libcuda.so.1" ] \
    || { echo "  ERROR: no libcuda.so.1 after extracting cuda-compat" >&2; exit 1; }
echo "  cuda-compat installed to $COMPAT_PREFIX"

echo "installed to $PREFIX"
echo "now: . scripts/nvidia_gl_env.sh"
