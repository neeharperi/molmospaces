#!/usr/bin/env bash
# Launch the full campaign: 7 policy servers + 7 eval lanes across 4 GPUs, in one tmux session.
#
#   DATE_TAG=20260828_full bash scripts/launch_campaign.sh servers   # start the servers only
#   DATE_TAG=20260828_full bash scripts/launch_campaign.sh lanes group-a
#   DATE_TAG=20260828_full bash scripts/launch_campaign.sh lanes group-b
#   bash scripts/launch_campaign.sh status
#
# Servers and lanes are separate subcommands on purpose: servers must be up and warmed before
# any lane starts (scripts/eval.py TCP-probes and hard-fails otherwise), and servers outlive
# the group-a -> group-b transition.
#
# GPU ASSIGNMENT. Weights total ~158 GB against 4 x ~92 GB usable, so all seven co-reside with
# room to spare. Placement is by footprint and by which pairs never spike together:
#
#   GPU0  dreamzero  45 GB                            + 1 render lane
#   GPU1  cosmos_nano 35 + cosmos_edge 10             + 2 render lanes
#   GPU2  molmoact2 22 + tiptop 10 + m2t2 5           + 2 render lanes
#   GPU3  pi05 ~19 + pi0 ~19 (JAX capped)             + 2 render lanes
#
# DreamZero gets a card to itself: it is the largest and the slowest, so it sets the critical
# path and should not queue behind anyone. TiPToP and M2T2 are deliberately together -- TiPToP
# posts a point cloud to M2T2 for every planning call, and keeping that off the cross-NUMA
# path matters more than balancing bytes. pi0/pi0.5 only fit together because
# serve_openpi.sh caps JAX's allocator; unpatched, either one would take 75% of the card.
#
# NUMA: GPU0/1 are on node 0, GPU2/3 on node 1 (verified via each device's
# /sys/bus/pci/devices/<addr>/numa_node). Lanes are CPU-pinned to their GPU's node so a
# lane's MuJoCo workers do not sit on the far socket from the card they render on.
#
# Pinning uses `taskset`, not `numactl` -- numactl is NOT installed on this host. taskset
# gives CPU affinity but not memory binding, so this is the weaker half of what numactl would
# do; it is still the half that matters for 28 render workers. The node CPU lists are read
# from /sys rather than hardcoded, so this survives a different machine.
#
# NOTE this host is SHARED. Another user's jobs sit on all four GPUs. Memory headroom is
# ample, but they consume compute, so wall-clock will exceed a dedicated-node estimate.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
SESSION="${SESSION:-campaign}"
DATE_TAG="${DATE_TAG:?set DATE_TAG, e.g. 20260828_full}"
ENVS="${MLSPACES_ENVS:-$HOME/anaconda3/envs}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/mlspaces-assets}"

# policy  LANE_GPU  NUMA_NODE   -- EGL device id is resolved per-GPU at launch (see below)
# NUMA node is asserted at launch against /sys, so a wrong entry here fails loudly.
LANES=(
  "dreamzero      0 0"
  "cosmos_nano    1 0"
  "cosmos_edge    1 0"
  "molmoact2_droid 2 1"
  "tiptop         2 1"
  "pi05_droid     3 1"
  "pi0_droid      3 1"
)

# MUJOCO_EGL_DEVICE_ID for a given nvidia-smi GPU index. Populated by probe_egl_mapping.py --
# NOT assumed to be the identity, and NOT the stale reversed constant the 2-GPU host used.
# CPU list for a NUMA node, e.g. "0-55,112-167". Empty if the node does not exist.
cpus_for_node() { cat "/sys/devices/system/node/node$1/cpulist" 2>/dev/null; }

# The NUMA node a given nvidia-smi GPU index actually sits on, per /sys.
numa_for_gpu() {
    local bus; bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$1" \
                     | tr 'A-Z' 'a-z' | sed 's/^0*\(....:..:..\..\)$/\1/')
    cat "/sys/bus/pci/devices/${bus}/numa_node" 2>/dev/null
}

egl_for() {
    "$ENVS/mlspaces-classic/bin/python" scripts/probe_egl_mapping.py 2>/dev/null \
      | awk -v g="$1" '/-> *MUJOCO_EGL_DEVICE_ID=/ && $5 == g {sub(/.*=/,"",$NF); print $NF}' \
      | head -1
}

start_servers() {
    tmux has-session -t "$SESSION" 2>/dev/null || tmux new-session -d -s "$SESSION" -n idle
    mkdir -p runs/_servers
    _srv() {  # name, then command
        local name="$1"; shift
        tmux new-window -t "$SESSION" -n "srv-$name" \
            "cd $REPO && DATE_TAG=$DATE_TAG bash scripts/serve_supervised.sh $name $*; read"
        echo "  started server: $name"
    }
    _srv dreamzero   "env GPUS=0 DIT_SPLIT=0 PORT=5000 bash scripts/serve_dreamzero.sh"
    _srv cosmos_nano "env GPU=1 PORT=8004 CKPT=nvidia/Cosmos3-Nano-Policy-DROID bash scripts/serve_cosmos.sh"
    _srv cosmos_edge "env GPU=1 PORT=8003 bash scripts/serve_cosmos.sh"
    _srv molmoact2   "env GPU=2 PORT=8000 bash scripts/serve_molmoact2.sh"
    _srv m2t2        "env GPU=2 PORT=8123 bash scripts/serve_m2t2.sh"
    # TiPToP must run from the REPO ROOT: with cwd=third_party/tiptop, sys.path[0] contains a
    # cutamp/ directory with no __init__.py, which shadows the installed editable package as a
    # PEP 420 namespace package and fails startup with a misleading "cuTAMP version mismatch".
    _srv tiptop      "env CUDA_VISIBLE_DEVICES=2 $ENVS/mlspaces-tiptop/bin/python -m tiptop.tiptop_websocket_server --port 8765"
    _srv openpi_pi05 "env GPU=3 PORT=8080 CONFIG=pi05_droid_jointpos bash scripts/serve_openpi.sh"
    _srv openpi_pi0  "env GPU=3 PORT=8081 CONFIG=pi0_droid_jointpos  bash scripts/serve_openpi.sh"
    echo
    echo "Servers starting. Wait for all 7 ports before launching lanes:"
    echo "  bash scripts/launch_campaign.sh wait-servers"
}

wait_servers() {
    local ports="8080 8081 8000 8765 5000 8003 8004 8123"
    echo "waiting for: $ports"
    while :; do
        local down=""
        for p in $ports; do
            (exec 3<>/dev/tcp/127.0.0.1/"$p") 2>/dev/null && exec 3<&- || down="$down $p"
        done
        [ -z "$down" ] && { echo "all servers up"; return 0; }
        echo "  still down:$down"
        sleep 30
    done
}

start_lanes() {
    local group="${1:-all}"
    mkdir -p runs/_lanes
    for spec in "${LANES[@]}"; do
        set -- $spec
        local policy="$1" gpu="$2" numa="$3"
        local egl; egl="$(egl_for "$gpu")"
        [ -n "$egl" ] || { echo "could not resolve EGL id for GPU $gpu; run scripts/probe_egl_mapping.py" >&2; exit 1; }
        # Assert the table's NUMA node against reality rather than trusting it.
        local actual_numa; actual_numa="$(numa_for_gpu "$gpu")"
        if [ -n "$actual_numa" ] && [ "$actual_numa" != "$numa" ]; then
            echo "GPU $gpu is on NUMA node $actual_numa, not $numa as this script's LANES table says" >&2
            exit 1
        fi
        local cpus; cpus="$(cpus_for_node "$numa")"
        local pin=""; [ -n "$cpus" ] && pin="taskset -c $cpus"
        tmux new-window -t "$SESSION" -n "lane-$policy" \
          "cd $REPO && source $HOME/anaconda3/etc/profile.d/conda.sh && \
           LANE_GPU=$gpu MUJOCO_EGL_DEVICE_ID=$egl DATE_TAG=$DATE_TAG \
           $pin bash scripts/run_full_matrix.sh $policy all $group 2>&1 | tee -a runs/_lanes/${policy}_${DATE_TAG}.log; read"
        echo "  lane $policy -> GPU $gpu (EGL $egl, NUMA $numa cpus=${cpus:-unpinned}), group=$group"
        sleep 60   # stagger: seven simultaneous model warmups thrash the HF cache and the PCIe bus
    done
}

case "${1:-}" in
    servers)      start_servers ;;
    wait-servers) wait_servers ;;
    lanes)        start_lanes "${2:-all}" ;;
    status)       python3 scripts/campaign_status.py --date "$DATE_TAG" ;;
    *) echo "usage: $0 {servers|wait-servers|lanes [all|group-a|group-b]|status}" >&2; exit 2 ;;
esac
