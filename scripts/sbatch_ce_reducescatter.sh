#!/usr/bin/env bash
#SBATCH --job-name=ce-rs-sdma
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Build the CE reduce-scatter RCCL image on the allocated node and run the
# SDMA on/off overlap A/B against it.
#
# The image is rebuilt here rather than shipped from the login node because
# each node runs its own docker daemon and the image is ~41 GB -- larger than
# the free space on the NFS home, so `docker save` to a shared file is not an
# option.
#
#   sbatch -A yaof -p Compute-Interactive scripts/sbatch_ce_reducescatter.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_IMAGE="${BASE_IMAGE:-unifiedtrainingdockers.azurecr.io/utd/nightly:primus_rocm10.2_20260929}"
RCCL_COMMIT="${RCCL_COMMIT:-f214805997b44f8fe589de54b4ff3890c9c8cd4a}"
IMAGE="${IMAGE:-primus-rocm10.2-rccl-ce-rs:latest}"
GPU_TARGETS="${GPU_TARGETS:-gfx942;gfx950}"

echo "node: $(hostname)  cores: $(nproc)"
df -h /var/lib/docker | tail -1

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "=== building ${IMAGE} ==="
    docker build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "RCCL_COMMIT=${RCCL_COMMIT}" \
        --build-arg "PARALLEL_JOBS=$(nproc)" \
        --build-arg PYTHON_VERSION=3.12 \
        --build-arg "GPU_TARGETS=${GPU_TARGETS}" \
        --tag "${IMAGE}" \
        --file "${REPO_ROOT}/docker/Dockerfile.primus_rccl_ce" \
        "${REPO_ROOT}/docker"
else
    echo "=== ${IMAGE} already present, skipping build ==="
fi

IMAGE="${IMAGE}" exec "${REPO_ROOT}/scripts/run_1node_ce_reducescatter.sh"
