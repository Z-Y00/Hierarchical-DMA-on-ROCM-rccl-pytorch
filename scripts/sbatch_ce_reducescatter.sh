#!/usr/bin/env bash
#SBATCH --job-name=ce-rs-sdma
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Fetch the CE reduce-scatter RCCL image onto the allocated node and run the
# SDMA on/off overlap A/B against it.
#
# --gres=gpu:8 is not optional: without it Slurm hands out a couple of cores
# and no GPU cgroup, and the benchmark dies at device init.
#
#   sbatch -A yaof -p Compute-Interactive scripts/sbatch_ce_reducescatter.sh
set -euo pipefail

# Slurm copies the batch script to /var/spool/slurmd/job<id>/slurm_script, so
# BASH_SOURCE does not point into the repo here the way it does for a plain
# shell invocation. Prefer the submit directory, and fail loudly rather than
# burning the allocation on a path that has no benchmark in it.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
RUNNER="${REPO_ROOT}/scripts/run_1node_ce_reducescatter.sh"
if [[ ! -x "${RUNNER}" ]]; then
    echo "REPO_ROOT=${REPO_ROOT} has no executable ${RUNNER}" >&2
    echo "Submit from the repo root, or pass REPO_ROOT=/path/to/repo." >&2
    exit 1
fi

BASE_IMAGE="${BASE_IMAGE:-unifiedtrainingdockers.azurecr.io/utd/nightly:primus_rocm10.2_20260929}"
RCCL_COMMIT="${RCCL_COMMIT:-f214805997b44f8fe589de54b4ff3890c9c8cd4a}"
IMAGE="${IMAGE:-lorrisync/therock-main:ce-rs-primus-rocm10.2-f214805}"
GPU_TARGETS="${GPU_TARGETS:-gfx942;gfx950}"

echo "node: $(hostname)  cores: $(nproc)  gpus: ${SLURM_GPUS_ON_NODE:-?}"
df -h /var/lib/docker | tail -1
rocm-smi --showid 2>/dev/null | tail -12 || true

# Each node runs its own docker daemon, so the image built on the login node is
# not visible here. Pulling is much cheaper than a rebuild; fall back to
# building from source if the registry copy is missing.
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "=== ${IMAGE} already present ==="
elif docker pull "${IMAGE}"; then
    echo "=== pulled ${IMAGE} ==="
else
    echo "=== pull failed, building ${IMAGE} from source ==="
    docker build \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        --build-arg "RCCL_COMMIT=${RCCL_COMMIT}" \
        --build-arg "PARALLEL_JOBS=$(nproc)" \
        --build-arg PYTHON_VERSION=3.12 \
        --build-arg "GPU_TARGETS=${GPU_TARGETS}" \
        --tag "${IMAGE}" \
        --file "${REPO_ROOT}/docker/Dockerfile.primus_rccl_ce" \
        "${REPO_ROOT}/docker"
fi

IMAGE="${IMAGE}" exec "${RUNNER}"
