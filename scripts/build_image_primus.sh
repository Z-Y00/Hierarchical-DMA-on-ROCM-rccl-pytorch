#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-rocm/primus:v26.7-pytorch2.12-te2.17}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-primus-v26.7-rccl-ce:latest}"
RCCL_COMMIT="${RCCL_COMMIT:-b43491c83b588e07d99f7f20ed43ed4f0170ed1c}"
PARALLEL_JOBS="${PARALLEL_JOBS:-32}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
GPU_TARGETS="${GPU_TARGETS:-gfx942;gfx950}"

docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "RCCL_COMMIT=${RCCL_COMMIT}" \
    --build-arg "PARALLEL_JOBS=${PARALLEL_JOBS}" \
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
    --build-arg "GPU_TARGETS=${GPU_TARGETS}" \
    --tag "${OUTPUT_IMAGE}" \
    --file "${REPO_ROOT}/docker/Dockerfile.primus_rccl_ce" \
    "${REPO_ROOT}/docker"

docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --env MASTER_ADDR=127.0.0.1 \
    --env MASTER_PORT=29601 \
    --env NCCL_DEBUG=VERSION \
    "${OUTPUT_IMAGE}" \
    python -c '
import torch
import torch.distributed as dist

torch.cuda.set_device(0)
dist.init_process_group("nccl", rank=0, world_size=1)
x = torch.ones(1, device="cuda")
dist.all_reduce(x)
print("all_reduce", x.item())
dist.destroy_process_group()
'
