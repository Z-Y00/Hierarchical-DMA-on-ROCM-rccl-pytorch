#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch:rocm7.14_ubuntu26.04_py3.14_pytorch_release_2.12.0}"
OUTPUT_IMAGE="${OUTPUT_IMAGE:-rocm714-py314-rocr-rccl-ce:latest}"
RCCL_COMMIT="${RCCL_COMMIT:-69d542a2162dfbb2e4c3ad8721d443d3a282c37c}"
ROCR_COMMIT="${ROCR_COMMIT:-${RCCL_COMMIT}}"
PARALLEL_JOBS="${PARALLEL_JOBS:-32}"

docker build \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "ROCR_COMMIT=${ROCR_COMMIT}" \
    --build-arg "RCCL_COMMIT=${RCCL_COMMIT}" \
    --build-arg "PARALLEL_JOBS=${PARALLEL_JOBS}" \
    --tag "${OUTPUT_IMAGE}" \
    --file "${REPO_ROOT}/docker/Dockerfile.rocm714_py314_rccl_ce" \
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
