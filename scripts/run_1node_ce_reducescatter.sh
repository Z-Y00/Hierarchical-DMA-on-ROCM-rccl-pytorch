#!/usr/bin/env bash
# Compare GEMM interference from a reduce-scatter with and without the RCCL CE
# (SDMA copy-engine) path, on one node.
#
# CE reduce-scatter refuses to dispatch when comm->nNodes != 1, so this is
# deliberately single-node rather than a variant of run_2node_overlap.sh.
# Run it from inside a Slurm allocation, e.g.
#   salloc -A yaof -p Compute-Interactive -N1 -t 04:00:00
#   srun --pty scripts/run_1node_ce_reducescatter.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-primus-rocm10.2-rccl-ce-rs:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-ce-rs-overlap-${USER}}"
MASTER_PORT="${MASTER_PORT:-29551}"
NPROC="${NPROC:-8}"
# 128 MiB total stays inside the 256 MiB CE 2-shot window on gfx942/gfx950.
MESSAGE_BYTES="${MESSAGE_BYTES:-134217728}"
GEMM_M="${GEMM_M:-8192}"
GEMM_N="${GEMM_N:-8192}"
GEMM_K="${GEMM_K:-8192}"
SHAPE_SCAN="${SHAPE_SCAN:-single}"
SHAPE_START="${SHAPE_START:-0}"
SHAPE_COUNT="${SHAPE_COUNT:-0}"
# CE reduce-scatter requires CTA policy ZERO unless forced, so policy 0 is the
# only policy where the SDMA and non-SDMA arms are otherwise comparable.
POLICIES="${POLICIES:-0}"
WARMUP="${WARMUP:-20}"
ITERATIONS="${ITERATIONS:-30}"
GEMM_REPEATS="${GEMM_REPEATS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-INFO}"
NCCL_DEBUG_SUBSYSTEMS="${NCCL_DEBUG_SUBSYSTEMS:-INIT,COLL,TUNING}"

RUN_NAME="${RUN_NAME:-ce_rs_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/runs/${RUN_NAME}}"
mkdir -p "${RESULTS_DIR}"

docker image inspect "${IMAGE}" >/dev/null
docker rm --force "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run --detach --rm \
    --name "${CONTAINER_NAME}" \
    --network=host \
    --ipc=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \
    --security-opt seccomp=unconfined \
    --volume "${REPO_ROOT}:/workspace/repro:ro" \
    --volume "${RESULTS_DIR}:/results" \
    --workdir /workspace/repro \
    "${IMAGE}" \
    sleep infinity >/dev/null

cleanup() { docker rm --force "${CONTAINER_NAME}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

read -r -a policy_args <<< "${POLICIES}"

# arm_env <ce_reducescatter_on>
run_arm() {
    local arm="$1"
    local ce_enable="$2"
    local force="$3"
    local log_file="${RESULTS_DIR}/${arm}.log"

    echo "=== arm: ${arm} (RCCL_CE_REDUCESCATTER=${ce_enable}) ==="
    docker exec \
        --env NCCL_CUMEM_ENABLE=1 \
        --env TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \
        --env NCCL_IGNORE_CPU_AFFINITY=1 \
        --env RCCL_MSCCL_ENABLE=0 \
        --env RCCL_DDA_ENABLE=0 \
        --env "RCCL_CE_REDUCESCATTER=${ce_enable}" \
        --env "RCCL_FORCE_CE_REDUCESCATTER=${force}" \
        --env "NCCL_DEBUG=${NCCL_DEBUG_LEVEL}" \
        --env "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYSTEMS}" \
        "${CONTAINER_NAME}" \
        timeout --signal=TERM "${TIMEOUT_SECONDS}s" \
        torchrun \
        --nnodes=1 \
        "--nproc_per_node=${NPROC}" \
        --master_addr=127.0.0.1 \
        "--master_port=${MASTER_PORT}" \
        /workspace/repro/benchmarks/gemm_iteration_impact.py \
        --collective=reducescatter \
        "--message-bytes=${MESSAGE_BYTES}" \
        "--gemm-m=${GEMM_M}" \
        "--gemm-n=${GEMM_N}" \
        "--gemm-k=${GEMM_K}" \
        "--shape-scan=${SHAPE_SCAN}" \
        "--shape-start=${SHAPE_START}" \
        "--shape-count=${SHAPE_COUNT}" \
        "--gemm-repeats=${GEMM_REPEATS}" \
        "--warmup=${WARMUP}" \
        "--iterations=${ITERATIONS}" \
        --policies "${policy_args[@]}" \
        >"${log_file}" 2>&1

    # The CE path logs its dispatch decision; without this the two arms can
    # silently run the same symmetric kernel and report a null result.
    if grep -qE 'CE: rank .* -> |Taking CE collective path' "${log_file}"; then
        echo "  CE/SDMA path: ACTIVE"
    else
        echo "  CE/SDMA path: not taken"
    fi
    grep -E '^RESULT ' "${log_file}" || true
}

run_arm sdma_off 0 0
run_arm sdma_on 1 1

echo "Results: ${RESULTS_DIR}"
