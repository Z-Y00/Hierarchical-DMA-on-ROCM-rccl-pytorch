#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${HOST_A:?Set HOST_A to the first benchmark node}"
: "${HOST_B:?Set HOST_B to the second benchmark node}"

IMAGE="${IMAGE:-rocm714-py314-rocr-rccl-ce:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-gemm-ag-overlap-${USER}}"
MASTER_PORT="${MASTER_PORT:-29541}"
MESSAGE_BYTES="${MESSAGE_BYTES:-134217728}"
GEMM_M="${GEMM_M:-8192}"
GEMM_N="${GEMM_N:-8192}"
GEMM_K="${GEMM_K:-8192}"
POLICIES="${POLICIES:-0 2}"
RCCL_DEBUG_MARKERS="${RCCL_DEBUG_MARKERS:-0}"
BENCHMARK_SCRIPT=/workspace/repro/benchmarks/gemm_iteration_impact.py
GEMM_REPEATS="${GEMM_REPEATS:-1}"
WARMUP="${WARMUP:-20}"
ITERATIONS="${ITERATIONS:-30}"
extra_args=(
    "--pre-gemms=${PRE_GEMMS:-8}"
    "--post-gemms=${POST_GEMMS:-16}"
    "--allgather-repeats=${ALLGATHER_REPEATS:-1}"
)

RUN_NAME="${RUN_NAME:-continuous_$(date +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/runs/${RUN_NAME}}"
SSH_OPTIONS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new)

if [[ "${RCCL_DEBUG_MARKERS}" == "1" ]]; then
    NCCL_DEBUG_LEVEL=INFO
    NCCL_DEBUG_SUBSYSTEMS=INIT,COLL,TUNING,NET,ENV
    echo "RCCL marker logging enabled; use only for short diagnostics."
else
    NCCL_DEBUG_LEVEL=VERSION
    NCCL_DEBUG_SUBSYSTEMS=INIT
fi

check_idle() {
    local host="$1"
    local state
    state="$(
        ssh "${SSH_OPTIONS[@]}" "${host}" bash -s <<'REMOTE'
set -euo pipefail
gpu_busy="$(
    rocm-smi --showuse --showmemuse --csv 2>/dev/null |
        awk -F',' '
            /^card[0-9]+/ {
                gsub(/[^0-9.]/, "", $2)
                gsub(/[^0-9.]/, "", $4)
                if (($2 + 0) > 1 || ($4 + 0) > 1) busy = 1
            }
            END { print busy + 0 }
        '
)"
containers="$(docker ps --format '{{.Names}}' | paste -sd, -)"
if [[ "${gpu_busy}" == "0" && -z "${containers}" ]]; then
    echo IDLE
else
    echo "BUSY gpu=${gpu_busy} containers=${containers:-none}"
fi
REMOTE
    )"
    echo "${host}: ${state}"
    [[ "${state}" == "IDLE" ]]
}

check_idle "${HOST_A}"
check_idle "${HOST_B}"
mkdir -p "${RESULTS_DIR}/node0" "${RESULTS_DIR}/node1"

start_container() {
    local host="$1"
    local node_results="$2"
    ssh "${SSH_OPTIONS[@]}" "${host}" bash -s -- \
        "${IMAGE}" \
        "${CONTAINER_NAME}" \
        "${REPO_ROOT}" \
        "${node_results}" <<'REMOTE'
set -euo pipefail

image="$1"
container_name="$2"
repo_root="$3"
node_results="$4"

docker image inspect "${image}" >/dev/null
test -f /etc/libibverbs.d/ionic.driver
ionic_lib="$(
    ldconfig -p |
        awk '$1 == "libionic.so.1" && !found {value=$NF; found=1} END {print value}'
)"
ibverbs_lib="$(
    ldconfig -p |
        awk '$1 == "libibverbs.so.1" && !found {value=$NF; found=1} END {print value}'
)"
test -f "${ionic_lib}"
test -f "${ibverbs_lib}"

docker rm --force "${container_name}" >/dev/null 2>&1 || true
docker run --detach --rm \
    --name "${container_name}" \
    --network=host \
    --uts=host \
    --ipc=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1:-1 \
    --ulimit nofile=1048576:1048576 \
    --security-opt seccomp=unconfined \
    --env PYTORCH_ROCM_ARCH=gfx950 \
    --volume "${repo_root}:/workspace/repro:ro" \
    --volume "${node_results}:/results" \
    --volume /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro \
    --volume "${ionic_lib}:${ionic_lib}:ro" \
    --volume "${ionic_lib}:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro" \
    --volume "${ionic_lib}:/usr/lib/x86_64-linux-gnu/libionic.so:ro" \
    --volume "${ionic_lib}:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro" \
    --volume "${ibverbs_lib}:/usr/lib/x86_64-linux-gnu/libibverbs.so.1:ro" \
    --workdir /workspace/repro \
    "${image}" \
    sleep infinity
REMOTE
}

cleanup() {
    for host in "${HOST_A}" "${HOST_B}"; do
        ssh "${SSH_OPTIONS[@]}" "${host}" \
            docker rm --force "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT

start_container "${HOST_A}" "${RESULTS_DIR}/node0"
start_container "${HOST_B}" "${RESULTS_DIR}/node1"

read -r -a policy_args <<< "${POLICIES}"
run_node() {
    local host="$1"
    local node_rank="$2"
    local log_file="$3"
    local -a command=(
        docker exec
        --env NCCL_CUMEM_ENABLE=1
        --env NCCL_LOCAL_REGISTER=0
        --env TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true
        --env NCCL_SOCKET_IFNAME=fenic
        --env NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
        --env NCCL_IB_TC=104
        --env NCCL_IB_TIMEOUT=14
        --env NCCL_IGNORE_CPU_AFFINITY=1
        --env "NCCL_DEBUG=${NCCL_DEBUG_LEVEL}"
        --env "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYSTEMS}"
        --env RCCL_MSCCL_ENABLE=0
        "${CONTAINER_NAME}"
        timeout --signal=TERM 600s
        torchrun
        --nnodes=2
        --nproc_per_node=8
        "--node_rank=${node_rank}"
        "--master_addr=${HOST_A}"
        "--master_port=${MASTER_PORT}"
        "${BENCHMARK_SCRIPT}"
        "--message-bytes=${MESSAGE_BYTES}"
        "--gemm-m=${GEMM_M}"
        "--gemm-n=${GEMM_N}"
        "--gemm-k=${GEMM_K}"
        "--gemm-repeats=${GEMM_REPEATS}"
        "--warmup=${WARMUP}"
        "--iterations=${ITERATIONS}"
        --policies
        "${policy_args[@]}"
        "${extra_args[@]}"
    )
    local remote_command
    printf -v remote_command '%q ' "${command[@]}"
    ssh "${SSH_OPTIONS[@]}" "${host}" "${remote_command}" >"${log_file}" 2>&1
}

run_node "${HOST_A}" 0 "${RESULTS_DIR}/node0.stdout.log" &
pid_a=$!
run_node "${HOST_B}" 1 "${RESULTS_DIR}/node1.stdout.log" &
pid_b=$!

status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?
if (( status != 0 )); then
    echo "Benchmark failed; inspect ${RESULTS_DIR}" >&2
    exit "${status}"
fi

awk '/^RESULT |^SUMMARY / { print }' "${RESULTS_DIR}/node0.stdout.log"
echo "Results: ${RESULTS_DIR}"
