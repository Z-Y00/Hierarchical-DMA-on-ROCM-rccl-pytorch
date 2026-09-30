# Hierarchical DMA on ROCm: GEMM/AllGather overlap reproduction

This repository reproduces the two-node GEMM/AllGather overlap behavior of the
hierarchical Copy Engine (CE) AllGather implementation in RCCL commit
`69d542a2162dfbb2e4c3ad8721d443d3a282c37c`.

## Reference system

- 2 nodes × 8 AMD Instinct MI355X GPUs
- 8 × 400 Gb/s Pensando/AMD AINIC RoCE rails per node
- ROCm 7.14
- PyTorch 2.12
- `rocm/pytorch:rocm7.14_ubuntu26.04_py3.14_pytorch_release_2.12.0`
- ROCr, RCCL, and rccl-tests rebuilt from the same pinned commit
- RCCL `2.30.7-HEAD:69d542a+`
- Symmetric registered AllGather buffers
- CTA policy 0 versus CTA policy 2 (hierarchical CE)
- `NCCL_DEBUG=VERSION` to avoid hot-path logging overhead

The reference runs used:

- AllGather output: 128 MiB per rank
- GEMM: BF16 `8192 × 8192 × 8192`
- 16 total ranks

## Continuous GEMM iteration impact

`benchmarks/gemm_iteration_impact.py` runs a continuous GEMM stream and injects
one AllGather after eight GEMM iterations. It records every GEMM iteration
before, during, and after the collective over 30 trials.

Reference result:

| Policy | Baseline GEMM | Overlapped median | p95 | Maximum | Median increase |
|---|---:|---:|---:|---:|---:|
| 0 | 0.841 ms | 1.203 ms | 1.260 ms | 1.399 ms | +43.0% |
| 2 CE | 0.841 ms | 0.879 ms | 0.921 ms | 0.939 ms | +4.5% |

Hierarchical CE reduces median GEMM interference by **89.5%**. Its longer
AllGather window shows that the result is compute protection rather than
improved standalone collective latency.

## Build the image

The exact test image is defined in
`docker/Dockerfile.rocm714_py314_rccl_ce`. It uses the official ROCm/PyTorch
image above, rebuilds ROCr, RCCL, and rccl-tests from commit `69d542a`, then
replaces PyTorch's wheel-private ROCr and RCCL libraries. No Flux, NeMo,
Megatron, or Transformer Engine packages are included.

```bash
./scripts/build_image.sh
```

Build or load the resulting image on both test nodes before running.

## Run

The launcher requires passwordless SSH and exclusive access to both nodes. It
refuses to run when either node has GPU utilization, allocated VRAM, or a
running container. It also mounts the host AINIC provider and matching
`libibverbs.so.1`; Ubuntu 26.04's packaged libibverbs is not ABI-compatible with
the host provider's `IBVERBS_PRIVATE_34` interface.

```bash
export HOST_A=<first-node>
export HOST_B=<second-node>
export IMAGE=rocm714-py314-rocr-rccl-ce:latest

./scripts/run_2node_overlap.sh
```

### Verify hierarchical CE dispatch

Marker logging is disabled by default because per-collective INFO output
substantially distorts performance. Run a short policy-2 diagnostic with:

```bash
RCCL_DEBUG_MARKERS=1 \
POLICIES=2 \
WARMUP=1 \
ITERATIONS=1 \
RUN_NAME=ce_dispatch_debug \
./scripts/run_2node_overlap.sh
```

For a real cross-node hierarchical DMA dispatch, the per-node logs are expected
to contain all of:

```text
NCCL INFO AllGather: opCount ...
NCCL INFO AllGather [Hierarchical CE]: ... -> RMA proxy + CE
NCCL INFO CE: rank ... -> Batch path ... hipMemcpyBatchAsync
```

Check both logs with:

```bash
rg 'AllGather: opCount|Hierarchical CE|CE: rank' \
  results/runs/ce_dispatch_debug/node*.stdout.log
```

`CTA policy flags set to 2` or `finished init RMA CE contexts` alone only
confirms configuration/initialization; it does not prove CE dispatch.

Useful overrides include `MESSAGE_BYTES`, `GEMM_M`, `GEMM_N`, `GEMM_K`,
`WARMUP`, and `ITERATIONS`.

New logs are written under `results/runs/`. The checked-in reference logs and
machine-readable metrics are under `results/reference/`.

## Primus-Turbo model shape scan

The benchmark can scan the MI355X BF16 training shapes introduced by
[Primus-Turbo PR #265](https://github.com/AMD-AGI/Primus-Turbo/pull/265).
It covers attention QKV/output, MLP gate-up/down, and LM-head projections.
The sequence length is multiplied by each model's MI355X micro-batch sizes.
Identical `(M, N, K)` shapes are measured once while all model/projection
aliases remain in the result.

```bash
HOST_A=lorrirao@<first-node> \
HOST_B=lorrirao@<second-node> \
SHAPE_SCAN=primus-mi355x-bf16 \
TIMEOUT_SECONDS=7200 \
./scripts/run_2node_overlap.sh
```

The full scan contains 160 model/projection/batch aliases and 127 unique
shapes. `SHAPE_START` and `SHAPE_COUNT` can run a contiguous subset for a
smoke test or resumable chunks. Scan runs generate:

- `report/overlap_report.json` with complete structured results.
- `report/overlap_report.csv` with one row per unique shape.
- `node0.stdout.log` and `node1.stdout.log` with raw benchmark output.

The report compares policy 0 with hierarchical CE policy 2. Its primary
benefit metric is the reduction in GEMM slowdown during overlap:

```text
100 × (policy-0 slowdown − policy-2 slowdown) / policy-0 slowdown
```

## Interpretation

Policy 2 uses a hierarchical path:

1. RMA/NIC PUTs transfer one slice per rail between nodes.
2. Copy Engines fan out local and received slices across the GPUs in each node.

The debug markers confirm this path is selected. Moving local fan-out through
Copy Engines substantially reduces GEMM interference on the aligned runtime
stack, while making the AllGather completion window longer.
