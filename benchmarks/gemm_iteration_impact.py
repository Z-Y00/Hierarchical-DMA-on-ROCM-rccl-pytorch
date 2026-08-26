#!/usr/bin/env python3
"""Measure per-GEMM latency while one AllGather overlaps a continuous loop."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


@dataclass
class ImpactResult:
    policy: int
    baseline_gemm_ms: float
    overlapped_gemm_median_ms: float
    overlapped_gemm_p95_ms: float
    overlapped_gemm_max_ms: float
    post_gemm_ms: float
    allgather_window_ms: float
    median_latency_increase_pct: float
    p95_latency_increase_pct: float
    max_latency_increase_pct: float
    overlapped_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--message-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--gemm-m", type=int, default=8192)
    parser.add_argument("--gemm-n", type=int, default=8192)
    parser.add_argument("--gemm-k", type=int, default=8192)
    parser.add_argument("--gemm-repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--pre-gemms", type=int, default=8)
    parser.add_argument("--post-gemms", type=int, default=16)
    parser.add_argument("--allgather-repeats", type=int, default=1)
    parser.add_argument("--policies", type=int, nargs="+", default=[0, 2])
    return parser.parse_args()


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def max_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def min_count_across_ranks(value: int, device: torch.device) -> int:
    tensor = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def make_policy_group(policy: int, world_size: int) -> dist.ProcessGroup:
    options = dist.ProcessGroupNCCL.Options()
    options.config.cta_policy = policy
    options.config.split_share = 0
    return dist.new_group(
        ranks=list(range(world_size)),
        backend="nccl",
        pg_options=options,
        group_desc=f"GEMM_ITERATION_IMPACT_POLICY_{policy}",
    )


def make_symmetric_buffer(
    group: dist.ProcessGroup,
    device: torch.device,
    total_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor, object, object]:
    if total_bytes % group.size():
        raise ValueError("message bytes must be divisible by world size")

    symm_mem.enable_symm_mem_for_group(group.group_name)
    dist.barrier(group=group)
    pool = symm_mem.get_mem_pool(device)
    with torch.cuda.use_mem_pool(pool):
        output = torch.empty(total_bytes, dtype=torch.uint8, device=device)
    symmetric_memory = symm_mem.rendezvous(output, group=group.group_name)

    input_bytes = total_bytes // group.size()
    local_input = output.narrow(0, group.rank() * input_bytes, input_bytes)
    local_input.fill_(group.rank() % 251)
    dist.barrier(group=group)
    return output, local_input, pool, symmetric_memory


def allgather(
    output: torch.Tensor,
    local_input: torch.Tensor,
    group: dist.ProcessGroup,
) -> None:
    work = dist.all_gather_into_tensor(
        output,
        local_input,
        group=group,
        async_op=True,
    )
    work.wait()


def launch_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    repeats: int,
) -> None:
    for _ in range(repeats):
        torch.mm(a, b, out=out)


def run_trial(
    control_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    gemm_fn,
    allgather_fn,
    pre_gemms: int,
    post_gemms: int,
    allgather_repeats: int,
) -> tuple[list[float], list[float], list[float], float]:
    torch.cuda.synchronize()

    origin = torch.cuda.Event(enable_timing=True)
    trigger = torch.cuda.Event()
    comm_start = torch.cuda.Event(enable_timing=True)
    comm_end = torch.cuda.Event(enable_timing=True)
    final = torch.cuda.Event()
    pre_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    post_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    with torch.cuda.stream(control_stream):
        origin.record()

    compute_stream.wait_event(origin)
    with torch.cuda.stream(compute_stream):
        for _ in range(pre_gemms):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            gemm_fn()
            end.record()
            pre_events.append((start, end))
        trigger.record()

    comm_stream.wait_event(trigger)
    with torch.cuda.stream(comm_stream):
        comm_start.record()
        for _ in range(allgather_repeats):
            allgather_fn()
        comm_end.record()

    with torch.cuda.stream(compute_stream):
        for _ in range(post_gemms):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            gemm_fn()
            end.record()
            post_events.append((start, end))
        final.record()

    with torch.cuda.stream(control_stream):
        control_stream.wait_event(final)
        control_stream.wait_event(comm_end)
    control_stream.synchronize()

    comm_start_ms = origin.elapsed_time(comm_start)
    comm_end_ms = origin.elapsed_time(comm_end)
    pre_latencies = [start.elapsed_time(end) for start, end in pre_events]
    overlapped_latencies: list[float] = []
    post_latencies: list[float] = []
    for start, end in post_events:
        start_ms = origin.elapsed_time(start)
        end_ms = origin.elapsed_time(end)
        latency = start.elapsed_time(end)
        if start_ms < comm_end_ms and end_ms > comm_start_ms:
            overlapped_latencies.append(latency)
        else:
            post_latencies.append(latency)

    if not overlapped_latencies:
        raise RuntimeError("No GEMM iteration overlapped the AllGather window")
    return (
        pre_latencies,
        overlapped_latencies,
        post_latencies,
        comm_start.elapsed_time(comm_end),
    )


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    symm_mem.set_backend("NCCL")
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()

    torch.manual_seed(1234 + rank)
    a = torch.randn(
        (args.gemm_m, args.gemm_k),
        dtype=torch.bfloat16,
        device=device,
    )
    b = torch.randn(
        (args.gemm_k, args.gemm_n),
        dtype=torch.bfloat16,
        device=device,
    )
    out = torch.empty(
        (args.gemm_m, args.gemm_n),
        dtype=torch.bfloat16,
        device=device,
    )

    control_stream = torch.cuda.default_stream(device)
    compute_stream = torch.cuda.Stream(device=device, priority=0)
    comm_stream = torch.cuda.Stream(device=device, priority=-1)
    gemm_fn = lambda: launch_gemm(a, b, out, args.gemm_repeats)

    with torch.cuda.stream(compute_stream):
        for _ in range(args.warmup):
            gemm_fn()
    torch.cuda.synchronize()

    results: list[ImpactResult] = []
    live_resources: list[tuple[object, ...]] = []
    for policy in args.policies:
        group = make_policy_group(policy, world_size)
        output, local_input, pool, symmetric_memory = make_symmetric_buffer(
            group,
            device,
            args.message_bytes,
        )
        live_resources.append(
            (group, output, local_input, pool, symmetric_memory)
        )
        allgather_fn = lambda: allgather(output, local_input, group)

        for _ in range(args.warmup):
            with torch.cuda.stream(comm_stream):
                allgather_fn()
        torch.cuda.synchronize()
        dist.barrier()

        pre_samples: list[float] = []
        overlap_samples: list[float] = []
        post_samples: list[float] = []
        comm_samples: list[float] = []
        for _ in range(args.iterations):
            pre, overlap, post, comm_ms = run_trial(
                control_stream,
                compute_stream,
                comm_stream,
                gemm_fn,
                allgather_fn,
                args.pre_gemms,
                args.post_gemms,
                args.allgather_repeats,
            )
            pre_samples.extend(pre)
            overlap_samples.extend(overlap)
            post_samples.extend(post)
            comm_samples.append(comm_ms)

        baseline_local = float(statistics.median(pre_samples + post_samples))
        overlap_median_local = float(statistics.median(overlap_samples))
        overlap_p95_local = percentile(overlap_samples, 95.0)
        overlap_max_local = max(overlap_samples)
        post_local = float(statistics.median(post_samples))
        comm_local = float(statistics.median(comm_samples))

        baseline_ms = max_across_ranks(baseline_local, device)
        overlap_median_ms = max_across_ranks(overlap_median_local, device)
        overlap_p95_ms = max_across_ranks(overlap_p95_local, device)
        overlap_max_ms = max_across_ranks(overlap_max_local, device)
        post_ms = max_across_ranks(post_local, device)
        comm_ms = max_across_ranks(comm_local, device)
        overlap_count = min_count_across_ranks(len(overlap_samples), device)

        result = ImpactResult(
            policy=policy,
            baseline_gemm_ms=baseline_ms,
            overlapped_gemm_median_ms=overlap_median_ms,
            overlapped_gemm_p95_ms=overlap_p95_ms,
            overlapped_gemm_max_ms=overlap_max_ms,
            post_gemm_ms=post_ms,
            allgather_window_ms=comm_ms,
            median_latency_increase_pct=(
                100.0 * (overlap_median_ms - baseline_ms) / baseline_ms
            ),
            p95_latency_increase_pct=(
                100.0 * (overlap_p95_ms - baseline_ms) / baseline_ms
            ),
            max_latency_increase_pct=(
                100.0 * (overlap_max_ms - baseline_ms) / baseline_ms
            ),
            overlapped_samples=overlap_count,
        )
        results.append(result)
        if rank == 0:
            print("RESULT " + json.dumps(asdict(result), sort_keys=True), flush=True)

        torch.cuda.synchronize()
        dist.barrier()

    if rank == 0:
        payload = {
            "world_size": world_size,
            "message_bytes": args.message_bytes,
            "gemm": {
                "m": args.gemm_m,
                "n": args.gemm_n,
                "k": args.gemm_k,
                "repeats_per_iteration": args.gemm_repeats,
                "dtype": "bfloat16",
            },
            "trials": args.iterations,
            "pre_gemms": args.pre_gemms,
            "post_gemms": args.post_gemms,
            "allgather_repeats": args.allgather_repeats,
            "results": [asdict(result) for result in results],
        }
        print("SUMMARY " + json.dumps(payload, sort_keys=True), flush=True)

    torch.cuda.synchronize()
    dist.barrier()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
