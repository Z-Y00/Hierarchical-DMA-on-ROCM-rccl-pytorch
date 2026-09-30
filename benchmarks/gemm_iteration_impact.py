#!/usr/bin/env python3
"""Measure per-GEMM latency while one collective overlaps a continuous loop."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from primus_gemm_shapes import (
    PRIMUS_TURBO_COMMIT,
    PRIMUS_TURBO_PR,
    GemmShape,
    get_primus_mi355x_bf16_shapes,
)


@dataclass
class ImpactResult:
    shape_id: int
    m: int
    n: int
    k: int
    aliases: list[dict[str, object]]
    policy: int
    collective: str
    baseline_gemm_ms: float
    overlapped_gemm_median_ms: float
    overlapped_gemm_p95_ms: float
    overlapped_gemm_max_ms: float
    post_gemm_ms: float
    collective_window_ms: float
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
    parser.add_argument("--collective-repeats", type=int, default=1)
    parser.add_argument(
        "--collective",
        choices=["allgather", "reducescatter"],
        default="allgather",
    )
    parser.add_argument("--policies", type=int, nargs="+", default=[0, 2])
    parser.add_argument(
        "--shape-scan",
        choices=["single", "primus-mi355x-bf16"],
        default="single",
    )
    parser.add_argument("--shape-start", type=int, default=0)
    parser.add_argument(
        "--shape-count",
        type=int,
        default=0,
        help="Number of unique shapes to run; zero runs the rest of the scan.",
    )
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


def make_symmetric_buffers(
    collective: str,
    group: dist.ProcessGroup,
    device: torch.device,
    total_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor, object, list[object]]:
    """Allocate the send/recv pair for `collective` inside the symmetric pool.

    Both buffers are rendezvous'd so RCCL sees NCCL_WIN_COLL_SYMMETRIC on each
    window; the CE (SDMA) reduce-scatter path refuses to dispatch otherwise.
    """
    world_size = group.size()
    if total_bytes % world_size:
        raise ValueError("message bytes must be divisible by world size")

    symm_mem.enable_symm_mem_for_group(group.group_name)
    dist.barrier(group=group)
    pool = symm_mem.get_mem_pool(device)

    if collective == "allgather":
        with torch.cuda.use_mem_pool(pool):
            recv = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        handles = [symm_mem.rendezvous(recv, group=group.group_name)]
        shard_bytes = total_bytes // world_size
        send = recv.narrow(0, group.rank() * shard_bytes, shard_bytes)
        send.fill_(group.rank() % 251)
    else:
        # CE reduce-scatter rejects Float8 and needs a real reduction dtype.
        elements = total_bytes // torch.bfloat16.itemsize
        if elements % world_size:
            raise ValueError("message elements must be divisible by world size")
        with torch.cuda.use_mem_pool(pool):
            send = torch.empty(elements, dtype=torch.bfloat16, device=device)
            recv = torch.empty(
                elements // world_size, dtype=torch.bfloat16, device=device
            )
        handles = [
            symm_mem.rendezvous(send, group=group.group_name),
            symm_mem.rendezvous(recv, group=group.group_name),
        ]
        send.fill_(1.0 + group.rank())

    dist.barrier(group=group)
    return send, recv, pool, handles


def run_collective(
    collective: str,
    send: torch.Tensor,
    recv: torch.Tensor,
    group: dist.ProcessGroup,
) -> None:
    if collective == "allgather":
        work = dist.all_gather_into_tensor(recv, send, group=group, async_op=True)
    else:
        work = dist.reduce_scatter_tensor(recv, send, group=group, async_op=True)
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
    collective_fn,
    pre_gemms: int,
    post_gemms: int,
    collective_repeats: int,
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
        for _ in range(collective_repeats):
            collective_fn()
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
        raise RuntimeError("No GEMM iteration overlapped the collective window")
    return (
        pre_latencies,
        overlapped_latencies,
        post_latencies,
        comm_start.elapsed_time(comm_end),
    )


def select_shapes(
    args: argparse.Namespace,
) -> tuple[list[tuple[int, GemmShape]], int]:
    if args.shape_scan == "single":
        shapes = [GemmShape(args.gemm_m, args.gemm_n, args.gemm_k, [])]
    else:
        shapes = get_primus_mi355x_bf16_shapes()

    if args.shape_start < 0 or args.shape_start >= len(shapes):
        raise ValueError(
            f"shape-start must be in [0, {len(shapes) - 1}], "
            f"got {args.shape_start}"
        )
    if args.shape_count < 0:
        raise ValueError("shape-count cannot be negative")

    end = len(shapes)
    if args.shape_count:
        end = min(end, args.shape_start + args.shape_count)
    return list(enumerate(shapes))[args.shape_start:end], len(shapes)


def benchmark_shape(
    *,
    shape_id: int,
    shape: GemmShape,
    args: argparse.Namespace,
    rank: int,
    device: torch.device,
    control_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    policy_resources: list[tuple[int, dist.ProcessGroup, torch.Tensor, torch.Tensor]],
) -> list[ImpactResult]:
    torch.manual_seed(1234 + rank + shape_id * 17)
    a = torch.randn((shape.m, shape.k), dtype=torch.bfloat16, device=device)
    b = torch.randn((shape.k, shape.n), dtype=torch.bfloat16, device=device)
    out = torch.empty((shape.m, shape.n), dtype=torch.bfloat16, device=device)
    gemm_fn = lambda: launch_gemm(a, b, out, args.gemm_repeats)

    with torch.cuda.stream(compute_stream):
        for _ in range(args.warmup):
            gemm_fn()
    torch.cuda.synchronize()
    dist.barrier()

    shape_results: list[ImpactResult] = []
    for policy, group, send, recv in policy_resources:
        collective_fn = lambda: run_collective(args.collective, send, recv, group)
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
                collective_fn,
                args.pre_gemms,
                args.post_gemms,
                args.collective_repeats,
            )
            pre_samples.extend(pre)
            overlap_samples.extend(overlap)
            post_samples.extend(post)
            comm_samples.append(comm_ms)

        baseline_samples = pre_samples + post_samples
        if not baseline_samples:
            raise RuntimeError("No non-overlapped GEMM samples for baseline")
        baseline_local = float(statistics.median(baseline_samples))
        overlap_median_local = float(statistics.median(overlap_samples))
        overlap_p95_local = percentile(overlap_samples, 95.0)
        overlap_max_local = max(overlap_samples)
        post_local = (
            float(statistics.median(post_samples))
            if post_samples
            else baseline_local
        )
        comm_local = float(statistics.median(comm_samples))

        baseline_ms = max_across_ranks(baseline_local, device)
        overlap_median_ms = max_across_ranks(overlap_median_local, device)
        overlap_p95_ms = max_across_ranks(overlap_p95_local, device)
        overlap_max_ms = max_across_ranks(overlap_max_local, device)
        post_ms = max_across_ranks(post_local, device)
        comm_ms = max_across_ranks(comm_local, device)
        overlap_count = min_count_across_ranks(len(overlap_samples), device)

        result = ImpactResult(
            shape_id=shape_id,
            m=shape.m,
            n=shape.n,
            k=shape.k,
            aliases=[asdict(alias) for alias in shape.aliases],
            policy=policy,
            collective=args.collective,
            baseline_gemm_ms=baseline_ms,
            overlapped_gemm_median_ms=overlap_median_ms,
            overlapped_gemm_p95_ms=overlap_p95_ms,
            overlapped_gemm_max_ms=overlap_max_ms,
            post_gemm_ms=post_ms,
            collective_window_ms=comm_ms,
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
        shape_results.append(result)
        if rank == 0:
            print("RESULT " + json.dumps(asdict(result), sort_keys=True), flush=True)

        torch.cuda.synchronize()
        dist.barrier()

    del a, b, out, gemm_fn
    gc.collect()
    torch.cuda.empty_cache()
    return shape_results


def main() -> None:
    args = parse_args()
    selected_shapes, total_shapes = select_shapes(args)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    symm_mem.set_backend("NCCL")
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()

    control_stream = torch.cuda.default_stream(device)
    compute_stream = torch.cuda.Stream(device=device, priority=0)
    comm_stream = torch.cuda.Stream(device=device, priority=-1)

    live_resources: list[tuple[object, ...]] = []
    policy_resources: list[
        tuple[int, dist.ProcessGroup, torch.Tensor, torch.Tensor]
    ] = []
    for policy in args.policies:
        group = make_policy_group(policy, world_size)
        send, recv, pool, handles = make_symmetric_buffers(
            args.collective,
            group,
            device,
            args.message_bytes,
        )
        live_resources.append((group, send, recv, pool, handles))
        policy_resources.append((policy, group, send, recv))
        for _ in range(args.warmup):
            with torch.cuda.stream(comm_stream):
                run_collective(args.collective, send, recv, group)
        torch.cuda.synchronize()
        dist.barrier()

    results: list[ImpactResult] = []
    for shape_id, shape in selected_shapes:
        if rank == 0:
            progress = {
                "shape_id": shape_id,
                "shapes_total": total_shapes,
                **shape.to_dict(),
            }
            print(
                "SHAPE_START " + json.dumps(progress, sort_keys=True),
                flush=True,
            )
        results.extend(
            benchmark_shape(
                shape_id=shape_id,
                shape=shape,
                args=args,
                rank=rank,
                device=device,
                control_stream=control_stream,
                compute_stream=compute_stream,
                comm_stream=comm_stream,
                policy_resources=policy_resources,
            )
        )

    if rank == 0:
        payload = {
            "world_size": world_size,
            "message_bytes": args.message_bytes,
            "collective": args.collective,
            "rccl_env": {
                name: value
                for name, value in sorted(os.environ.items())
                if name.startswith(("RCCL_", "NCCL_"))
            },
            "shape_scan": args.shape_scan,
            "shape_source": (
                {
                    "pull_request": PRIMUS_TURBO_PR,
                    "commit": PRIMUS_TURBO_COMMIT,
                }
                if args.shape_scan == "primus-mi355x-bf16"
                else None
            ),
            "shapes_total": total_shapes,
            "shape_start": args.shape_start,
            "shapes_run": len(selected_shapes),
            "gemm": {
                "repeats_per_iteration": args.gemm_repeats,
                "dtype": "bfloat16",
            },
            "trials": args.iterations,
            "pre_gemms": args.pre_gemms,
            "post_gemms": args.post_gemms,
            "collective_repeats": args.collective_repeats,
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
