#!/usr/bin/env python3
"""Aggregate GEMM overlap RESULT records into JSON and CSV reports."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_results(logs: list[Path]) -> dict[tuple[int, int], dict[str, object]]:
    results: dict[tuple[int, int], dict[str, object]] = {}
    for log in logs:
        with log.open() as stream:
            for line in stream:
                if not line.startswith("RESULT "):
                    continue
                result = json.loads(line.removeprefix("RESULT "))
                key = (int(result["shape_id"]), int(result["policy"]))
                results[key] = result
    return results


def reduction(before: float, after: float) -> float | None:
    if before <= 0.0:
        return None
    return 100.0 * (before - after) / before


def build_rows(
    results: dict[tuple[int, int], dict[str, object]],
) -> list[dict[str, object]]:
    shape_ids = sorted({shape_id for shape_id, _ in results})
    rows: list[dict[str, object]] = []
    for shape_id in shape_ids:
        policy_0 = results.get((shape_id, 0))
        policy_2 = results.get((shape_id, 2))
        if policy_0 is None or policy_2 is None:
            continue

        aliases = policy_0["aliases"]
        models = sorted({str(alias["model"]) for alias in aliases})
        projections = sorted({str(alias["projection"]) for alias in aliases})
        batch_sizes = sorted({int(alias["batch_size"]) for alias in aliases})
        p0_median = float(policy_0["median_latency_increase_pct"])
        p2_median = float(policy_2["median_latency_increase_pct"])
        p0_p95 = float(policy_0["p95_latency_increase_pct"])
        p2_p95 = float(policy_2["p95_latency_increase_pct"])
        p0_overlap = float(policy_0["overlapped_gemm_median_ms"])
        p2_overlap = float(policy_2["overlapped_gemm_median_ms"])

        rows.append(
            {
                "shape_id": shape_id,
                "m": int(policy_0["m"]),
                "n": int(policy_0["n"]),
                "k": int(policy_0["k"]),
                "models": models,
                "projections": projections,
                "batch_sizes": batch_sizes,
                "aliases": aliases,
                "policy_0_baseline_ms": float(policy_0["baseline_gemm_ms"]),
                "policy_2_baseline_ms": float(policy_2["baseline_gemm_ms"]),
                "policy_0_overlap_median_ms": p0_overlap,
                "policy_2_overlap_median_ms": p2_overlap,
                "policy_0_median_slowdown_pct": p0_median,
                "policy_2_median_slowdown_pct": p2_median,
                "median_interference_reduction_pct": reduction(
                    p0_median, p2_median
                ),
                "overlap_latency_reduction_pct": reduction(
                    p0_overlap, p2_overlap
                ),
                "policy_0_p95_slowdown_pct": p0_p95,
                "policy_2_p95_slowdown_pct": p2_p95,
                "p95_interference_reduction_pct": reduction(p0_p95, p2_p95),
                "policy_0_collective_ms": float(
                    policy_0["collective_window_ms"]
                ),
                "policy_2_collective_ms": float(
                    policy_2["collective_window_ms"]
                ),
                "policy_0_overlapped_samples": int(
                    policy_0["overlapped_samples"]
                ),
                "policy_2_overlapped_samples": int(
                    policy_2["overlapped_samples"]
                ),
            }
        )
    return rows


def finite_values(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row[key] is not None]


def summarize(
    rows: list[dict[str, object]],
    raw_results: dict[tuple[int, int], dict[str, object]],
) -> dict[str, object]:
    median_reductions = finite_values(
        rows, "median_interference_reduction_pct"
    )
    p95_reductions = finite_values(rows, "p95_interference_reduction_pct")
    return {
        "shapes_with_any_result": len(
            {shape_id for shape_id, _ in raw_results}
        ),
        "shapes_compared": len(rows),
        "shapes_missing_policy_pair": len(
            {shape_id for shape_id, _ in raw_results}
        )
        - len(rows),
        "shapes_with_lower_median_slowdown": sum(
            float(row["policy_2_median_slowdown_pct"])
            < float(row["policy_0_median_slowdown_pct"])
            for row in rows
        ),
        "median_interference_reduction_pct": (
            statistics.median(median_reductions)
            if median_reductions
            else None
        ),
        "median_p95_interference_reduction_pct": (
            statistics.median(p95_reductions) if p95_reductions else None
        ),
        "primus_turbo_pull_request": (
            "https://github.com/AMD-AGI/Primus-Turbo/pull/265"
        ),
    }


def csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def main() -> None:
    args = parse_args()
    raw_results = load_results(args.logs)
    rows = build_rows(raw_results)
    summary = summarize(rows, raw_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "overlap_report.json"
    csv_path = args.output_dir / "overlap_report.csv"
    with json_path.open("w") as stream:
        json.dump({"summary": summary, "shapes": rows}, stream, indent=2)
        stream.write("\n")

    if rows:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {key: csv_value(value) for key, value in row.items()}
                for row in rows
            )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"JSON report: {json_path}")
    if rows:
        print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
