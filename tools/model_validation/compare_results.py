#!/usr/bin/env python3
"""Compare inference results between PyTorch and ExecuTorch/Android.

This script compares saved inference results from different runtimes
to validate model accuracy and consistency.

Usage:
    # Compare PyTorch vs ExecuTorch results
    uv run python tools/model_validation/compare_results.py \
        --pytorch-results pytorch_outputs.json \
        --executorch-results executorch_outputs.json \
        --tolerance 0.01

    # Compare with Android results
    uv run python tools/model_validation/compare_results.py \
        --pytorch-results pytorch_outputs.json \
        --android-results android_benchmark_results.json \
        --tolerance 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two sets of inference outputs."""

    source_a: str
    source_b: str
    total_samples: int
    matched_samples: int
    within_tolerance: int
    pass_rate: float
    max_diff: float
    avg_diff: float
    std_diff: float
    tolerance: float
    per_sample_diffs: list[dict[str, Any]]


def load_pytorch_results(path: Path) -> dict[str, list[float]]:
    """Load PyTorch inference results from JSON.

    Expected format:
    {
        "sample_0001": [0.1, 0.2, ...],
        "sample_0002": [0.3, 0.4, ...],
        ...
    }
    """
    with open(path) as f:
        return json.load(f)


def load_android_results(path: Path) -> dict[str, list[float]]:
    """Load Android benchmark results and extract outputs.

    Expected format (from ResultsReporter):
    {
        "results": [
            {"sampleId": "sample_0001", "outputs": [0.1, 0.2, ...], "latencyMs": 10},
            ...
        ]
    }
    """
    with open(path) as f:
        data = json.load(f)

    results = {}
    for item in data.get("results", []):
        sample_id = item["sampleId"]
        outputs = item["outputs"]
        results[sample_id] = outputs

    return results


def compare_outputs(
    results_a: dict[str, list[float]],
    results_b: dict[str, list[float]],
    source_a: str,
    source_b: str,
    tolerance: float,
) -> ComparisonResult:
    """Compare two sets of inference outputs.

    Args:
        results_a: First set of results (reference).
        results_b: Second set of results (comparison).
        source_a: Name of first source.
        source_b: Name of second source.
        tolerance: Maximum allowed difference.

    Returns:
        ComparisonResult with detailed comparison metrics.
    """
    matched_samples = 0
    within_tolerance = 0
    diffs = []
    per_sample = []

    for sample_id, output_a in results_a.items():
        if sample_id not in results_b:
            continue

        output_b = results_b[sample_id]
        matched_samples += 1

        # Calculate differences
        arr_a = np.array(output_a)
        arr_b = np.array(output_b)

        if len(arr_a) != len(arr_b):
            logger.warning(
                f"Sample {sample_id}: output length mismatch "
                f"({len(arr_a)} vs {len(arr_b)})"
            )
            continue

        abs_diff = np.abs(arr_a - arr_b)
        max_diff = float(abs_diff.max())
        mean_diff = float(abs_diff.mean())
        mse = float(np.mean((arr_a - arr_b) ** 2))

        diffs.append(max_diff)

        is_within = max_diff <= tolerance
        if is_within:
            within_tolerance += 1

        per_sample.append({
            "sample_id": sample_id,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "mse": mse,
            "within_tolerance": is_within,
        })

    if not diffs:
        return ComparisonResult(
            source_a=source_a,
            source_b=source_b,
            total_samples=len(results_a),
            matched_samples=0,
            within_tolerance=0,
            pass_rate=0.0,
            max_diff=0.0,
            avg_diff=0.0,
            std_diff=0.0,
            tolerance=tolerance,
            per_sample_diffs=[],
        )

    return ComparisonResult(
        source_a=source_a,
        source_b=source_b,
        total_samples=len(results_a),
        matched_samples=matched_samples,
        within_tolerance=within_tolerance,
        pass_rate=within_tolerance / matched_samples if matched_samples > 0 else 0.0,
        max_diff=max(diffs),
        avg_diff=float(np.mean(diffs)),
        std_diff=float(np.std(diffs)),
        tolerance=tolerance,
        per_sample_diffs=sorted(per_sample, key=lambda x: -x["max_diff"]),
    )


def print_comparison(result: ComparisonResult) -> None:
    """Print comparison results in a formatted table."""
    print("\n" + "=" * 60)
    print(f"COMPARISON: {result.source_a} vs {result.source_b}")
    print("=" * 60)
    print(f"Total samples in {result.source_a}:  {result.total_samples}")
    print(f"Matched samples:                     {result.matched_samples}")
    print(f"Within tolerance ({result.tolerance}):         {result.within_tolerance}")
    print(f"Pass rate:                           {result.pass_rate:.2%}")
    print("-" * 60)
    print(f"Max difference:                      {result.max_diff:.6f}")
    print(f"Avg difference:                      {result.avg_diff:.6f}")
    print(f"Std difference:                      {result.std_diff:.6f}")
    print("-" * 60)

    # Show worst samples
    print("\nWorst samples (highest difference):")
    for i, sample in enumerate(result.per_sample_diffs[:5]):
        status = "✓" if sample["within_tolerance"] else "✗"
        print(
            f"  {status} {sample['sample_id']}: "
            f"max={sample['max_diff']:.6f}, "
            f"mean={sample['mean_diff']:.6f}"
        )

    print("\n" + "=" * 60)
    passed = result.pass_rate >= 0.99
    print(f"RESULT: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare inference results between runtimes"
    )
    parser.add_argument(
        "--pytorch-results",
        type=Path,
        help="Path to PyTorch results JSON",
    )
    parser.add_argument(
        "--executorch-results",
        type=Path,
        help="Path to ExecuTorch results JSON",
    )
    parser.add_argument(
        "--android-results",
        type=Path,
        help="Path to Android benchmark results JSON",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed difference (default: 0.01)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for comparison results JSON",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all sample comparisons, not just worst",
    )
    args = parser.parse_args()

    # Need at least two result sets to compare
    if not args.pytorch_results:
        parser.error("--pytorch-results is required")

    if not args.executorch_results and not args.android_results:
        parser.error("Either --executorch-results or --android-results is required")

    # Load PyTorch results
    logger.info(f"Loading PyTorch results from {args.pytorch_results}")
    pytorch_results = load_pytorch_results(args.pytorch_results)
    logger.info(f"Loaded {len(pytorch_results)} samples")

    comparisons = []

    # Compare with ExecuTorch
    if args.executorch_results:
        logger.info(f"Loading ExecuTorch results from {args.executorch_results}")
        executorch_results = load_pytorch_results(args.executorch_results)
        logger.info(f"Loaded {len(executorch_results)} samples")

        comparison = compare_outputs(
            pytorch_results,
            executorch_results,
            "PyTorch",
            "ExecuTorch",
            args.tolerance,
        )
        comparisons.append(comparison)
        print_comparison(comparison)

    # Compare with Android
    if args.android_results:
        logger.info(f"Loading Android results from {args.android_results}")
        android_results = load_android_results(args.android_results)
        logger.info(f"Loaded {len(android_results)} samples")

        comparison = compare_outputs(
            pytorch_results,
            android_results,
            "PyTorch",
            "Android",
            args.tolerance,
        )
        comparisons.append(comparison)
        print_comparison(comparison)

    # Save results
    if args.output:
        output_data = {
            "comparisons": [
                {
                    "source_a": c.source_a,
                    "source_b": c.source_b,
                    "total_samples": c.total_samples,
                    "matched_samples": c.matched_samples,
                    "within_tolerance": c.within_tolerance,
                    "pass_rate": c.pass_rate,
                    "max_diff": c.max_diff,
                    "avg_diff": c.avg_diff,
                    "std_diff": c.std_diff,
                    "tolerance": c.tolerance,
                    "per_sample_diffs": c.per_sample_diffs if args.show_all else c.per_sample_diffs[:20],
                }
                for c in comparisons
            ]
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Saved comparison results to {args.output}")

    # Return exit code based on all comparisons passing
    all_passed = all(c.pass_rate >= 0.99 for c in comparisons)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
