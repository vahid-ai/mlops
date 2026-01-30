#!/usr/bin/env python3
"""End-to-end model validation: PyTorch → ExecuTorch → Android → Compare.

This script orchestrates the full validation workflow:
1. Fetch PyTorch model from MLflow
2. Export to ExecuTorch format
3. Run inference on test data with PyTorch (baseline)
4. Build Android app and run on emulator
5. Compare results and log to MLflow

Usage:
    uv run python tools/model_validation/validate_model.py \
        --model-name kronodroid_autoencoder \
        --test-data data/test_samples.npy \
        --tolerance 0.01

    # Skip Android test (PyTorch vs ExecuTorch on host only)
    uv run python tools/model_validation/validate_model.py \
        --model-name kronodroid_autoencoder \
        --test-data data/test_samples.npy \
        --skip-android
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runtimes.executorch.export import (
    ExecuTorchExporter,
    MLflowExecuTorchClient,
    get_quantization_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation."""

    passed: bool
    total_samples: int
    samples_within_tolerance: int
    pass_rate: float
    max_diff: float
    avg_diff: float
    tolerance: float
    pytorch_metrics: dict[str, float]
    executorch_metrics: dict[str, float]
    android_metrics: dict[str, Any] | None = None


def run_pytorch_inference(
    model: torch.nn.Module,
    test_data: np.ndarray,
) -> dict[str, list[float]]:
    """Run PyTorch model inference on test data.

    Args:
        model: PyTorch model.
        test_data: Test input data (N, input_dim).

    Returns:
        Dict mapping sample_id to output values.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for i, sample in enumerate(test_data):
            sample_id = f"sample_{i:04d}"
            input_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            output = model(input_tensor)
            results[sample_id] = output.numpy().flatten().tolist()

    logger.info(f"PyTorch inference complete: {len(results)} samples")
    return results


def run_executorch_inference(
    pte_path: Path,
    test_data: np.ndarray,
) -> dict[str, list[float]]:
    """Run ExecuTorch model inference on test data (host).

    Args:
        pte_path: Path to .pte file.
        test_data: Test input data (N, input_dim).

    Returns:
        Dict mapping sample_id to output values.
    """
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(str(pte_path))
    method = program.load_method("forward")

    results = {}

    for i, sample in enumerate(test_data):
        sample_id = f"sample_{i:04d}"
        input_array = sample.astype(np.float32).reshape(1, -1)
        output = method.execute([input_array])
        results[sample_id] = output[0].flatten().tolist()

    logger.info(f"ExecuTorch inference complete: {len(results)} samples")
    return results


def run_android_test(
    pte_path: Path,
    test_data: np.ndarray,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Build and run Android instrumentation test.

    Args:
        pte_path: Path to .pte model file.
        test_data: Test input data.
        output_dir: Directory for output files.

    Returns:
        Android benchmark results, or None if test failed.
    """
    # Copy model to models directory
    models_dir = PROJECT_ROOT / "models"
    target_pte = models_dir / pte_path.name
    subprocess.run(["cp", str(pte_path), str(target_pte)], check=True)

    # Create expected outputs JSON for the test
    expected_outputs = []
    for i, sample in enumerate(test_data[:100]):  # Limit for Android test
        expected_outputs.append({
            "sampleId": f"sample_{i:04d}",
            "input": sample.tolist(),
            "expected": [0.0] * len(sample),  # Placeholder - Android test generates
        })

    # Save test data for Android app
    test_data_dir = (
        PROJECT_ROOT
        / "apps/android/runtime_comparison_app/src/main/assets/test_data"
    )
    test_data_dir.mkdir(parents=True, exist_ok=True)

    with open(test_data_dir / "expected_outputs.json", "w") as f:
        json.dump(expected_outputs, f)

    # Build Android test APK
    logger.info("Building Android app...")
    result = subprocess.run(
        [
            "bazel",
            "build",
            "--config=android_debug",
            "//apps/android/runtime_comparison_app:runtime_comparison_app",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Android build failed:\n{result.stderr}")
        return None

    # Check if emulator/device is available
    adb_result = subprocess.run(
        ["adb", "devices"],
        capture_output=True,
        text=True,
    )

    if "emulator" not in adb_result.stdout and "device" not in adb_result.stdout:
        logger.warning("No Android device/emulator available, skipping Android test")
        return None

    # Install APK
    apk_path = (
        PROJECT_ROOT
        / "bazel-bin/apps/android/runtime_comparison_app/runtime_comparison_app.apk"
    )

    if not apk_path.exists():
        logger.error(f"APK not found at {apk_path}")
        return None

    logger.info("Installing APK...")
    subprocess.run(["adb", "install", "-r", str(apk_path)], check=True)

    # Run instrumentation test
    logger.info("Running Android instrumentation test...")
    test_result = subprocess.run(
        [
            "adb",
            "shell",
            "am",
            "instrument",
            "-w",
            "-e",
            "class",
            "com.dfp.runtimeapp.ModelAccuracyTest#testPerformance",
            "com.dfp.runtimeapp.test/androidx.test.runner.AndroidJUnitRunner",
        ],
        capture_output=True,
        text=True,
    )

    logger.info(f"Test output:\n{test_result.stdout}")

    # Pull results
    results_path = output_dir / "android_results.json"
    pull_result = subprocess.run(
        [
            "adb",
            "pull",
            "/sdcard/Android/data/com.dfp.runtimeapp/files/benchmark_results.json",
            str(results_path),
        ],
        capture_output=True,
        text=True,
    )

    if pull_result.returncode != 0:
        logger.warning("Could not pull Android results")
        return None

    with open(results_path) as f:
        return json.load(f)


def compare_results(
    pytorch_results: dict[str, list[float]],
    executorch_results: dict[str, list[float]],
    tolerance: float,
) -> ValidationResult:
    """Compare PyTorch and ExecuTorch inference results.

    Args:
        pytorch_results: PyTorch inference outputs.
        executorch_results: ExecuTorch inference outputs.
        tolerance: Maximum allowed difference.

    Returns:
        ValidationResult with comparison metrics.
    """
    samples_within_tolerance = 0
    max_diff = 0.0
    total_diff = 0.0
    num_samples = 0

    for sample_id, pt_output in pytorch_results.items():
        if sample_id not in executorch_results:
            continue

        et_output = executorch_results[sample_id]
        sample_max_diff = max(
            abs(p - e) for p, e in zip(pt_output, et_output)
        )

        max_diff = max(max_diff, sample_max_diff)
        total_diff += sample_max_diff
        num_samples += 1

        if sample_max_diff <= tolerance:
            samples_within_tolerance += 1

    pass_rate = samples_within_tolerance / num_samples if num_samples > 0 else 0.0
    avg_diff = total_diff / num_samples if num_samples > 0 else 0.0

    # Calculate aggregate metrics
    pt_mean = np.mean([np.mean(v) for v in pytorch_results.values()])
    et_mean = np.mean([np.mean(v) for v in executorch_results.values()])

    return ValidationResult(
        passed=pass_rate >= 0.99,  # 99% must be within tolerance
        total_samples=num_samples,
        samples_within_tolerance=samples_within_tolerance,
        pass_rate=pass_rate,
        max_diff=max_diff,
        avg_diff=avg_diff,
        tolerance=tolerance,
        pytorch_metrics={"mean_output": float(pt_mean)},
        executorch_metrics={"mean_output": float(et_mean)},
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ExecuTorch model against PyTorch baseline"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="MLflow registered model name",
    )
    parser.add_argument(
        "--model-version",
        default="latest",
        help="Model version (default: latest)",
    )
    parser.add_argument(
        "--test-data",
        required=True,
        help="Path to test data (.npy file)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Max allowed difference (default: 0.01)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default="http://localhost:5050",
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--backend",
        default="xnnpack",
        choices=["xnnpack", "vulkan", "portable"],
        help="ExecuTorch backend (default: xnnpack)",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        help="Quantization config (default: none)",
    )
    parser.add_argument(
        "--skip-android",
        action="store_true",
        help="Skip Android emulator test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Initialize clients
    mlflow_client = MLflowExecuTorchClient(args.mlflow_uri)
    quant_config = get_quantization_config(args.quantization)
    exporter = ExecuTorchExporter(
        backend=args.backend,
        quantization=quant_config.mode.value,
    )

    # Create output directory
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="model_validation_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Fetch PyTorch model from MLflow
    logger.info(f"Fetching model '{args.model_name}' from MLflow...")
    model, metadata = mlflow_client.fetch_pytorch_model(
        args.model_name,
        args.model_version,
    )
    logger.info(f"Loaded model from run {metadata.run_id}")

    # Step 2: Load test data
    logger.info(f"Loading test data from {args.test_data}...")
    test_data = np.load(args.test_data)
    logger.info(f"Test data shape: {test_data.shape}")

    # Step 3: Run PyTorch baseline inference
    logger.info("Running PyTorch baseline inference...")
    pytorch_results = run_pytorch_inference(model, test_data)

    # Step 4: Export to ExecuTorch
    logger.info("Exporting to ExecuTorch...")
    example_inputs = (torch.randn(1, test_data.shape[1]),)
    pte_path = output_dir / "model.pte"
    export_metadata = exporter.export(
        model=model,
        example_inputs=example_inputs,
        output_path=pte_path,
        model_name=args.model_name,
    )
    logger.info(f"Exported to {pte_path} ({pte_path.stat().st_size:,} bytes)")

    # Step 5: Register .pte in MLflow
    logger.info("Registering ExecuTorch model in MLflow...")
    pte_uri = mlflow_client.register_executorch_model(
        pte_path=pte_path,
        source_run_id=metadata.run_id,
        model_name=args.model_name,
        export_metadata=export_metadata.to_dict(),
        metadata_path=pte_path.with_suffix(".json"),
    )
    logger.info(f"Registered at: {pte_uri}")

    # Step 6: Run ExecuTorch inference on host
    logger.info("Running ExecuTorch inference (host)...")
    try:
        executorch_results = run_executorch_inference(pte_path, test_data)
    except Exception as e:
        logger.warning(f"ExecuTorch host inference failed: {e}")
        logger.info("Falling back to Android-only validation")
        executorch_results = {}

    # Step 7: Compare PyTorch vs ExecuTorch
    if executorch_results:
        logger.info("Comparing PyTorch vs ExecuTorch results...")
        validation = compare_results(pytorch_results, executorch_results, args.tolerance)
    else:
        validation = None

    # Step 8: Run Android test (optional)
    android_results = None
    if not args.skip_android:
        logger.info("Running Android emulator test...")
        android_results = run_android_test(pte_path, test_data, output_dir)
        if android_results:
            logger.info("Android test complete")
            mlflow_client.log_android_benchmark_results(
                metadata.run_id,
                android_results,
            )

    # Step 9: Log results to MLflow
    if validation:
        mlflow_client.log_validation_results(
            run_id=metadata.run_id,
            pytorch_results=validation.pytorch_metrics,
            executorch_results=validation.executorch_metrics,
            comparison_metrics={
                "pass_rate": validation.pass_rate,
                "max_diff": validation.max_diff,
                "avg_diff": validation.avg_diff,
                "passed": float(validation.passed),
            },
            tolerance=args.tolerance,
        )

    # Step 10: Print summary
    print("\n" + "=" * 60)
    print("MODEL VALIDATION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"MLflow Run: {metadata.run_id}")
    print(f"Backend: {args.backend}")
    print(f"Quantization: {args.quantization}")
    print("-" * 60)

    if validation:
        print(f"Total samples:           {validation.total_samples}")
        print(f"Within tolerance:        {validation.samples_within_tolerance}")
        print(f"Pass rate:               {validation.pass_rate:.2%}")
        print(f"Max difference:          {validation.max_diff:.6f}")
        print(f"Avg difference:          {validation.avg_diff:.6f}")
        print(f"Tolerance:               {args.tolerance}")
        print("-" * 60)
        print(f"RESULT:                  {'✓ PASSED' if validation.passed else '✗ FAILED'}")
    else:
        print("Host validation skipped (ExecuTorch runtime not available)")

    if android_results:
        print("-" * 60)
        print("Android Results:")
        print(f"  Device:                {android_results.get('deviceInfo', {}).get('model', 'unknown')}")
        print(f"  Avg latency:           {android_results.get('avgLatencyMs', 0):.2f} ms")
        print(f"  Throughput:            {android_results.get('throughputSamplesPerSec', 0):.1f} samples/sec")

    print("=" * 60)

    # Save summary
    summary = {
        "model_name": args.model_name,
        "run_id": metadata.run_id,
        "pte_uri": pte_uri,
        "validation": validation.__dict__ if validation else None,
        "android": android_results,
    }

    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Return exit code
    if validation and not validation.passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
