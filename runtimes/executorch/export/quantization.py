"""Quantization configurations for ExecuTorch export.

Provides pre-configured quantization settings for different use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class QuantizationMode(str, Enum):
    """Supported quantization modes."""

    NONE = "none"
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"
    INT8 = "int8"
    INT4 = "int4"


class QuantizationBackend(str, Enum):
    """Quantization backends."""

    PYTORCH = "pytorch"
    XNNPACK = "xnnpack"
    QNNPACK = "qnnpack"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        mode: Quantization mode (none, dynamic, static, qat).
        dtype: Data type for quantized weights (int8, int4).
        backend: Quantization backend to use.
        calibration_samples: Number of samples for static quantization calibration.
        per_channel: Whether to use per-channel quantization.
        symmetric: Whether to use symmetric quantization.
    """

    mode: QuantizationMode = QuantizationMode.NONE
    dtype: str = "int8"
    backend: QuantizationBackend = QuantizationBackend.XNNPACK
    calibration_samples: int = 100
    per_channel: bool = True
    symmetric: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "dtype": self.dtype,
            "backend": self.backend.value,
            "calibration_samples": self.calibration_samples,
            "per_channel": self.per_channel,
            "symmetric": self.symmetric,
        }


# Pre-configured quantization configs for common use cases
QUANTIZATION_CONFIGS: dict[str, QuantizationConfig] = {
    # No quantization - full precision
    "none": QuantizationConfig(mode=QuantizationMode.NONE),
    # Dynamic quantization - weights quantized, activations in fp32
    # Good balance of accuracy and speed
    "dynamic_int8": QuantizationConfig(
        mode=QuantizationMode.DYNAMIC,
        dtype="int8",
        backend=QuantizationBackend.XNNPACK,
    ),
    # Static quantization - both weights and activations quantized
    # Best performance, requires calibration data
    "static_int8": QuantizationConfig(
        mode=QuantizationMode.STATIC,
        dtype="int8",
        backend=QuantizationBackend.XNNPACK,
        calibration_samples=200,
    ),
    # Quantization-aware training
    # Highest accuracy for quantized models, requires QAT during training
    "qat_int8": QuantizationConfig(
        mode=QuantizationMode.QAT,
        dtype="int8",
        backend=QuantizationBackend.XNNPACK,
    ),
    # 4-bit quantization - maximum compression
    # Significant accuracy loss, use for very large models
    "dynamic_int4": QuantizationConfig(
        mode=QuantizationMode.DYNAMIC,
        dtype="int4",
        backend=QuantizationBackend.PYTORCH,
    ),
    # Mobile-optimized config
    "mobile": QuantizationConfig(
        mode=QuantizationMode.DYNAMIC,
        dtype="int8",
        backend=QuantizationBackend.XNNPACK,
        per_channel=True,
        symmetric=False,
    ),
}


def get_quantization_config(name: str) -> QuantizationConfig:
    """Get a pre-configured quantization config by name.

    Args:
        name: Config name (none, dynamic_int8, static_int8, qat_int8, mobile).

    Returns:
        QuantizationConfig instance.

    Raises:
        ValueError: If config name is not found.
    """
    if name not in QUANTIZATION_CONFIGS:
        available = ", ".join(QUANTIZATION_CONFIGS.keys())
        raise ValueError(
            f"Unknown quantization config: {name}. Available: {available}"
        )
    return QUANTIZATION_CONFIGS[name]


def list_quantization_configs() -> list[str]:
    """List available quantization config names."""
    return list(QUANTIZATION_CONFIGS.keys())
