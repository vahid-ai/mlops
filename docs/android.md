# Android Development with Bazel

This document covers the Android build system, project structure, and development workflow for the DFP MLOps project.

## Table of Contents

- [Overview](#overview)
- [What is Bazel?](#what-is-bazel)
- [Why Bazel for Android?](#why-bazel-for-android)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Build Commands](#build-commands)
- [Testing](#testing)
- [ExecuTorch Model Integration](#executorch-model-integration)
- [Configuration Reference](#configuration-reference)

---

## Overview

This project uses **Bazel** as the build system for Android applications. The Android apps are designed to run machine learning models (trained via Kubeflow pipelines) on-device using **ExecuTorch** (PyTorch's mobile runtime successor).

### Key Components

| Component | Purpose |
|-----------|---------|
| `apps/android/main_app` | Production Android application |
| `apps/android/runtime_comparison_app` | Benchmarking app for comparing inference runtimes |
| `models/` | ExecuTorch models (.pte files) exported from ML pipelines |
| `runtimes/` | Runtime bindings for ExecuTorch, ONNX, TFLite |

---

## What is Bazel?

[Bazel](https://bazel.build/) is a fast, scalable, multi-language build system developed by Google. It's the open-source version of Google's internal build tool (Blaze) that builds all of Google's software.

### Key Concepts

#### WORKSPACE / MODULE.bazel

The root of a Bazel project is defined by either:
- **`WORKSPACE`** (legacy) - Defines external dependencies
- **`MODULE.bazel`** (modern/bzlmod) - Modern dependency management (what this project uses)

```python
# MODULE.bazel - declares the module and its dependencies
module(
    name = "dfp_mlops",
    version = "0.1.0",
)

bazel_dep(name = "rules_android", version = "0.6.0")
bazel_dep(name = "rules_kotlin", version = "2.0.0")
```

#### BUILD.bazel Files

Every directory that contains buildable code has a `BUILD.bazel` file that defines **targets**:

```python
# BUILD.bazel
android_binary(
    name = "main_app",           # Target name (bazel build //:main_app)
    manifest = "AndroidManifest.xml",
    deps = [":main_app_lib"],    # Dependencies
)

kt_android_library(
    name = "main_app_lib",
    srcs = glob(["src/**/*.kt"]),  # Source files
    deps = ["@maven//:androidx_core_core_ktx"],
)
```

#### Targets and Labels

Bazel uses **labels** to identify targets:

```
//apps/android/main_app:main_app
│ │                    │ │
│ │                    │ └─ Target name
│ │                    └─── Colon separator
│ └──────────────────────── Package path (from WORKSPACE root)
└────────────────────────── Repository (// = current repo)
```

Common label patterns:
- `//apps/android/main_app:main_app` - Specific target
- `//apps/android/main_app` - Default target (same as directory name)
- `//apps/android/...` - All targets in package and subpackages
- `//:main_app` - Target in root package
- `@maven//:junit_junit` - Target from external repository

#### Dependencies

Dependencies are declared explicitly in `deps`:

```python
deps = [
    ":local_library",                    # Same package
    "//apps/android:common",             # Another package
    "@maven//:androidx_core_core_ktx",   # External Maven artifact
]
```

---

## Why Bazel for Android?

### Advantages Over Gradle

| Feature | Bazel | Gradle |
|---------|-------|--------|
| **Build speed** | Extremely fast (aggressive caching) | Slower, less caching |
| **Reproducibility** | Hermetic builds (same inputs = same outputs) | Can vary by environment |
| **Incremental builds** | Fine-grained (file-level) | Coarse-grained (module-level) |
| **Multi-language** | Native support for many languages | Plugin-based |
| **Scalability** | Designed for massive codebases | Struggles at scale |
| **Remote caching** | Built-in | Requires Enterprise/plugins |
| **Remote execution** | Built-in | Limited |

### When to Use Bazel

Bazel excels when you have:
- **Monorepos** with multiple languages (Kotlin, Python, C++)
- **Shared code** between Android apps and backend
- **Native code** (NDK, ExecuTorch C++ bindings)
- **Large codebases** where build times matter
- **CI/CD pipelines** that benefit from caching

### Bazel + ExecuTorch

ExecuTorch models require:
1. Native C++ runtime libraries
2. JNI bindings for Android
3. Model files bundled as assets

Bazel handles this elegantly:

```python
android_binary(
    name = "app",
    assets = ["//models:exported_models"],  # Bundle .pte models
    assets_dir = "assets/models",
    deps = [
        "@maven//:org_pytorch_executorch",  # Java bindings
        "//runtimes/executorch:native",     # Native libs (if custom)
    ],
)
```

---

## Project Structure

```
mlops/
├── MODULE.bazel              # Bzlmod dependencies (Android, Kotlin, Maven)
├── BUILD.bazel               # Root BUILD with config_settings
├── .bazelversion             # Pins Bazel version (7.4.1)
├── .bazelrc                  # Build configurations
├── .bazelignore              # Directories Bazel should ignore
│
├── apps/
│   ├── BUILD.bazel           # App-level aliases
│   │
│   ├── android/
│   │   ├── BUILD.bazel       # Shared Android config, Kotlin toolchain
│   │   │
│   │   ├── main_app/
│   │   │   ├── BUILD.bazel   # android_binary + kt_android_library
│   │   │   └── src/
│   │   │       └── main/
│   │   │           ├── AndroidManifest.xml
│   │   │           ├── java/com/dfp/app/
│   │   │           │   └── MainActivity.kt
│   │   │           └── res/
│   │   │
│   │   └── runtime_comparison_app/
│   │       ├── BUILD.bazel
│   │       └── src/...
│   │
│   └── api/
│       └── BUILD.bazel       # Backend API (if applicable)
│
├── models/
│   ├── BUILD.bazel           # filegroup for .pte models
│   ├── *.pte                 # ExecuTorch model files
│   └── *.json                # Model metadata
│
└── runtimes/
    ├── executorch/
    │   └── BUILD.bazel       # ExecuTorch native bindings
    ├── onnx/
    │   └── BUILD.bazel       # ONNX Runtime bindings
    └── tflite/
        └── BUILD.bazel       # TFLite bindings
```

### Key Files Explained

#### `.bazelversion`

Specifies which Bazel version to use. Bazelisk (the Bazel version manager) reads this file:

```
7.4.1
```

#### `.bazelrc`

Build configurations and flags:

```bash
# Common settings
common --enable_bzlmod              # Use modern dependency management

# Android builds
build --android_sdk=@androidsdk//:sdk-34
build:android_release --fat_apk_cpu=arm64-v8a,armeabi-v7a,x86_64
build:android_debug --fat_apk_cpu=arm64-v8a  # Faster debug builds

# Test settings
test --test_output=errors
test:robolectric --define=robolectric=true
```

#### `.bazelignore`

Directories Bazel should not traverse (improves performance):

```
notebooks
engines
infra
.venv
```

#### `MODULE.bazel`

Modern dependency management (replaces WORKSPACE):

```python
module(name = "dfp_mlops", version = "0.1.0")

# Android
bazel_dep(name = "rules_android", version = "0.6.0")
bazel_dep(name = "rules_kotlin", version = "2.0.0")

# Maven dependencies
bazel_dep(name = "rules_jvm_external", version = "6.6")

maven = use_extension("@rules_jvm_external//:extensions.bzl", "maven")
maven.install(
    name = "maven",
    artifacts = [
        "androidx.core:core-ktx:1.13.1",
        "org.pytorch:executorch:0.4.0",
        # ...
    ],
)
use_repo(maven, "maven")
```

---

## Getting Started

### Prerequisites

1. **Install Bazel** via Bazelisk (cross-platform version manager):

   ```bash
   # Using the project's task runner
   task bazel:install

   # Or manually on macOS
   brew install bazelisk

   # Or on Linux
   curl -fsSL https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64 -o /usr/local/bin/bazel
   chmod +x /usr/local/bin/bazel
   ```

2. **Install Android SDK** (Bazel will download NDK automatically):

   ```bash
   # Set ANDROID_HOME
   export ANDROID_HOME=$HOME/Android/Sdk  # Linux
   export ANDROID_HOME=$HOME/Library/Android/sdk  # macOS
   ```

3. **Verify installation**:

   ```bash
   bazel --version
   # bazel 7.4.1
   ```

### First Build

```bash
# Build the main app
bazel build //apps/android/main_app:main_app

# The APK will be at:
# bazel-bin/apps/android/main_app/main_app.apk
```

---

## Build Commands

### Building Apps

```bash
# Build specific target
bazel build //apps/android/main_app:main_app

# Build using root alias
bazel build //:main_app

# Build all Android apps
bazel build //apps/android/...

# Build with debug config (faster, single ABI)
bazel build --config=android_debug //apps/android/main_app:main_app

# Build release (multi-ABI)
bazel build --config=android_release //apps/android/main_app:main_app
```

### Installing on Device

```bash
# Build and install
bazel mobile-install //apps/android/main_app:main_app

# With device serial (if multiple devices)
bazel mobile-install //apps/android/main_app:main_app --adb_arg=-s --adb_arg=DEVICE_SERIAL
```

### Querying the Build Graph

```bash
# List all targets in a package
bazel query //apps/android/main_app:all

# Show dependencies of a target
bazel query "deps(//apps/android/main_app:main_app)" --output=graph

# Find reverse dependencies (what depends on this)
bazel query "rdeps(//..., //models:exported_models)"

# List all Android apps
bazel query "kind(android_binary, //...)"
```

### Cleaning

```bash
# Clean build outputs
bazel clean

# Clean everything including caches
bazel clean --expunge
```

---

## Testing

### Unit Tests (Robolectric)

Robolectric runs Android tests on the JVM without an emulator:

```bash
# Run all tests
bazel test --config=robolectric //apps/android/main_app:main_app_test

# Run with verbose output
bazel test --config=robolectric --test_output=all //apps/android/main_app:main_app_test
```

### Instrumentation Tests

Run on real device or emulator:

```bash
# Run instrumentation tests
bazel test --config=android_test //apps/android/main_app:main_app_instrumentation_test
```

### Test Coverage

```bash
# Generate coverage report
bazel coverage --config=robolectric //apps/android/...
```

---

## ExecuTorch Model Integration

### Workflow Overview

```
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  Kubeflow Pipeline  │ ───▶ │   torch.export()    │ ───▶ │  ExecuTorch .pte    │
│  (Train PyTorch)    │      │  (Export model)     │      │  (Mobile format)    │
└─────────────────────┘      └─────────────────────┘      └─────────────────────┘
                                                                    │
                                                                    ▼
                                                          ┌─────────────────────┐
                                                          │   models/*.pte      │
                                                          │   (In repository)   │
                                                          └─────────────────────┘
                                                                    │
                                                                    ▼
                                                          ┌─────────────────────┐
                                                          │   Android App       │
                                                          │   (Bundle as asset) │
                                                          └─────────────────────┘
```

### Adding a New Model

1. **Export from PyTorch**:

   ```python
   import torch
   from executorch.exir import to_edge

   model = MyModel()
   example_inputs = (torch.randn(1, 289),)

   # Export
   exported = torch.export.export(model, example_inputs)
   edge = to_edge(exported)

   # Save
   edge.to_executorch().save("my_model.pte")
   ```

2. **Place in models directory**:

   ```bash
   cp my_model.pte /path/to/mlops/models/
   ```

3. **Reference in BUILD.bazel** (automatic via glob, or explicit):

   ```python
   # models/BUILD.bazel
   filegroup(
       name = "my_model",
       srcs = ["my_model.pte"],
       visibility = ["//visibility:public"],
   )
   ```

4. **Use in Android app**:

   ```python
   # apps/android/main_app/BUILD.bazel
   android_binary(
       name = "main_app",
       assets = ["//models:my_model"],
       assets_dir = "assets/models",
       # ...
   )
   ```

5. **Load in Kotlin**:

   ```kotlin
   val module = Module.load(
       assetFilePath(context, "models/my_model.pte")
   )
   val result = module.forward(inputTensor)
   ```

### Model Metadata

Include JSON metadata alongside models:

```json
// models/my_model.json
{
  "name": "kronodroid_autoencoder",
  "version": "1.0.0",
  "input_shape": [1, 289],
  "output_shape": [1, 289],
  "input_dtype": "float32",
  "description": "Autoencoder for malware detection"
}
```

---

## Configuration Reference

### Build Configurations

| Config | Usage | Description |
|--------|-------|-------------|
| `android_debug` | `--config=android_debug` | Single ABI (arm64), faster builds |
| `android_release` | `--config=android_release` | Multi-ABI, optimized |
| `robolectric` | `--config=robolectric` | JVM tests with Robolectric |
| `ci` | `--config=ci` | CI-optimized settings |
| `executorch` | `--config=executorch` | Enable ExecuTorch features |

### Maven Dependencies

Key dependencies configured in `MODULE.bazel`:

| Artifact | Version | Purpose |
|----------|---------|---------|
| `androidx.core:core-ktx` | 1.13.1 | Kotlin extensions |
| `androidx.appcompat:appcompat` | 1.7.0 | Compatibility library |
| `org.pytorch:executorch` | 0.4.0 | ExecuTorch Android runtime |
| `org.jetbrains.kotlinx:kotlinx-coroutines-android` | 1.9.0 | Async/coroutines |
| `org.robolectric:robolectric` | 4.14.1 | JVM Android testing |

### Adding New Maven Dependencies

Edit `MODULE.bazel`:

```python
maven.install(
    name = "maven",
    artifacts = [
        # Existing deps...
        "com.example:new-library:1.0.0",  # Add new dep
    ],
)
```

Then reference in BUILD files:

```python
deps = [
    "@maven//:com_example_new_library",  # Dots become underscores
]
```

---

## Troubleshooting

### Common Issues

#### "no such package" Error

```
ERROR: no such package '@maven//': ...
```

**Fix**: Run `bazel fetch //...` to download dependencies.

#### SDK Not Found

```
ERROR: ANDROID_HOME not set
```

**Fix**: Set `ANDROID_HOME` environment variable or create `local.properties`:

```properties
sdk.dir=/path/to/Android/Sdk
```

#### Kotlin Compilation Errors

**Fix**: Ensure Kotlin toolchain is configured in `apps/android/BUILD.bazel`.

#### Slow First Build

First build downloads dependencies and builds from scratch. Subsequent builds are much faster due to caching.

### Useful Debug Commands

```bash
# Verbose build output
bazel build --verbose_failures //...

# Show what would be built
bazel build --nobuild //apps/android/main_app:main_app

# Analyze build performance
bazel analyze-profile /path/to/profile.json

# Check dependency graph
bazel query "deps(//apps/android/main_app:main_app)" --output=graph | dot -Tpng > deps.png
```

---

## Model Validation Workflow

The project includes a complete workflow for validating ExecuTorch models against PyTorch baselines, including running on Android emulators.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Model Testing Pipeline                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   MLflow     │    │  ExecuTorch  │    │   Android    │    │   MLflow     │
│  Model Reg   │───▶│   Export     │───▶│  Emulator    │───▶│   Results    │
│  (PyTorch)   │    │  (.pte)      │    │   Test       │    │   Logging    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  ┌─────────┐        ┌─────────┐        ┌─────────┐        ┌─────────┐
  │ Weights │        │  .pte   │        │ Results │        │ Compare │
  │ + Model │        │ Artifact│        │  JSON   │        │ Report  │
  └─────────┘        └─────────┘        └─────────┘        └─────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **ExecuTorch export** | `runtimes/executorch/export/` | PyTorch → .pte conversion, MLflow integration |
| **Android app/tests** | `apps/android/runtime_comparison_app/` | Run inference, collect results |
| **Orchestration** | `tools/model_validation/` | End-to-end validation, comparisons |
| **Models** | `models/` | Local cache for .pte files |

### Export Package (`runtimes/executorch/export/`)

The export package provides tools for converting PyTorch models to ExecuTorch format:

```python
from runtimes.executorch.export import (
    ExecuTorchExporter,
    MLflowExecuTorchClient,
    get_quantization_config,
)

# Initialize
client = MLflowExecuTorchClient("http://localhost:5050")
exporter = ExecuTorchExporter(backend="xnnpack", quantization="none")

# Fetch model from MLflow
model, metadata = client.fetch_pytorch_model("kronodroid_autoencoder")

# Export to ExecuTorch
example_inputs = (torch.randn(1, 289),)
export_metadata = exporter.export(model, example_inputs, "model.pte")

# Register in MLflow
client.register_executorch_model(
    pte_path="model.pte",
    source_run_id=metadata.run_id,
    model_name="kronodroid_autoencoder",
    export_metadata=export_metadata.to_dict(),
)
```

### Available Quantization Configs

| Config | Description |
|--------|-------------|
| `none` | No quantization (fp32) |
| `dynamic_int8` | Dynamic quantization with int8 weights |
| `static_int8` | Static quantization (requires calibration) |
| `qat_int8` | Quantization-aware training |
| `mobile` | Mobile-optimized settings |

List configs:
```bash
task model:export:list-quantization
```

### CLI Usage

```bash
# Export model from MLflow
uv run python -m runtimes.executorch.export.cli export \
    --model-name kronodroid_autoencoder \
    --output models/kronodroid.pte \
    --backend xnnpack \
    --quantization none \
    --register

# Validate exported model
uv run python -m runtimes.executorch.export.cli validate \
    --model-name kronodroid_autoencoder \
    --pte-path models/kronodroid.pte \
    --tolerance 0.01

# Show model info
uv run python -m runtimes.executorch.export.cli info models/kronodroid.pte
```

---

## Task Reference

### Model Validation Tasks

| Task | Description |
|------|-------------|
| `task model:validate` | Full validation: PyTorch → ExecuTorch → Android → Compare |
| `task model:validate:no-android` | Host-only validation (skip emulator) |
| `task model:export` | Export PyTorch model to ExecuTorch format |
| `task model:export:list-quantization` | List quantization configurations |
| `task model:compare` | Compare inference results between runtimes |

#### Examples

```bash
# Full validation with Android emulator
task model:validate \
    MODEL_NAME=kronodroid_autoencoder \
    TEST_DATA=data/test_samples.npy \
    TOLERANCE=0.01

# Export with quantization
task model:export \
    MODEL_NAME=kronodroid_autoencoder \
    OUTPUT=models/kronodroid_int8.pte \
    QUANTIZATION=dynamic_int8

# Compare results
task model:compare \
    PYTORCH_RESULTS=results/pytorch.json \
    ANDROID_RESULTS=results/android.json \
    TOLERANCE=0.01
```

### Android Build Tasks

| Task | Description |
|------|-------------|
| `task android:build` | Build all Android apps |
| `task android:build:main` | Build main app |
| `task android:build:runtime` | Build runtime comparison app |
| `task android:install` | Install app on device/emulator |
| `task android:test` | Run Robolectric tests |
| `task android:test:device` | Run instrumentation tests on device |
| `task android:emulator:start` | Start Android emulator |
| `task android:emulator:stop` | Stop Android emulator |
| `task android:results:pull` | Pull benchmark results from device |

#### Examples

```bash
# Build debug APK
task android:build:runtime CONFIG=android_debug

# Build release APK (multi-ABI)
task android:build:runtime CONFIG=android_release

# Install on connected device
task android:install APP=runtime_comparison_app

# Run tests on emulator
task android:emulator:start AVD=Pixel_6_API_34
task android:test:device

# Pull results
task android:results:pull OUTPUT_DIR=./results
```

---

## End-to-End Validation Workflow

### Step-by-Step Guide

1. **Ensure MLflow is running with a trained model**:
   ```bash
   task port-forward  # Start port-forwards
   # MLflow should be at http://localhost:5050
   ```

2. **Prepare test data**:
   ```bash
   # Create test data as numpy array
   python -c "
   import numpy as np
   # Generate or load test samples
   test_data = np.random.randn(100, 289).astype(np.float32)
   np.save('data/test_samples.npy', test_data)
   "
   ```

3. **Export model to ExecuTorch**:
   ```bash
   task model:export \
       MODEL_NAME=kronodroid_autoencoder \
       OUTPUT=models/kronodroid_autoencoder.pte
   ```

4. **Run host-only validation** (no emulator needed):
   ```bash
   task model:validate:no-android \
       MODEL_NAME=kronodroid_autoencoder \
       TEST_DATA=data/test_samples.npy
   ```

5. **Or run full validation with Android emulator**:
   ```bash
   # Start emulator
   task android:emulator:start

   # Run full validation
   task model:validate \
       MODEL_NAME=kronodroid_autoencoder \
       TEST_DATA=data/test_samples.npy
   ```

6. **View results in MLflow**:
   - Open http://localhost:5050
   - Find the training run
   - Check artifacts for `executorch/model.pte`
   - Check metrics for `validation.*` and `android.*`

### MLflow Integration

The validation workflow logs everything to MLflow:

```
MLflow Run (original training)
├── Artifacts
│   ├── model/                    # PyTorch model
│   │   ├── model.pth
│   │   └── MLmodel
│   └── executorch/               # ExecuTorch export
│       ├── model.pte
│       └── model.json
├── Parameters
│   ├── learning_rate: 0.001
│   ├── executorch.backend: xnnpack
│   └── executorch.quantization: none
├── Metrics
│   ├── train_loss: 0.023
│   ├── validation.pytorch.mean_output: 0.45
│   ├── validation.executorch.mean_output: 0.449
│   ├── validation.pass_rate: 0.99
│   ├── validation.max_diff: 0.008
│   ├── android.avg_latency_ms: 12.5
│   └── android.throughput_samples_per_sec: 80.0
└── Tags
    ├── executorch.exported: true
    ├── executorch.backend: xnnpack
    └── android.tested: true
```

### Validation Criteria

The validation passes when:
- **99%+ samples** are within tolerance (default: 0.01)
- Model loads successfully on Android
- Inference latency is reasonable (< 100ms avg, < 500ms p99)

Failed validations return exit code 1 and log details to MLflow.

---

## Runtime Comparison App

The `runtime_comparison_app` is designed to benchmark model inference on Android:

### Features

- Load ExecuTorch models from assets
- Run inference with configurable warmup
- Collect detailed latency metrics (avg, p50, p95, p99)
- Save results as JSON for analysis
- Support for instrumentation tests

### Key Classes

| Class | Purpose |
|-------|---------|
| `ModelBenchmark` | Load models, run inference, collect metrics |
| `ResultsReporter` | Save results to file, post to server, log to Logcat |
| `ModelAccuracyTest` | Instrumentation tests for accuracy validation |

### Running Benchmarks Manually

```kotlin
// In Android app
val benchmark = ModelBenchmark(context)
benchmark.loadModelFromAssets("models/kronodroid.pte")

val testData = listOf(
    "sample_001" to FloatArray(289) { 0.5f },
    "sample_002" to FloatArray(289) { 0.3f },
)

val results = benchmark.runInference(testData, warmupRuns = 5)

val reporter = ResultsReporter(context)
reporter.saveToFile(results, "benchmark_results.json")
reporter.logSummary(results)
```

### Pulling Results from Device

```bash
# After running the app
task android:results:pull OUTPUT_DIR=./results

# Results include:
# - benchmark_results.json  (full inference results)
# - performance_test_results.json  (from instrumentation tests)
# - accuracy_test_results.json  (accuracy metrics)
```

---

## Further Reading

- [Bazel Documentation](https://bazel.build/docs)
- [rules_android](https://github.com/bazelbuild/rules_android)
- [rules_kotlin](https://github.com/bazelbuild/rules_kotlin)
- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [Bazel Android Tutorial](https://bazel.build/tutorials/android-app)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
