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

## Further Reading

- [Bazel Documentation](https://bazel.build/docs)
- [rules_android](https://github.com/bazelbuild/rules_android)
- [rules_kotlin](https://github.com/bazelbuild/rules_kotlin)
- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [Bazel Android Tutorial](https://bazel.build/tutorials/android-app)
