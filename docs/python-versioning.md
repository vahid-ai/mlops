# Python Versioning Strategy

This document explains the Python versioning strategy for the DFP monorepo, which uses [uv](https://docs.astral.sh/uv/) for fast, reproducible Python environment management.

## Overview

The project uses **Python 3.12** as the "blessed" version. This choice provides the optimal balance between:

- **ExecuTorch compatibility**: ExecuTorch 0.4.0 supports Python 3.10-3.12 (we pin to 3.12 to avoid pandas conflicts on 3.10/3.11)
- **PyTorch compatibility**: PyTorch 2.5.x works with ExecuTorch 0.4.0
- **Modern features**: Python 3.12 has performance improvements and modern syntax
- **Ecosystem support**: Most ML/data libraries fully support 3.12

## Quick Start

```bash
# Install/sync the default environment
task uv:sync

# Or with specific dependency groups
task uv:sync:dev      # Add dev tools (linting, testing)
task uv:sync:android  # Add ExecuTorch for Android
task uv:sync:kubeflow # Add Kubeflow pipeline deps
task uv:sync:all      # All dependency groups
```

## Project Configuration

### Key Files

| File | Purpose |
|------|---------|
| `.python-version` | Specifies Python 3.12 for uv and other tools |
| `pyproject.toml` | Defines dependencies and version constraints |
| `uv.toml` | uv-specific configuration (cache, index) |
| `uv.lock` | Lock file for reproducible installs |

### Dependency Groups

The project defines several dependency groups in `pyproject.toml`:

```toml
[dependency-groups]
dev = [...]           # Development tools (pyrefly, pytest)
android = [...]       # ExecuTorch for mobile deployment
kubeflow = [...]      # Kubeflow pipeline dependencies
spark-container = [...] # Docker container dependencies (pandas<2)
```

## Version Constraints

### Python Version

```toml
requires-python = ">=3.12,<3.13"
```

**Why Python 3.12 specifically?**
- ExecuTorch 0.4.0 has pandas version constraints on Python 3.10/3.11 that conflict with our pandas>=2.1 requirement
- Python 3.12 avoids these conflicts while still being supported by ExecuTorch
- Python 3.13 is not supported by ExecuTorch

### PyTorch Version

```toml
"torch>=2.5.0,<2.6"
```

**Why pin to 2.5.x?**
- ExecuTorch 0.4.0 requires torch==2.5.0
- Models saved with older PyTorch versions need to be re-saved with 2.5.x
- PyTorch's pickle format changes between major versions

### Pandas Version (Containers)

The main project uses `pandas>=2.1`, but Docker containers for Spark/Feast use `pandas>=1.5,<2`:

```toml
spark-container = [
    "pandas>=1.5,<2",  # Feast 0.36.0 requires pandas<2
    ...
]
```

## Environment Management

### Local Development

```bash
# Create/sync environment with Python 3.12
task uv:sync

# Activate the environment (optional - uv run works without activation)
source .venv/bin/activate

# Run commands in the environment
uv run python script.py
uv run pytest tests/
```

### With Specific Groups

```bash
# Development (adds linting and testing tools)
task uv:sync:dev
uv run pytest tests/
uv run pyrefly check .

# Android/ExecuTorch development
task uv:sync:android
uv run python -m runtimes.executorch.export.cli --help

# Kubeflow pipeline development
task uv:sync:kubeflow
uv run python tools/scripts/run_kronodroid_pipeline.py --help
```

### Environment Information

```bash
# Show current environment status
task uv:info

# Update lock file after changing pyproject.toml
task uv:lock

# Clean and recreate environment
task uv:clean
task uv:sync
```

## Docker Containers

Docker containers use uv for fast, reproducible builds:

```dockerfile
# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies
RUN uv pip install --system --no-cache <packages>
```

### Container Python Versions

| Container | Base Image | Python | Notes |
|-----------|------------|--------|-------|
| dfp-spark | apache/spark:3.5.0-python3 | 3.11 | Spark jobs |
| dfp-autoencoder-train | apache/spark:3.5.0-python3 | 3.11 | ML training with pandas<2 |

## Troubleshooting

### "No solution found" when syncing

This usually means a dependency conflict. Check:

1. Python version matches constraints:
   ```bash
   python --version  # Should be 3.10-3.12
   ```

2. Try clearing the cache:
   ```bash
   task uv:clean
   task uv:sync
   ```

### "ExecuTorch not found" or import errors

ExecuTorch requires Python 3.10-3.12:

```bash
# Check Python version
python --version

# If using 3.13, create a 3.12 environment
task uv:sync:android
```

### PyTorch version mismatch when loading models

If you see errors like `TypeError: code() argument 13 must be str, not int`:

1. The model was saved with a different PyTorch version (e.g., 2.4.x)
2. Re-save the model with PyTorch 2.5.x:
   ```python
   # Load with old PyTorch, save state_dict, reload with new PyTorch
   import torch

   # Option 1: If you have access to the training code
   # Re-run training with PyTorch 2.5.x

   # Option 2: Load weights only (if architecture is defined in code)
   model = YourModel()
   model.load_state_dict(torch.load("old_model.pt", weights_only=True))
   torch.save(model, "new_model.pt")
   ```
3. Or load with the same PyTorch version that saved it (not recommended for ExecuTorch)

### Feast/Pandas compatibility in containers

Feast 0.36.0 requires `pandas<2`. The `spark-container` dependency group handles this:

```bash
# For local development with Feast (pandas>=2 works)
task uv:sync

# For building container images (use spark-container group)
# See tools/docker/Dockerfile.autoencoder_train
```

## Best Practices

1. **Always use `uv run`** instead of activating the environment:
   ```bash
   uv run python script.py  # Recommended
   ```

2. **Commit `uv.lock`** for reproducible builds:
   ```bash
   task uv:lock
   git add uv.lock
   git commit -m "Update uv.lock"
   ```

3. **Use dependency groups** for different use cases rather than multiple environments

4. **Pin critical packages** in `pyproject.toml` to avoid breaking changes

5. **Test in containers** before deploying to Kubeflow/Kubernetes

## Reference

### Task Commands

| Task | Description |
|------|-------------|
| `task uv:sync` | Sync default Python 3.12 environment |
| `task uv:sync:dev` | Sync with dev tools |
| `task uv:sync:android` | Sync with ExecuTorch |
| `task uv:sync:kubeflow` | Sync with Kubeflow deps |
| `task uv:sync:all` | Sync with all groups |
| `task uv:info` | Show environment info |
| `task uv:lock` | Update lock file |
| `task uv:clean` | Remove venv and cache |

### Related Documentation

- [uv Documentation](https://docs.astral.sh/uv/)
- [ExecuTorch Installation](https://pytorch.org/executorch/stable/getting-started-setup.html)
- [Android Development](./android.md)
