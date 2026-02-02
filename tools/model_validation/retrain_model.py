#!/usr/bin/env python3
"""Re-train Kronodroid autoencoder with current PyTorch version.

This script creates a model compatible with the current PyTorch version
for ExecuTorch export testing.

Usage:
    uv run python tools/model_validation/retrain_model.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LightningAutoencoder(nn.Module):
    """Autoencoder matching the Kubeflow component architecture."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Tuple[int, ...],
    ):
        super().__init__()

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def main():
    """Re-train and register model with current PyTorch version."""
    print("=" * 60)
    print("Re-training Kronodroid Autoencoder")
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 60)

    # Configuration matching the Kubeflow component
    input_dim = 289  # Number of syscall features
    latent_dim = 32
    hidden_dims = (128, 64)
    batch_size = 256
    epochs = 10
    lr = 0.001
    model_name = "kronodroid_autoencoder"

    # MLflow config
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:19000"

    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("kronodroid-autoencoder-retrain")

    print(f"\n1. Creating model...")
    print(f"   Architecture: {input_dim} -> {hidden_dims} -> {latent_dim}")

    model = LightningAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")

    # Generate synthetic training data (simulating syscall features)
    print(f"\n2. Generating synthetic training data...")
    n_train = 5000
    n_val = 1000

    # Create data with some structure (not pure random)
    np.random.seed(42)
    train_data = np.random.randn(n_train, input_dim).astype(np.float32) * 0.5
    val_data = np.random.randn(n_val, input_dim).astype(np.float32) * 0.5

    # Add some correlations between features
    for i in range(0, input_dim - 1, 2):
        train_data[:, i + 1] = train_data[:, i] * 0.8 + np.random.randn(n_train).astype(np.float32) * 0.2
        val_data[:, i + 1] = val_data[:, i] * 0.8 + np.random.randn(n_val).astype(np.float32) * 0.2

    # Normalize
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0) + 1e-8
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std

    train_tensor = torch.from_numpy(train_data)
    val_tensor = torch.from_numpy(val_data)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size)

    print(f"   Train samples: {n_train}")
    print(f"   Val samples: {n_val}")

    # Training
    print(f"\n3. Training for {epochs} epochs...")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"   MLflow run: {run_id}")

        # Log parameters
        mlflow.log_params({
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dims": str(hidden_dims),
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "pytorch_version": torch.__version__,
            "retrained_for_executorch": True,
        })

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            for (batch,) in train_loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    output = model(batch)
                    loss = criterion(output, batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, step=epoch)

            print(f"   Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # Log final metrics
        mlflow.log_metric("best_val_loss", best_val_loss)

        # Save normalization params
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "mean": mean.tolist(),
                "std": std.tolist(),
            }, f)
            mlflow.log_artifact(f.name, "normalization")
            os.unlink(f.name)

        # Register model
        print(f"\n4. Registering model to MLflow...")
        model.eval()

        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=model_name,
            pip_requirements=[f"torch>={torch.__version__}"],
        )

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        model_version = str(max([int(v.version) for v in versions])) if versions else "1"

        print(f"   Registered: {model_name} v{model_version}")
        print(f"   PyTorch version: {torch.__version__}")

    print(f"\n" + "=" * 60)
    print(f"Model re-trained successfully!")
    print(f"Run: task model:export MODEL_NAME={model_name}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
