"""PyTorch Lightning modules for DFP models."""

from typing import Tuple

try:
    import torch
    from torch import nn
    import lightning as L
except ImportError:  # pragma: no cover - tooling only
    torch = None
    nn = None
    L = None


class LightningAutoencoder(L.LightningModule if L else object):
    """Lightning-wrapped autoencoder for Kronodroid syscall features.

    Architecture:
        Encoder: input_dim -> hidden_dims[0] -> ... -> hidden_dims[-1] -> latent_dim
        Decoder: latent_dim -> hidden_dims[-1] -> ... -> hidden_dims[0] -> input_dim

    Args:
        input_dim: Number of input features (default 22 for syscall features)
        latent_dim: Dimension of latent space
        hidden_dims: Tuple of hidden layer dimensions
        learning_rate: Optimizer learning rate
        reconstruction_loss: 'mse' or 'mae'
    """

    def __init__(
        self,
        input_dim: int = 22,
        latent_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (128, 64),
        learning_rate: float = 1e-3,
        reconstruction_loss: str = "mse",
    ):
        if L is None:
            raise RuntimeError("lightning not installed")
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate

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

        # Build decoder (reverse of encoder)
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

        # Loss function
        self.loss_fn = nn.MSELoss() if reconstruction_loss == "mse" else nn.L1Loss()

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        """Decode latent representation to reconstruction."""
        return self.decoder(z)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass: encode then decode."""
        z = self.encode(x)
        return self.decode(z)

    def _shared_step(self, batch: "torch.Tensor", stage: str) -> "torch.Tensor":
        """Shared step for train/val/test."""
        x = batch
        x_hat = self(x)
        loss = self.loss_fn(x_hat, x)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """Configure Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent_representation(self, x: "torch.Tensor") -> "torch.Tensor":
        """Get latent representation for input data."""
        self.eval()
        with torch.no_grad():
            return self.encode(x)

    def compute_reconstruction_error(self, x: "torch.Tensor") -> "torch.Tensor":
        """Compute per-sample reconstruction error (for anomaly detection)."""
        self.eval()
        with torch.no_grad():
            x_hat = self(x)
            # MSE per sample
            return torch.mean((x - x_hat) ** 2, dim=1)
