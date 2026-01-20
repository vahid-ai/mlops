"""Model registry for PyTorch experiments."""

from typing import Tuple, Union

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - tooling only
    torch = None
    nn = None

# Import Lightning module if available
try:
    from core.dfp_core.ml.lightning_modules import LightningAutoencoder
except ImportError:
    LightningAutoencoder = None


class SmallAutoencoder(nn.Module if nn else object):
    """Simple autoencoder for basic experiments (plain PyTorch)."""

    def __init__(self, input_dim: int = 128, latent_dim: int = 16):
        super().__init__()
        if nn:
            self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, input_dim))

    def forward(self, x):  # type: ignore[override]
        if nn is None:
            raise RuntimeError("torch not installed")
        latent = self.encoder(x)
        return self.decoder(latent)


def create_autoencoder(
    input_dim: int = 22,
    latent_dim: int = 16,
    hidden_dims: Tuple[int, ...] = (128, 64),
    use_lightning: bool = True,
    learning_rate: float = 1e-3,
) -> Union["LightningAutoencoder", SmallAutoencoder]:
    """Factory function for creating autoencoder models.

    Args:
        input_dim: Number of input features
        latent_dim: Dimension of latent space
        hidden_dims: Tuple of hidden layer dimensions (only for Lightning)
        use_lightning: If True, return LightningAutoencoder; else SmallAutoencoder
        learning_rate: Learning rate (only for Lightning)

    Returns:
        Autoencoder model (Lightning or plain PyTorch)
    """
    if use_lightning:
        if LightningAutoencoder is None:
            raise ImportError(
                "LightningAutoencoder not available. "
                "Install lightning: pip install lightning"
            )
        return LightningAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
        )
    return SmallAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
