"""Model registry for PyTorch experiments."""
try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - tooling only
    torch = None
    nn = None

class SmallAutoencoder(nn.Module if nn else object):
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
