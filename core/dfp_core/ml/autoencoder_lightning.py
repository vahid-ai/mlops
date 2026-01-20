"""PyTorch Lightning modules for autoencoder training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 16
    hidden_dims: tuple[int, ...] = (128, 64)
    lr: float = 1e-3
    weight_decay: float = 0.0


def build_mlp(layer_dims: list[int]):
    import torch
    from torch import nn

    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(layer_dims, layer_dims[1:], strict=True):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
    layers.pop()  # drop last ReLU
    return nn.Sequential(*layers)


def make_autoencoder(cfg: AutoencoderConfig):
    import torch
    from torch import nn

    encoder_dims = [cfg.input_dim, *cfg.hidden_dims, cfg.latent_dim]
    decoder_dims = [cfg.latent_dim, *reversed(cfg.hidden_dims), cfg.input_dim]
    encoder = build_mlp(encoder_dims)
    decoder = build_mlp(decoder_dims)
    return nn.ModuleDict({"encoder": encoder, "decoder": decoder})


def make_lightning_module(cfg: AutoencoderConfig):
    try:
        import lightning.pytorch as pl
    except Exception:  # pragma: no cover - optional dependency at dev time
        import pytorch_lightning as pl  # type: ignore[no-redef]

    import torch
    from torch import nn

    model = make_autoencoder(cfg)

    class AutoencoderLightningModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.save_hyperparameters(cfg.__dict__)
            self.model = model
            self.loss_fn = nn.MSELoss()

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            z = self.model["encoder"](x)
            return self.model["decoder"](z)

        def training_step(self, batch, batch_idx: int):  # type: ignore[override]
            (x,) = batch
            x_hat = self(x)
            loss = self.loss_fn(x_hat, x)
            self.log("train/recon_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx: int):  # type: ignore[override]
            (x,) = batch
            x_hat = self(x)
            loss = self.loss_fn(x_hat, x)
            self.log("val/recon_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss

        def test_step(self, batch, batch_idx: int):  # type: ignore[override]
            (x,) = batch
            x_hat = self(x)
            loss = self.loss_fn(x_hat, x)
            self.log("test/recon_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            return loss

        def configure_optimizers(self):  # type: ignore[override]
            return torch.optim.AdamW(
                self.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )

    return AutoencoderLightningModule()

