from __future__ import annotations

import torch
from torch import nn


class LinearOperatorNet(nn.Module):
    """Hypernetwork that outputs eigenvalues and eigenvectors for linear dynamics."""

    def __init__(
        self,
        latent_dim: int = 5,
        hidden_dim: int = 96,
        num_layers: int = 4,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2.")
        self.latent_dim = latent_dim
        input_dim = latent_dim + 1

        layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(negative_slope)]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope))
        self.backbone = nn.Sequential(*layers)

        self.lambda_head = nn.Linear(hidden_dim, latent_dim)
        self.p_head = nn.Linear(hidden_dim, latent_dim * latent_dim)
        self._scale_output_layers(scale=0.01)

    def _scale_output_layers(self, scale: float) -> None:
        with torch.no_grad():
            self.lambda_head.weight.mul_(scale)
            self.lambda_head.bias.mul_(scale)
            self.p_head.weight.mul_(scale)
            self.p_head.bias.mul_(scale)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if z.dim() != 2:
            raise ValueError("z must be 2D (batch, latent_dim).")
        if t.shape[0] != z.shape[0]:
            raise ValueError("t and z must have the same batch size.")
        x = torch.cat([z, t], dim=-1)
        hidden = self.backbone(x)
        lambdas = self.lambda_head(hidden)
        p_flat = self.p_head(hidden)
        p = p_flat.view(z.shape[0], self.latent_dim, self.latent_dim)
        return lambdas, p
