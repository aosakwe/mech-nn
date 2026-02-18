from __future__ import annotations

import torch
from torch import nn


def _laplacian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    distances = torch.cdist(x, y, p=1)
    return torch.exp(-distances / bandwidth)


class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy with Laplacian kernel."""

    def __init__(self, bandwidth: float = 1.0) -> None:
        super().__init__()
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k_xx = _laplacian_kernel(x, x, self.bandwidth).mean()
        k_yy = _laplacian_kernel(y, y, self.bandwidth).mean()
        k_xy = _laplacian_kernel(x, y, self.bandwidth).mean()
        return k_xx + k_yy - 2.0 * k_xy


def regularization_terms(
    z: torch.Tensor,
    p: torch.Tensor,
    lambdas: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute kinetic energy and inverse consistency regularizers."""
    identity = torch.eye(p.shape[-1], device=p.device, dtype=p.dtype)
    identity = identity.expand(p.shape[0], -1, -1)
    p_reg = p + eps * identity
    p_reg = torch.nan_to_num(p_reg, nan=0.0, posinf=1e4, neginf=-1e4)
    p_inv = torch.linalg.solve(p_reg, identity)
    p_inv = torch.nan_to_num(p_inv, nan=0.0, posinf=0.0, neginf=0.0)
    a = torch.bmm(p, torch.bmm(torch.diag_embed(lambdas), p_inv))
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    dz = torch.bmm(a, z.unsqueeze(-1)).squeeze(-1)
    kinetic = (dz.pow(2).sum(dim=-1)).mean()
    sign, logabsdet = torch.linalg.slogdet(p_reg)
    valid_sign = sign != 0
    inv_det = torch.exp(-torch.clamp(logabsdet, min=-20.0, max=20.0))
    inv_det = torch.where(valid_sign, inv_det, torch.full_like(inv_det, 1e4))
    inv_det = torch.nan_to_num(inv_det, nan=1e4, posinf=1e4, neginf=0.0)
    inverse_consistency = inv_det.mean()
    return kinetic, inverse_consistency
