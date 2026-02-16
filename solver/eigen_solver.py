from __future__ import annotations

import torch
from torch import nn


class EigenSolver(nn.Module):
    """Analytical solver using eigen-decomposition of linear dynamics."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        lambdas: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)
        diag_exp = torch.exp(lambdas * delta_t)
        identity = torch.eye(p.shape[-1], device=p.device, dtype=p.dtype)
        identity = identity.expand(p.shape[0], -1, -1)
        p_reg = p + self.eps * identity
        try:
            z_eig = torch.linalg.solve(p_reg, z.unsqueeze(-1))
        except RuntimeError:
            z_eig = torch.linalg.lstsq(p_reg, z.unsqueeze(-1)).solution
        z_scaled = diag_exp.unsqueeze(-1) * z_eig
        z_next = torch.bmm(p_reg, z_scaled)
        return z_next.squeeze(-1)
