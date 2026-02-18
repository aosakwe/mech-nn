from __future__ import annotations

import torch
from torch import nn


class EigenSolver(nn.Module):
    """Analytical solver using eigen-decomposition of linear dynamics."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.max_exp_arg = 40.0

    def forward(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        lambdas: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(-1)
        exp_arg = lambdas * delta_t
        exp_arg = torch.clamp(exp_arg, min=-self.max_exp_arg, max=self.max_exp_arg)
        diag_exp = torch.exp(exp_arg)
        diag_exp = torch.nan_to_num(diag_exp, nan=1.0, posinf=1.0, neginf=0.0)
        identity = torch.eye(p.shape[-1], device=p.device, dtype=p.dtype)
        identity = identity.expand(p.shape[0], -1, -1)
        p_reg = p + self.eps * identity
        p_reg = torch.nan_to_num(p_reg, nan=0.0, posinf=1e4, neginf=-1e4)
        try:
            z_eig = torch.linalg.solve(p_reg, z.unsqueeze(-1))
        except RuntimeError:
            z_eig = torch.linalg.lstsq(p_reg, z.unsqueeze(-1)).solution
        z_eig = torch.nan_to_num(z_eig, nan=0.0, posinf=0.0, neginf=0.0)
        z_scaled = diag_exp.unsqueeze(-1) * z_eig
        z_next = torch.bmm(p_reg, z_scaled)
        z_next = torch.nan_to_num(z_next, nan=0.0, posinf=0.0, neginf=0.0)
        return z_next.squeeze(-1)
