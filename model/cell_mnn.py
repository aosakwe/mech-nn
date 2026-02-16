from __future__ import annotations

import torch
from torch import nn

from model.linear_operator_net import LinearOperatorNet
from solver.eigen_solver import EigenSolver


def _safe_solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(a, b)
    except RuntimeError:
        return torch.linalg.lstsq(a, b).solution


class CellMNN(nn.Module):
    """Cell-MNN model using state-dependent linear operators in latent space."""

    def __init__(
        self,
        latent_dim: int = 5,
        pca_components: torch.Tensor | None = None,
        pca_mean: torch.Tensor | None = None,
        solver_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.operator_net = LinearOperatorNet(latent_dim=latent_dim)
        self.solver = EigenSolver(eps=solver_eps)
        if pca_components is not None:
            self.register_buffer("pca_components", pca_components)
        else:
            self.pca_components = None
        if pca_mean is not None:
            self.register_buffer("pca_mean", pca_mean)
        else:
            self.pca_mean = None

    def forward(
        self, z: torch.Tensor, t: torch.Tensor, delta_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lambdas, p = self.operator_net(z, t)
        z_next = self.solver(z, p, lambdas, delta_t)
        return z_next, p, lambdas

    def compute_operator(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lambdas, p = self.operator_net(z, t)
        identity = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        identity = identity.expand(p.shape[0], -1, -1)
        p_reg = p + self.solver.eps * identity
        p_inv = _safe_solve(p_reg, identity)
        a = torch.bmm(p, torch.bmm(torch.diag_embed(lambdas), p_inv))
        return a

    def extract_grn(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Project latent dynamics to gene space for GRN extraction."""
        if self.pca_components is None:
            raise RuntimeError("PCA components are required for GRN extraction.")
        lambdas, p = self.operator_net(z, t)
        identity = torch.eye(self.latent_dim, device=z.device, dtype=z.dtype)
        identity = identity.expand(p.shape[0], -1, -1)
        p_reg = p + self.solver.eps * identity
        p_inv = _safe_solve(p_reg, identity)
        a = torch.bmm(p, torch.bmm(torch.diag_embed(lambdas), p_inv))
        v = self.pca_components
        v_t = v.transpose(0, 1)
        v_batch = v.unsqueeze(0).expand(a.shape[0], -1, -1)
        v_t_batch = v_t.unsqueeze(0).expand(a.shape[0], -1, -1)
        w_genes = torch.bmm(v_batch, torch.bmm(a, v_t_batch))
        return w_genes
