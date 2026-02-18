from __future__ import annotations

from typing import Iterable, List

import torch

from data.snapshot_sampler import SnapshotSampler
from losses.mmd import MMDLoss, regularization_terms
from model.cell_mnn import CellMNN


def rollout_mmd_loss(
    model: CellMNN,
    sampler: SnapshotSampler,
    timepoints: Iterable[float],
    batch_size: int,
    mmd_loss: MMDLoss,
    gamma: float = 0.1,
    kinetic_weight: float = 1.0,
    inverse_weight: float = 1.0,
) -> torch.Tensor:
    times = list(timepoints)
    if len(times) < 2:
        raise ValueError("At least two timepoints are required for rollout.")

    z_current = sampler.sample_batch(times[0], batch_size)
    discount = 1.0
    total_loss = 0.0

    for idx in range(len(times) - 1):
        t_start = times[idx]
        t_end = times[idx + 1]
        z_target = sampler.sample_batch(t_end, batch_size)
        t_tensor = torch.full(
            (batch_size, 1),
            float(t_start),
            device=z_current.device,
            dtype=z_current.dtype,
        )
        delta_t = torch.full(
            (batch_size, 1),
            float(t_end - t_start),
            device=z_current.device,
            dtype=z_current.dtype,
        )

        z_pred, p, lambdas = model(z_current, t_tensor, delta_t)
        mmd = mmd_loss(z_pred, z_target)
        kinetic, inverse = regularization_terms(z_current, p, lambdas)
        step_loss = mmd + kinetic_weight * kinetic + inverse_weight * inverse
        total_loss = total_loss + discount * step_loss
        z_current = z_pred
        discount *= gamma

    return total_loss


def train_cell_mnn(
    model: CellMNN,
    sampler: SnapshotSampler,
    timepoints: Iterable[float],
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    gamma: float = 0.1,
    kinetic_weight: float = 1.0,
    inverse_weight: float = 1.0,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-5,
    device: torch.device | None = None,
) -> List[float]:
    if device is not None:
        model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    mmd_loss = MMDLoss()
    losses: List[float] = []
    max_grad_norm = 5.0

    for _ in range(epochs):
        epoch_loss = 0.0
        model.train()
        for _ in range(steps_per_epoch):
            loss = rollout_mmd_loss(
                model,
                sampler,
                timepoints,
                batch_size,
                mmd_loss,
                gamma=gamma,
                kinetic_weight=kinetic_weight,
                inverse_weight=inverse_weight,
            )
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = torch.nan_to_num(
                        param.grad, nan=0.0, posinf=0.0, neginf=0.0
                    )
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        losses.append(epoch_loss / steps_per_epoch)

    return losses
