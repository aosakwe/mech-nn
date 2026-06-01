from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import logging
import torch
from torch.utils.data import IterableDataset


@dataclass
class PCAPreprocessor:
    latent_dim: int = 5
    dtype: torch.dtype = torch.float32
    device: torch.device | None = None

    def fit(self, data: torch.Tensor) -> "PCAPreprocessor":
        """Fit PCA on pooled data.

        Args:
            data: Tensor of shape (n_samples, n_features)
        """
        if data.dim() != 2:
            raise ValueError("PCA input must be 2D (n_samples, n_features).")
        target_device = self.device or data.device
        data = data.to(dtype=self.dtype, device=target_device)
        if not torch.isfinite(data).all():
            logging.warning("PCA input has non-finite values; replacing with zeros.")
            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        mean = data.mean(dim=0, keepdim=True)
        centered = data - mean
        # Use SVD for PCA projection matrix
        try:
            _, _, v_h = torch.linalg.svd(centered, full_matrices=False)
        except RuntimeError as exc:
            logging.warning(
                "PCA SVD failed on %s; retrying in float64 on CPU. Error: %s",
                target_device,
                exc,
            )
            centered64 = centered.to(device="cpu", dtype=torch.float64)
            try:
                _, _, v_h = torch.linalg.svd(centered64, full_matrices=False)
                v_h = v_h.to(
                    device=target_device,
                    dtype=target_device and data.dtype or data.dtype,
                )
            except RuntimeError as exc2:
                logging.warning(
                    "PCA SVD failed on CPU; falling back to pca_lowrank. Error: %s",
                    exc2,
                )
                q = min(self.latent_dim, centered64.shape[1], centered64.shape[0])
                _u, _s, v = torch.pca_lowrank(centered64, q=q, center=False)
                v_h = v[:, : self.latent_dim].T.contiguous().to(
                    device=target_device,
                    dtype=target_device and data.dtype or data.dtype,
                )
        components = v_h[: self.latent_dim].T.contiguous()  # (n_features, latent_dim)
        self.mean_ = mean
        self.components_ = components
        return self

    def to(self, device: torch.device) -> "PCAPreprocessor":
        if hasattr(self, "mean_"):
            self.mean_ = self.mean_.to(device)
        if hasattr(self, "components_"):
            self.components_ = self.components_.to(device)
        self.device = device
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "components_"):
            raise RuntimeError("PCA must be fit before calling transform.")
        data = data.to(dtype=self.components_.dtype, device=self.components_.device)
        if not torch.isfinite(data).all():
            logging.warning(
                "PCA transform input has non-finite values; replacing with zeros."
            )
            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        centered = data - self.mean_
        transformed = centered @ self.components_
        if not torch.isfinite(transformed).all():
            logging.warning(
                "PCA transform produced non-finite values; replacing with zeros."
            )
            transformed = torch.nan_to_num(
                transformed, nan=0.0, posinf=0.0, neginf=0.0
            )
        return transformed

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)


class SnapshotSampler:
    """Sample independent snapshots for any time pair."""

    def __init__(self, time_to_data: Dict[float, torch.Tensor]):
        self.time_to_data = {
            float(t): v for t, v in time_to_data.items()
        }
        self.times = sorted(self.time_to_data.keys())

    def sample_batch(self, timepoint: float, batch_size: int) -> torch.Tensor:
        data = self.time_to_data[float(timepoint)]
        indices = torch.randint(0, data.shape[0], (batch_size,), device=data.device)
        return data[indices]

    def sample_pair(
        self, time_start: float, time_end: float, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_start = self.sample_batch(time_start, batch_size)
        z_end = self.sample_batch(time_end, batch_size)
        delta_t = torch.full(
            (batch_size, 1),
            float(time_end - time_start),
            device=z_start.device,
            dtype=z_start.dtype,
        )
        return z_start, z_end, delta_t


class SnapshotPairDataset(IterableDataset):
    """Iterable dataset yielding snapshot pairs and time deltas."""

    def __init__(
        self,
        sampler: SnapshotSampler,
        time_pairs: Iterable[Tuple[float, float]],
        batch_size: int,
        steps_per_epoch: int,
    ):
        super().__init__()
        self.sampler = sampler
        self.time_pairs = list(time_pairs)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        for idx in range(self.steps_per_epoch):
            time_start, time_end = self.time_pairs[idx % len(self.time_pairs)]
            yield self.sampler.sample_pair(time_start, time_end, self.batch_size)


def build_snapshot_sampler_with_pca(
    time_to_data: Dict[float, torch.Tensor],
    latent_dim: int = 5,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tuple[SnapshotSampler, PCAPreprocessor]:
    """Fit PCA on pooled data and return snapshot sampler in latent space."""
    any_tensor = next(iter(time_to_data.values()))
    target_device = device or any_tensor.device
    fit_device = torch.device("cpu") if target_device.type == "cuda" else target_device

    pooled = torch.cat(
        [v.to(device=fit_device, dtype=dtype) for v in time_to_data.values()],
        dim=0,
    )
    pca = PCAPreprocessor(latent_dim=latent_dim, dtype=dtype, device=fit_device)
    pca.fit(pooled)
    if fit_device != target_device:
        pca.to(target_device)
    latent_time_to_data = {t: pca.transform(v) for t, v in time_to_data.items()}
    return SnapshotSampler(latent_time_to_data), pca
