"""Contrastive regularization utilities.

This module provides `ContrastiveRegularizedLoss`, a wrapper around a standard
supervised loss (e.g., CrossEntropyLoss) that adds a contrastive term computed
from intermediate model features and per-sample metadata.

High-level:
- Base loss: any nn.Module taking `(outputs, targets)` returning a scalar loss.
- Contrastive term: encourages feature distances to match metadata distances.
- Total loss: `L_total = L_base + lambda_contrastive * L_contrastive`.

Key properties:
- Pluggable: pass any base criterion and feed the feature tensor explicitly in
  the forward call (no forward hooks required).
- Vectorized: computes all pairwise distances in-batch (O(B^2)).
- Flexible metadata: accepts either a dict with the requested key or a tensor
  per sample. Optional per-dimension weights allow mixing discrete and
  continuous metadata on a common scale.
- Robustness: guards against NaNs/Infs and handles missing metadata
  gracefully.
- Logging: exposes a `last_components` dict after each forward for
  monitoring.

Usage tips:
- Start with `lambda_contrastive = 0.0` to verify base training, then
  increase gradually (optionally via a warm-up schedule).
- Normalise features (`feature_norm=True`) when using cosine-like geometry.
- Tune the weighting (`lambda_contrastive`) and any metadata scaling to balance the regression term.
- Batch size matters: the term is O(B^2); use smaller batches or subsample
  pairs when necessary.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveRegularizedLoss(nn.Module):
    """Cross-entropy style criterion plus a metadata-driven contrastive term.

    The loss expects the caller to supply model features explicitly:

    >>> logits, features = model(images, return_features=True)
    >>> loss = criterion(logits, labels, metadata=batch_meta, features=features)

    Parameters
    ----------
    base_criterion:
        The "main" supervised loss (e.g. `nn.CrossEntropyLoss`).
    lambda_contrastive:
        Weight applied to the contrastive distance-matching component.
    feature_norm:
        If True, L2-normalise the features before computing pairwise distances.
    metadata_key / metadata_distance / metadata_weights:
        Controls how metadata is fetched, transformed and scaled before the
        contrastive distances are computed.
    feature_layer:
        Kept for backwards compatibility; no longer used because features are
        passed explicitly.
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        lambda_contrastive: float = 0.1,
        pos_threshold: float = 0.1,
        neg_threshold: float = 0.5,
        margin: float = 1.0,
        feature_norm: bool = True,
        metadata_key: Optional[str] = None,
        metadata_distance: str = 'l2',  # 'l2' or 'cosine'
        metadata_weights: Optional[Sequence[float]] = None,
        feature_layer: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Store the supervised criterion and the hyper-parameters that shape the
        # contrastive term.
        self.base = base_criterion
        self.lambda_contrastive = float(lambda_contrastive)
        self.pos_threshold = float(pos_threshold)
        self.neg_threshold = float(neg_threshold)
        self.margin = float(margin)
        self.feature_norm = feature_norm
        self.metadata_key = metadata_key
        self.metadata_distance = metadata_distance
        self.feature_layer = feature_layer  # kept for backwards compatibility

        # Optional per-dimension scaling for metadata. Storing the weights as a
        # buffer allows the tensor to move with the loss module across devices.
        if metadata_weights is not None:
            weights = torch.as_tensor(metadata_weights, dtype=torch.float32)
            if weights.ndim != 1:
                raise ValueError('metadata_weights must be a 1D sequence')
            self.register_buffer('_metadata_weights', weights)
        else:
            self.register_buffer('_metadata_weights', torch.tensor([], dtype=torch.float32))

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        metadata: Optional[Dict[str, Any] | torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute total loss = base loss + lambda * contrastive loss."""

        # Reset the accounting dictionary that is used downstream for logging.
        self.last_components = {
            'base': 0.0,
            'contrastive': 0.0,
            'total': 0.0,
            'lambda_contrastive': float(self.lambda_contrastive),
            'pos_pairs': 0,
            'neg_pairs': 0,
            'pos_term': 0.0,
            'neg_term': 0.0,
        }

        # --- Base supervised loss -------------------------------------------------
        # Guard against NaNs/infs in the logits and compute the supervised loss.
        if not torch.isfinite(outputs).all():
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
        base_loss = self.base(outputs, targets)
        if not torch.isfinite(base_loss):
            # If the base loss is invalid we abort early; returning zero prevents
            # gradients from going haywire.
            total = torch.zeros((), device=outputs.device, dtype=outputs.dtype)
            return total

        # If the contrastive term is disabled (lambda=0) or we do not have
        # metadata/features, we simply return the base loss.
        if self.lambda_contrastive == 0.0 or metadata is None or features is None:
            total = base_loss
            self.last_components.update({
                'base': float(base_loss.detach().item()),
                'total': float(total.detach().item()),
            })
            return total

        # --- Metadata processing --------------------------------------------------
        m = self._resolve_metadata(metadata, device=outputs.device, dtype=outputs.dtype)
        if m is None:
            # No metadata available -> revert to base loss only.
            total = base_loss
            self.last_components.update({
                'base': float(base_loss.detach().item()),
                'total': float(total.detach().item()),
            })
            return total

        # Apply optional per-dimension weights so the user can emphasise or
        # de-emphasise certain metadata components.
        if self._metadata_weights.numel() > 0:
            if m.size(1) != self._metadata_weights.numel():
                raise ValueError(
                    f"metadata_weights length {self._metadata_weights.numel()} does not match metadata dimension {m.size(1)}"
                )
            m = m * self._metadata_weights.to(device=m.device, dtype=m.dtype)

        # --- Feature processing ---------------------------------------------------
        # Ensure the feature tensor is 2-D (B × D) and optionally L2-normalised so
        # that Euclidean distance corresponds to cosine distance.
        z = features
        if not torch.is_tensor(z):
            z = torch.as_tensor(z, device=outputs.device, dtype=outputs.dtype)
        else:
            z = z.to(device=outputs.device, dtype=outputs.dtype)
        if z.dim() > 2:
            z = z.view(z.size(0), -1)
        if self.feature_norm:
            z = F.normalize(z, p=2, dim=1, eps=1e-8)
        if not torch.isfinite(z).all():
            total = base_loss
            self.last_components.update({
                'base': float(base_loss.detach().item()),
                'total': float(total.detach().item()),
            })
            return total

        # --- Pairwise distances ---------------------------------------------------
        d_feat = self._pairwise_euclidean(z)
        d_meta = self._pairwise_distance_metadata(m)
        if not torch.isfinite(d_meta).all():
            total = base_loss
            self.last_components.update({
                'base': float(base_loss.detach().item()),
                'total': float(total.detach().item()),
            })
            return total

        # Convert distances to a comparable scale and minimise their difference.
        scale = d_meta.max().clamp(min=1e-6)
        target = d_meta / scale
        d_feat_norm = d_feat / scale

        contrastive_loss = (d_feat_norm - target).pow(2).mean()
        if not torch.isfinite(contrastive_loss):
            total = base_loss
            self.last_components.update({
                'base': float(base_loss.detach().item()),
                'total': float(total.detach().item()),
            })
            return total

        total = base_loss + self.lambda_contrastive * contrastive_loss
        self.last_components.update({
            'base': float(base_loss.detach().item()),
            'contrastive': float(contrastive_loss.detach().item()),
            'total': float(total.detach().item()),
        })
        return total

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _resolve_metadata(
        self,
        metadata: Dict[str, Any] | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Convert metadata payload (dict or tensor) to a dense tensor."""
        if isinstance(metadata, dict):
            if self.metadata_key is None:
                if len(metadata) != 1:
                    return None
                m = next(iter(metadata.values()))
            else:
                if self.metadata_key not in metadata:
                    return None
                m = metadata[self.metadata_key]
        else:
            m = metadata

        m = torch.as_tensor(m, device=device, dtype=dtype)
        if m.dim() == 1:
            m = m.unsqueeze(1)
        return m

    @staticmethod
    def _pairwise_euclidean(z: torch.Tensor) -> torch.Tensor:
        """Return pairwise Euclidean distances for a batch of vectors."""
        z_norm2 = (z ** 2).sum(dim=1, keepdim=True)
        d2 = z_norm2 + z_norm2.t() - 2.0 * (z @ z.t())
        return torch.clamp(d2, min=0.0).sqrt()

    def _pairwise_distance_metadata(self, m: torch.Tensor) -> torch.Tensor:
        """Distance function applied to metadata (L2 or cosine)."""
        if self.metadata_distance == 'l2':
            return self._pairwise_euclidean(m)
        if self.metadata_distance == 'cosine':
            m_n = F.normalize(m, p=2, dim=1, eps=1e-8)
            sim = m_n @ m_n.t()
            return 1.0 - sim
        raise ValueError("metadata_distance must be 'l2' or 'cosine'")
