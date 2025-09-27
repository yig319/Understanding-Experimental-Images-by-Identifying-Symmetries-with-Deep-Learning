"""
Contrastive regularization utilities.

This module provides `ContrastiveRegularizedLoss`, a wrapper around a standard
supervised loss (e.g., CrossEntropyLoss) that adds a contrastive term computed
from intermediate model features and per-sample metadata.

High-level:
- Base loss: any nn.Module taking `(outputs, targets)` returning a scalar loss.
- Contrastive term: encourages feature distances to match metadata distances.
- Total loss: `L_total = L_base + lambda_contrastive * L_contrastive`.

Key properties:
- Pluggable: pass any model and specify `feature_layer` to hook features.
- Vectorized: computes all pairwise distances in-batch (O(B^2)).
- Flexible metadata: either a dict with a chosen key or a tensor per sample.
- Robustness: guards against NaNs/Infs and handles missing metadata gracefully.
- Logging: exposes a `last_components` dict after each forward for monitoring.

Usage tips:
- Start with `lambda_contrastive = 0.0` to verify base training, then increase.
- Consider normalizing features via `feature_norm=True` (default) for stability.
- Tune `pos_threshold`, `neg_threshold`, and `margin` to your metadata scale.
- Batch size matters: O(B^2) cost; consider smaller batches or subsampling when needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class ContrastiveRegularizedLoss(nn.Module):
    """
    Wraps a base criterion (e.g., CrossEntropyLoss) and adds a contrastive
    regularization term that encourages feature distances to reflect provided
    metadata distances.

    Algorithm
    - Register a forward hook on `feature_layer` of the model to capture a feature
      tensor `f in R^{B x D}` (flattened if necessary).
    - Normalize features if `feature_norm=True` for cosine-like geometry.
    - Build pairwise feature distances `d_feat(i,j) = ||f_i - f_j||_2`.
    - Build pairwise metadata distances `d_meta` (L2 or cosine distance over `m`).
    - Create masks for positive and negative pairs based on thresholds over `d_meta`.
    - Contrastive loss:
        L_pos = mean_{(i,j) in pos} ||f_i - f_j||^2
        L_neg = mean_{(i,k) in neg} [max(0, margin - ||f_i - f_k||)]^2
        L_contrastive = L_pos + L_neg
    - Total loss: `L = L_base + lambda_contrastive * L_contrastive`.

    Notes on thresholds and margin
    - `pos_threshold`: samples with metadata distance <= this are pulled together.
    - `neg_threshold`: samples with metadata distance >= this are pushed apart.
    - `margin`: radius to enforce separation for negatives. If features are L2
      normalized, a margin in [0.5, 1.5] is often a good starting point.

    Metadata handling
    - If `metadata_key` is provided, expects a dict and uses that key.
    - If not, accepts either a single-key dict or a tensor with shape [B, M] or [B].
    - Distance metric over metadata can be 'l2' or 'cosine', controlled via
      `metadata_distance`.

    Parallelism
    - Supports `nn.DataParallel`/`nn.DistributedDataParallel` by concatenating
      feature chunks from per-replica hooks in forward.

    Logging
    - After forward, `self.last_components` stores a plain dict with base,
      contrastive, and total losses, along with pair counts and per-term values.
      This is used by the Trainer to print/log loss breakdown.

    Complexity
    - O(B^2 D) memory/time due to pairwise distance matrices. Use smaller batches
      or subsample pairs if necessary for very large B.
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        model: nn.Module,
        lambda_contrastive: float = 0.1,
        feature_layer: str = 'avgpool',
        pos_threshold: float = 0.1,
        neg_threshold: float = 0.5,
        margin: float = 1.0,
        feature_norm: bool = True,
        metadata_key: Optional[str] = None,
        metadata_distance: str = 'l2',  # 'l2' or 'cosine'
    ) -> None:
        super().__init__()
        self.base = base_criterion
        self.lambda_contrastive = float(lambda_contrastive)
        self.pos_threshold = float(pos_threshold)
        self.neg_threshold = float(neg_threshold)
        self.margin = float(margin)
        self.feature_norm = feature_norm
        self.metadata_key = metadata_key
        self.metadata_distance = metadata_distance

        # Register a forward hook on the specified feature layer
        self._features = None
        self._feature_chunks = []  # collects per-replica feature tensors (for DataParallel)
        real_model = model.module if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)) else model
        if not hasattr(real_model, feature_layer):
            raise AttributeError(f"Model has no layer '{feature_layer}'.")
        layer = getattr(real_model, feature_layer)
        layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: nn.Module, inputs, output):
        # For ResNet avgpool: output shape [B, C, 1, 1]; flatten to [B, C]
        if isinstance(output, torch.Tensor) and output.dim() == 4 and output.size(-1) == 1 and output.size(-2) == 1:
            feat = torch.flatten(output, 1)
        elif isinstance(output, torch.Tensor) and output.dim() == 2:
            feat = output
        else:
            # Fallback: flatten last dims
            feat = torch.flatten(output, 1)

        # Accumulate chunk features to support DataParallel (one hook call per replica)
        self._feature_chunks.append(feat)
        # Also keep the last features for non-parallel case backward compatibility
        self._features = feat

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Compute total loss and capture components for logging.

        Parameters
        - outputs: model logits or predictions consumed by base criterion
        - targets: ground-truth labels for base criterion
        - metadata: optional; either a dict or tensor supplying per-sample
                    metadata vectors used to construct `d_meta`.

        Returns
        - Scalar tensor total loss.

        Side effects
        - Populates `self.last_components` with floats for logging purposes:
          {'base', 'contrastive', 'total', 'lambda_contrastive',
           'pos_pairs', 'neg_pairs', 'pos_term', 'neg_term'}
        """
        # Reset last-components snapshot for logging/inspection
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

        # Compute base loss robustly (clean any non-finite logits first)
        if not torch.isfinite(outputs).all():
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
        base_loss = self.base(outputs, targets)
        if not torch.isfinite(base_loss):
            # Bail out if CE is not finite; return zero to avoid poisoning optimizer
            total = torch.zeros((), device=outputs.device, dtype=outputs.dtype)
            # snapshot
            self.last_components.update({
                'base': 0.0,
                'contrastive': 0.0,
                'total': 0.0,
            })
            return total
        if self.lambda_contrastive == 0.0:
            total = base_loss
            self.last_components.update({
                'base': float(base_loss.detach().item()),
                'contrastive': 0.0,
                'total': float(total.detach().item()),
            })
            return total

        # If no features captured or no metadata, return base loss
        # Concatenate feature chunks collected by hooks (handles DataParallel)
        feats = None
        if len(self._feature_chunks) > 0:
            # Move all chunks to outputs.device for concatenation
            target_device = outputs.device
            chunks = [c.to(target_device) for c in self._feature_chunks]
            feats = torch.cat(chunks, dim=0)
        elif self._features is not None:
            feats = self._features.to(outputs.device)

        # Clear chunks for next forward pass
        self._feature_chunks = []

        if feats is None or metadata is None:
            total = base_loss
            self.last_components.update({
                'base': float(base_loss.detach().item()),
                'contrastive': 0.0,
                'total': float(total.detach().item()),
            })
            return total

        # Resolve metadata vector per-sample
        if self.metadata_key is None:
            # If only one key provided in dict, use it; else skip reg
            if isinstance(metadata, dict) and len(metadata) == 1:
                m = next(iter(metadata.values()))
            elif isinstance(metadata, torch.Tensor):
                m = metadata
            else:
                return base_loss
        else:
            if not isinstance(metadata, dict) or self.metadata_key not in metadata:
                return base_loss
            m = metadata[self.metadata_key]

        m = torch.as_tensor(m, device=feats.device, dtype=feats.dtype)
        if m.dim() == 1:
            m = m.unsqueeze(1)  # [B, 1]

        # Align feature and metadata batch sizes to outputs batch size
        B_out = outputs.size(0)
        if feats.size(0) != B_out:
            # Slice to smallest common size to avoid mismatches
            Bc = min(B_out, feats.size(0))
            feats = feats[:Bc]
            m = m[:Bc]
        if m.size(0) != B_out:
            # Ensure m matches feats length
            Bc = min(B_out, m.size(0), feats.size(0))
            feats = feats[:Bc]
            m = m[:Bc]

        # Normalize features if requested (default True). This often improves
        # stability and makes margin scales more intuitive.
        z = feats
        if self.feature_norm:
            z = F.normalize(z, p=2, dim=1, eps=1e-8)
        # Early bail if features contain non-finite values
        if not torch.isfinite(z).all():
            return base_loss

        # Pairwise feature distances
        # d_ij = ||z_i - z_j||_2
        # Efficient computation via (a-b)^2 = |a|^2 + |b|^2 - 2 a.b
        z_norm2 = (z ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        d2 = z_norm2 + z_norm2.t() - 2.0 * (z @ z.t())
        d_feat = torch.clamp(d2, min=0.0).sqrt()  # [B, B]

        # Pairwise metadata distance matrix
        if self.metadata_distance == 'l2':
            m_norm2 = (m ** 2).sum(dim=1, keepdim=True)
            dm2 = m_norm2 + m_norm2.t() - 2.0 * (m @ m.t())
            d_meta = torch.clamp(dm2, min=0.0).sqrt()
        elif self.metadata_distance == 'cosine':
            m_n = F.normalize(m, p=2, dim=1, eps=1e-8)
            sim = m_n @ m_n.t()
            d_meta = 1.0 - sim  # cosine distance in [0, 2]
        else:
            raise ValueError("metadata_distance must be 'l2' or 'cosine'")

        # If metadata contains non-finite values, skip contrastive term
        if not torch.isfinite(d_meta).all():
            total = base_loss
            self.last_components.update({
                'base': float(base_loss.detach().item()),
                'contrastive': 0.0,
                'total': float(total.detach().item()),
            })
            return total

        # Masks for positive/negative pairs (exclude diagonal)
        B = d_meta.size(0)
        device = d_meta.device
        eye = torch.eye(B, device=device, dtype=torch.bool)
        pos_mask = (d_meta <= self.pos_threshold) & (~eye)
        neg_mask = (d_meta >= self.neg_threshold) & (~eye)

        # Positive loss: pull close
        if pos_mask.any():
            pos_d = d_feat[pos_mask]
            pos_loss = (pos_d ** 2).mean()
        else:
            pos_loss = torch.zeros((), device=device, dtype=z.dtype)

        # Negative loss: push apart with margin
        if neg_mask.any():
            neg_d = d_feat[neg_mask]
            neg_loss = F.relu(self.margin - neg_d).pow(2).mean()
        else:
            neg_loss = torch.zeros((), device=device, dtype=z.dtype)

        contrastive_loss = pos_loss + neg_loss
        # Guard against non-finite contrastive loss
        if not torch.isfinite(contrastive_loss):
            total = base_loss
            self.last_components.update({
                'base': float(base_loss.detach().item()),
                'contrastive': 0.0,
                'total': float(total.detach().item()),
                'pos_pairs': int(pos_mask.sum().item()),
                'neg_pairs': int(neg_mask.sum().item()),
                'pos_term': float(pos_loss.detach().item()) if torch.isfinite(pos_loss) else 0.0,
                'neg_term': float(neg_loss.detach().item()) if torch.isfinite(neg_loss) else 0.0,
            })
            return total

        total = base_loss + self.lambda_contrastive * contrastive_loss

        # Snapshot components for logging/inspection (used by Trainer)
        self.last_components.update({
            'base': float(base_loss.detach().item()),
            'contrastive': float(contrastive_loss.detach().item()),
            'total': float(total.detach().item()),
            'pos_pairs': int(pos_mask.sum().item()),
            'neg_pairs': int(neg_mask.sum().item()),
            'pos_term': float(pos_loss.detach().item()),
            'neg_term': float(neg_loss.detach().item()),
        })

        return total
