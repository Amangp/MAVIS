"""Losses for the CVPR 2025 QACap implementation.

The paper defines the joint objective as

    L = L_cap + lambda_JS * L_JS,

where L_cap is caption negative log-likelihood and L_JS is the Jensen-Shannon
Divergence between the QAFD and caption-generation regional attention maps.

For QARCL, Figure 5 shows each token-to-region attention map being max-pooled
over its query-token axis to obtain a vector over the shared image regions.
Those two region vectors are normalized as probability distributions before
computing JS divergence. This allows the question-token and caption-token
axes to have different lengths while comparing the same image regions.

This module deliberately does not supply a default lambda_JS: the paper says
its value is discussed in the supplementary material, so it must be provided
from a verified source/configuration rather than invented here.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def caption_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Caption negative log-likelihood for teacher-forced decoder outputs.

    Args:
        logits: Decoder vocabulary scores, shape ``[B, T, V]``.
        labels: Next-token targets aligned with decoder logits, shape ``[B, T]``.
            Prompt-only and padded positions must contain ``ignore_index``.
        ignore_index: Target value ignored by cross entropy. The paired
            ``data_updated.py`` uses ``-100`` for PET prompt and pad positions.
        label_smoothing: Explicit loss smoothing value; use 0.0 unless a
            paper/supplementary source specifies otherwise.

    Returns:
        Scalar mean cross-entropy loss over non-ignored target tokens.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B,T,V], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B,T], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "The first two logits dimensions must match labels: "
            f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
        )
    if not 0.0 <= float(label_smoothing) < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")

    flat_labels = labels.reshape(-1)
    valid_count = flat_labels.ne(int(ignore_index)).sum()
    if int(valid_count.item()) == 0:
        raise ValueError("All caption labels are ignored; cannot calculate caption NLL")

    # Compute cross entropy in FP32 for numerical stability under mixed precision.
    flat_logits = logits.float().reshape(-1, logits.size(-1))
    return F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=int(ignore_index),
        label_smoothing=float(label_smoothing),
    )


def _to_query_region_map(
    attention: torch.Tensor,
    *,
    query_mask: Optional[torch.Tensor] = None,
    name: str = "attention",
) -> torch.Tensor:
    """Convert attention weights to ``[B, query_tokens, image_regions]``.

    Accepted shapes:
      * ``[B, Q, V]``: attention averaged over heads already;
      * ``[B, H, Q, V]``: per-head maps, averaged over heads here;
      * ``[B, V]``: already region pooled.

    ``query_mask`` is a boolean/0-1 tensor ``[B, Q]`` where True/1 means a
    query position is valid and should participate in regional max pooling.
    """
    if not isinstance(attention, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if attention.ndim == 4:
        # The paper depicts one attention map per branch, rather than separate
        # maps per head. Average heads before the token-wise regional max.
        attention = attention.float().mean(dim=1)
    elif attention.ndim == 3:
        attention = attention.float()
    elif attention.ndim == 2:
        attention = attention.float().unsqueeze(1)
    else:
        raise ValueError(
            f"{name} must have rank 2, 3 or 4; got shape {tuple(attention.shape)}"
        )

    if attention.shape[-1] < 1:
        raise ValueError(f"{name} has no image-region positions")
    if not torch.isfinite(attention).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if (attention < -1e-7).any():
        raise ValueError(f"{name} contains negative values; expected attention weights")
    attention = attention.clamp_min(0.0)

    if query_mask is not None:
        if query_mask.ndim != 2:
            raise ValueError(
                f"{name} query_mask must have shape [B,Q], "
                f"got {tuple(query_mask.shape)}"
            )
        if attention.shape[0] != query_mask.shape[0] or attention.shape[1] != query_mask.shape[1]:
            raise ValueError(
                f"{name} query_mask shape {tuple(query_mask.shape)} does not match "
                f"attention query dimensions {tuple(attention.shape[:2])}"
            )
        valid_queries = query_mask.to(device=attention.device, dtype=torch.bool)
        if (~valid_queries).all(dim=1).any():
            raise ValueError(f"{name} has a sample with no valid query positions")
        attention = attention.masked_fill(~valid_queries.unsqueeze(-1), float("-inf"))

    # Figure 5 depicts max pooling over question/caption tokens to produce
    # region-wise attention weights on a common image-region axis.
    region_scores = attention.amax(dim=1)
    if not torch.isfinite(region_scores).all():
        raise ValueError(f"{name} produced non-finite regional scores")
    if (region_scores.sum(dim=-1) <= 0).any():
        raise ValueError(f"{name} produced an all-zero regional attention distribution")
    return region_scores


def _normalize_region_scores(scores: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize nonnegative region scores into probability distributions."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    scores = scores.clamp_min(0.0)
    denominator = scores.sum(dim=-1, keepdim=True)
    if (denominator <= eps).any():
        raise ValueError("Cannot normalize an all-zero regional attention map")
    return scores / denominator.clamp_min(eps)


def regional_attention_distribution(
    attention: torch.Tensor,
    *,
    query_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    name: str = "attention",
) -> torch.Tensor:
    """Create a normalized regional distribution from a QAFD/decoder map."""
    regional_scores = _to_query_region_map(
        attention, query_mask=query_mask, name=name
    )
    return _normalize_region_scores(regional_scores, eps=eps)


def js_divergence(
    attn_q: torch.Tensor,
    attn_c: torch.Tensor,
    eps: float = 1e-8,
    *,
    q_query_mask: Optional[torch.Tensor] = None,
    c_query_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Jensen-Shannon divergence between QAFD and caption regional maps.

    Args:
        attn_q: QAFD attention, normally ``[B, Lq, Lv]``.
        attn_c: Final caption-decoder cross-attention over image regions,
            normally ``[B, Lc, Lv]``.
        q_query_mask: Optional valid-query mask for QAFD map.
        c_query_mask: Optional valid-query mask for caption map. For QACap,
            pass ``labels.ne(-100)`` so PET/BOS-only decoder positions are not
            included in the caption-region pooling.
        eps: Numerical stability constant.

    Returns:
        Scalar batch-mean JS divergence.
    """
    p = regional_attention_distribution(
        attn_q, query_mask=q_query_mask, eps=eps, name="QAFD attention"
    )
    q = regional_attention_distribution(
        attn_c, query_mask=c_query_mask, eps=eps, name="caption attention"
    )
    if p.shape != q.shape:
        raise ValueError(
            "QARCL requires matching batch and image-region dimensions after "
            f"regional pooling; got QAFD {tuple(p.shape)} and caption {tuple(q.shape)}"
        )

    # Symmetric Jensen-Shannon divergence: 1/2 KL(P||M) + 1/2 KL(Q||M).
    # Inputs are probability distributions and remain differentiable.
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    midpoint = 0.5 * (p + q)
    kl_pm = (p * (p.log() - midpoint.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - midpoint.log())).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).mean()


def qacap_total_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    qafd_attention: torch.Tensor,
    caption_attention: torch.Tensor,
    *,
    lambda_js: float,
    caption_query_mask: Optional[torch.Tensor] = None,
    question_query_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Calculate the complete paper objective ``L_cap + lambda_JS * L_JS``.

    ``lambda_js`` is mandatory because the main paper delegates its value to
    the supplementary material. Do not use a guessed default for reproduction.

    If ``caption_query_mask`` is omitted, it is derived from ``labels`` by
    treating every non-ignored label position as a caption-generation query.
    With the paired data module, ignored positions correspond to the PET/BOS
    context and batch padding.

    Returns a dictionary with ``loss``, ``caption_loss`` and ``js_loss``.
    """
    if lambda_js < 0:
        raise ValueError(f"lambda_js must be nonnegative, got {lambda_js}")

    caption_loss = caption_nll(
        logits=logits,
        labels=labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    if caption_query_mask is None:
        caption_query_mask = labels.ne(int(ignore_index))
    js_loss = js_divergence(
        qafd_attention,
        caption_attention,
        q_query_mask=question_query_mask,
        c_query_mask=caption_query_mask,
    )
    total = caption_loss + float(lambda_js) * js_loss
    return {
        "loss": total,
        "caption_loss": caption_loss,
        "js_loss": js_loss,
    }


# A descriptive alias for training scripts that use a generic name.
compute_qacap_loss = qacap_total_loss
