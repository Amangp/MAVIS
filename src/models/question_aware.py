
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestionAwareFeatureDistilling(nn.Module):
    def __init__(
        self,
        visual_dim: int = 768,
        question_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.5,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        self.alpha = alpha

        # Project visual and question features.
        self.visual_projection = nn.Linear(
            visual_dim, hidden_dim
        )
        self.question_projection = nn.Linear(
            question_dim, hidden_dim
        )

        # Question self-attention.
        self.question_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.question_norm = nn.LayerNorm(hidden_dim)
        self.question_dropout = nn.Dropout(dropout)

        # Question-to-image cross-attention.
        self.question_to_image_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_dropout = nn.Dropout(dropout)
        self.visual_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        question_features: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        if not torch.isfinite(visual_features).all():
            raise ValueError(
                "visual_features contains NaN or infinity."
            )

        if not torch.isfinite(question_features).all():
            raise ValueError(
                "question_features contains NaN or infinity."
            )

        # 1. Project features into a shared space.
        image_tokens = self.visual_projection(visual_features)
        question_tokens = self.question_projection(
            question_features
        )

        question_padding_mask = (
            ~question_mask.bool()
            if question_mask is not None
            else None
        )

        visual_padding_mask = (
            ~visual_mask.bool()
            if visual_mask is not None
            else None
        )

        # 2. Contextualize question tokens.
        question_context, _ = self.question_self_attention(
            question_tokens,
            question_tokens,
            question_tokens,
            key_padding_mask=question_padding_mask,
            need_weights=False,
        )

        question_tokens = self.question_norm(
            question_tokens
            + self.question_dropout(question_context)
        )

        # 3. Let question tokens attend to image patches.
        question_visual_context, attention_map = (
            self.question_to_image_attention(
                question_tokens,
                image_tokens,
                image_tokens,
                key_padding_mask=visual_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
        )

        question_tokens = self.cross_norm(
            question_tokens
            + self.cross_dropout(question_visual_context)
        )

        # 4. Token-wise relevance: maximum attention per patch.
        token_relevance = attention_map.max(dim=1).values

        token_relevance = token_relevance / (
            token_relevance.amax(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        )

        # 5. Sentence-wise relevance using cosine similarity.
        if question_mask is not None:
            pooled_question = question_tokens.masked_fill(
                ~question_mask.bool().unsqueeze(-1),
                float("-inf"),
            ).max(dim=1).values

            pooled_question = torch.nan_to_num(
                pooled_question,
                nan=0.0,
                neginf=0.0,
                posinf=0.0,
            )
        else:
            pooled_question = question_tokens.max(dim=1).values

        sentence_relevance = F.cosine_similarity(
            image_tokens,
            pooled_question.unsqueeze(1),
            dim=-1,
        )

        sentence_relevance = (
            sentence_relevance + 1.0
        ) / 2.0

        # Ignore padded image patches.
        if visual_mask is not None:
            valid_patches = visual_mask.bool()

            token_relevance = token_relevance.masked_fill(
                ~valid_patches, 0.0
            )
            sentence_relevance = sentence_relevance.masked_fill(
                ~valid_patches, 0.0
            )

        # 6. Combine token-wise and sentence-wise relevance.
        relevance_scores = (
            token_relevance + sentence_relevance
        ) / 2.0

        relevance_scores = relevance_scores.clamp(0.0, 1.0)

        # 7. Reweight features and retain original information.
        weighted_visual = (
            image_tokens * relevance_scores.unsqueeze(-1)
        )

        image_tokens = (
            self.alpha * image_tokens
            + (1.0 - self.alpha) * weighted_visual
        )

        image_tokens = self.visual_norm(image_tokens)

        if visual_mask is not None:
            image_tokens = image_tokens * (
                visual_mask.unsqueeze(-1).to(image_tokens.dtype)
            )

        return {
            "image_tokens": image_tokens,
            "question_tokens": question_tokens,
            "attention_map": attention_map,
            "relevance_scores": relevance_scores,
        }


QuestionAware = QuestionAwareFeatureDistilling
