from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from src.models.answer_decoder import AnswerDecoder
from src.models.question_aware import QuestionAwareFeatureDistilling
from src.models.text_encoder import TextEncoder
from src.models.vision_encoder import VisionTransformerEncoder


class MAVISCrossModalFusion(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        num_answers: int = 7500,
        image_size: int = 224,
        patch_size: int = 16,
        visual_dim: int = 768,
        text_embed_dim: int = 300,
        hidden_dim: int = 512,
        vision_depth: int = 14,
        vision_heads: int = 12,
        text_depth: int = 2,
        text_heads: int = 4,
        fusion_heads: int = 8,
        max_question_length: int = 32,
        dropout:  float = 0.1,
        pad_token_id: int = 0,
        alpha: float = 0.5,
        vision_mlp_ratio: float = 4.0,
        vision_dropout: float | None = None,
        text_dropout: float | None = None,
        fusion_dropout: float | None = None,
        answer_hidden_dim: int | None = None,
        answer_dropout: float | None = None,
    ):
        super().__init__()
        if hidden_dim % fusion_heads != 0:
            raise ValueError("hidden_dim must be divisible by fusion_heads.")
        if num_answers <= 0:
            raise ValueError("num_answers must be positive.")

        self.pad_token_id = pad_token_id
        vision_dropout = dropout if vision_dropout is None else vision_dropout
        text_dropout = dropout if text_dropout is None else text_dropout
        fusion_dropout = dropout if fusion_dropout is None else fusion_dropout
        answer_hidden_dim = hidden_dim if answer_hidden_dim is None else answer_hidden_dim
        answer_dropout = dropout if answer_dropout is None else answer_dropout
        if answer_hidden_dim != hidden_dim:
            raise ValueError(
                "answer_hidden_dim must equal hidden_dim because the fused joint feature "
                "contains four hidden_dim vectors."
            )
        self.vision_encoder = VisionTransformerEncoder(
            image_size=image_size,
            patch_size=patch_size,
            dim=visual_dim,
            depth=vision_depth,
            num_heads=vision_heads,
            mlp_ratio=vision_mlp_ratio,
            dropout=vision_dropout,
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=hidden_dim,
            num_heads=text_heads,
            num_layers=text_depth,
            dropout=text_dropout,
            padding_idx=pad_token_id,
            max_length=max_question_length,
        )
        self.cross_modal_fusion = QuestionAwareFeatureDistilling(
            visual_dim=visual_dim,
            question_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
            alpha=alpha
        )

        # Pool visual and question sequences after question-aware interaction.
        # Concatenation preserves both modalities for answer prediction.
        self.answer_decoder = AnswerDecoder(
            hidden_dim=answer_hidden_dim,
            num_answers=num_answers,
            dropout=answer_dropout,
        )
    @property
    def answer_head(self):
        """Backward-compatible access to the answer classification layers."""
        return self.answer_decoder.decoder
    @staticmethod
    def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        images: torch.Tensor,
        question_ids: torch.Tensor,
        return_attention: bool = True,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, 3, H, W].")
        if question_ids.ndim != 2:
            raise ValueError("question_ids must have shape [B, L].")
        if images.shape[0] != question_ids.shape[0]:
            raise ValueError("Image and question batch sizes do not match.")

        question_mask = question_ids.ne(self.pad_token_id)
        if not question_mask.any(dim=1).all():
            raise ValueError("Each question must contain at least one non-padding token.")

        visual_tokens = self.vision_encoder(images)  # [B, N+1, visual_dim]
        question_tokens = self.text_encoder(question_ids, return_tokens=True)

        # The vision encoder prepends a CLS token, which is a valid visual token.
        visual_mask = torch.ones(
            visual_tokens.shape[:2],
            dtype=torch.bool,
            device=visual_tokens.device,
        )

        fused = self.cross_modal_fusion(
            visual_features=visual_tokens,
            question_features=question_tokens,
            question_mask=question_mask,
            visual_mask=visual_mask,
        )

        v_tokens = fused["image_tokens"]
        q_tokens = fused["question_tokens"]

        # Keep padded question-token features zero after QAFD fusion.
        q_tokens = q_tokens.masked_fill(
            ~question_mask.unsqueeze(-1), 0.0
        )

        v_pooled = v_tokens.mean(dim=1)
        q_pooled = self.masked_mean(q_tokens, question_mask)

        joint = torch.cat(
            [v_pooled, q_pooled, v_pooled * q_pooled, torch.abs(v_pooled - q_pooled)],
            dim=-1,
        )
        logits = self.answer_decoder(joint)

        output = {
            "logits": logits,
            "image_tokens": v_tokens,
            "question_tokens": q_tokens,
            "relevance_scores": fused["relevance_scores"],
        }
        if return_attention:
            output["attention_map"] = fused["attention_map"]
        return output
