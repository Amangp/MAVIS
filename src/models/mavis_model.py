
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.caption_decoder import CaptionDecoder
from src.models.cross_modal_fusion import MAVISCrossModalFusion


class MAVIS(nn.Module):
    """
    MAVIS: Multi-task Visual Question Answering and Image Captioning.

    Shared component:
        VisionTransformerEncoder

    VQA branch:
        Shared ViT -> TextEncoder -> QuestionAwareFeatureDistilling
        -> AnswerDecoder

    Caption branch:
        Shared ViT -> CaptionDecoder
    """

    def __init__(
        self,
        config: dict[str, Any],
        question_vocab_size: int,
        answer_vocab_size: int,
        caption_vocab_size: int,
    ) -> None:
        super().__init__()

        if question_vocab_size <= 0:
            raise ValueError("question_vocab_size must be positive.")
        if answer_vocab_size <= 0:
            raise ValueError("answer_vocab_size must be positive.")
        if caption_vocab_size <= 0:
            raise ValueError("caption_vocab_size must be positive.")

        vision = config["model"]["vision"]
        text = config["model"]["text"]
        fusion = config["model"]["question_aware_fusion"]
        answer = config["model"]["answer_decoder"]
        caption = config["model"]["caption_decoder"]
        data = config["data"]

        # -------------------------------------------------
        # VQA branch
        # -------------------------------------------------
        # The existing MAVISCrossModalFusion creates the ViT,
        # TextEncoder, QuestionAwareFeatureDistilling and
        # AnswerDecoder using their existing implementations.
        self.vqa = MAVISCrossModalFusion(
            vocab_size=question_vocab_size,
            num_answers=answer_vocab_size,
            image_size=vision["image_size"],
            patch_size=vision["patch_size"],
            visual_dim=vision["visual_dim"],
            text_embed_dim=text["embedding_dim"],
            hidden_dim=fusion["hidden_dim"],
            vision_depth=vision["depth"],
            vision_heads=vision["num_heads"],
            text_depth=text["depth"],
            text_heads=text["num_heads"],
            fusion_heads=fusion["num_heads"],
            max_question_length=data["max_question_length"],
            dropout=text["dropout"],
            pad_token_id=data.get("pad_token_id", data["caption_special_token_ids"]["pad"]),
            alpha=fusion["alpha"],
            vision_mlp_ratio=vision["mlp_ratio"],
            vision_dropout=vision["dropout"],
            text_dropout=text["dropout"],
            fusion_dropout=fusion["dropout"],
            answer_hidden_dim=answer["hidden_dim"],
            answer_dropout=answer["dropout"],
        )

        # -------------------------------------------------
        # Shared vision encoder
        # -------------------------------------------------
        # Reuse the exact same ViT module for captioning.
        # Do not create a second vision encoder.
        self.vision_encoder = self.vqa.vision_encoder

        # -------------------------------------------------
        # Caption branch
        # -------------------------------------------------
        special_ids = data["caption_special_token_ids"]

        self.caption_decoder = CaptionDecoder(
            caption_vocab_size=caption_vocab_size,
            visual_dim=vision["visual_dim"],
            hidden_dim=caption["hidden_dim"],
            num_heads=caption["num_heads"],
            num_layers=caption["num_layers"],
            max_length=data["max_caption_length"],
            dropout=caption["dropout"],
            pad_token_id=special_ids["pad"],
            bos_token_id=special_ids["bos"],
            eos_token_id=special_ids["eos"],
        )

        self.num_answers = answer_vocab_size
        self.caption_vocab_size = caption_vocab_size
        self.config = config

    # -----------------------------------------------------
    # VQA forward pass
    # -----------------------------------------------------
    def forward_vqa(
        self,
        images: torch.Tensor,
        question_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]
            question_ids: [B, question_length]

        Returns:
            logits: [B, num_answers]
            image_tokens: [B, num_patches + 1, hidden_dim]
            question_tokens: [B, question_length, hidden_dim]
            relevance_scores: [B, num_patches + 1]
            attention_map: optional attention weights
        """
        return self.vqa(
            images,
            question_ids,
            return_attention=return_attention,
        )

    # -----------------------------------------------------
    # Caption training forward pass
    # -----------------------------------------------------
    def forward_caption(
        self,
        images: torch.Tensor,
        caption_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forced caption training.

        Args:
            images: [B, 3, H, W]
            caption_input_ids: [B, caption_length]

        Returns:
            Caption logits: [B, caption_length, caption_vocab_size]
        """
        visual_tokens = self.vision_encoder(images)

        return self.caption_decoder(
            visual_tokens,
            caption_input_ids,
        )

    # -----------------------------------------------------
    # Caption inference
    # -----------------------------------------------------
    @torch.no_grad()
    def generate_caption(
        self,
        images: torch.Tensor,
        max_new_tokens: int | None = None,
    ) -> torch.Tensor:
        """
        Generate captions using greedy decoding.

        Returns token IDs including BOS and generated tokens.
        """
        visual_tokens = self.vision_encoder(images)

        if max_new_tokens is None:
            max_new_tokens = int(
                self.config["model"]["caption_decoder"].get(
                    "max_new_tokens", self.caption_decoder.max_length - 1
                )
            )
        max_new_tokens = min(int(max_new_tokens), self.caption_decoder.max_length - 1)

        return self.caption_decoder.generate(
            visual_tokens,
            max_new_tokens=max_new_tokens,
        )

    # -----------------------------------------------------
    # Optional task-based forward interface
    # -----------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        task: str,
        question_ids: torch.Tensor | None = None,
        caption_input_ids: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Run one task at a time.

        task="vqa":
            Requires question_ids.

        task="caption":
            Requires caption_input_ids.

        Use generate_caption() separately for caption inference.
        """
        if task == "vqa":
            if question_ids is None:
                raise ValueError(
                    "question_ids are required when task='vqa'."
                )

            return self.forward_vqa(
                images,
                question_ids,
                return_attention=return_attention,
            )

        if task == "caption":
            if caption_input_ids is None:
                raise ValueError(
                    "caption_input_ids are required when task='caption'."
                )

            return self.forward_caption(
                images,
                caption_input_ids,
            )

        raise ValueError(
            f"Unsupported task: {task!r}. Use 'vqa' or 'caption'."
        )
