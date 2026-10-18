from __future__ import annotations

import torch
import torch.nn as nn
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
class CaptionDecoder(nn.Module):
    def __init__(
        self,
        caption_vocab_size: int,
        visual_dim=config["model"]["vision"]["visual_dim"],
        hidden_dim=config["model"]["caption_decoder"]["hidden_dim"],
        num_heads=config["model"]["caption_decoder"]["num_heads"],
        num_layers=config["model"]["caption_decoder"]["num_layers"],
        max_length=config["data"]["max_caption_length"],
        dropout=config["model"]["caption_decoder"]["dropout"],
        pad_token_id=config["data"]["caption_special_token_ids"]["pad"],
        bos_token_id=config["data"]["caption_special_token_ids"]["bos"],
        eos_token_id=config["data"]["caption_special_token_ids"]["eos"],
    ) -> None:
        super().__init__()

        if caption_vocab_size <= 0:
            raise ValueError("caption_vocab_size must be positive.")
        if visual_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Feature dimensions must be positive.")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by positive num_heads.")
        if num_layers <= 0 or max_length <= 0:
            raise ValueError("num_layers and max_length must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        token_ids = {
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
        }
        for name, token_id in token_ids.items():
            if not isinstance(token_id, int):
                raise TypeError(f"{name} must be an integer.")
            if not 0 <= token_id < caption_vocab_size:
                raise ValueError(f"{name} must be within the caption vocabulary.")

        self.caption_vocab_size = caption_vocab_size
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        self.token_embedding = nn.Embedding(
            caption_vocab_size, hidden_dim, padding_idx=pad_token_id
        )
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.output_projection = nn.Linear(hidden_dim, caption_vocab_size)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        caption_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(visual_tokens, torch.Tensor):
            raise TypeError("visual_tokens must be a torch.Tensor.")
        if not isinstance(caption_input_ids, torch.Tensor):
            raise TypeError("caption_input_ids must be a torch.Tensor.")
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [B, N, visual_dim].")
        if caption_input_ids.ndim != 2:
            raise ValueError("caption_input_ids must have shape [B, T].")
        if visual_tokens.shape[0] == 0:
            raise ValueError("Batch size must be greater than zero.")
        if visual_tokens.shape[1] == 0:
            raise ValueError("At least one visual token is required.")
        if caption_input_ids.shape[0] == 0:
            raise ValueError("Batch size must be greater than zero.")
        if caption_input_ids.shape[1] == 0:
            raise ValueError("Caption input cannot be empty.")
        if visual_tokens.shape[0] != caption_input_ids.shape[0]:
            raise ValueError("Image and caption batch sizes do not match.")
        if visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                f"Expected visual dimension {self.visual_dim}, "
                f"received {visual_tokens.shape[-1]}."
            )
        if caption_input_ids.shape[1] > self.max_length:
            raise ValueError("Caption input exceeds max_length.")
        if not torch.is_floating_point(visual_tokens):
            raise TypeError("visual_tokens must have a floating-point dtype.")
        if caption_input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("Caption token IDs must use torch.int32 or torch.int64.")
        if visual_tokens.device != caption_input_ids.device:
            raise ValueError("Visual tokens and caption IDs must be on the same device.")
        if not torch.isfinite(visual_tokens).all():
            raise ValueError("Visual tokens contain NaN or infinity.")
        if ((caption_input_ids < 0) |
                (caption_input_ids >= self.caption_vocab_size)).any():
            raise ValueError("Caption token ID is outside the vocabulary.")
        if not caption_input_ids.ne(self.pad_token_id).any(dim=1).all():
            raise ValueError("Each caption input must contain a non-padding token.")

        batch_size, seq_len = caption_input_ids.shape
        memory = self.visual_projection(visual_tokens)

        positions = torch.arange(
            seq_len, device=caption_input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        target = self.token_embedding(caption_input_ids) + self.position_embedding(positions)

        # Boolean True means a future position is blocked.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool,
                       device=caption_input_ids.device),
            diagonal=1,
        )
        padding_mask = caption_input_ids.eq(self.pad_token_id)

        decoded = self.transformer_decoder(
            tgt=target,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )
        logits = self.output_projection(decoded)

        if not torch.isfinite(logits).all():
            raise FloatingPointError("Caption decoder produced non-finite logits.")
        return logits

    @torch.no_grad()
    def generate(
        self,
        visual_tokens: torch.Tensor,
        max_new_tokens: int = 30,
    ) -> torch.Tensor:
        """Greedy generation; output includes BOS and generated tokens."""
        if not isinstance(visual_tokens, torch.Tensor):
            raise TypeError("visual_tokens must be a torch.Tensor.")
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [B, N, visual_dim].")
        if visual_tokens.shape[0] == 0:
            raise ValueError("Batch size must be greater than zero.")
        if visual_tokens.shape[1] == 0:
            raise ValueError("At least one visual token is required.")
        if visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError("Unexpected visual feature dimension.")
        if not torch.is_floating_point(visual_tokens):
            raise TypeError("visual_tokens must have a floating-point dtype.")
        if not torch.isfinite(visual_tokens).all():
            raise ValueError("Visual tokens contain NaN or infinity.")
        if not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        # BOS consumes one decoder position.
        if max_new_tokens > self.max_length:
            raise ValueError("max_new_tokens cannot exceed max_length.")

        was_training = self.training
        self.eval()
        try:
            batch_size = visual_tokens.shape[0]
            generated = torch.full(
                (batch_size, 1),
                self.bos_token_id,
                dtype=torch.long,
                device=visual_tokens.device,
            )
            finished = torch.zeros(
                batch_size, dtype=torch.bool, device=visual_tokens.device
            )

            for _ in range(max_new_tokens):
                # The input sequence must fit in the positional embedding table.
                logits = self.forward(visual_tokens, generated)
                next_token = logits[:, -1, :].argmax(dim=-1)
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, self.eos_token_id),
                    next_token,
                )
                generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
                finished |= next_token.eq(self.eos_token_id)
                if finished.all():
                    break
            return generated
        finally:
            self.train(was_training)
