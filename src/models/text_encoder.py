import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        padding_idx: int = 0,
        max_length: int = 32,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if vocab_size <= 0 or max_length <= 0:
            raise ValueError("vocab_size and max_length must be positive.")

        self.padding_idx = padding_idx
        self.max_length = max_length
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, question_ids: torch.Tensor, return_tokens: bool = False):
        if question_ids.ndim != 2:
            raise ValueError("question_ids must have shape [B, L].")
        batch_size, seq_len = question_ids.shape
        if seq_len > self.max_length:
            raise ValueError(
                f"Question length {seq_len} exceeds max_length={self.max_length}."
            )

        positions = torch.arange(
            seq_len, device=question_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        padding_mask = question_ids.eq(self.padding_idx)  # True means padding.
        x = self.embedding(question_ids) + self.position_embedding(positions)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.output_norm(self.proj(self.norm(x)))

        # Transformer output at padded positions can be nonzero due to biases.
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        if return_tokens:
            return x

        valid = (~padding_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled
