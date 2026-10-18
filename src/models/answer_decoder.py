
from __future__ import annotations

import torch
import torch.nn as nn


class AnswerDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 512,
        num_answers: int = 7500,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_answers <= 0:
            raise ValueError("num_answers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = hidden_dim * 4
        self.num_answers = num_answers

        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_answers),
        )

    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        if not isinstance(fused_features, torch.Tensor):
            raise TypeError("fused_features must be a torch.Tensor.")

        if fused_features.ndim != 2:
            raise ValueError(
                "fused_features must have shape [B, input_dim]."
            )

        if fused_features.shape[0] == 0:
            raise ValueError("Batch size must be greater than zero.")

        if fused_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"received {fused_features.shape[-1]}."
            )

        if not torch.is_floating_point(fused_features):
            raise TypeError("fused_features must have a floating-point dtype.")

        if not torch.isfinite(fused_features).all():
            raise ValueError("fused_features contains NaN or infinity.")

        return self.decoder(fused_features)

