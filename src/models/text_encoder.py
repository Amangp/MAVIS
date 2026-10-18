
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        hidden_dim=512,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, ques):
        x = self.embedding(ques)

        x = self.encoder(x)

        x = x.mean(dim=1)

        x = self.proj(x)

        return x
