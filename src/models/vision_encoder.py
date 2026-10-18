
import torch
import torch.nn as nn


class VisionTransformerEncoder(nn.Module):
    def __init__(
        
        self,
        image_size=224,
        patch_size=16,
        dim=768,
        depth=14,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, dim)
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                "Input must have shape [batch_size, 3, height, width]"
            )

        if x.shape[1] != 3:
            raise ValueError("Input must have 3 color channels")

        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            raise ValueError(
                f"Input images must be {self.image_size}x{self.image_size}"
            )

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)
        x = self.proj(x)

        return x
