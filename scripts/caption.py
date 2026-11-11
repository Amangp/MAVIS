"""
QACap caption-only inference.

This script performs:
    image + question -> PET-conditioned QACap caption

It deliberately does not call the LLM. Use answer.py for the complete
caption-to-answer pipeline.

Paper fidelity:
- Rebuilds the model from the architecture settings saved in the checkpoint.
- Uses the BLIP processor owned by QACap, matching the training dataset.
- Calls QACap.generate_caption with the same PET question prefix.
- Does not invent a default generation length: the paper's main text does not
  specify max_new_tokens. Supply --max-new-tokens or set
  inference.max_new_tokens in the saved configuration.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image, ImageOps


# Allow `python scripts/caption.py ...` from the repository root without
# requiring an editable install. If qacap is installed as a package, this
# still resolves to the active package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (PROJECT_ROOT / "qacap").is_dir() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_CONFIG_KEYS = {
    "blip_name",
    "bert_name",
    "alpha",
    "num_heads",
    "qafd_layers",
    "max_question_length",
}
REQUIRED_MODEL_CONFIG_KEYS = {"alpha", "num_heads", "qafd_layers"}


def _mapping(value: Any, section_name: str) -> dict[str, Any]:
    """Convert a saved config section into a plain dictionary."""
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(
        f"Checkpoint config section '{section_name}' must be a mapping, "
        f"got {type(value).__name__}."
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted project checkpoint and validate its required fields."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path.resolve()}")

    # The project checkpoint may contain Python/NumPy RNG state and a config
    # object, so weights_only=False is necessary. Only load trusted files.
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with PyTorch versions predating weights_only.
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a dictionary.")
    if not isinstance(checkpoint.get("model"), dict):
        raise KeyError("Checkpoint is missing its model state dictionary.")
    if checkpoint.get("config") is None:
        raise KeyError(
            "Checkpoint does not contain the training configuration. "
            "A QACap model cannot be reconstructed safely without it."
        )
    return checkpoint


def _resolve_model_config(
    checkpoint_config: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve constructor kwargs, data settings and inference settings."""
    root = _mapping(checkpoint_config, "config")
    if "model" not in root:
        raise KeyError("Saved configuration is missing the 'model' section.")
    model_config = _mapping(root["model"], "model")
    data_config = _mapping(root.get("data", {}), "data")
    inference_config = _mapping(root.get("inference", {}), "inference")

    missing = sorted(REQUIRED_MODEL_CONFIG_KEYS - model_config.keys())
    if missing:
        raise KeyError(
            "Saved model configuration is missing required QAFD settings: "
            f"{missing}. Do not guess these values; verify them against the "
            "paper supplementary material or the recorded experiment config."
        )

    model_kwargs = {
        key: model_config[key]
        for key in MODEL_CONFIG_KEYS
        if key in model_config
    }

    legacy_keys = sorted(set(model_config) - MODEL_CONFIG_KEYS)
    if legacy_keys:
        # These may come from the earlier prototype (e.g. hidden_dim, dropout,
        # freeze flags). The updated paper-aligned QACap constructor does not
        # accept them. Strict state-dict loading below will still detect an
        # incompatible checkpoint architecture.
        import warnings

        warnings.warn(
            "Ignoring legacy/non-constructor model config keys: "
            f"{legacy_keys}. Ensure the checkpoint was trained with the "
            "current QACap architecture; state-dict loading is strict.",
            RuntimeWarning,
            stacklevel=2,
        )

    return model_kwargs, data_config, inference_config


def _resolve_generation_length(
    cli_value: Optional[int],
    inference_config: Mapping[str, Any],
) -> int:
    """Resolve an explicit inference length; never assume an undocumented value."""
    value = cli_value
    if value is None:
        value = inference_config.get("max_new_tokens")
    if value is None:
        raise ValueError(
            "Maximum caption generation length is not specified in the paper's "
            "main text. Supply --max-new-tokens or add "
            "inference.max_new_tokens to the saved configuration."
        )
    value = int(value)
    if value < 1:
        raise ValueError("max_new_tokens must be a positive integer.")
    return value


def _prepare_image(
    image_path: Path,
    processor: Any,
    image_size: Optional[int],
) -> torch.Tensor:
    """Preprocess the image using the same BLIP processor as QACapDataset."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path.resolve()}")

    if image_size is not None:
        image_size = int(image_size)
        if image_size < 1:
            raise ValueError("data.image_size must be a positive integer.")
        processor.image_processor.size = {
            "height": image_size,
            "width": image_size,
        }

    with Image.open(image_path) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")

    encoded = processor(images=image, return_tensors="pt")
    pixel_values = encoded.get("pixel_values")
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
        raise RuntimeError(
            "The BLIP processor did not return pixel_values with shape [B,C,H,W]."
        )
    if pixel_values.shape[0] != 1:
        raise RuntimeError(
            f"Expected one image, but processor returned batch size {pixel_values.shape[0]}."
        )
    return pixel_values


def generate_caption(
    checkpoint_path: str | Path,
    image_path: str | Path,
    question: str,
    *,
    device: str = "auto",
    max_new_tokens: Optional[int] = None,
) -> str:
    """Load QACap and produce one PET-conditioned caption."""
    checkpoint_path = Path(checkpoint_path)
    image_path = Path(image_path)

    question = question.strip() if isinstance(question, str) else ""
    if not question:
        raise ValueError("Question must be a non-empty string.")

    checkpoint = _load_checkpoint(checkpoint_path)
    model_kwargs, data_config, inference_config = _resolve_model_config(
        checkpoint["config"]
    )
    generation_length = _resolve_generation_length(
        max_new_tokens,
        inference_config,
    )

    if device == "auto":
        selected_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and check the GPU driver."
            )
        selected_device = torch.device("cuda")
    elif device == "cpu":
        selected_device = torch.device("cpu")
    else:
        raise ValueError("device must be one of: auto, cuda, cpu")

    # Import here so config validation can be used without loading Transformers.
    try:
        from qacap.model import QACap
    except ImportError as exc:
        raise ImportError(
            "Could not import qacap.model. Run this script from the project "
            "root and install the project's dependencies."
        ) from exc

    model = QACap(**model_kwargs)
    # Strict loading is intentional: an old prototype checkpoint with extra
    # projection/freeze modules is not the same model as this updated version.
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(selected_device)
    model.eval()

    # Reuse QACap's own BLIP processor rather than constructing a second,
    # potentially different processor instance.
    pixel_values = _prepare_image(
        image_path=image_path,
        processor=model.blip_processor,
        image_size=data_config.get("image_size"),
    ).to(selected_device)

    with torch.inference_mode():
        captions = model.generate_caption(
            pixel_values=pixel_values,
            questions=[question],
            max_new_tokens=generation_length,
        )

    if len(captions) != 1 or not isinstance(captions[0], str) or not captions[0].strip():
        raise RuntimeError("QACap did not produce a non-empty caption.")
    return captions[0].strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a question-aware caption using the trained QACap model."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a trained QACap checkpoint.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question that guides QACap caption generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of new caption tokens. Required unless "
            "inference.max_new_tokens is saved in the checkpoint config."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device (default: automatic CUDA detection).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional text file where the generated caption will be saved.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    caption = generate_caption(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        question=args.question,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    print(caption)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(caption + os.linesep, encoding="utf-8")


if __name__ == "__main__":
    main()
