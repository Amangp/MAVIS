"""
Inference entry point for the CVPR 2025 QACap pipeline.

Pipeline:
    image + question
        -> QACap question-aware caption
        -> Claude 3.5 using Task Instruction / Context / Question prompt

This script loads a checkpoint produced by checkpoint.py. It expects the
checkpoint's `config` field to contain a `model` mapping with the required
QAFD settings (`alpha`, `num_heads`, `qafd_layers`). The paper's main text does
not state these numerical values, so they must come from the verified project
configuration rather than being invented here.

The paper does not state a maximum caption-generation length or an exact
Anthropic API snapshot/token budget. These may be supplied through the saved
inference config, CLI options, or the environment variables used by llm.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Allow `python scripts/answer.py` from the repository root without an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (PROJECT_ROOT / "qacap").is_dir() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Mapping, Optional

import torch
from PIL import Image, ImageOps


MODEL_ARGUMENTS = (
    "blip_name",
    "bert_name",
    "alpha",
    "num_heads",
    "qafd_layers",
    "qafd_ffn_ratio",
    "qafd_ffn_activation",
    "qafd_ffn_residual",
    "max_question_length",
)


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    """Validate a config section without silently accepting malformed values."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    # Support OmegaConf-like objects without importing OmegaConf.
    if hasattr(value, "items"):
        return dict(value.items())
    raise TypeError(f"Checkpoint config section '{name}' must be a mapping.")


def _load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint written by the project's checkpoint.py utility."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # checkpoint.py may save Python/NumPy RNG states and a config object.
    # weights_only=False is therefore necessary. Only load trusted local files.
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with PyTorch versions before the weights_only argument.
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint contents must be a dictionary.")
    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain the required 'model' weights.")
    if "config" not in checkpoint or checkpoint["config"] is None:
        raise KeyError(
            "Checkpoint has no saved configuration. A model cannot be rebuilt "
            "reliably without the architecture settings used during training."
        )
    return checkpoint


def _resolve_config(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return model, data and inference config sections."""
    root = _as_mapping(checkpoint["config"], "config")
    model_config = dict(_as_mapping(root.get("model"), "model"))
    data_config = dict(_as_mapping(root.get("data"), "data"))
    inference_config = dict(_as_mapping(root.get("inference"), "inference"))

    required = ("alpha", "num_heads", "qafd_layers")
    missing = [key for key in required if key not in model_config]
    if missing:
        raise KeyError(
            "Saved model config is missing required QACap architecture setting(s): "
            f"{missing}. Do not guess these values; verify them against the paper "
            "supplementary material or the experiment configuration."
        )

    # The current paper-faithful QACap constructor accepts these arguments.
    # Old prototype-only keys (e.g. hidden_dim/dropout/freeze flags) are not
    # forwarded, because the updated architecture intentionally removed them.
    model_kwargs = {
        key: model_config[key]
        for key in MODEL_ARGUMENTS
        if key in model_config
    }
    return model_kwargs, data_config, inference_config


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Check that the installed PyTorch build has CUDA support."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_image(
    image_path: Path,
    processor: Any,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    """Use the same BLIP image processor as the training dataset."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")

    if image_size is not None:
        size = int(image_size)
        if size < 1:
            raise ValueError("Configured data.image_size must be positive.")
        processor.image_processor.size = {"height": size, "width": size}

    encoded = processor(images=image, return_tensors="pt")
    pixel_values = encoded.get("pixel_values")
    if pixel_values is None or pixel_values.ndim != 4:
        raise RuntimeError("BLIP processor did not return [B, C, H, W] pixel_values.")
    return pixel_values


def run_answer(
    checkpoint_path: str | Path,
    image_path: str | Path,
    question: str,
    *,
    device: str = "auto",
    max_new_tokens: Optional[int] = None,
    llm_model: Optional[str] = None,
    llm_max_tokens: Optional[int] = None,
    answer_mode: str = "short",
    choices: Optional[list[str]] = None,
) -> tuple[str, str]:
    """Run QACap caption generation followed by the paper's Claude answering step."""
    checkpoint_path = Path(checkpoint_path)
    image_path = Path(image_path)

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")
    if answer_mode not in {"short", "open-ended", "multiple-choice"}:
        raise ValueError("answer_mode must be short, open-ended, or multiple-choice")
    if answer_mode == "multiple-choice":
        if not choices or len(choices) < 2:
            raise ValueError("multiple-choice mode requires at least two choices")
        choices = [str(choice).strip() for choice in choices]
        if any(not choice for choice in choices):
            raise ValueError("multiple-choice choices must be non-empty strings")
    elif choices is not None:
        raise ValueError("choices should only be supplied for multiple-choice mode")

    checkpoint = _load_trusted_checkpoint(checkpoint_path)
    model_kwargs, data_config, inference_config = _resolve_config(checkpoint)

    if max_new_tokens is None:
        configured_max_new_tokens = inference_config.get("max_new_tokens")
        if configured_max_new_tokens is None:
            raise ValueError(
                "Maximum caption-generation length is not specified by the paper. "
                "Supply --max-new-tokens or set inference.max_new_tokens in the "
                "saved configuration."
            )
        max_new_tokens = int(configured_max_new_tokens)
    if max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be a positive integer.")

    if llm_model is None:
        llm_model = inference_config.get("llm_model")
    if llm_max_tokens is None:
        llm_max_tokens = inference_config.get("llm_max_tokens")

    # Imports are delayed so this file's config/argument helpers remain usable
    # without downloading model weights or installing the Anthropic SDK.
    try:
        from qacap.model import QACap
        from qacap.llm import answer_with_llm
    except ImportError as exc:
        raise ImportError(
            "Could not import qacap.model or qacap.llm. Run this script from "
            "the project root and ensure the qacap package is installed."
        ) from exc

    model = QACap(**model_kwargs)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint weights do not exactly match the current QACap model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    selected_device = _select_device(device)
    model.to(selected_device)
    model.eval()

    # Reuse the processor owned by QACap, matching the same pretrained
    # processor/tokenizer pair used by QACapDataset.
    pixel_values = _prepare_image(
        image_path=image_path,
        processor=model.blip_processor,
        image_size=data_config.get("image_size"),
    ).to(selected_device)

    with torch.inference_mode():
        captions = model.generate_caption(
            pixel_values=pixel_values,
            questions=[question.strip()],
            max_new_tokens=max_new_tokens,
        )
    if len(captions) != 1 or not captions[0].strip():
        raise RuntimeError("QACap did not generate a non-empty caption.")
    caption = captions[0].strip()

    # llm.py pins the method to Anthropic/Claude. The exact Claude API snapshot
    # and max token budget are required there or via its environment variables.
    answer = answer_with_llm(
        question=question.strip(),
        caption=caption,
        model=llm_model,
        max_tokens=llm_max_tokens,
        answer_mode=answer_mode,
        choices=choices,
    )
    return caption, answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the QACap + Claude knowledge-based VQA inference pipeline."
    )
    parser.add_argument("--checkpoint", required=True, help="Trained QACap checkpoint.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--question", required=True, help="Question about the image.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device (default: auto-detect CUDA).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Maximum new caption tokens. Required unless inference.max_new_tokens "
            "is present in the checkpoint config; not specified in the main paper."
        ),
    )
    parser.add_argument(
        "--answer-mode",
        choices=("short", "open-ended", "multiple-choice"),
        default="short",
        help="Answer format described in the paper's evaluation protocol.",
    )
    parser.add_argument(
        "--choices-json",
        default=None,
        help="JSON array of multiple-choice options; required for multiple-choice mode.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help=(
            "Exact Claude 3.5 API model ID/snapshot. Alternatively set "
            "QACAP_LLM_MODEL; the paper does not specify a snapshot."
        ),
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=None,
        help=(
            "Anthropic response-token budget. Alternatively set "
            "QACAP_LLM_MAX_TOKENS; this is not specified in the paper."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    choices = None
    if args.choices_json is not None:
        try:
            choices = json.loads(args.choices_json)
        except json.JSONDecodeError as exc:
            raise ValueError("--choices-json must be a valid JSON array") from exc
        if not isinstance(choices, list):
            raise ValueError("--choices-json must decode to a JSON array")
    caption, answer = run_answer(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        question=args.question,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        llm_model=args.llm_model,
        llm_max_tokens=args.llm_max_tokens,
        answer_mode=args.answer_mode,
        choices=choices,
    )
    print(f"QACap caption: {caption}")
    print(f"LLM answer: {answer}")


if __name__ == "__main__":
    main()
