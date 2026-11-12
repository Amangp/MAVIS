"""
Unified two-stage trainer for the CVPR 2025 QACap implementation.

One command runs the paper-described sequence:
  Stage 1: fine-tune on the PromptCap dataset for 30 epochs.
  Stage 2: continue the SAME model on the constructed QACap dataset for 50 epochs.

The main paper specifies an initial learning rate of 2e-5 for both stages.
It defines the joint objective:
    L = L_cap + lambda_JS * L_JS

This script applies that objective in both phases because the paper describes
end-to-end joint training and sequential fine-tuning of the model. Confirm the
exact phase-wise loss treatment against the supplementary material if it is
available; the main text does not describe a phase-specific change to the loss.

Research-integrity notes:
- The exact values of alpha, lambda_JS, QAFD block count, batch size, optimizer
  details, weight decay, clipping and decoding/truncation limits are not all
  specified in the main paper. This script reads these values from YAML and
  reports which are not paper-confirmed; it does not invent missing values.
- The paper's main text specifies 30 + 50 epochs and initial LR 2e-5. In
  paper mode, any other epoch/LR values are rejected unless the user explicitly
  supplies --allow-non-paper-settings (intended only for smoke tests).
- Scheduler behavior is not added because the main paper does not specify a
  scheduler. Checkpoint utility support for a scheduler remains available.
- Validation is optional and is used only for checkpoint selection. It does
  not change the training objective.
- The script saves checkpoints at epoch boundaries. Resuming is supported
  from the latest checkpoint, including automatic transition from PromptCap
  to QACap after Stage 1 completes.

Expected YAML sections:
  seed
  model:
    blip_name, bert_name, alpha, num_heads, qafd_layers
    optional max_question_length, max_caption_length
    lambda_js (required; numerical value must be verified from supplementary)
  training:
    batch_size, num_workers, learning_rate, promptcap_epochs, qacap_epochs
    optional weight_decay, amp, gradient_accumulation_steps,
             grad_clip, label_smoothing
  data:
    root, promptcap_train, qacap_train
    optional promptcap_val, qacap_val or val, image_size, max_samples

Example:
    python train.py --config configs/paper_reproduction.yaml

Resume:
    python train.py --config configs/paper_reproduction.yaml --resume outputs/qacap/last.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import random
import warnings
from contextlib import nullcontext
from functools import partial
from pathlib import Path


# Allow `python scripts/train.py` from the repository root without an editable install.
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if (PROJECT_ROOT / "qacap").is_dir() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Mapping, Optional

import torch
import yaml
from torch.utils.data import DataLoader


PAPER_LEARNING_RATE = 2e-5
PAPER_PROMPTCAP_EPOCHS = 30
PAPER_QACAP_EPOCHS = 50
VALID_PHASES = ("promptcap", "qacap")

MODEL_CONSTRUCTOR_KEYS = (
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


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("YAML configuration root must be a mapping.")
    for section in ("model", "training", "data"):
        if not isinstance(config.get(section), dict):
            raise KeyError(f"Configuration must contain a '{section}' mapping.")
    return config


def _require(config: Mapping[str, Any], key: str, section: str) -> Any:
    value = config.get(key)
    if value is None:
        raise KeyError(f"Missing required configuration value: {section}.{key}")
    return value


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy (when installed), CPU and CUDA RNGs."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers from the DataLoader's torch Generator."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_paper_settings(
    config: Mapping[str, Any],
    *,
    allow_non_paper_settings: bool,
) -> None:
    training = config["training"]
    model = config["model"]

    learning_rate = float(_require(training, "learning_rate", "training"))
    promptcap_epochs = int(_require(training, "promptcap_epochs", "training"))
    qacap_epochs = int(_require(training, "qacap_epochs", "training"))

    deviations = []
    if not math.isclose(learning_rate, PAPER_LEARNING_RATE, rel_tol=0.0, abs_tol=1e-12):
        deviations.append(
            f"learning_rate={learning_rate:g} (paper: {PAPER_LEARNING_RATE:g})"
        )
    if promptcap_epochs != PAPER_PROMPTCAP_EPOCHS:
        deviations.append(
            f"promptcap_epochs={promptcap_epochs} (paper: {PAPER_PROMPTCAP_EPOCHS})"
        )
    if qacap_epochs != PAPER_QACAP_EPOCHS:
        deviations.append(
            f"qacap_epochs={qacap_epochs} (paper: {PAPER_QACAP_EPOCHS})"
        )

    if deviations and not allow_non_paper_settings:
        raise ValueError(
            "Configuration differs from the main-paper training schedule:\n  - "
            + "\n  - ".join(deviations)
            + "\nUse --allow-non-paper-settings only for debugging/smoke tests. "
            "Do not report such a run as paper reproduction."
        )

    if deviations:
        warnings.warn(
            "NON-PAPER DEBUG/OVERRIDE SETTINGS ENABLED:\n  - "
            + "\n  - ".join(deviations),
            RuntimeWarning,
        )

    if "lambda_js" not in model:
        raise KeyError(
            "model.lambda_js is required. The main paper says its value is "
            "provided in the supplementary material; do not invent a default."
        )
    lambda_js = float(model["lambda_js"])
    if not math.isfinite(lambda_js) or lambda_js < 0:
        raise ValueError("model.lambda_js must be finite and nonnegative.")
    alpha = float(_require(model, "alpha", "model"))
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("model.alpha must be finite and in [0, 1].")
    if int(_require(model, "num_heads", "model")) < 1:
        raise ValueError("model.num_heads must be positive.")
    if int(_require(model, "qafd_layers", "model")) < 1:
        raise ValueError("model.qafd_layers must be positive.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("training.learning_rate must be finite and positive.")
    if promptcap_epochs < 1 or qacap_epochs < 1:
        raise ValueError("Both training epoch counts must be positive.")

    if float(training.get("label_smoothing", 0.0)) != 0.0:
        raise ValueError(
            "The paper defines caption negative log-likelihood. "
            "Nonzero label_smoothing changes that objective."
        )

    if int(training.get("batch_size", 0)) < 1:
        raise ValueError("training.batch_size must be positive.")
    if int(training.get("num_workers", 0)) < 0:
        raise ValueError("training.num_workers cannot be negative.")
    if int(training.get("gradient_accumulation_steps", 1)) < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    weight_decay = float(training.get("weight_decay", 0.0))
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("training.weight_decay must be finite and nonnegative.")
    grad_clip = training.get("grad_clip")
    if grad_clip is not None and (
        not math.isfinite(float(grad_clip)) or float(grad_clip) <= 0
    ):
        raise ValueError("training.grad_clip must be finite and positive when supplied.")

    for key in ("promptcap_train", "qacap_train"):
        manifest = Path(_require(config["data"], key, "data"))
        if not manifest.is_file():
            raise FileNotFoundError(f"Training manifest does not exist: {manifest.resolve()}")

    # These values are present in the initial project config, but the paper's
    # main text does not state them. Surface that distinction in the log.
    unconfirmed = []
    for key in (
        "alpha", "num_heads", "qafd_layers",
        "qafd_ffn_ratio", "qafd_ffn_activation", "qafd_ffn_residual",
    ):
        if model.get(key) is None:
            raise KeyError(f"model.{key} is required; do not infer this value.")
        unconfirmed.append(f"model.{key}={model[key]!r}")

    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be 'adam' or 'adamw'.")
    unconfirmed.append(f"training.optimizer={optimizer_name!r}")
    if "weight_decay" in training:
        unconfirmed.append(f"training.weight_decay={training['weight_decay']!r}")
    if training.get("grad_clip") is not None:
        unconfirmed.append(f"training.grad_clip={training['grad_clip']!r}")
    if training.get("amp", False):
        unconfirmed.append("training.amp=True (hardware/numerical execution setting)")

    print(
        "Paper-main-text-unconfirmed settings (verify with supplementary/config): "
        + (", ".join(unconfirmed) if unconfirmed else "none supplied")
    )
    print(
        f"Training schedule: PromptCap={promptcap_epochs} epochs, "
        f"QACap={qacap_epochs} epochs, initial LR={learning_rate:g}"
    )
    print(f"QARCL lambda_JS from config: {float(model['lambda_js']):g}")


def _build_model(config: Mapping[str, Any]) -> torch.nn.Module:
    from qacap.model import QACap

    model_config = config["model"]
    model_kwargs = {
        key: model_config[key]
        for key in MODEL_CONSTRUCTOR_KEYS
        if key in model_config
    }
    required = ("alpha", "num_heads", "qafd_layers")
    missing = [key for key in required if key not in model_kwargs]
    if missing:
        raise KeyError(
            "Updated QACap constructor requires explicit architecture settings: "
            f"{missing}"
        )

    ignored_legacy_keys = sorted(
        set(model_config)
        - set(MODEL_CONSTRUCTOR_KEYS)
        - {"lambda_js", "max_caption_length"}
    )
    if ignored_legacy_keys:
        warnings.warn(
            "The following legacy/prototype model config entries are not accepted "
            "by the updated QACap architecture and will not be forwarded: "
            f"{ignored_legacy_keys}. Check that the checkpoint/config belongs to "
            "the updated model.py.",
            RuntimeWarning,
        )
    return QACap(**model_kwargs)


def _make_dataset(
    manifest: str | Path,
    config: Mapping[str, Any],
    *,
    max_samples: Optional[int],
):
    from qacap.data import QACapDataset

    model_config = config["model"]
    data_config = config["data"]
    max_caption_length = model_config.get(
        "max_caption_length",
        data_config.get("max_caption_length"),
    )
    return QACapDataset(
        jsonl=manifest,
        root=data_config.get("root", "."),
        image_size=data_config.get("image_size"),
        max_samples=max_samples,
        blip_name=model_config.get(
            "blip_name", "Salesforce/blip-image-captioning-base"
        ),
        max_caption_length=max_caption_length,
    )


def _make_loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    generator: torch.Generator,
):
    from qacap.data import collate_qacap

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(
            collate_qacap,
            pad_token_id=dataset.pad_token_id,
        ),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
    )


def _make_grad_scaler(enabled: bool):
    """Create an AMP scaler across supported PyTorch API versions."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> tuple:
    pixel_values = batch["pixel_values"].to(
        device, non_blocking=(device.type == "cuda")
    )
    decoder_input_ids = batch["decoder_input_ids"].to(
        device, non_blocking=(device.type == "cuda")
    )
    decoder_attention_mask = batch["decoder_attention_mask"].to(
        device, non_blocking=(device.type == "cuda")
    )
    labels = batch["labels"].to(device, non_blocking=(device.type == "cuda"))
    return pixel_values, decoder_input_ids, decoder_attention_mask, labels


def _forward_and_loss(
    model: torch.nn.Module,
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    lambda_js: float,
    label_smoothing: float,
) -> dict[str, torch.Tensor]:
    from qacap.losses import qacap_total_loss

    pixel_values, decoder_ids, decoder_mask, labels = _move_batch(batch, device)

    logits, qafd_attention = model(
        pixel_values=pixel_values,
        questions=batch["questions"],
        decoder_input_ids=decoder_ids,
        decoder_attention_mask=decoder_mask,
        labels=labels,
    )

    # In paper-reproduction mode, QARCL is mandatory. The model raises during
    # forward if the final decoder cross-attention is unavailable.
    caption_attention = getattr(model, "last_decoder_attention", None)
    if caption_attention is None:
        raise RuntimeError(
            "QACap did not expose final decoder cross-attention. Refusing to "
            "continue with caption loss alone because that omits QARCL."
        )

    caption_query_mask = getattr(model, "last_caption_query_mask", None)
    if caption_query_mask is None:
        raise RuntimeError(
            "Model did not expose last_caption_query_mask. The QARCL caption "
            "query mask is needed to exclude PET prompt positions."
        )

    return qacap_total_loss(
        logits=logits,
        labels=labels,
        qafd_attention=qafd_attention,
        caption_attention=caption_attention,
        lambda_js=lambda_js,
        caption_query_mask=caption_query_mask,
        label_smoothing=label_smoothing,
        ignore_index=-100,
    )


@torch.no_grad()
def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    lambda_js: float,
    label_smoothing: float,
    amp_enabled: bool,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_caption_loss = 0.0
    total_js_loss = 0.0
    batches = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        with _autocast_context(device, amp_enabled):
            losses = _forward_and_loss(
                model,
                batch,
                device,
                lambda_js=lambda_js,
                label_smoothing=label_smoothing,
            )
        total_loss += float(losses["loss"].detach().float().item())
        total_caption_loss += float(losses["caption_loss"].detach().float().item())
        total_js_loss += float(losses["js_loss"].detach().float().item())
        batches += 1

    if batches == 0:
        raise RuntimeError("Validation loader produced no batches.")

    return {
        "loss": total_loss / batches,
        "caption_loss": total_caption_loss / batches,
        "js_loss": total_js_loss / batches,
    }


def _save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_metric: Optional[float],
    config: Mapping[str, Any],
    phase: str,
    phase_complete: bool,
    loader_generator: torch.Generator,
    metrics: Mapping[str, Any],
) -> None:
    from qacap.checkpoint import save_checkpoint

    save_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        config=dict(config),
        scaler=scaler,
        phase=phase,
        extra={
            "phase_complete": bool(phase_complete),
            "loader_generator_state": loader_generator.get_state(),
            "metrics": dict(metrics),
        },
        save_rng_state=True,
    )


def _train_one_phase(
    *,
    phase: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    config: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
    global_step: int,
    start_epoch: int = 0,
    best_metric: Optional[float] = None,
    resume_loader_state: Optional[torch.Tensor] = None,
    allow_non_paper_settings: bool = False,
    max_train_batches: Optional[int] = None,
) -> tuple[int, Optional[float]]:
    if phase not in VALID_PHASES:
        raise ValueError(f"Unknown training phase: {phase}")

    training = config["training"]
    data_config = config["data"]
    model_config = config["model"]

    train_key = "promptcap_train" if phase == "promptcap" else "qacap_train"
    train_manifest = _require(data_config, train_key, "data")
    if phase == "promptcap":
        epoch_count = int(_require(training, "promptcap_epochs", "training"))
        stage_seed_offset = 0
        max_samples = data_config.get(
            "promptcap_max_samples", data_config.get("max_samples")
        )
    else:
        epoch_count = int(_require(training, "qacap_epochs", "training"))
        stage_seed_offset = 1
        max_samples = data_config.get(
            "qacap_max_samples", data_config.get("max_samples")
        )

    if start_epoch < 0 or start_epoch > epoch_count:
        raise ValueError(
            f"Resume epoch {start_epoch} is outside phase {phase} "
            f"range [0, {epoch_count}]."
        )

    dataset = _make_dataset(
        train_manifest,
        config,
        max_samples=max_samples,
    )
    batch_size = int(training["batch_size"])
    num_workers = int(training.get("num_workers", 0))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(config.get("seed", 42)) + stage_seed_offset)
    if resume_loader_state is not None:
        loader_generator.set_state(resume_loader_state.cpu())

    train_loader = _make_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        device=device,
        generator=loader_generator,
    )

    validation_key = (
        "promptcap_val" if phase == "promptcap" else "qacap_val"
    )
    validation_manifest = data_config.get(
        validation_key,
        data_config.get("val"),
    )
    validation_loader = None
    if validation_manifest:
        if not Path(validation_manifest).is_file():
            raise FileNotFoundError(
                f"Validation manifest for {phase} does not exist: {validation_manifest}"
            )
        validation_dataset = _make_dataset(
            validation_manifest,
            config,
            max_samples=data_config.get("val_max_samples"),
        )
        validation_generator = torch.Generator()
        validation_generator.manual_seed(
            int(config.get("seed", 42)) + 100 + stage_seed_offset
        )
        validation_loader = _make_loader(
            validation_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            device=device,
            generator=validation_generator,
        )

    lambda_js = float(model_config["lambda_js"])
    label_smoothing = float(training.get("label_smoothing", 0.0))
    amp_enabled = bool(training.get("amp", False)) and device.type == "cuda"
    accumulation_steps = int(training.get("gradient_accumulation_steps", 1))
    grad_clip = training.get("grad_clip")
    if grad_clip is not None:
        grad_clip = float(grad_clip)
        if grad_clip <= 0:
            grad_clip = None

    if validation_loader is None:
        warnings.warn(
            f"No validation manifest configured for {phase}; only last.pt will "
            "be saved. A best checkpoint cannot be selected without validation.",
            RuntimeWarning,
        )

    phase_dir = output_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    best_value = float("inf") if best_metric is None else float(best_metric)

    print(
        f"\n===== {phase.upper()} PHASE: {start_epoch}/{epoch_count} epochs complete "
        f"at resume ====="
    )
    print(f"train_manifest={Path(train_manifest).resolve()}")
    print(f"training_examples={len(dataset)}")
    if validation_loader is not None:
        print(f"validation_examples={len(validation_loader.dataset)}")
    print(f"batch_size={batch_size}, num_workers={num_workers}, device={device}")
    print(
        f"lambda_JS={lambda_js}; gradient_accumulation_steps={accumulation_steps}; "
        f"AMP={amp_enabled}"
    )

    for epoch in range(start_epoch, epoch_count):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_caption_loss = 0.0
        running_js_loss = 0.0
        batches_seen = 0
        optimizer_updates = 0

        total_batches = len(train_loader)
        if max_train_batches is not None:
            if max_train_batches < 1:
                raise ValueError("max_train_batches must be positive.")
            total_batches = min(total_batches, max_train_batches)

        if total_batches == 0:
            raise RuntimeError(f"{phase} training loader is empty.")

        from tqdm.auto import tqdm
        progress = tqdm(
            enumerate(train_loader),
            total=total_batches,
            desc=f"{phase} epoch {epoch + 1}/{epoch_count}",
            dynamic_ncols=True,
        )

        for batch_index, batch in progress:
            if max_train_batches is not None and batch_index >= max_train_batches:
                break

            # Divide each loss by the size of its accumulation group. This
            # handles an incomplete final accumulation group correctly.
            group_start = (batch_index // accumulation_steps) * accumulation_steps
            current_group_size = min(
                accumulation_steps,
                total_batches - group_start,
            )
            with _autocast_context(device, amp_enabled):
                losses = _forward_and_loss(
                    model,
                    batch,
                    device,
                    lambda_js=lambda_js,
                    label_smoothing=label_smoothing,
                )
                scaled_loss = losses["loss"] / current_group_size

            scaler.scale(scaled_loss).backward()

            running_loss += float(losses["loss"].detach().float().item())
            running_caption_loss += float(
                losses["caption_loss"].detach().float().item()
            )
            running_js_loss += float(losses["js_loss"].detach().float().item())
            batches_seen += 1

            is_accumulation_boundary = (
                (batch_index + 1) % accumulation_steps == 0
            )
            is_last_batch = batch_index + 1 == total_batches
            if is_accumulation_boundary or is_last_batch:
                scaler.unscale_(optimizer)
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        max_norm=grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                optimizer_updates += 1

            progress.set_postfix(
                loss=f"{losses['loss'].detach().float().item():.4f}",
                cap=f"{losses['caption_loss'].detach().float().item():.4f}",
                js=f"{losses['js_loss'].detach().float().item():.4f}",
                updates=optimizer_updates,
            )

        train_metrics = {
            "loss": running_loss / batches_seen,
            "caption_loss": running_caption_loss / batches_seen,
            "js_loss": running_js_loss / batches_seen,
        }

        validation_metrics: Optional[dict[str, float]] = None
        if validation_loader is not None:
            validation_metrics = _run_validation(
                model,
                validation_loader,
                device,
                lambda_js=lambda_js,
                label_smoothing=label_smoothing,
                amp_enabled=amp_enabled,
            )
            metric_for_best = validation_metrics["loss"]
            is_best = metric_for_best < best_value
            if is_best:
                best_value = metric_for_best
                _save_training_checkpoint(
                    phase_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_value,
                    config=config,
                    phase=phase,
                    phase_complete=(epoch == epoch_count - 1),
                    loader_generator=loader_generator,
                    metrics={
                        "train": train_metrics,
                        "validation": validation_metrics,
                    },
                )
        else:
            is_best = False

        phase_complete = epoch == epoch_count - 1
        metrics = {
            "train": train_metrics,
            "validation": validation_metrics,
        }

        _save_training_checkpoint(
            phase_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=(best_value if validation_loader is not None else None),
            config=config,
            phase=phase,
            phase_complete=phase_complete,
            loader_generator=loader_generator,
            metrics=metrics,
        )
        # A root-level latest checkpoint lets the complete two-phase run resume
        # with a single path regardless of the active phase.
        _save_training_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=(best_value if validation_loader is not None else None),
            config=config,
            phase=phase,
            phase_complete=phase_complete,
            loader_generator=loader_generator,
            metrics=metrics,
        )

        val_text = (
            f", val_loss={validation_metrics['loss']:.6f}"
            if validation_metrics is not None
            else ", validation=not-configured"
        )
        print(
            f"{phase} epoch {epoch + 1}/{epoch_count}: "
            f"train_loss={train_metrics['loss']:.6f}, "
            f"caption_loss={train_metrics['caption_loss']:.6f}, "
            f"js_loss={train_metrics['js_loss']:.6f}"
            f"{val_text}, optimizer_updates={optimizer_updates}, "
            f"global_step={global_step}"
        )

    return global_step, (best_value if validation_loader is not None else None)


def _resume_training(
    resume_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
) -> dict[str, Any]:
    from qacap.checkpoint import load_checkpoint

    checkpoint = load_checkpoint(
        path=resume_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        map_location=device,
        scaler=scaler,
        restore_rng=True,
        strict_model=True,
        require_training_state=True,
    )
    phase = checkpoint.get("phase")
    if phase not in VALID_PHASES:
        raise ValueError(
            "Resume checkpoint does not contain a valid 'phase' field. "
            "Use a checkpoint produced by this updated train.py."
        )
    if not isinstance(checkpoint.get("epoch"), int):
        raise ValueError("Resume checkpoint is missing its completed epoch index.")
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete two-stage QACap training sequence: PromptCap then QACap."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/paper_reproduction.yaml",
        help="Training YAML configuration.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from an epoch-boundary checkpoint (prefer outputs/qacap/last.pt).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/qacap",
        help="Directory for phase and latest checkpoints.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device.",
    )
    parser.add_argument(
        "--allow-non-paper-settings",
        action="store_true",
        help="Allow non-paper LR/epoch counts for debugging only; never report as reproduction.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Debug-only cap per epoch. Not suitable for research training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("--max-train-batches must be positive.")
    if args.max_train_batches is not None and not args.allow_non_paper_settings:
        raise ValueError(
            "--max-train-batches truncates the training data and is only valid "
            "with --allow-non-paper-settings. Such a run is a smoke test, not a reproduction."
        )

    config = _load_config(Path(args.config))
    _validate_paper_settings(
        config,
        allow_non_paper_settings=args.allow_non_paper_settings,
    )

    seed = int(config.get("seed", 42))
    _set_seed(seed)
    device = _resolve_device(args.device)
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(
            f"cuda_version={torch.version.cuda}, "
            f"torch_version={torch.__version__}"
        )

    model = _build_model(config).to(device)
    training = config["training"]
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.0))
    if weight_decay < 0:
        raise ValueError("training.weight_decay cannot be negative.")

    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    optimizer_class = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[optimizer_name]
    optimizer = optimizer_class(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    print(f"optimizer={optimizer_name}, weight_decay={weight_decay:g}")

    amp_enabled = bool(training.get("amp", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)

    global_step = 0
    resume_checkpoint: Optional[dict[str, Any]] = None
    if args.resume:
        resume_checkpoint = _resume_training(
            Path(args.resume), model, optimizer, scaler, device
        )
        global_step = int(resume_checkpoint.get("global_step", 0))
        print(
            f"resumed_phase={resume_checkpoint['phase']}, "
            f"completed_epoch={resume_checkpoint['epoch'] + 1}, "
            f"global_step={global_step}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    promptcap_epochs = int(training["promptcap_epochs"])
    qacap_epochs = int(training["qacap_epochs"])

    # Determine where a resumed run should continue. A checkpoint written at
    # the end of PromptCap automatically advances to QACap; one written during
    # QACap resumes that phase. The same model and optimizer objects are used
    # throughout, preserving the sequential fine-tuning design.
    phases_to_run: list[tuple[str, int, Optional[torch.Tensor], Optional[float]]] = []
    if resume_checkpoint is None:
        phases_to_run = [
            ("promptcap", 0, None, None),
            ("qacap", 0, None, None),
        ]
    else:
        phase = resume_checkpoint["phase"]
        epoch = int(resume_checkpoint["epoch"])
        extra = resume_checkpoint.get("extra") or {}
        phase_complete = bool(extra.get("phase_complete", False))
        loader_state = extra.get("loader_generator_state")
        best_metric = resume_checkpoint.get("best_metric")

        if phase == "promptcap":
            next_epoch = epoch + 1
            if phase_complete or next_epoch >= promptcap_epochs:
                phases_to_run = [("qacap", 0, None, None)]
            else:
                phases_to_run = [
                    ("promptcap", next_epoch, loader_state, best_metric),
                    ("qacap", 0, None, None),
                ]
        elif phase == "qacap":
            next_epoch = epoch + 1
            if phase_complete or next_epoch >= qacap_epochs:
                print("Both training phases are already complete; nothing remains to run.")
                return
            phases_to_run = [("qacap", next_epoch, loader_state, best_metric)]

    for phase, start_epoch, loader_state, best_metric in phases_to_run:
        global_step, _ = _train_one_phase(
            phase=phase,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            device=device,
            output_dir=output_dir,
            global_step=global_step,
            start_epoch=start_epoch,
            best_metric=best_metric,
            resume_loader_state=loader_state,
            allow_non_paper_settings=args.allow_non_paper_settings,
            max_train_batches=args.max_train_batches,
        )

    # Final model checkpoint is a copy of the latest complete checkpoint.
    latest_path = output_dir / "last.pt"
    if not latest_path.is_file():
        raise RuntimeError("Training ended without producing a latest checkpoint.")
    final_path = output_dir / "final.pt"
    import shutil
    shutil.copy2(latest_path, final_path)
    print(f"\nTraining sequence finished. Final checkpoint: {final_path.resolve()}")


if __name__ == "__main__":
    main()
