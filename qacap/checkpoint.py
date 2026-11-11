"""
QACap checkpoint utilities.

Saves model/training state, two-stage phase metadata, optional AMP scaler state,
RNG states, and optional extra training-loop state. Checkpoints are written
atomically so an interrupted save does not replace a valid previous checkpoint.
"""

from __future__ import annotations

import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

try:
    import numpy as np
except ImportError:
    np = None


CHECKPOINT_FORMAT_VERSION = 2


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DataParallel/DistributedDataParallel-style model wrappers."""
    return model.module if hasattr(model, "module") else model


def _capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch CPU and available CUDA RNG states."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if np is not None:
        state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    """Restore RNG state entries present in the checkpoint."""
    if not state:
        return

    if state.get("python") is not None:
        random.setstate(state["python"])

    if state.get("numpy") is not None and np is not None:
        np.random.set_state(state["numpy"])

    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "Cannot exactly restore CUDA RNG state: checkpoint contains "
                f"{len(cuda_states)} device state(s), while this runtime has "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    global_step: int = 0,
    best_metric: Optional[float] = None,
    config: Optional[Any] = None,
    *,
    scaler: Optional[Any] = None,
    phase: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    save_rng_state: bool = True,
) -> Path:
    """
    Save a complete training checkpoint atomically.

    `epoch` is stored exactly as supplied. The training loop must consistently
    define whether it represents the last completed or next-to-run epoch.

    `phase` can identify the current training phase (for example, "promptcap"
    or "qacap"). `extra` can store phase-local epoch, sampler state, or other
    loop metadata without changing this utility's interface.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    module = _unwrap_model(model)
    checkpoint: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "phase": phase,
        "global_step": int(global_step),
        "best_metric": best_metric,
        "config": config,
        "extra": dict(extra) if extra is not None else {},
    }
    if save_rng_state:
        checkpoint["rng_state"] = _capture_rng_state()

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        torch.save(checkpoint, temp_path)
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    return destination


def load_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    *,
    scaler: Optional[Any] = None,
    restore_rng: bool = True,
    strict_model: bool = True,
    require_training_state: bool = False,
) -> dict[str, Any]:
    """
    Restore a checkpoint into model and optional training objects.

    `require_training_state=True` makes missing optimizer/scheduler/scaler
    state an error when the corresponding object is supplied.

    Security: checkpoints are loaded with weights_only=False because saved
    Python/NumPy RNG states and config objects may require pickle support.
    Only load checkpoint files that you trust.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Explicit False is needed for Python/NumPy RNG state and arbitrary
    # serializable config snapshots. Load only trusted checkpoint files.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint contents must be a dictionary.")
    if "model" not in checkpoint:
        raise KeyError("Checkpoint is missing the required 'model' state.")

    _unwrap_model(model).load_state_dict(
        checkpoint["model"],
        strict=strict_model,
    )

    def restore_optional(
        key: str,
        target: Optional[Any],
        description: str,
    ) -> None:
        if target is None:
            return
        saved_state = checkpoint.get(key)
        if saved_state is None:
            if require_training_state:
                raise KeyError(
                    f"Cannot resume strictly: checkpoint has no {description} state."
                )
            return
        target.load_state_dict(saved_state)

    restore_optional("optimizer", optimizer, "optimizer")
    restore_optional("scheduler", scheduler, "scheduler")
    restore_optional("scaler", scaler, "GradScaler")

    if restore_rng:
        _restore_rng_state(checkpoint.get("rng_state"))

    return checkpoint
