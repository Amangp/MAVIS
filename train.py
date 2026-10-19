from __future__ import annotations
import math
import re
from collections import Counter
import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.data_loader import (
    MAVISVQAv2Dataset,
    mavis_collate_fn,
    default_image_transform,
)
from src.data.caption_loader import (
    COCOCaptionDataset,
    caption_collate_fn,
    load_caption_vocab,
)
from src.models.mavis_model import MAVIS

ROOT = Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def resolve_image_dir(configured_path: str | Path, split: str) -> Path:
    base = resolve_path(configured_path)
    subtype = "train2014" if split == "train" else "val2014"
    candidates = [base, base / subtype]
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.glob("COCO_*.jpg"), None):
            return candidate
    raise FileNotFoundError(
        "No COCO images found. Checked: " + ", ".join(map(str, candidates))
    )


def resolve_caption_annotations(configured_path: str | Path, split: str) -> Path:
    configured = resolve_path(configured_path)
    if configured.is_file():
        return configured
    filename = f"captions_{split}2014.json"
    fallback = ROOT / "datasets" / "annotations_trainval2014" / "annotations" / filename
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"Caption annotations not found: {configured}; fallback: {fallback}"
    )


def load_config() -> dict[str, Any]:
    path = ROOT / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "training" not in config:
        raise ValueError(f"{path} must contain a top-level 'training' section.")
    return config


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def make_loader(dataset, batch_size, shuffle, workers, pin_memory, collate_fn):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        persistent_workers=int(workers) > 0,
        collate_fn=collate_fn,
    )


def build_dataloaders(config: dict[str, Any], args):
    paths, data = config["paths"], config["data"]
    answer_cfg, caption_cfg = (
        config["training"]["answer"],
        config["training"]["caption"],
    )
    evaluation = config.get("evaluation", {})
    transform = default_image_transform(int(data["image_size"]))
    loaders, sizes = {}, {}
    question_vocab_path = resolve_path(paths["vocab"]["question"])
    answer_vocab_path = resolve_path(paths["vocab"]["answer"])
    caption_vocab_path = resolve_path(paths["vocab"]["caption"])

    if answer_cfg.get("enabled", True):
        train_ds = MAVISVQAv2Dataset(
            questions_file=resolve_path(paths["vqa"]["train_questions"]),
            annotations_file=resolve_path(paths["vqa"]["train_annotations"]),
            image_root=resolve_image_dir(paths["vqa"]["train_images"], "train"),
            word_vocab_file=question_vocab_path,
            answer_vocab_file=answer_vocab_path,
            max_question_length=int(data["max_question_length"]),
            image_size=int(data["image_size"]),
            transform=transform,
            strict_images=True,
        )
        loaders["vqa_train"] = make_loader(
            train_ds,
            answer_cfg["batch_size"],
            True,
            answer_cfg.get("num_workers", 0),
            answer_cfg.get("pin_memory", True),
            mavis_collate_fn,
        )
        print(f"VQA training examples: {len(train_ds):,}")
        if config.get("validation", {}).get("enabled", True):
            val_ds = MAVISVQAv2Dataset(
                questions_file=resolve_path(paths["vqa"]["val_questions"]),
                annotations_file=resolve_path(paths["vqa"]["val_annotations"]),
                image_root=resolve_image_dir(paths["vqa"]["val_images"], "val"),
                word_vocab_file=question_vocab_path,
                answer_vocab_file=answer_vocab_path,
                max_question_length=int(data["max_question_length"]),
                image_size=int(data["image_size"]),
                transform=transform,
                strict_images=True,
            )
            loaders["vqa_val"] = make_loader(
                val_ds,
                evaluation.get("batch_size", answer_cfg["batch_size"]),
                False,
                evaluation.get("num_workers", answer_cfg.get("num_workers", 0)),
                False,
                mavis_collate_fn,
            )
            print(f"VQA validation examples: {len(val_ds):,}")
        with question_vocab_path.open("rb") as f:
            qv = pickle.load(f)
        with answer_vocab_path.open("rb") as f:
            av = pickle.load(f)
        sizes["question"] = len(qv)
        sizes["answer"] = len(av)

    if caption_cfg.get("enabled", True):
        caption_vocab = load_caption_vocab(caption_vocab_path)
        pad_id = int(data["caption_special_token_ids"]["pad"])
        train_ds = COCOCaptionDataset(
            annotation_json=resolve_caption_annotations(
                paths["captions"]["train_annotations"], "train"
            ),
            image_root=resolve_image_dir(paths["captions"]["train_images"], "train"),
            vocab=caption_vocab,
            transform=transform,
            max_length=int(data["max_caption_length"]),
            strict_images=True,
        )
        cap_collate = lambda batch: caption_collate_fn(batch, pad_token_id=pad_id)
        loaders["caption_train"] = make_loader(
            train_ds,
            caption_cfg["batch_size"],
            True,
            caption_cfg.get("num_workers", 0),
            caption_cfg.get("pin_memory", True),
            cap_collate,
        )
        print(f"Caption training examples: {len(train_ds):,}")
        if config.get("validation", {}).get("enabled", True):
            val_ds = COCOCaptionDataset(
                annotation_json=resolve_caption_annotations(
                    paths["captions"]["val_annotations"], "val"
                ),
                image_root=resolve_image_dir(paths["captions"]["val_images"], "val"),
                vocab=caption_vocab,
                transform=transform,
                max_length=int(data["max_caption_length"]),
                strict_images=True,
            )
            loaders["caption_val"] = make_loader(
                val_ds,
                evaluation.get("batch_size", caption_cfg["batch_size"]),
                False,
                evaluation.get("num_workers", caption_cfg.get("num_workers", 0)),
                False,
                cap_collate,
            )
            print(f"Caption validation examples: {len(val_ds):,}")
        sizes["caption"] = len(caption_vocab)

    if not any(k.endswith("_train") for k in loaders):
        raise ValueError("Both VQA and caption training are disabled in config.yaml.")
    return loaders, sizes

def vqa_soft_accuracy(logits, targets):
    """VQA consensus-style accuracy from the dataset's soft answer targets."""
    predictions = logits.argmax(dim=1)
    return targets.gather(1, predictions.unsqueeze(1)).mean()


def caption_token_accuracy(logits, targets, pad_id):
    """Token accuracy excluding padding positions."""
    predictions = logits.argmax(dim=-1)
    valid = targets.ne(pad_id)

    if not valid.any():
        return 0.0

    return (
        (predictions.eq(targets) & valid).sum().float()
        / valid.sum()
    ).item()


def tokenize_caption(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def corpus_bleu(references, hypotheses, max_n=4):
    """Lightweight corpus BLEU-1 through BLEU-4; no extra package required."""
    clipped = [0] * max_n
    totals = [0] * max_n
    ref_length = 0
    hyp_length = 0

    for reference, hypothesis in zip(references, hypotheses):
        ref = tokenize_caption(reference)
        hyp = tokenize_caption(hypothesis)

        ref_length += len(ref)
        hyp_length += len(hyp)

        for n in range(1, max_n + 1):
            hyp_ngrams = Counter(
                tuple(hyp[i:i+n])
                for i in range(max(0, len(hyp) - n + 1))
            )
            ref_ngrams = Counter(
                tuple(ref[i:i+n])
                for i in range(max(0, len(ref) - n + 1))
            )

            clipped[n - 1] += sum(
                min(count, ref_ngrams[gram])
                for gram, count in hyp_ngrams.items()
            )
            totals[n - 1] += sum(hyp_ngrams.values())

    brevity = (
        1.0 if hyp_length >= ref_length
        else math.exp(1 - ref_length / max(hyp_length, 1))
    )

    scores = []
    for n in range(1, max_n + 1):
        precisions = [
            max(clipped[i], 1e-12) / max(totals[i], 1)
            for i in range(n)
        ]
        scores.append(
            brevity * math.exp(
                sum(math.log(p) for p in precisions) / n
            )
        )

    return {
        f"caption_val_bleu_{n}": scores[n - 1]
        for n in range(1, max_n + 1)
    }


def train_vqa_epoch(
    model, loader, optimizer, device, scaler, use_amp, grad_clip, max_batches
):
    model.train()
    total = 0.0
    batches = 0
    accuracy_sum = 0.0
    example_count = 0
    progress = tqdm(loader, desc="VQA", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        questions = batch["question_ids"].to(device, non_blocking=True)
        targets = batch["answer_target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model.forward_vqa(images, questions, return_attention=False)[
                "logits"
            ]
            loss = (
                F.binary_cross_entropy_with_logits(
                    logits.float(), targets.float(), reduction="none"
                )
                .sum(dim=1)
                .mean()
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite VQA loss: {loss.item()}")
        with torch.no_grad():
            batch_accuracy = vqa_soft_accuracy(
                logits.detach().float(), targets.float()
            )
            accuracy_sum += batch_accuracy.item() * images.size(0)
            example_count += images.size(0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
        batches += 1
        progress.set_postfix(
            loss=f"{loss.item():.3f}",
            soft_acc=f"{accuracy_sum / max(example_count, 1) * 100:.2f}%"
        )
        if max_batches and batches >= max_batches:
            break
    if batches == 0:
        raise RuntimeError("VQA training processed zero batches.")
    return {
        "vqa_train_loss": total / batches,
        "vqa_train_soft_accuracy": accuracy_sum / max(example_count, 1),
    }


def train_caption_epoch(
    model, loader, optimizer, device, scaler, use_amp, grad_clip, pad_id, max_batches
):
    model.train()
    total = 0.0
    batches = 0
    correct_tokens = 0
    valid_tokens = 0
    progress = tqdm(loader, desc="Caption", leave=False)
    for batch in progress:
        images = batch["images"].to(device, non_blocking=True)
        inputs = batch["caption_input_ids"].to(device, non_blocking=True)
        targets = batch["caption_targets"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model.forward_caption(images, inputs)
            if logits.shape[:2] != targets.shape:
                raise ValueError(
                    f"Caption logits {tuple(logits.shape)} do not match targets {tuple(targets.shape)}"
                )
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=pad_id,
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite caption loss: {loss.item()}")
        predictions = logits.detach().argmax(dim=-1)
        valid = targets.ne(pad_id)

        correct_tokens += int(
            (predictions.eq(targets) & valid).sum()
        )
        valid_tokens += int(valid.sum())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
        batches += 1
        progress.set_postfix(
            loss=f"{loss.item():.3f}",
            token_acc=f"{correct_tokens / max(valid_tokens, 1) * 100:.2f}%"
        )
        if max_batches and batches >= max_batches:
            break
    if batches == 0:
        raise RuntimeError("Caption training processed zero batches.")
    return {
        "caption_train_loss": total / batches,
        "caption_train_token_accuracy": (
            correct_tokens / max(valid_tokens, 1)
        ),
    }


@torch.no_grad()
def validate_vqa(model, loader, device, max_batches=0):
    model.eval()
    score_sum = 0.0
    loss_sum = 0.0
    count = 0
    for i, batch in enumerate(tqdm(loader, desc="Validate VQA", leave=False), start=1):
        images = batch["image"].to(device, non_blocking=True)
        questions = batch["question_ids"].to(device, non_blocking=True)
        targets = batch["answer_target"].to(device, non_blocking=True)
        logits = model.forward_vqa(images, questions, return_attention=False)["logits"]
        loss = (
            F.binary_cross_entropy_with_logits(
                logits.float(), targets.float(), reduction="none"
            )
            .sum(dim=1)
            .mean()
        )
        pred = logits.argmax(dim=1)
        scores = targets.gather(1, pred.unsqueeze(1)).squeeze(1)
        batch_n = images.size(0)
        count += batch_n
        score_sum += float(scores.sum())
        loss_sum += float(loss) * batch_n
        if max_batches and i >= max_batches:
            break
    if not count:
        raise RuntimeError("VQA validation processed zero examples.")
    return {
        "vqa_val_soft_accuracy": score_sum / count,
        "vqa_val_loss": loss_sum / count,
    }


@torch.no_grad()
def validate_caption(model, loader, device, pad_id, max_batches=0):
    model.eval()
    loss_sum = 0.0
    count = 0
    correct_tokens = 0
    for i, batch in enumerate(
        tqdm(loader, desc="Validate caption", leave=False), start=1
    ):
        images = batch["images"].to(device, non_blocking=True)
        inputs = batch["caption_input_ids"].to(device, non_blocking=True)
        targets = batch["caption_targets"].to(device, non_blocking=True)
        logits = model.forward_caption(images, inputs)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
            reduction="sum",
        )
        valid_tokens = int(targets.ne(pad_id).sum())
        loss_sum += float(loss)
        count += valid_tokens
        if max_batches and i >= max_batches:
            break
        predictions = logits.argmax(dim=-1)
        valid = targets.ne(pad_id)
    
        correct_tokens += int(
            (predictions.eq(targets) & valid).sum()
        )
    
    if not count:
        raise RuntimeError("Caption validation processed zero non-padding tokens.")
    
    return {
        "caption_val_token_loss": loss_sum / count,
        "caption_val_token_accuracy": correct_tokens / count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and validate the MAVIS multitask model."
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit batches per task for a smoke test; 0 means full datasets.",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Checkpoint path to resume."
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Start fresh history; does not delete existing checkpoints.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="Limit validation batches per task; 0 means the full validation datasets.",
    )
    args = parser.parse_args()
    if args.max_batches < 0:
        parser.error("--max-batches must be >= 0")
    if args.max_val_batches < 0:
        parser.error("--max-val-batches must be >= 0")

    config = load_config()
    set_seed(
        int(config.get("project", {}).get("seed", 42)),
        bool(config.get("runtime", {}).get("deterministic", False)),
    )
    runtime = config.get("runtime", {})
    requested = runtime.get("device", "auto")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "auto"
        else requested
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    print("Project: MAVIS")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    loaders, sizes = build_dataloaders(config, args)
    model = MAVIS(
        config=config,
        question_vocab_size=sizes.get("question", 1),
        answer_vocab_size=sizes.get(
            "answer", int(config["data"].get("num_answers", 7500))
        ),
        caption_vocab_size=sizes.get("caption", 1),
    ).to(device)
    answer_cfg, caption_cfg = (
        config["training"]["answer"],
        config["training"]["caption"],
    )
    optim_cfg = config.get("optimizer", {})
    # Independent AdamW states allow task-specific learning rates while preserving one shared model/ViT.
    answer_opt = (
        torch.optim.AdamW(
            model.parameters(),
            lr=float(answer_cfg["learning_rate"]),
            weight_decay=float(answer_cfg.get("weight_decay", 1e-4)),
        )
        if "vqa_train" in loaders
        else None
    )
    caption_opt = (
        torch.optim.AdamW(
            model.parameters(),
            lr=float(caption_cfg["learning_rate"]),
            weight_decay=float(caption_cfg.get("weight_decay", 1e-4)),
        )
        if "caption_train" in loaders
        else None
    )
    use_amp = device.type == "cuda" and (
        bool(answer_cfg.get("mixed_precision", False))
        or bool(caption_cfg.get("mixed_precision", False))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    if args.resume:
        cp = resolve_path(args.resume)
        if not cp.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {cp}")
        checkpoint = torch.load(cp, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if answer_opt and "answer_optimizer_state_dict" in checkpoint:
            answer_opt.load_state_dict(checkpoint["answer_optimizer_state_dict"])
        elif answer_opt and "optimizer_state_dict" in checkpoint:
            answer_opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if caption_opt and "caption_optimizer_state_dict" in checkpoint:
            caption_opt.load_state_dict(checkpoint["caption_optimizer_state_dict"])
        elif caption_opt and "optimizer_state_dict" in checkpoint:
            caption_opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resumed from {cp} at epoch {start_epoch}")
    epochs = max(
        int(answer_cfg.get("epochs", 0)) if "vqa_train" in loaders else 0,
        int(caption_cfg.get("epochs", 0)) if "caption_train" in loaders else 0,
    )
    if args.max_batches > 0:
        epochs = min(epochs, 1)
    if start_epoch > epochs:
        raise ValueError(
            f"Checkpoint epoch leads to start_epoch={start_epoch}, beyond configured epochs={epochs}. Increase epochs or use a different checkpoint."
        )
    output = resolve_path(config["project"].get("output_dir", "outputs"))
    ckpt_dir = resolve_path(
        config["project"].get("checkpoint_dir", "outputs/checkpoints")
    )
    results_dir = resolve_path(config["project"].get("results_dir", "outputs/results"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    resolve_path(config["project"].get("log_dir", "outputs/logs")).mkdir(
        parents=True, exist_ok=True
    )
    history_path = results_dir / "training_history.json"
    if history_path.exists() and args.resume and not args.new_run:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = []
    best_vqa = float("-inf")
    best_caption = float("inf")
    pad_id = int(config["data"]["caption_special_token_ids"]["pad"])
    validation_cfg = config.get("validation", {})
    validate_enabled = bool(validation_cfg.get("enabled", True))
    val_every = max(1, int(validation_cfg.get("every_n_epochs", 1)))
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*18} Epoch {epoch}/{epochs} {'='*18}")
        record = {"epoch": epoch}
        if "vqa_train" in loaders and epoch <= int(answer_cfg["epochs"]):
            print("\nTraining VQA branch...")
            metrics = train_vqa_epoch(
                model,
                loaders["vqa_train"],
                answer_opt,
                device,
                scaler,
                use_amp,
                answer_cfg.get("gradient_clip_norm", 1.0),
                args.max_batches,
            )

            record.update(metrics)

            print(f"VQA training loss: {metrics['vqa_train_loss']:.4f}")
            print(
                "VQA training soft accuracy: "
                f"{metrics['vqa_train_soft_accuracy']:.4f}"
            )
        if "caption_train" in loaders and epoch <= int(caption_cfg["epochs"]):
            print("\nTraining caption branch...")
            metrics = train_caption_epoch(
                model,
                loaders["caption_train"],
                caption_opt,
                device,
                scaler,
                use_amp,
                caption_cfg.get("gradient_clip_norm", 1.0),
                pad_id,
                args.max_batches,
            )

            record.update(metrics)

            print(f"Caption training loss: {metrics['caption_train_loss']:.4f}")
            print(
                "Caption training token accuracy: "
                f"{metrics['caption_train_token_accuracy']:.4f}"
            )
        if validate_enabled and epoch % val_every == 0:
            if "vqa_val" in loaders:
                metrics = validate_vqa(
                    model,
                    loaders["vqa_val"],
                    device,
                    max_batches=args.max_val_batches,
                )
                record.update(metrics)
                print(
                    f"VQA validation soft accuracy: {metrics['vqa_val_soft_accuracy']:.4f}; loss: {metrics['vqa_val_loss']:.4f}"
                )
                if metrics["vqa_val_soft_accuracy"] > best_vqa:
                    best_vqa = metrics["vqa_val_soft_accuracy"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "metric": best_vqa,
                        },
                        resolve_path(
                            config["paths"]
                            .get("outputs", {})
                            .get(
                                "answer_checkpoint",
                                "outputs/checkpoints/mavis_answer_best.pt",
                            )
                        ),
                    )
            if "caption_val" in loaders:
                metrics = validate_caption(
                    model,
                    loaders["caption_val"],
                    device,
                    pad_id,
                    max_batches=args.max_val_batches,
                )
                record.update(metrics)
                print(
                    f"Caption validation token loss: {metrics['caption_val_token_loss']:.4f}"
                )
                if metrics["caption_val_token_loss"] < best_caption:
                    best_caption = metrics["caption_val_token_loss"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "config": config,
                            "metric": best_caption,
                        },
                        resolve_path(
                            config["paths"]
                            .get("outputs", {})
                            .get(
                                "caption_checkpoint",
                                "outputs/checkpoints/mavis_caption_best.pt",
                            )
                        ),
                    )
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "answer_optimizer_state_dict": (
                answer_opt.state_dict() if answer_opt else None
            ),
            "caption_optimizer_state_dict": (
                caption_opt.state_dict() if caption_opt else None
            ),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "history": history,
        }
        latest_path = ckpt_dir / "mavis_multitask_last.pt"
        torch.save(checkpoint, latest_path)
        if answer_cfg.get("save_every_epoch", True) or caption_cfg.get(
            "save_every_epoch", True
        ):
            torch.save(checkpoint, ckpt_dir / f"mavis_multitask_epoch_{epoch:02d}.pt")
        print(f"Checkpoint saved: {latest_path}")
        print(f"Loss history saved: {history_path}")
    print("\nMAVIS training completed.")
    if args.max_batches > 0:
        print(
            "Sm  oke test only: validation skipped; no full-dataset or model-quality claim should be inferred."
        )


if __name__ == "__main__":
    main()
