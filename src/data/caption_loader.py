from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = SPECIAL_TOKENS


def tokenize_caption(text: str) -> list[str]:
    """Simple lowercase tokenizer that keeps words and punctuation as tokens."""
    if not isinstance(text, str):
        raise TypeError("Caption must be a string.")
    text = text.lower().strip()
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def build_caption_vocab(
    annotation_json: str | Path,
    min_freq: int = 5,
    max_vocab_size: Optional[int] = 20000,
) -> dict[str, int]:
    """Build a deterministic vocabulary from a COCO captions JSON file."""
    if min_freq < 1:
        raise ValueError("min_freq must be >= 1.")
    if max_vocab_size is not None and max_vocab_size < len(SPECIAL_TOKENS):
        raise ValueError("max_vocab_size must allow all special tokens.")

    annotation_json = Path(annotation_json)
    if not annotation_json.is_file():
        raise FileNotFoundError(f"Caption annotation file not found: {annotation_json}")

    with annotation_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
        raise ValueError("Expected COCO captions JSON with an 'annotations' list.")

    counter: Counter[str] = Counter()
    for ann in data["annotations"]:
        caption = ann.get("caption")
        if not isinstance(caption, str):
            continue
        counter.update(tokenize_caption(caption))

    # Special tokens always occupy stable, documented IDs.
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    candidates = [
        (word, count)
        for word, count in counter.items()
        if count >= min_freq and word not in vocab
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))

    if max_vocab_size is not None:
        candidates = candidates[: max_vocab_size - len(vocab)]

    for word, _ in candidates:
        vocab[word] = len(vocab)

    return vocab


def save_caption_vocab(vocab: dict[str, int], path: str | Path) -> None:
    """Save vocabulary as JSON so it is portable and inspectable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_caption_vocab(path: str | Path) -> dict[str, int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Caption vocabulary not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        vocab = json.load(f)

    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Caption vocabulary must be a non-empty token-to-ID object.")

    try:
        vocab = {str(token): int(idx) for token, idx in vocab.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Vocabulary IDs must be integers.") from exc

    if len(set(vocab.values())) != len(vocab):
        raise ValueError("Vocabulary IDs must be unique.")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary IDs must be contiguous from 0.")
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            raise ValueError(f"Vocabulary is missing required token {token}.")
    return vocab


class COCOCaptionDataset(Dataset):

    def __init__(
        self,
        annotation_json: str | Path,
        image_root: str | Path,
        vocab: Optional[dict[str, int]] = None,
        vocab_json: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
        max_length: int = 40,
        strict_images: bool = True,
    ) -> None:
        self.annotation_json = Path(annotation_json)
        self.image_root = Path(image_root)
        self.transform = transform
        self.max_length = max_length
        self.strict_images = strict_images

        if max_length < 2:
            raise ValueError("max_length must be at least 2 for BOS and EOS.")
        if not self.annotation_json.is_file():
            raise FileNotFoundError(f"Caption annotation file not found: {self.annotation_json}")
        if vocab is not None and vocab_json is not None:
            raise ValueError("Provide vocab or vocab_json, not both.")
        if vocab is None:
            if vocab_json is None:
                raise ValueError("Provide a caption vocabulary or vocab_json.")
            vocab = load_caption_vocab(vocab_json)

        self.vocab = {str(k): int(v) for k, v in vocab.items()}
        if len(set(self.vocab.values())) != len(self.vocab):
            raise ValueError("Vocabulary IDs must be unique.")
        if sorted(self.vocab.values()) != list(range(len(self.vocab))):
            raise ValueError("Vocabulary IDs must be contiguous from 0.")
        for token in SPECIAL_TOKENS:
            if token not in self.vocab:
                raise ValueError(f"Vocabulary is missing required token {token}.")

        self.pad_id = self.vocab[PAD_TOKEN]
        self.bos_id = self.vocab[BOS_TOKEN]
        self.eos_id = self.vocab[EOS_TOKEN]
        self.unk_id = self.vocab[UNK_TOKEN]

        with self.annotation_json.open("r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images")
        annotations = data.get("annotations")
        if not isinstance(images, list) or not isinstance(annotations, list):
            raise ValueError("Expected COCO captions JSON with 'images' and 'annotations' lists.")

        self.image_names: dict[int, str] = {}
        for image in images:
            try:
                image_id = int(image["id"])
                filename = str(image["file_name"])
            except (KeyError, TypeError, ValueError):
                continue
            self.image_names[image_id] = filename

        self.samples: list[tuple[int, str]] = []
        skipped = 0
        for ann in annotations:
            caption = ann.get("caption")
            try:
                image_id = int(ann["image_id"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if not isinstance(caption, str) or not caption.strip():
                skipped += 1
                continue
            if image_id not in self.image_names:
                skipped += 1
                continue
            self.samples.append((image_id, caption))

        if not self.samples:
            raise ValueError("No usable caption annotations were found.")
        self.skipped_annotations = skipped

    def __len__(self) -> int:
        return len(self.samples)

    def _image_path(self, image_id: int) -> Path:
        filename = self.image_names[image_id]
        path = self.image_root / filename
        if path.is_file():
            return path
        
        raise FileNotFoundError(
            f"Image file not found for image_id={image_id}: {path}"
        )

    def encode_caption(self, caption: str) -> torch.Tensor:
        tokens = tokenize_caption(caption)
        # Reserve two slots for BOS and EOS; truncate content but retain EOS.
        tokens = tokens[: self.max_length - 2]
        ids = [self.bos_id]
        ids.extend(self.vocab.get(token, self.unk_id) for token in tokens)
        ids.append(self.eos_id)
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        if not isinstance(index, int):
            raise TypeError("Dataset index must be an integer.")
        if index < 0 or index >= len(self.samples):
            raise IndexError("Caption dataset index out of range.")

        image_id, caption = self.samples[index]
        image_path = self._image_path(image_id)

        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        caption_ids = self.encode_caption(caption)

        return {
            "image": image,
            "caption_ids": caption_ids,
            "image_id": image_id,
            "caption": caption,
        }


def caption_collate_fn(
    batch: list[dict],
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor | list[str]]:
    """Pad captions in a batch and prepare teacher-forcing inputs/targets."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    images = [item["image"] for item in batch]
    captions = [item["caption_ids"] for item in batch]

    if not all(isinstance(image, torch.Tensor) for image in images):
        raise TypeError("Apply a transform that converts every image to a tensor.")
    if not all(image.ndim == 3 for image in images):
        raise ValueError("Each image tensor must have shape [C, H, W].")
    image_shapes = {tuple(image.shape) for image in images}
    if len(image_shapes) != 1:
        raise ValueError("Images in a batch must have identical tensor shapes.")
    if not all(isinstance(caption, torch.Tensor) and caption.ndim == 1 for caption in captions):
        raise TypeError("caption_ids must be one-dimensional tensors.")

    images_tensor = torch.stack(images, dim=0)
    max_len = max(caption.numel() for caption in captions)
    padded = torch.full(
        (len(captions), max_len),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    for row, caption in enumerate(captions):
        padded[row, : caption.numel()] = caption

    if max_len < 2:
        raise ValueError("Caption sequences must include at least BOS and EOS.")

    # Teacher forcing: input predicts the next token.
    caption_input_ids = padded[:, :-1]
    caption_targets = padded[:, 1:]

    return {
        "images": images_tensor,
        "caption_ids": padded,
        "caption_input_ids": caption_input_ids,
        "caption_targets": caption_targets,
        "image_ids": torch.tensor([int(item["image_id"]) for item in batch]),
        "captions": [str(item["caption"]) for item in batch],
    }
