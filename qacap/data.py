"""Dataset and collation utilities for QACap caption training.

The dataset supplies the three inputs required by the QACap captioning path:
image pixels, the original question (for the question encoder/QAFD), and a
BLIP-decoder sequence prefixed with the paper's PET question prompt.

Expected JSONL fields per row:
    image:   path to the image (absolute or relative to ``root``)
    question: original KBVQA question
    caption:  selected question-aware target caption (without the PET prompt)
Optional fields:
    answers:  list of reference answers
    image_id: source dataset image identifier

Important: the caption decoder sequence is built using the BLIP tokenizer,
not the separate BERT tokenizer used by QAFD. Prompt tokens are context only;
caption tokens and the end-of-sequence token are the supervised targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


PET_PROMPT_TEMPLATE = (
    "Please describe this image according to the following question: {question}"
)


class QACapDataset(Dataset):
    """Load QACap image/question/caption triplets from a JSONL file.

    ``caption`` must contain only the target question-aware caption. The PET
    question prompt is added here as a decoder prefix and must not be duplicated
    inside the stored caption.

    Args:
        jsonl: JSONL manifest containing one example per line.
        root: Base directory for relative image paths.
        image_size: Optional square image size override. When ``None``, the
            pretrained BLIP image processor's configured size is retained.
        max_samples: Optional positive limit. ``None`` or ``0`` uses all rows.
        blip_name: BLIP checkpoint whose processor/tokenizer is used for images
            and caption-decoder tokens.
        max_caption_length: Optional maximum number of caption target tokens,
            including EOS. ``None`` keeps the complete caption. This is a data
            truncation setting, not a value specified in the paper's main text.
    """

    def __init__(
        self,
        jsonl: Union[str, Path],
        root: Union[str, Path] = ".",
        image_size: Optional[int] = None,
        max_samples: Optional[int] = None,
        blip_name: str = "Salesforce/blip-image-captioning-base",
        max_caption_length: Optional[int] = None,
    ) -> None:
        self.manifest_path = Path(jsonl)
        self.root = Path(root)

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"QACap JSONL manifest not found: {self.manifest_path.resolve()}"
            )
        if max_samples is not None and int(max_samples) < 0:
            raise ValueError("max_samples must be None, 0, or a positive integer")
        if max_caption_length is not None and int(max_caption_length) < 1:
            raise ValueError("max_caption_length must be None or at least 1")
        if image_size is not None and int(image_size) < 1:
            raise ValueError("image_size must be None or a positive integer")

        self.rows: List[Dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {self.manifest_path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise TypeError(
                        f"Expected a JSON object at line {line_number} of {self.manifest_path}"
                    )
                missing = [key for key in ("image", "question", "caption") if key not in row]
                if missing:
                    raise KeyError(
                        f"Missing required field(s) {missing} at line {line_number}"
                    )
                self.rows.append(row)

        if max_samples:
            self.rows = self.rows[: int(max_samples)]
        if not self.rows:
            raise ValueError(f"No usable examples found in {self.manifest_path}")

        try:
            from transformers import AutoConfig, BlipProcessor
        except ImportError as exc:
            raise ImportError(
                "QACapDataset requires Hugging Face Transformers. "
                "Install the project's requirements before loading datasets."
            ) from exc

        self.processor = BlipProcessor.from_pretrained(blip_name)
        self.tokenizer = self.processor.tokenizer
        self.blip_config = AutoConfig.from_pretrained(blip_name)
        text_config = getattr(self.blip_config, "text_config", self.blip_config)
        self.max_caption_length = (
            int(max_caption_length) if max_caption_length is not None else None
        )

        if image_size is not None:
            # BLIP's image processor expects a height/width size dictionary.
            self.processor.image_processor.size = {
                "height": int(image_size),
                "width": int(image_size),
            }

        # BLIP uses a BERT-derived tokenizer in common checkpoints. Such a
        # tokenizer may expose CLS/SEP rather than BOS/EOS, while the decoder
        # config may provide decoder_start_token_id. Resolve IDs in the same
        # priority order as QACap.model to avoid data/model token mismatches.
        self.pad_token_id = self._first_defined(
            self.tokenizer.pad_token_id,
            getattr(text_config, "pad_token_id", None),
        )
        self.bos_token_id = self._first_defined(
            self.tokenizer.bos_token_id,
            getattr(text_config, "decoder_start_token_id", None),
            getattr(text_config, "bos_token_id", None),
            self.tokenizer.cls_token_id,
        )
        self.eos_token_id = self._first_defined(
            self.tokenizer.eos_token_id,
            self.tokenizer.sep_token_id,
            getattr(text_config, "eos_token_id", None),
        )
        self.decoder_max_positions = getattr(
            text_config, "max_position_embeddings", None
        )

        if self.pad_token_id is None:
            raise ValueError("Could not resolve the BLIP decoder PAD token ID")
        if self.bos_token_id is None:
            raise ValueError("Could not resolve the BLIP decoder BOS/decoder-start token ID")
        if self.eos_token_id is None:
            raise ValueError("Could not resolve the BLIP decoder EOS/SEP token ID")

        # Guard against accidentally pairing a tokenizer and decoder with
        # incompatible special-token conventions.
        tokenizer_vocab_size = len(self.tokenizer)
        if self.eos_token_id >= tokenizer_vocab_size or self.bos_token_id >= tokenizer_vocab_size:
            raise ValueError("BLIP tokenizer special-token IDs are outside its vocabulary")

    @staticmethod
    def _first_defined(*values: Optional[int]) -> Optional[int]:
        for value in values:
            if value is not None:
                return int(value)
        return None

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image_path(self, image_value: Union[str, Path], index: int) -> Path:
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = self.root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image for dataset row {index} was not found: {image_path.resolve()}"
            )
        return image_path

    def _encode_caption_with_pet(
        self, question: str, caption: str, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create BLIP decoder inputs and aligned next-token labels.

        Decoder input: [BOS] + PET(question) + caption_tokens
        Labels:        [-100 for prompt context] + caption_tokens + [EOS]

        The label at the final decoder position teaches the model to emit EOS.
        Prompt tokens are never treated as caption targets.
        """
        prompt = PET_PROMPT_TEMPLATE.format(question=question)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        caption_ids = self.tokenizer.encode(caption, add_special_tokens=False)

        # max_caption_length includes the EOS target token.
        if self.max_caption_length is not None:
            content_limit = max(self.max_caption_length - 1, 0)
            caption_ids = caption_ids[:content_limit]

        if not caption_ids:
            raise ValueError(
                f"Caption at dataset row {index} is empty after tokenization/truncation"
            )

        decoder_ids = [self.bos_token_id] + prompt_ids + caption_ids
        labels = ([-100] * len(prompt_ids)) + caption_ids + [self.eos_token_id]

        if len(decoder_ids) != len(labels):
            raise RuntimeError(
                "Internal decoder/label alignment error: "
                f"{len(decoder_ids)} input tokens vs {len(labels)} labels"
            )

        # Do not silently truncate the PET prompt. If the combined sequence is
        # too long for the pretrained BLIP text decoder, report the offending row.
        limits = []
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            limits.append(tokenizer_limit)
        if isinstance(self.decoder_max_positions, int) and self.decoder_max_positions > 0:
            limits.append(self.decoder_max_positions)
        if limits:
            effective_limit = min(limits)
            if len(decoder_ids) > effective_limit:
                raise ValueError(
                    f"Decoder sequence at dataset row {index} has {len(decoder_ids)} tokens, "
                    f"exceeding the BLIP decoder limit of {effective_limit}. "
                    "Set an explicit max_caption_length or review the question length; "
                    "the PET prompt is not silently removed."
                )

        decoder_input_ids = torch.tensor(decoder_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long)
        return decoder_input_ids, decoder_attention_mask, labels_tensor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        question = str(row["question"]).strip()
        caption = str(row["caption"]).strip()
        if not question:
            raise ValueError(f"Question at dataset row {idx} is empty")
        if not caption:
            raise ValueError(f"Caption at dataset row {idx} is empty")

        image_path = self._resolve_image_path(row["image"], idx)
        # Load/copy the image while the file handle is open, and honor EXIF
        # orientation before sending it to the pretrained BLIP image processor.
        with Image.open(image_path) as image_file:
            image = ImageOps.exif_transpose(image_file).convert("RGB")

        image_encoding = self.processor(images=image, return_tensors="pt")
        pixel_values = image_encoding.pixel_values[0]

        decoder_input_ids, decoder_attention_mask, labels = (
            self._encode_caption_with_pet(question, caption, idx)
        )

        answers = row.get("answers", [])
        if answers is None:
            answers = []
        elif isinstance(answers, str):
            answers = [answers]

        pet_prompt = PET_PROMPT_TEMPLATE.format(question=question)
        return {
            "pixel_values": pixel_values,
            "question": question,
            "pet_prompt": pet_prompt,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "caption": caption,
            "answers": answers,
            "image_id": row.get("image_id", idx),
        }


def collate_qacap(
    batch: Sequence[Dict[str, Any]], pad_token_id: int
) -> Dict[str, Any]:
    """Pad variable-length QACap examples into a batch.

    ``labels`` use -100 for PET-prompt positions and batch padding, matching
    PyTorch cross-entropy's default ignore index and Hugging Face BLIP labels.
    """
    if not batch:
        raise ValueError("Cannot collate an empty QACap batch")
    if pad_token_id is None:
        raise ValueError("pad_token_id must be the BLIP tokenizer's padding ID")

    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    decoder_input_ids = pad_sequence(
        [item["decoder_input_ids"] for item in batch],
        batch_first=True,
        padding_value=int(pad_token_id),
    )
    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    decoder_attention_mask = decoder_input_ids.ne(int(pad_token_id)).long()

    # Ensure padding is ignored even if a future data-preparation change adds
    # padded decoder inputs before collation.
    labels = labels.masked_fill(decoder_attention_mask.eq(0), -100)

    return {
        "pixel_values": pixel_values,
        "questions": [item["question"] for item in batch],
        "pet_prompts": [item["pet_prompt"] for item in batch],
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
        "captions": [item["caption"] for item in batch],
        "answers": [item["answers"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
    }
