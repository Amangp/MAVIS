from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def normalize_question(text: str) -> str:
    """Must match the tokenization used when creating word2idx.pkl."""
    text = text.lower().strip()
    text = re.sub(r"([?.!,;:])", r" \1 ", text)
    return re.sub(r"\s+", " ", text)


def normalize_answer(text: str) -> str:
    """Must match the normalization used when creating ans2idx.pkl."""
    return " ".join(str(text).lower().strip().split())


def load_pickle(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def default_image_transform(image_size: int = 224):
    # Generic ImageNet normalization; change only if your model's training
    # setup specifies another normalization.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


class MAVISVQAv2Dataset(Dataset):

    def __init__(
        self,
        questions_file: str | Path,
        annotations_file: str | Path,
        image_root: str | Path,
        word_vocab_file: str | Path,
        answer_vocab_file: str | Path,
        max_question_length: int = 32,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        strict_images: bool = True,
        return_all_answers: bool = False,
    ):
        self.questions_file = Path(questions_file)
        self.annotations_file = Path(annotations_file)
        self.image_root = Path(image_root)
        self.word2idx = load_pickle(Path(word_vocab_file))
        self.ans2idx = load_pickle(Path(answer_vocab_file))
        self.max_question_length = int(max_question_length)
        self.transform = transform or default_image_transform(image_size)
        self.strict_images = strict_images
        self.return_all_answers = return_all_answers

        if self.max_question_length <= 0:
            raise ValueError("max_question_length must be positive.")
        if PAD_TOKEN not in self.word2idx or UNK_TOKEN not in self.word2idx:
            raise ValueError("word2idx.pkl must contain <PAD> and <UNK> tokens.")
        if not self.ans2idx:
            raise ValueError("ans2idx.pkl is empty.")
        ids = sorted(self.ans2idx.values())
        if ids != list(range(len(ids))):
            raise ValueError("ans2idx values must be contiguous IDs starting at 0.")

        qdata = load_json(self.questions_file)
        adata = load_json(self.annotations_file)
        question_records = qdata.get("questions", [])
        annotation_records = adata.get("annotations", [])
        if not question_records or not annotation_records:
            raise ValueError("Question or annotation list is empty.")

        annotations_by_qid = {
            a["question_id"]: a
            for a in annotation_records
            if "question_id" in a
        }

        self.samples = []
        missing_annotations = 0
        for q in question_records:
            qid = q.get("question_id")
            ann = annotations_by_qid.get(qid)
            if ann is None:
                missing_annotations += 1
                continue

            image_id = int(q.get("image_id", ann.get("image_id", -1)))
            if image_id < 0:
                continue

            data_subtype = qdata.get("data_subtype")
            if not data_subtype:
                data_subtype = adata.get("data_subtype")
            if not data_subtype:
                # Most common official files use train2014 / val2014.
                data_subtype = "train2014" if "train" in self.questions_file.name.lower() else "val2014"

            image_path = self.image_root / f"COCO_{data_subtype}_{image_id:012d}.jpg"
            if self.strict_images and not image_path.is_file():
                raise FileNotFoundError(
                    f"Image missing for question_id={qid}: {image_path}\n"
                    "Pass image_root as the directory containing COCO_*.jpg files."
                )

            answers = []
            for item in ann.get("answers", []):
                raw = item.get("answer", "") if isinstance(item, dict) else item
                answer = normalize_answer(raw)
                if answer:
                    answers.append(answer)

            self.samples.append({
                "question_id": int(qid),
                "image_id": image_id,
                "question": str(q.get("question", "")),
                "image_path": image_path,
                "answers": answers,
            })

        if not self.samples:
            raise ValueError("No usable question/annotation pairs were found.")

        self.missing_annotations = missing_annotations
        self.num_answer_classes = len(self.ans2idx)
        self.pad_idx = int(self.word2idx[PAD_TOKEN])
        self.unk_idx = int(self.word2idx[UNK_TOKEN])

    def __len__(self):
        return len(self.samples)

    def encode_question(self, question: str) -> tuple[torch.Tensor, int]:
        tokens = normalize_question(question).split()
        tokens = tokens[: self.max_question_length]
        ids = [self.word2idx.get(token, self.unk_idx) for token in tokens]
        length = len(ids)
        if length < self.max_question_length:
            ids.extend([self.pad_idx] * (self.max_question_length - length))
        return torch.tensor(ids, dtype=torch.long), length

    def make_soft_target(self, answers: list[str]) -> torch.Tensor:
        """
        VQA consensus target: for each reference answer a,
        score(a) = min(number of matching answers / 3, 1).
        Only answers represented in ans2idx can be output by the classifier.
        """
        counts = Counter(answer for answer in answers if answer in self.ans2idx)
        target = torch.zeros(self.num_answer_classes, dtype=torch.float32)
        for answer, count in counts.items():
            target[self.ans2idx[answer]] = min(count / 3.0, 1.0)
        return target

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        if not sample["image_path"].is_file():
            raise FileNotFoundError(f"Image not found: {sample['image_path']}")

        with Image.open(sample["image_path"]) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        question_ids, question_length = self.encode_question(sample["question"])
        soft_target = self.make_soft_target(sample["answers"])

        result = {
            "image": image,
            "question_ids": question_ids,
            "question_length": torch.tensor(question_length, dtype=torch.long),
            "answer_target": soft_target,
            "question_id": torch.tensor(sample["question_id"], dtype=torch.long),
            "image_id": torch.tensor(sample["image_id"], dtype=torch.long),
            "question_text": sample["question"],
        }
        if self.return_all_answers:
            result["answers"] = sample["answers"]
        return result


def mavis_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors while keeping question text and optional answers as lists."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    result = {}
    tensor_keys = [
        "image", "question_ids", "question_length", "answer_target",
        "question_id", "image_id",
    ]
    for key in tensor_keys:
        result[key] = torch.stack([item[key] for item in batch], dim=0)
    result["question_text"] = [item["question_text"] for item in batch]
    if "answers" in batch[0]:
        result["answers"] = [item.get("answers", []) for item in batch]
    return result


def build_vqa_v2_datasets(
    project_root: str | Path,
    split: str = "train",
    image_size: int = 224,
    max_question_length: int = 32,
    strict_images: bool = True,
):
    """
    Convenience constructor for official VQA v2 COCO 2014 files.
    split='train' or 'val'. Adjust folder names here if your dataset differs.
    """
    root = Path(project_root)
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")

    subtype = "train2014" if split == "train" else "val2014"
    question_dir = root / "datasets" / f"v2_Questions_{'Train' if split == 'train' else 'Val'}_mscoco"
    annotation_dir = root / "datasets" / f"v2_Annotations_{'Train' if split == 'train' else 'Val'}_mscoco"
    questions_file = question_dir / f"v2_OpenEnded_mscoco_{subtype}_questions.json"
    annotations_file = annotation_dir / f"v2_mscoco_{subtype}_annotations.json"
    image_root = root / "datasets" / subtype

    return MAVISVQAv2Dataset(
        questions_file=questions_file,
        annotations_file=annotations_file,
        image_root=image_root,
        word_vocab_file=root / "vocab" / "word2idx.pkl",
        answer_vocab_file=root / "vocab" / "ans2idx.pkl",
        max_question_length=max_question_length,
        image_size=image_size,
        strict_images=strict_images,
    )


# Backward-compatible aliases for projects using these names.
VQADataset = MAVISVQAv2Dataset
collate_fn = mavis_collate_fn
