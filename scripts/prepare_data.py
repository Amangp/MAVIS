"""
Prepare and validate the final QACap training manifest.

This script operates on the selected-caption JSONL produced by
generate_qacap_candidates.py. It validates the fields consumed by QACapDataset,
checks that each sample has the five human COCO captions used by the paper,
resolves image paths, verifies image readability, checks candidate-selection
metadata when present, and writes a normalized JSONL atomically.

Important scope boundary:
- This script does NOT infer or merge the raw OK-VQA, A-OKVQA, and FVQA source
  annotation formats. The original datasets use different schemas, and the
  paper-specific 29,569-example construction must be implemented against the
  actual files and their exact formats.
- Run the source-specific import/merge stage first, then candidate generation
  and selection, then this manifest validation step.
- No train/validation split is created here: the paper describes using the
  KBVQA training splits to construct the QACap dataset, and its main text does
  not specify a new random split for the 29,569 synthesized examples.

Expected input JSONL fields:
    image          image path, absolute or relative to --root
    question       original KBVQA question
    answers        non-empty answer annotations
    coco_captions  exactly five human-written COCO captions
    caption        final selected question-aware caption

Recommended additional provenance:
    question_id, image_id, source_dataset, raw_caption_candidates,
    caption_candidates, candidate_selection

Output retains all fields and normalizes image paths to a path relative to
--root when possible. The resulting `image`, `question`, `caption`, `answers`
and `image_id` fields are compatible with QACapDataset.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageOps, UnidentifiedImageError


REQUIRED_FIELDS = ("image", "question", "answers", "coco_captions", "caption")
EXPECTED_COCO_CAPTION_COUNT = 5
SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}
SOURCE_DATASETS = {"okvqa", "a-okvqa", "aokvqa", "fvqa"}


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read JSONL with line-numbered errors."""
    if not path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}, "
                    f"got {type(row).__name__}."
                )
            rows.append((line_number, row))
    return rows


def _nonempty_text(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Line {line_number}: '{field}' must be a non-empty string."
        )
    return value.strip()


def _normalize_answers(value: Any, line_number: int) -> list[Any]:
    """Validate supported answer representations without changing annotations."""
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"Line {line_number}: 'answers' is empty.")
        return [value.strip()]

    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError(
            f"Line {line_number}: 'answers' must be a string or a sequence."
        )

    answers = []
    for index, answer in enumerate(value):
        if isinstance(answer, str):
            if answer.strip():
                answers.append(answer.strip())
        elif isinstance(answer, Mapping):
            answer_text = answer.get("answer")
            if answer_text is None or not str(answer_text).strip():
                raise ValueError(
                    f"Line {line_number}: answers[{index}] has no non-empty 'answer'."
                )
            # Preserve original annotation fields (e.g. answer confidence).
            item = dict(answer)
            item["answer"] = str(answer_text).strip()
            answers.append(item)
        else:
            raise ValueError(
                f"Line {line_number}: answers[{index}] must be a string "
                "or an object containing 'answer'."
            )

    if not answers:
        raise ValueError(f"Line {line_number}: 'answers' contains no valid answers.")
    return answers


def _normalize_coco_captions(value: Any, line_number: int) -> list[str]:
    """Extract the five human COCO captions while preserving their order."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(
            f"Line {line_number}: 'coco_captions' must be a sequence of five captions."
        )

    captions: list[str] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            caption = item.strip()
        elif isinstance(item, Mapping):
            caption = str(item.get("caption", "")).strip()
        else:
            caption = ""
        if not caption:
            raise ValueError(
                f"Line {line_number}: coco_captions[{index}] is empty or malformed."
            )
        captions.append(caption)

    if len(captions) != EXPECTED_COCO_CAPTION_COUNT:
        raise ValueError(
            f"Line {line_number}: expected exactly five human COCO captions "
            f"as described in the paper; found {len(captions)}."
        )
    return captions


def _resolve_image_path(
    image_value: Any,
    root: Path,
    line_number: int,
    *,
    verify_image: bool,
) -> tuple[str, Path]:
    """Resolve image path and optionally verify that PIL can decode it."""
    if not isinstance(image_value, str) or not image_value.strip():
        raise ValueError(f"Line {line_number}: 'image' must be a non-empty path.")

    supplied = Path(image_value)
    resolved = supplied if supplied.is_absolute() else root / supplied
    resolved = resolved.resolve()

    if not resolved.is_file():
        raise FileNotFoundError(
            f"Line {line_number}: image file does not exist: {resolved}"
        )

    if resolved.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(
            f"Line {line_number}: unsupported image extension "
            f"'{resolved.suffix}' for {resolved}."
        )

    if verify_image:
        try:
            with Image.open(resolved) as image:
                image.verify()
            # Re-open and force a decode; verify() alone checks file structure.
            with Image.open(resolved) as image:
                image = ImageOps.exif_transpose(image)
                image.convert("RGB").load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError(
                f"Line {line_number}: image cannot be decoded: {resolved}: {exc}"
            ) from exc

    try:
        normalized_path = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        # Keep an absolute path if the image is outside the declared root.
        normalized_path = str(resolved)

    return normalized_path, resolved


def _validate_selection_metadata(
    row: Mapping[str, Any],
    selected_caption: str,
    line_number: int,
) -> None:
    """Check that stored candidate-selection metadata agrees with final caption."""
    candidates = row.get("caption_candidates")
    selection = row.get("candidate_selection")

    # Input may be a manually prepared manifest; metadata is checked when present.
    if selection is None:
        return
    if not isinstance(selection, Mapping):
        raise ValueError(
            f"Line {line_number}: 'candidate_selection' must be a JSON object."
        )
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            f"Line {line_number}: candidate_selection exists but "
            "'caption_candidates' is missing or empty."
        )

    selected_index = selection.get("selected_index")
    if not isinstance(selected_index, int) or not (0 <= selected_index < len(candidates)):
        raise ValueError(
            f"Line {line_number}: candidate_selection.selected_index is invalid."
        )

    selected_record = candidates[selected_index]
    if isinstance(selected_record, Mapping):
        selected_record_caption = selected_record.get("caption")
    else:
        selected_record_caption = selected_record

    if selected_record_caption != selected_caption:
        raise ValueError(
            f"Line {line_number}: final 'caption' does not match the caption "
            "at candidate_selection.selected_index."
        )


def _sample_key(row: Mapping[str, Any]) -> str:
    """Question-level key: image/question pair, optionally namespaced by dataset."""
    dataset = str(row.get("source_dataset", "")).strip().lower()
    image_id = row.get("image_id", row.get("image", ""))
    question_id = row.get("question_id", row.get("questionId", row.get("qid")))
    if question_id is not None:
        return f"{dataset}|qid:{question_id}"
    return f"{dataset}|image:{image_id}|question:{row.get('question', '')}"


def normalize_record(
    original: Mapping[str, Any],
    line_number: int,
    root: Path,
    *,
    verify_images: bool,
    require_candidate_selection: bool,
) -> dict[str, Any]:
    """Validate and normalize one selected-caption training record."""
    missing = [field for field in REQUIRED_FIELDS if field not in original]
    if missing:
        raise ValueError(
            f"Line {line_number}: missing required field(s): {', '.join(missing)}"
        )

    row = dict(original)
    row["question"] = _nonempty_text(row["question"], "question", line_number)
    row["caption"] = _nonempty_text(row["caption"], "caption", line_number)
    row["answers"] = _normalize_answers(row["answers"], line_number)
    row["coco_captions"] = _normalize_coco_captions(
        row["coco_captions"], line_number
    )

    if "source_dataset" in row:
        source = str(row["source_dataset"]).strip().lower()
        if source not in SOURCE_DATASETS:
            raise ValueError(
                f"Line {line_number}: unknown source_dataset '{row['source_dataset']}'. "
                "Expected OK-VQA, A-OKVQA or FVQA."
            )
        row["source_dataset"] = source

    normalized_image, _ = _resolve_image_path(
        row["image"],
        root,
        line_number,
        verify_image=verify_images,
    )
    row["image"] = normalized_image

    if require_candidate_selection and not row.get("_qacap_candidate_generation_complete"):
        raise ValueError(
            f"Line {line_number}: record is not marked as completed candidate "
            "generation/selection. Run generate_qacap_candidates.py first or "
            "omit --require-candidate-selection for a manually audited manifest."
        )

    _validate_selection_metadata(row, row["caption"], line_number)

    if row.get("image_id") is None:
        # Retain an image identifier for downstream logging. Do not invent a
        # question ID; use the normalized image path as a stable image key.
        row["image_id"] = row["image"]

    return row


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and normalize the selected-caption JSONL consumed by QACapDataset."
        )
    )
    parser.add_argument("jsonl", help="Selected-caption JSONL input.")
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory used to resolve relative image paths.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional normalized JSONL output. If omitted, perform validation only.",
    )
    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Check image existence/extensions but do not fully decode image pixels.",
    )
    parser.add_argument(
        "--require-candidate-selection",
        action="store_true",
        help="Require completion marker written by generate_qacap_candidates.py.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Do not fail when duplicate image/question samples are found.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help=(
            "Optional expected record count. The paper reports 29,569 QACap "
            "entries; use this only for the complete merged dataset."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_count is not None and args.expected_count <= 0:
        raise ValueError("--expected-count must be a positive integer.")

    input_path = Path(args.jsonl)
    root = Path(args.root).resolve()
    rows_with_lines = _read_jsonl(input_path)
    if not rows_with_lines:
        raise ValueError(f"No JSONL records found in {input_path}.")

    normalized_rows: list[dict[str, Any]] = []
    duplicate_counts: Counter[str] = Counter()
    errors: list[str] = []

    for line_number, row in rows_with_lines:
        try:
            normalized = normalize_record(
                row,
                line_number,
                root,
                verify_images=not args.skip_image_decode,
                require_candidate_selection=args.require_candidate_selection,
            )
            normalized_rows.append(normalized)
            duplicate_counts[_sample_key(normalized)] += 1
        except Exception as exc:
            errors.append(str(exc))

    duplicates = [key for key, count in duplicate_counts.items() if count > 1]
    if duplicates and not args.allow_duplicates:
        errors.append(
            f"Found {len(duplicates)} duplicate image/question key(s); "
            f"examples: {duplicates[:5]}. Use --allow-duplicates only after review."
        )

    if args.expected_count is not None and len(normalized_rows) != args.expected_count:
        errors.append(
            f"Expected {args.expected_count} valid records, found {len(normalized_rows)}."
        )

    if errors:
        print("DATA VALIDATION FAILED")
        for error in errors:
            print(f" - {error}")
        raise SystemExit(1)

    if args.output:
        _write_jsonl_atomic(Path(args.output), normalized_rows)

    source_counts = Counter(
        str(row.get("source_dataset", "unspecified")).lower()
        for row in normalized_rows
    )
    image_counts = Counter(str(row["image"]) for row in normalized_rows)

    print("DATA VALIDATION PASSED")
    print(f"records={len(normalized_rows)}")
    print(f"unique_images={len(image_counts)}")
    print(f"duplicate_image_question_keys={len(duplicates)}")
    print(f"source_dataset_counts={dict(source_counts)}")
    print(f"image_root={root}")
    if args.output:
        print(f"normalized_output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
