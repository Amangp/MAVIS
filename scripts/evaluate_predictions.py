"""Evaluate normalized QACap predictions with the metrics described in the paper.

Input JSONL schema (one item per question):
  {"dataset":"okvqa", "mode":"soft", "prediction":"...", "answers":[10 human answers]}
  {"dataset":"aokvqa", "mode":"open-ended", "prediction":"...", "answers":[...]}
  {"dataset":"aokvqa", "mode":"multiple-choice", "prediction":"B", "ground_truth":"B"}

This script intentionally consumes prediction records, not raw benchmark
annotation files. A source-specific dataset adapter is still required to
produce these records from official validation JSON/JSONL files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "scripts" else _SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qacap.metrics import (
    exact_answer_accuracy,
    mean_vqa_soft_accuracy,
    multiple_choice_accuracy,
    open_answer_accuracy,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No prediction records found in {path}")
    return rows


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows, 1):
        dataset_value = row.get("dataset")
        dataset = str(dataset_value).strip().lower() if dataset_value is not None else ""
        mode = str(row.get("mode", "")).strip().lower()
        if not dataset:
            raise ValueError(f"Record {idx} is missing a non-empty 'dataset'")
        if not mode:
            raise ValueError(f"Record {idx} is missing 'mode'")
        prediction = row.get("prediction")
        if not isinstance(prediction, str) or not prediction.strip():
            raise ValueError(f"Record {idx} must contain a non-empty string 'prediction'")
        groups[(dataset, mode)].append(row)

    report: dict[str, Any] = {"count": len(rows), "groups": {}}
    for (dataset, mode), items in sorted(groups.items()):
        question_ids = [
            str(item["question_id"])
            for item in items
            if item.get("question_id") is not None
        ]
        duplicate_ids = sorted(
            value for value, count in Counter(question_ids).items() if count > 1
        )
        if duplicate_ids:
            raise ValueError(
                f"{dataset}/{mode}: duplicate question_id values: {duplicate_ids[:10]}"
            )
        predictions = [item["prediction"] for item in items]
        if mode in {"soft", "vqa-soft", "short-soft"}:
            references = [item.get("answers") for item in items]
            if any(
                not isinstance(refs, list)
                or not refs
                or any(not isinstance(answer, str) or not answer.strip() for answer in refs)
                for refs in references
            ):
                raise ValueError(
                    f"{dataset}/{mode}: every record needs a non-empty list of answer strings"
                )
            score = mean_vqa_soft_accuracy(predictions, references)
            metric_name = "vqa_soft_accuracy"
        elif mode in {"short", "exact"}:
            references = [item.get("answers", item.get("answer")) for item in items]
            if any(value is None or value == "" or value == [] for value in references):
                raise ValueError(f"{dataset}/{mode}: every record needs a non-empty answer reference")
            score = exact_answer_accuracy(predictions, references)
            metric_name = "exact_answer_accuracy"
        elif mode in {"open", "open-ended", "open_ended"}:
            references = [item.get("answers", item.get("answer")) for item in items]
            if any(value is None or value == "" or value == [] for value in references):
                raise ValueError(f"{dataset}/{mode}: every record needs a non-empty answer reference")
            score = open_answer_accuracy(predictions, references)
            metric_name = "open_answer_accuracy"
        elif mode in {"mc", "multiple-choice", "multiple_choice"}:
            references = [
                item.get("ground_truth", item.get("correct_choice")) for item in items
            ]
            if any(not isinstance(value, str) or not value.strip() for value in references):
                raise ValueError(
                    f"{dataset}/{mode}: every record needs a non-empty string ground_truth/correct_choice"
                )
            score = multiple_choice_accuracy(predictions, references)
            metric_name = "multiple_choice_accuracy"
        else:
            raise ValueError(f"Unsupported evaluation mode {mode!r} for {dataset}")

        report["groups"][f"{dataset}/{mode}"] = {
            "count": len(items),
            metric_name: score,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate normalized QACap prediction JSONL"
    )
    parser.add_argument(
        "predictions", help="Prediction JSONL using the documented schema"
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()
    report = evaluate_rows(load_jsonl(Path(args.predictions)))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved evaluation report: {output.resolve()}")


if __name__ == "__main__":
    main()
