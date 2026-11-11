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
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
        dataset = str(row.get("dataset", "unknown")).strip().lower()
        mode = str(row.get("mode", "")).strip().lower()
        if not mode:
            raise ValueError(f"Record {idx} is missing 'mode'")
        if "prediction" not in row:
            raise ValueError(f"Record {idx} is missing 'prediction'")
        groups[(dataset, mode)].append(row)

    report: dict[str, Any] = {"count": len(rows), "groups": {}}
    for (dataset, mode), items in sorted(groups.items()):
        predictions = [item["prediction"] for item in items]
        if mode in {"soft", "vqa-soft", "short-soft"}:
            references = [item.get("answers") for item in items]
            if any(not isinstance(refs, list) for refs in references):
                raise ValueError(
                    f"{dataset}/{mode}: every record must include an 'answers' list"
                )
            score = mean_vqa_soft_accuracy(predictions, references)
            metric_name = "vqa_soft_accuracy"
        elif mode in {"short", "exact"}:
            references = [item.get("answers", item.get("answer")) for item in items]
            score = exact_answer_accuracy(predictions, references)
            metric_name = "exact_answer_accuracy"
        elif mode in {"open", "open-ended", "open_ended"}:
            references = [item.get("answers", item.get("answer")) for item in items]
            score = open_answer_accuracy(predictions, references)
            metric_name = "open_answer_accuracy"
        elif mode in {"mc", "multiple-choice", "multiple_choice"}:
            references = [
                item.get("ground_truth", item.get("correct_choice")) for item in items
            ]
            if any(value is None for value in references):
                raise ValueError(
                    f"{dataset}/{mode}: every record needs ground_truth/correct_choice"
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
