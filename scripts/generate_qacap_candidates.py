"""
Build the paper's QACap question-aware caption training manifest.

Paper-described procedure (Section 3.2, CVPR 2025):
  1. Use the five human-written COCO captions together with a KBVQA question
     and its human answers as GPT-4 input.
  2. Generate three question-relevant candidate captions.
  3. Refine candidates to remove answer leakage, redundancy, verbosity and
     unnatural expressions. The paper gives "known for" and "demonstrating"
     as examples of expressions to filter.
  4. Evaluate candidate helpfulness by asking the paper's answering LLM to
     answer the question from each candidate caption. The answer score counts
     ground-truth answers appearing in the LLM response.
  5. Evaluate image-caption truthfulness with CLIP. Among candidates tied for
     the highest answer score, select the candidate with the highest CLIP score.

Important reproduction boundaries:
  - The supplementary prompts and complete phrase-filter list are not in the
    main paper available here. The prompts below operationalize the described
    procedure; they are not claimed to be verbatim author prompts.
  - The main paper does not identify the exact CLIP checkpoint or exact
    API snapshots/token budgets. The CLIP checkpoint and API settings are
    therefore explicit arguments/environment values, not guessed defaults.
  - The image is used for CLIP scoring. GPT-4 caption synthesis receives the
    five human COCO captions plus the QA pair, as described in Section 3.2.

Input JSONL fields:
  image:         image path (absolute or relative to --image-root)
  question:      question string
  answers:       ten human answers (strings or {"answer": "..."} objects)
  coco_captions: the five human COCO captions (strings or {"caption": "..."} objects)

Output JSONL additionally contains:
  raw_caption_candidates: the three initial GPT-4 captions
  caption_candidates:     refined candidates with answer and CLIP scores
  caption:                selected target caption (for QACapDataset)
  candidate_selection:   selection metadata

The script resumes from an existing output file by default. Completed records
are skipped to avoid repeated API calls. Use --overwrite to rebuild it.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import re
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image, ImageOps


# Resolve the project root when invoked as `python scripts/generate_qacap_candidates.py`.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = (
    _THIS_FILE.parent.parent if _THIS_FILE.parent.name == "scripts" else _THIS_FILE.parent
)
if (_PROJECT_ROOT / "qacap").is_dir() and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# The main paper names these two phrases as examples, but says the full list
# was collected separately. Add the complete supplementary list here when it
# becomes available; do not silently claim these two are the complete list.
PAPER_EXAMPLE_BLOCKED_PHRASES = (
    "known for",
    "demonstrating",
)

INITIAL_GENERATION_PROMPT = """You are constructing question-aware image captions for a visual question answering dataset.

Use the five human-written COCO captions as the visual scene descriptions. Consider the question and the ground-truth answer annotations to understand what information is relevant, but do not turn the answer or outside knowledge into a visual observation. A caption must describe image content supported by the supplied human captions. Do not state an answer that is not explicitly supported by the visible-scene descriptions.

Generate exactly THREE distinct candidate captions. Each candidate must:
- describe visible scene content relevant to the question;
- retain factual details supported by the supplied COCO captions;
- be concise, natural and non-redundant;
- provide useful visual context so a separate language model can reason about the answer;
- avoid unsupported objects, actions, attributes, identities or locations;
- avoid giving away an answer through inference or background knowledge.

Question:
{question}

Ground-truth answer annotations (for understanding/evaluation only; do not copy an answer unless it is explicitly visible in the supplied COCO captions):
{answers}

Five human-written COCO captions:
{coco_captions}

Return only a JSON object with this schema:
{{"candidates": ["caption 1", "caption 2", "caption 3"]}}
"""

REFINEMENT_PROMPT = """Refine the three GPT-4 candidate captions for a question-aware visual caption dataset.

The original five human COCO captions are the evidence for what is visibly present. The question and answer annotations are supplied to help identify relevance and possible answer leakage; do not infer an answer from world knowledge and insert it as if it were visible. Remove any detail that is not supported by the COCO captions.

Correct these problems:
- a caption directly states or implies a ground-truth answer that is not visibly supported;
- verbose, repetitive or redundant wording;
- unnatural or awkward expressions;
- unsupported objects, actions, properties, identities, locations or causal explanations.

Avoid the expressions in this filter list:
{blocked_phrases}

Keep each caption descriptive, natural and focused on question-relevant visual evidence. Preserve meaningful details from the original scene descriptions. Return exactly THREE refined captions, corresponding to the three candidates.

Question:
{question}

Ground-truth answer annotations (audit for leakage only):
{answers}

Five human-written COCO captions:
{coco_captions}

Initial GPT-4 candidates:
{candidates}

Return only a JSON object with this schema:
{{"candidates": ["refined caption 1", "refined caption 2", "refined caption 3"]}}
"""


def _extract_answer_texts(value: Any) -> list[str]:
    """Normalize answer annotations to a list of answer strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Sequence):
        raise TypeError("answers must be a string or a sequence of answer annotations")

    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            answer = item.strip()
        elif isinstance(item, Mapping):
            answer = str(item.get("answer", "")).strip()
        else:
            answer = str(item).strip()
        if answer:
            result.append(answer)
    return result


def _extract_caption_texts(value: Any) -> list[str]:
    """Normalize COCO caption annotations to a list of caption strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Sequence):
        raise TypeError(
            "coco_captions must be a string or a sequence of caption annotations"
        )

    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            caption = item.strip()
        elif isinstance(item, Mapping):
            caption = str(item.get("caption", "")).strip()
        else:
            caption = str(item).strip()
        if caption:
            result.append(caption)
    return result


def _require_three_candidates(
    value: Any,
    stage: str,
) -> list[str]:
    """Validate the exact three-candidate output requested by the paper."""
    if not isinstance(value, Mapping) or "candidates" not in value:
        raise ValueError(f"{stage} response must be a JSON object with 'candidates'.")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError(
            f"{stage} must return exactly three captions; received "
            f"{len(candidates) if isinstance(candidates, list) else type(candidates).__name__}."
        )
    cleaned: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError(f"{stage} candidate {index} is empty or not a string.")
        cleaned.append(candidate.strip())
    normalized = [" ".join(_normalize_for_match(item)) for item in cleaned]
    if len(set(normalized)) != 3:
        raise ValueError(f"{stage} returned duplicate captions; three distinct candidates are required.")
    return cleaned


def _parse_json_response(text: str, stage: str) -> dict[str, Any]:
    """Parse model JSON, tolerating a surrounding Markdown code fence."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{stage} returned an empty response.")
    content = text.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        # Try the outermost JSON object when the API returns a short preamble.
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"{stage} did not return valid JSON.")
        obj = json.loads(content[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"{stage} must return a JSON object.")
    return obj


def _openai_generate_three(
    client: Any,
    model: str,
    max_tokens: int,
    prompt: str,
    stage: str,
) -> list[str]:
    """Call GPT-4 and validate its three-caption JSON response."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "Follow the caption-dataset instructions precisely. "
                    "Return only valid JSON and exactly the requested number of captions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    message = response.choices[0].message
    content = getattr(message, "content", None)
    obj = _parse_json_response(content or "", stage)
    return _require_three_candidates(obj, stage)


def _normalize_for_match(text: str) -> list[str]:
    """Small normalized-token representation for answer-occurrence scoring."""
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def _answer_occurs_in_order(reference: str, response: str) -> bool:
    """Check whether a normalized answer phrase occurs contiguously in a response."""
    ref_tokens = _normalize_for_match(reference)
    response_tokens = _normalize_for_match(response)
    if not ref_tokens or len(ref_tokens) > len(response_tokens):
        return False
    width = len(ref_tokens)
    return any(
        response_tokens[index : index + width] == ref_tokens
        for index in range(len(response_tokens) - width + 1)
    )


def _score_helpfulness(
    question: str,
    caption: str,
    answers: Sequence[str],
    *,
    llm_model: str,
    llm_max_tokens: int,
) -> tuple[int, str]:
    """Use Claude 3.5 response and count ground-truth answers appearing in it."""
    try:
        from qacap.llm import answer_with_llm
    except ImportError as exc:
        raise ImportError(
            "Could not import qacap.llm.answer_with_llm. Run from the project "
            "root and use the updated llm.py."
        ) from exc

    response = answer_with_llm(
        question=question,
        caption=caption,
        model=llm_model,
        max_tokens=llm_max_tokens,
        answer_mode="short",
    )
    score = sum(
        1 for answer in answers
        if _answer_occurs_in_order(answer, response)
    )
    return score, response


class CLIPTruthfulnessScorer:
    """Compute image-caption cosine similarity using a configured CLIP checkpoint."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "Transformers is required for CLIP scoring. Install the project's "
                "pinned transformers dependency."
            ) from exc

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, image: Image.Image, captions: Sequence[str]) -> list[float]:
        if not captions:
            return []
        inputs = self.processor(
            text=list(captions),
            images=[image] * len(captions),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)

        image_embeds = torch.nn.functional.normalize(outputs.image_embeds.float(), dim=-1)
        text_embeds = torch.nn.functional.normalize(outputs.text_embeds.float(), dim=-1)
        similarities = (image_embeds * text_embeds).sum(dim=-1)
        scores = [float(value) for value in similarities.detach().cpu().tolist()]
        if len(scores) != len(captions) or not all(math.isfinite(value) for value in scores):
            raise RuntimeError("CLIP returned missing or non-finite image-caption scores.")
        return scores


def select_best_caption(
    candidates: Sequence[str],
    answer_scores: Sequence[int],
    clip_scores: Sequence[float],
) -> tuple[int, dict[str, Any]]:
    """Select by helpfulness first, using CLIP truthfulness to break ties."""
    if not candidates or not (
        len(candidates) == len(answer_scores) == len(clip_scores)
    ):
        raise ValueError("Candidates and score arrays must have equal non-zero lengths.")
    if any(not isinstance(score, int) or score < 0 for score in answer_scores):
        raise ValueError("Answer scores must be nonnegative integers.")
    if any(not math.isfinite(float(score)) for score in clip_scores):
        raise ValueError("CLIP scores must be finite numbers.")

    best_answer_score = max(answer_scores)
    top_indices = [
        index for index, score in enumerate(answer_scores)
        if score == best_answer_score
    ]
    selected_index = max(top_indices, key=lambda index: clip_scores[index])

    return selected_index, {
        "rule": (
            "highest_answer_score_then_highest_clip_score"
            if len(top_indices) > 1
            else "highest_answer_score"
        ),
        "best_answer_score": best_answer_score,
        "answer_score_tie_indices": top_indices,
        "selected_index": selected_index,
        "selected_clip_score": clip_scores[selected_index],
    }


def _record_key(row: Mapping[str, Any]) -> str:
    """Stable resume key, preferring dataset-scoped question IDs."""
    source = row.get("dataset", row.get("source_dataset", ""))
    prefix = f"dataset:{source}|" if source else ""
    for key in ("question_id", "questionId", "qid", "qa_id"):
        if row.get(key) is not None:
            return f"{prefix}{key}:{row[key]}"
    return f"{prefix}image:{row.get('image', '')}|question:{row.get('question', '')}"


def _input_fingerprint(row: Mapping[str, Any]) -> str:
    fields = {
        key: row.get(key)
        for key in ("dataset", "source_dataset", "image", "question", "answers", "coco_captions")
    }
    payload = json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Each JSONL record must be an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically rewrite output after each completed record for safe resumption."""
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
        ) as temp_handle:
            temp_path = Path(temp_handle.name)
            for row in rows:
                temp_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def process_record(
    row: dict[str, Any],
    *,
    image_root: Path,
    openai_client: Any,
    gpt4_model: str,
    gpt4_max_tokens: int,
    answer_model: str,
    answer_max_tokens: int,
    clip_scorer: CLIPTruthfulnessScorer,
    blocked_phrases: Sequence[str],
) -> dict[str, Any]:
    """Generate, refine, score and select one QACap training caption."""
    required_fields = ("image", "question", "answers", "coco_captions")
    missing = [field for field in required_fields if field not in row]
    if missing:
        raise KeyError(f"Input record is missing required fields: {missing}")

    question = str(row["question"]).strip()
    if not question:
        raise ValueError("Question is empty.")

    answers = _extract_answer_texts(row["answers"])
    coco_captions = _extract_caption_texts(row["coco_captions"])
    if not answers:
        raise ValueError("At least one ground-truth answer annotation is required.")
    if len(coco_captions) != 5:
        raise ValueError(
            "The paper describes using the original five human COCO captions; "
            f"this record contains {len(coco_captions)}."
        )

    image_path = Path(row["image"])
    if not image_path.is_absolute():
        image_path = image_root / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    initial_prompt = INITIAL_GENERATION_PROMPT.format(
        question=question,
        answers=json.dumps(answers, ensure_ascii=False),
        coco_captions=json.dumps(coco_captions, ensure_ascii=False),
    )
    raw_candidates = _openai_generate_three(
        openai_client,
        gpt4_model,
        gpt4_max_tokens,
        initial_prompt,
        "Initial GPT-4 generation",
    )

    refinement_prompt = REFINEMENT_PROMPT.format(
        blocked_phrases=json.dumps(list(blocked_phrases), ensure_ascii=False),
        question=question,
        answers=json.dumps(answers, ensure_ascii=False),
        coco_captions=json.dumps(coco_captions, ensure_ascii=False),
        candidates=json.dumps(raw_candidates, ensure_ascii=False),
    )
    refined_candidates = _openai_generate_three(
        openai_client,
        gpt4_model,
        gpt4_max_tokens,
        refinement_prompt,
        "GPT-4 refinement",
    )

    normalized_blocked = [phrase.casefold() for phrase in blocked_phrases]
    violations = {
        index: [
            phrase for phrase in normalized_blocked
            if phrase in caption.casefold()
        ]
        for index, caption in enumerate(refined_candidates)
    }
    violations = {index: phrases for index, phrases in violations.items() if phrases}
    if violations:
        raise ValueError(
            "Refined captions still contain blocked expressions. Review/refine "
            f"the prompt or supplementary phrase list. Violations: {violations}"
        )

    # Score every candidate for helpfulness by using the paper's Claude 3.5
    # question-answering setup. The score counts annotated answers appearing
    # in the LLM response, as described in Section 3.2.2.
    scored_candidates: list[dict[str, Any]] = []
    for index, caption in enumerate(refined_candidates):
        answer_score, llm_response = _score_helpfulness(
            question,
            caption,
            answers,
            llm_model=answer_model,
            llm_max_tokens=answer_max_tokens,
        )
        scored_candidates.append(
            {
                "candidate_index": index,
                "caption": caption,
                "answer_score": answer_score,
                "llm_response": llm_response,
            }
        )

    # CLIP evaluates image-caption truthfulness/semantic closeness.
    with Image.open(image_path) as image_file:
        image = ImageOps.exif_transpose(image_file).convert("RGB")
        clip_scores = clip_scorer.score(image, refined_candidates)

    for candidate, clip_score in zip(scored_candidates, clip_scores):
        candidate["clip_score"] = clip_score

    selected_index, selection_metadata = select_best_caption(
        candidates=refined_candidates,
        answer_scores=[item["answer_score"] for item in scored_candidates],
        clip_scores=clip_scores,
    )
    selected_caption = refined_candidates[selected_index]

    result = dict(row)
    result["raw_caption_candidates"] = raw_candidates
    result["caption_candidates"] = scored_candidates
    result["caption"] = selected_caption
    result["candidate_selection"] = selection_metadata
    result["_qacap_candidate_generation_complete"] = True
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and select question-aware QACap training captions."
    )
    parser.add_argument("input", help="Input JSONL with QA pairs and five COCO captions.")
    parser.add_argument("output", help="Output JSONL consumed by QACapDataset.")
    parser.add_argument("--image-root", default=".", help="Root for relative image paths.")
    parser.add_argument(
        "--gpt4-model",
        default=os.getenv("QACAP_GPT4_MODEL"),
        help="GPT-4 API model ID. Required; the paper specifies GPT-4 but not an API snapshot.",
    )
    parser.add_argument(
        "--gpt4-max-tokens",
        type=int,
        default=os.getenv("QACAP_GPT4_MAX_TOKENS"),
        help="GPT-4 response token budget (API setting, not specified in the paper).",
    )
    parser.add_argument(
        "--answer-model",
        default=os.getenv("QACAP_LLM_MODEL"),
        help="Claude 3.5 API model ID/snapshot; required for candidate helpfulness scoring.",
    )
    parser.add_argument(
        "--answer-max-tokens",
        type=int,
        default=os.getenv("QACAP_LLM_MAX_TOKENS"),
        help="Claude response token budget (API setting, not specified in the paper).",
    )
    parser.add_argument(
        "--clip-model",
        default=os.getenv("QACAP_CLIP_MODEL"),
        help="CLIP checkpoint ID; required because the main paper does not name the variant.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device for CLIP scoring.",
    )
    parser.add_argument(
        "--blocked-phrases",
        default=",".join(PAPER_EXAMPLE_BLOCKED_PHRASES),
        help=(
            "Comma-separated phrase filter. Defaults only to the two examples "
            "named in the main paper; add the full supplementary list when available."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing output and regenerate all records.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for GPT-4 caption generation.")

    required_values = {
        "--gpt4-model": args.gpt4_model,
        "--answer-model": args.answer_model,
        "--clip-model": args.clip_model,
        "--gpt4-max-tokens": args.gpt4_max_tokens,
        "--answer-max-tokens": args.answer_max_tokens,
    }
    missing = [name for name, value in required_values.items() if value is None or str(value).strip() == ""]
    if missing:
        raise ValueError(
            "Missing required model/API configuration: "
            + ", ".join(missing)
            + ". Supply CLI options or the documented QACAP_* environment variables."
        )

    gpt4_max_tokens = int(args.gpt4_max_tokens)
    answer_max_tokens = int(args.answer_max_tokens)
    if gpt4_max_tokens <= 0 or answer_max_tokens <= 0:
        raise ValueError("API response-token budgets must be positive integers.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when supplied.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    input_rows = _load_jsonl(input_path)
    if args.limit is not None:
        input_rows = input_rows[: args.limit]
    input_keys = [_record_key(row) for row in input_rows]
    duplicate_keys = sorted(key for key, count in Counter(input_keys).items() if count > 1)
    if duplicate_keys:
        raise ValueError(f"Input manifest contains duplicate question keys: {duplicate_keys[:10]}")

    if args.overwrite:
        completed_rows: list[dict[str, Any]] = []
    elif output_path.exists():
        completed_rows = _load_jsonl(output_path)
    else:
        completed_rows = []

    completed_by_key: dict[str, dict[str, Any]] = {}
    for saved_row in completed_rows:
        if saved_row.get("_qacap_candidate_generation_complete") is not True:
            continue
        saved_key = _record_key(saved_row)
        if saved_key in completed_by_key:
            raise ValueError(f"Existing output contains duplicate completed question key: {saved_key}")
        completed_by_key[saved_key] = saved_row

    # Preserve previous completed rows when resuming; keep input order in output.
    ordered_results: list[dict[str, Any]] = []
    device = _resolve_device(args.device)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI SDK with: pip install openai") from exc
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    clip_scorer = CLIPTruthfulnessScorer(args.clip_model, device)
    blocked_phrases = [
        phrase.strip()
        for phrase in args.blocked_phrases.split(",")
        if phrase.strip()
    ]

    for index, row in enumerate(input_rows, start=1):
        key = _record_key(row)
        fingerprint = _input_fingerprint(row)
        saved = completed_by_key.get(key)
        if saved is not None and saved.get("_qacap_input_fingerprint") == fingerprint:
            ordered_results.append(saved)
            print(f"[{index}/{len(input_rows)}] Already complete; skipped: {key}")
            continue

        print(f"[{index}/{len(input_rows)}] Processing: {key}")
        processed = process_record(
            row,
            image_root=Path(args.image_root),
            openai_client=openai_client,
            gpt4_model=str(args.gpt4_model),
            gpt4_max_tokens=gpt4_max_tokens,
            answer_model=str(args.answer_model),
            answer_max_tokens=answer_max_tokens,
            clip_scorer=clip_scorer,
            blocked_phrases=blocked_phrases,
        )
        processed["_qacap_input_fingerprint"] = fingerprint
        ordered_results.append(processed)
        _atomic_write_jsonl(output_path, ordered_results)
        print(
            f"  selected candidate {processed['candidate_selection']['selected_index']} "
            f"(answer score={processed['candidate_selection']['best_answer_score']}, "
            f"CLIP={processed['candidate_selection']['selected_clip_score']:.4f})"
        )

    if not input_rows:
        raise ValueError("Input JSONL contains no records to process.")

    _atomic_write_jsonl(output_path, ordered_results)
    print(f"Completed caption manifest: {output_path.resolve()}")


if __name__ == "__main__":
    main()
