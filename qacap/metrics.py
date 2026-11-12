"""
Evaluation metrics for the Separation of Powers / QACap implementation.

The paper describes:
1. VQA soft accuracy for answers with ten human annotations:
       min(number_of_matching_answers / 3, 1)
2. Exact/phrase evaluation for short answers.
3. For open-ended answers, the reference answer is considered correct when
   its words appear accurately and in order in the generated response. The
   paper's example ("another activity" in "doing another water activity")
   indicates ordered, not necessarily contiguous, word matching.
4. A-OKVQA direct-answer and multiple-choice accuracy are reported separately.

Where the paper does not define additional benchmark-specific details, this
module keeps the metric explicit rather than silently inventing one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


# Number-word normalization is part of common VQA answer normalization.
_NUMBER_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

_ARTICLES = {"a", "an", "the"}

# Common English contractions. This is intentionally kept local so that
# evaluation does not depend on an external NLP package.
_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "itll": "it'll",
    "its": "it's",
    "ive": "i've",
    "mightnt": "mightn't",
    "mustnt": "mustn't",
    "shant": "shan't",
    "shed": "she'd",
    "shes": "she's",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theres": "there's",
    "theyre": "they're",
    "theyve": "they've",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
    "youve": "you've",
}


def normalize_answer(answer: Any) -> str:
    """
    Normalize an answer for VQA-style text comparison.

    Operations:
      - lowercase and strip whitespace;
      - normalize common contractions;
      - normalize number words zero through nine;
      - remove English articles;
      - remove punctuation and collapse whitespace.

    This covers common VQA normalization cases. For benchmark leaderboard
    reproduction, prefer the benchmark's official evaluator if it differs.
    """
    if answer is None:
        return ""

    text = str(answer).strip().lower()
    if not text:
        return ""

    # Normalize apostrophe variants before contraction processing.
    text = text.replace("’", "'").replace("‘", "'")

    # Remove punctuation while preserving apostrophes inside contractions.
    text = "".join(
        ch if (ch.isalnum() or ch.isspace() or ch == "'") else " "
        for ch in text
    )

    tokens = []
    for token in text.split():
        bare = token.replace("'", "")
        if bare in _CONTRACTIONS:
            token = _CONTRACTIONS[bare]
        if token in _ARTICLES:
            continue
        token = _NUMBER_MAP.get(token, token)
        tokens.append(token)

    return " ".join(tokens)


def paper_vqa_soft_accuracy(
    prediction: Any,
    ten_gt_answers: Sequence[Any],
) -> float:
    """
    Soft accuracy as described in the paper's evaluation section.

    The paper describes ten human answers and gives full credit when at
    least three agree with the prediction; otherwise credit is proportional.
    Therefore the score is min(matches / 3, 1).

    Args:
        prediction: Model's predicted answer.
        ten_gt_answers: The ten human-provided reference answers.

    Returns:
        A score in [0, 1].
    """
    if isinstance(ten_gt_answers, (str, bytes)) or not isinstance(
        ten_gt_answers, Sequence
    ):
        raise TypeError("ten_gt_answers must be a sequence of reference answers.")

    if len(ten_gt_answers) != 10:
        raise ValueError(
            f"Expected exactly 10 human answers, received {len(ten_gt_answers)}."
        )

    pred = normalize_answer(prediction)
    if not pred:
        return 0.0

    matches = sum(normalize_answer(answer) == pred for answer in ten_gt_answers)
    return min(matches / 3.0, 1.0)


def vqa_soft_accuracy(
    prediction: Any,
    ten_gt_answers: Sequence[Any],
) -> float:
    """Backward-compatible name for the paper-described VQA soft accuracy."""
    return paper_vqa_soft_accuracy(prediction, ten_gt_answers)


def _ordered_phrase_match(reference: str, prediction: str) -> bool:
    """
    Return True when reference tokens occur in prediction in the same order.

    Tokens need not be adjacent. This follows the paper's example where the
    reference phrase "another activity" appears in "doing another water
    activity".
    """
    ref_tokens = normalize_answer(reference).split()
    pred_tokens = normalize_answer(prediction).split()

    if not ref_tokens:
        return False
    if len(ref_tokens) == 1:
        return ref_tokens[0] in pred_tokens

    pred_index = 0
    for ref_token in ref_tokens:
        try:
            pred_index = pred_tokens.index(ref_token, pred_index) + 1
        except ValueError:
            return False
    return True


def paper_open_answer_match(prediction: Any, answer: Any) -> bool:
    """
    Evaluate a concise/open-ended answer using the paper's described rule.

    The normalized reference answer's words must appear in order in the
    generated answer. This avoids false substring matches (e.g. "cat" in
    "catalogue") while allowing intervening words as in the paper's example.
    """
    if answer is None:
        return False
    return _ordered_phrase_match(str(answer), str(prediction))


def open_answer_accuracy(
    predictions: Sequence[Any],
    references: Sequence[Any],
) -> float:
    """
    Mean paper-style open-answer match over a collection.

    Each reference can be a single string or a sequence of acceptable
    reference strings. For multiple references, a prediction is correct when
    it matches at least one reference.
    """
    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have the same number of items."
        )
    if not predictions:
        return 0.0

    correct = 0
    for prediction, reference in zip(predictions, references):
        if isinstance(reference, str) or reference is None:
            refs = [reference]
        else:
            refs = list(reference)
        correct += any(
            paper_open_answer_match(prediction, ref) for ref in refs
        )
    return correct / len(predictions)



def exact_answer_accuracy(
    predictions: Sequence[Any],
    references: Sequence[Any],
) -> float:
    """Exact normalized accuracy for short-answer evaluation.

    Each reference can be a string or a sequence of acceptable human answers.
    This is separate from the ten-annotator VQA soft-accuracy metric above.
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same number of items.")
    if not predictions:
        return 0.0

    correct = 0
    for prediction, reference in zip(predictions, references):
        refs = [reference] if isinstance(reference, str) or reference is None else list(reference)
        normalized_prediction = normalize_answer(prediction)
        correct += bool(normalized_prediction) and any(
            normalized_prediction == normalize_answer(ref) for ref in refs
        )
    return float(correct) / len(predictions)


def multiple_choice_accuracy(
    predictions: Sequence[Any],
    ground_truth: Sequence[Any],
) -> float:
    """
    Exact normalized accuracy for multiple-choice predictions.

    Inputs must already represent the selected option (e.g. option text or
    option label) consistently on both sides. This function does not parse
    free-form LLM explanations or infer option letters from prose.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            "predictions and ground_truth must have the same number of items."
        )
    if not predictions:
        return 0.0

    matches = sum(
        normalize_answer(pred) == normalize_answer(gt)
        for pred, gt in zip(predictions, ground_truth)
    )
    return matches / len(predictions)


def mean_vqa_soft_accuracy(
    predictions: Sequence[Any],
    ten_gt_answers: Sequence[Sequence[Any]],
) -> float:
    """Mean paper-described VQA soft accuracy over a validation set."""
    if len(predictions) != len(ten_gt_answers):
        raise ValueError(
            "predictions and ten_gt_answers must have the same number of items."
        )
    if not predictions:
        return 0.0

    scores = [
        paper_vqa_soft_accuracy(pred, answers)
        for pred, answers in zip(predictions, ten_gt_answers)
    ]
    return sum(scores) / len(scores)
