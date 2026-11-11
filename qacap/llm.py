"""Claude answering stage for the Separation of Powers / QACap pipeline.

The paper's Figure 2 specifies three prompt elements: task instruction,
question-aware caption as context, and the original question. The reported
answering LLM is Claude 3.5. The prompt wording below is an implementation
reconstruction; exact supplementary prompt text and API snapshot are not
available in the supplied main-paper PDF.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence


TASK_INSTRUCTION = (
    "Based on the context, leverage the information in the context and your "
    "knowledge to answer the question. Maintain briefness and precision in "
    "the answer."
)

_MODE_INSTRUCTIONS = {
    "short": "Return only a one-word or short-phrase answer. Do not explain.",
    "open-ended": (
        "Give an accurate, concise open-ended response. Include enough wording "
        "to express the answer clearly, but do not add unnecessary explanation."
    ),
    "multiple-choice": (
        "Select exactly one of the provided options. Return only its option "
        "letter (for example, A or B), with no explanation."
    ),
}


def build_llm_prompt(
    question: str,
    caption: str,
    *,
    answer_mode: str = "short",
    choices: Optional[Sequence[str]] = None,
) -> str:
    """Construct Task Instruction / Context / Question input.

    The answer-mode instructions implement the paper's reported short-answer,
    open-ended and multiple-choice evaluation formats. Their exact wording is
    not claimed to be verbatim from the paper's supplementary appendix.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("caption must be a non-empty string")
    if answer_mode not in _MODE_INSTRUCTIONS:
        raise ValueError(
            f"answer_mode must be one of {tuple(_MODE_INSTRUCTIONS)}, got {answer_mode!r}"
        )

    instruction = TASK_INSTRUCTION + " " + _MODE_INSTRUCTIONS[answer_mode]
    choice_block = ""
    if answer_mode == "multiple-choice":
        if choices is None or len(choices) < 2:
            raise ValueError("multiple-choice mode requires at least two choices")
        labels = [chr(ord("A") + index) for index in range(len(choices))]
        choice_lines = []
        for label, choice in zip(labels, choices):
            text = str(choice).strip()
            if not text:
                raise ValueError("multiple-choice options must be non-empty")
            choice_lines.append(f"{label}. {text}")
        choice_block = "\n\nOptions:\n" + "\n".join(choice_lines)
    elif choices is not None:
        raise ValueError("choices are only valid when answer_mode='multiple-choice'")

    return (
        f"Task Instruction:\n{instruction}\n\n"
        f"Context:\n{caption.strip()}\n\n"
        f"Question:\n{question.strip()}"
        f"{choice_block}"
    )


def _required_value(value: Any, env_name: str) -> str:
    resolved = value if value is not None else os.getenv(env_name)
    if resolved is None or not str(resolved).strip():
        raise ValueError(
            f"{env_name} is required. Supply it as an argument or environment "
            "variable; the main paper does not specify the exact API value."
        )
    return str(resolved).strip()


def answer_with_llm(
    question: str,
    caption: str,
    *,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    answer_mode: str = "short",
    choices: Optional[Sequence[str]] = None,
    client: Any = None,
) -> str:
    """Answer using the configured Claude 3.5 model.

    `model` and `max_tokens` are required explicitly or through
    `QACAP_LLM_MODEL` and `QACAP_LLM_MAX_TOKENS`. They are deliberately not
    guessed because the main paper does not report the API snapshot/token cap.
    `client` may be supplied by a unit test to avoid an API call.
    """
    model_id = _required_value(model, "QACAP_LLM_MODEL")
    token_value = _required_value(
        str(max_tokens) if max_tokens is not None else None,
        "QACAP_LLM_MAX_TOKENS",
    )
    try:
        token_budget = int(token_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_tokens must be a positive integer") from exc
    if token_budget <= 0:
        raise ValueError("max_tokens must be a positive integer")

    prompt = build_llm_prompt(
        question,
        caption,
        answer_mode=answer_mode,
        choices=choices,
    )

    if client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "The Anthropic SDK is required. Install it with `pip install anthropic`."
            ) from exc
        client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model_id,
        max_tokens=token_budget,
        messages=[{"role": "user", "content": prompt}],
    )
    pieces = [
        block.text
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text" and getattr(block, "text", "")
    ]
    answer = "".join(pieces).strip()
    if not answer:
        raise RuntimeError("Claude returned no text answer")
    return answer
