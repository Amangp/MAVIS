import pytest

from qacap.metrics import (
    exact_answer_accuracy,
    mean_vqa_soft_accuracy,
    multiple_choice_accuracy,
    normalize_answer,
    open_answer_accuracy,
    paper_open_answer_match,
    paper_vqa_soft_accuracy,
)


def test_normalize_answer_handles_articles_numbers_punctuation_and_contractions():
    assert normalize_answer("The TWO cats, can't!") == "2 cats can't"


def test_vqa_soft_accuracy_uses_ten_answers_and_caps_at_one():
    refs = ["cat", "cat", "cat", "dog", "bird", "dog", "bird", "dog", "bird", "dog"]
    assert paper_vqa_soft_accuracy("cat", refs) == 1.0
    assert mean_vqa_soft_accuracy(["horse"], [refs]) == 0.0


def test_vqa_soft_accuracy_rejects_non_ten_reference_count():
    with pytest.raises(ValueError, match="exactly 10"):
        paper_vqa_soft_accuracy("cat", ["cat"])


def test_open_answer_matching_is_ordered_but_not_required_to_be_contiguous():
    assert paper_open_answer_match("doing another water activity", "another activity")
    assert not paper_open_answer_match("activity before another", "another activity")
    assert not paper_open_answer_match("catalogue", "cat")


def test_exact_and_multiple_choice_metrics():
    assert exact_answer_accuracy(["The red car"], ["red car"]) == 1.0
    assert multiple_choice_accuracy(["B"], ["b"]) == 1.0
    assert exact_answer_accuracy([], []) == 0.0
