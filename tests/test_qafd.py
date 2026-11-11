"""Unit tests for the paper-aligned QAFD implementation."""

import pytest
import torch

from qacap.model import QAFD


def _make_qafd():
    # Small dimensions keep these unit tests CPU-friendly. Numerical values
    # here are test fixtures, not claims about the paper's hidden settings.
    return QAFD(dim=32, num_heads=4, alpha=0.4, num_layers=2)


def test_qafd_output_shapes_and_finite_values():
    model = _make_qafd().eval()
    question = torch.randn(2, 7, 32)
    image = torch.randn(2, 16, 32)
    padding_mask = torch.zeros(2, 7, dtype=torch.bool)

    distilled, question_aware, attention = model(
        question, image, padding_mask
    )

    assert distilled.shape == (2, 16, 32)
    assert question_aware.shape == (2, 7, 32)
    assert attention.shape == (2, 7, 16)
    assert torch.isfinite(distilled).all()
    assert torch.isfinite(question_aware).all()
    assert torch.isfinite(attention).all()


def test_qafd_masks_padded_question_positions():
    model = _make_qafd().eval()
    question = torch.randn(2, 7, 32)
    image = torch.randn(2, 16, 32)
    mask = torch.tensor(
        [
            [False, False, False, True, True, True, True],
            [False, False, False, False, True, True, True],
        ],
        dtype=torch.bool,
    )

    question_changed = question.clone()
    question_changed[mask] = torch.randn_like(question_changed[mask]) * 1000.0

    first = model(question, image, mask)
    second = model(question_changed, image, mask)

    # Masked input tokens must not influence valid question representations,
    # regional attention maxima, or distilled image features.
    torch.testing.assert_close(first[0], second[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(first[1][~mask], second[1][~mask], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(first[2][~mask], second[2][~mask], rtol=1e-5, atol=1e-6)

    # Padding query rows are explicitly zeroed before pooling/return.
    assert torch.count_nonzero(first[1][mask]) == 0
    assert torch.count_nonzero(first[2][mask]) == 0


def test_qafd_attention_is_a_distribution_for_each_valid_question_token():
    model = _make_qafd().eval()
    question = torch.randn(2, 5, 32)
    image = torch.randn(2, 11, 32)
    mask = torch.tensor(
        [[False, False, False, True, True],
         [False, False, False, False, True]],
        dtype=torch.bool,
    )

    _, _, attention = model(question, image, mask)

    valid_sums = attention.sum(dim=-1)[~mask]
    torch.testing.assert_close(
        valid_sums,
        torch.ones_like(valid_sums),
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.count_nonzero(attention[mask]) == 0


def test_qafd_gradients_reach_question_and_image_features():
    model = _make_qafd().train()
    question = torch.randn(2, 6, 32, requires_grad=True)
    image = torch.randn(2, 13, 32, requires_grad=True)
    mask = torch.tensor(
        [[False, False, False, False, True, True],
         [False, False, False, False, False, True]],
        dtype=torch.bool,
    )

    distilled, question_aware, attention = model(question, image, mask)
    loss = distilled.square().mean() + question_aware.square().mean()
    loss.backward()

    assert question.grad is not None
    assert image.grad is not None
    assert torch.isfinite(question.grad).all()
    assert torch.isfinite(image.grad).all()
    assert question.grad.abs().sum() > 0
    assert image.grad.abs().sum() > 0


def test_qafd_rejects_mismatched_feature_dimensions():
    model = _make_qafd()
    question = torch.randn(2, 7, 32)
    image = torch.randn(2, 16, 24)

    with pytest.raises(ValueError, match="matching question/image feature dimensions"):
        model(question, image)


def test_qafd_rejects_invalid_padding_mask_shape():
    model = _make_qafd()
    question = torch.randn(2, 7, 32)
    image = torch.randn(2, 16, 32)
    invalid_mask = torch.zeros(2, 6, dtype=torch.bool)

    with pytest.raises(ValueError, match="question_padding_mask"):
        model(question, image, invalid_mask)


def test_qafd_rejects_all_padding_question():
    model = _make_qafd()
    question = torch.randn(2, 7, 32)
    image = torch.randn(2, 16, 32)
    all_padding = torch.ones(2, 7, dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one non-padding token"):
        model(question, image, all_padding)


def test_qafd_alpha_endpoints_match_equation_8():
    model_alpha_one = QAFD(dim=32, num_heads=4, alpha=1.0, num_layers=1).eval()
    model_alpha_zero = QAFD(dim=32, num_heads=4, alpha=0.0, num_layers=1).eval()
    question = torch.randn(2, 5, 32)
    image = torch.randn(2, 9, 32)

    distilled_one, _, _ = model_alpha_one(question, image)
    distilled_zero, _, _ = model_alpha_zero(question, image)

    # Eq. (8): alpha=1 preserves the original image features; alpha=0
    # returns the similarity-reweighted image features.
    torch.testing.assert_close(distilled_one, image, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(distilled_zero, image)
