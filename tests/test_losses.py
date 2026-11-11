"""Unit tests for caption NLL, QARCL JS divergence and the joint objective."""

import pytest
import torch

from qacap.losses import (
    caption_nll,
    js_divergence,
    qacap_total_loss,
    regional_attention_distribution,
)


def _random_attention(batch=2, queries=5, regions=7):
    return torch.softmax(torch.randn(batch, queries, regions), dim=-1)


def test_js_divergence_is_zero_for_identical_maps():
    attention = _random_attention(batch=2, queries=5, regions=7)
    loss = js_divergence(attention, attention.clone())
    assert loss.item() >= 0.0
    assert loss.item() < 1e-6


def test_js_divergence_is_nonnegative_for_different_maps():
    first = _random_attention()
    second = _random_attention()
    loss = js_divergence(first, second)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_js_accepts_different_query_lengths_with_same_image_regions():
    qafd_attention = _random_attention(batch=3, queries=6, regions=11)
    caption_attention = _random_attention(batch=3, queries=13, regions=11)

    loss = js_divergence(qafd_attention, caption_attention)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_js_accepts_per_head_attention_and_averages_heads():
    per_head = torch.softmax(torch.randn(2, 4, 5, 7), dim=-1)
    averaged = per_head.mean(dim=1)
    loss = js_divergence(per_head, averaged)

    assert loss.item() < 1e-6


def test_regional_distribution_uses_query_axis_max_pooling():
    attention = torch.tensor(
        [[[0.1, 0.8, 0.2],
          [0.7, 0.2, 0.3],
          [0.2, 0.4, 0.9]]],
        dtype=torch.float32,
    )
    distribution = regional_attention_distribution(attention)
    expected_scores = torch.tensor([[0.7, 0.8, 0.9]])
    expected = expected_scores / expected_scores.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(distribution, expected)


def test_query_masks_exclude_padded_positions_from_regional_max():
    first = torch.tensor(
        [[[0.8, 0.1, 0.1],
          [0.2, 0.7, 0.1],
          [0.0, 0.0, 1.0]]]
    )
    second = first.clone()
    # Last query is padding. Give it extreme attention to a different region.
    second[:, 2, :] = torch.tensor([1.0, 0.0, 0.0])
    valid = torch.tensor([[True, True, False]])

    first_distribution = regional_attention_distribution(
        first, query_mask=valid
    )
    second_distribution = regional_attention_distribution(
        second, query_mask=valid
    )
    torch.testing.assert_close(first_distribution, second_distribution)


def test_js_uses_caption_query_mask_to_exclude_pet_positions():
    qafd_attention = _random_attention(batch=1, queries=3, regions=5)
    caption_attention = torch.softmax(torch.randn(1, 6, 5), dim=-1)
    changed = caption_attention.clone()
    # First three positions are PET/BOS context and are masked out.
    changed[:, :3, :] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    caption_valid_mask = torch.tensor([[False, False, False, True, True, True]])

    first_loss = js_divergence(
        qafd_attention,
        caption_attention,
        c_query_mask=caption_valid_mask,
    )
    second_loss = js_divergence(
        qafd_attention,
        changed,
        c_query_mask=caption_valid_mask,
    )
    torch.testing.assert_close(first_loss, second_loss)


def test_caption_nll_ignores_minus_100_labels():
    logits = torch.randn(2, 4, 9, requires_grad=True)
    labels = torch.tensor(
        [[-100, 2, 3, -100],
         [-100, 4, 5, 6]],
        dtype=torch.long,
    )

    loss = caption_nll(logits, labels, ignore_index=-100)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_caption_nll_rejects_all_ignored_targets():
    logits = torch.randn(2, 4, 9)
    labels = torch.full((2, 4), -100, dtype=torch.long)

    with pytest.raises(ValueError, match="All caption labels are ignored"):
        caption_nll(logits, labels, ignore_index=-100)


def test_joint_loss_matches_paper_equation_and_backpropagates():
    logits = torch.randn(2, 6, 13, requires_grad=True)
    labels = torch.tensor(
        [[-100, -100, 3, 4, 5, -100],
         [-100, -100, 6, 7, 8, 9]],
        dtype=torch.long,
    )
    qafd_attention = _random_attention(batch=2, queries=4, regions=7).requires_grad_()
    caption_attention = _random_attention(batch=2, queries=6, regions=7).requires_grad_()
    caption_mask = labels.ne(-100)
    lambda_js = 0.25

    result = qacap_total_loss(
        logits,
        labels,
        qafd_attention,
        caption_attention,
        lambda_js=lambda_js,
        caption_query_mask=caption_mask,
    )

    assert set(result) == {"loss", "caption_loss", "js_loss"}
    torch.testing.assert_close(
        result["loss"],
        result["caption_loss"] + lambda_js * result["js_loss"],
    )
    result["loss"].backward()
    assert logits.grad is not None
    assert qafd_attention.grad is not None
    assert caption_attention.grad is not None


def test_joint_loss_requires_explicit_lambda_and_rejects_negative_lambda():
    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[-100, 2, 3]])
    qafd_attention = _random_attention(batch=1, queries=2, regions=4)
    caption_attention = _random_attention(batch=1, queries=3, regions=4)

    with pytest.raises(ValueError, match="lambda_js must be nonnegative"):
        qacap_total_loss(
            logits,
            labels,
            qafd_attention,
            caption_attention,
            lambda_js=-0.1,
        )


def test_js_rejects_all_zero_attention_maps():
    zero_attention = torch.zeros(1, 3, 5)
    with pytest.raises(ValueError, match="all-zero"):
        js_divergence(zero_attention, zero_attention)


def test_js_rejects_mismatched_region_dimensions():
    first = _random_attention(batch=1, queries=3, regions=5)
    second = _random_attention(batch=1, queries=4, regions=6)
    with pytest.raises(ValueError, match="matching batch and image-region dimensions"):
        js_divergence(first, second)


def test_js_rejects_samples_with_no_valid_query_positions():
    attention = _random_attention(batch=2, queries=3, regions=5)
    valid_mask = torch.tensor(
        [[True, True, True],
         [False, False, False]]
    )
    with pytest.raises(ValueError, match="no valid query positions"):
        js_divergence(attention, attention, q_query_mask=valid_mask)
