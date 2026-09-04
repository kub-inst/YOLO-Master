"""Tests for configurable, per-side STAL candidate-region expansion."""

import pytest
import torch

from ultralytics.utils.tal import TaskAlignedAssigner


def _assigner(**kwargs) -> TaskAlignedAssigner:
    """Create a small assigner with only the candidate-region logic configured."""
    return TaskAlignedAssigner(num_classes=1, stride=[8], **kwargs)


def _mask(assigner: TaskAlignedAssigner, bbox: list[float], points: list[list[float]]) -> torch.Tensor:
    """Return candidate eligibility for one valid GT box and a list of anchor centers."""
    return assigner.select_candidates_in_gts(
        torch.tensor(points, dtype=torch.float32),
        torch.tensor([[bbox]], dtype=torch.float32),
        torch.ones(1, 1, 1, dtype=torch.bool),
    )[0, 0]


def test_8_to_16_expansion_uses_an_explicit_target_and_preserves_the_other_side():
    """An enabled 8-16 target should expand only a 12-pixel side to 24."""
    mask = _mask(
        _assigner(candidate_expand_8_16=24),
        [40.0, 20.0, 52.0, 60.0],  # 12 x 40
        [[46.0, 40.0], [56.0, 40.0], [46.0, 65.0]],
    )
    assert mask.tolist() == [True, True, False]


def test_default_preserves_baseline_by_disabling_8_to_16_expansion():
    """The default -1 target must leave the new interval unchanged."""
    mask = _mask(
        _assigner(),
        [40.0, 40.0, 52.0, 52.0],  # 12 x 12
        [[46.0, 46.0], [54.0, 46.0]],
    )
    assert mask.tolist() == [True, False]


def test_0_to_8_target_is_independently_configurable():
    """The existing 0-8 expansion remains configurable and can be disabled independently."""
    points = [[22.0, 30.0], [29.0, 30.0]]
    expanded = _mask(_assigner(), [20.0, 20.0, 24.0, 40.0], points)  # 4 x 20
    disabled = _mask(_assigner(candidate_expand_0_8=-1), [20.0, 20.0, 24.0, 40.0], points)
    assert expanded.tolist() == [True, True]
    assert disabled.tolist() == [True, False]


def test_linear_decay_keeps_full_expansion_then_restores_original_box():
    """The 60+60 schedule is full at epoch 60, halfway at 90, and off at 120."""
    assigner = _assigner(
        candidate_expand_0_8=16,
        candidate_expand_8_16=24,
        candidate_expand_linear_decay=True,
        candidate_expand_full_epochs=60,
        candidate_expand_decay_epochs=60,
    )
    bbox = [20.0, 20.0, 24.0, 32.0]  # 4 x 12; full expansion targets 16 x 24

    assert assigner.set_epoch(60) == pytest.approx(1.0)
    assert _mask(assigner, bbox, [[29.0, 37.0]]).item()

    assert assigner.set_epoch(90) == pytest.approx(0.5)
    assert _mask(assigner, bbox, [[26.0, 34.0]]).item()
    assert not _mask(assigner, bbox, [[29.0, 37.0]]).item()

    assert assigner.set_epoch(120) == pytest.approx(0.0)
    assert not _mask(assigner, bbox, [[26.0, 34.0]]).item()


def test_linear_decay_starts_after_the_full_expansion_phase_and_stays_off():
    """Epoch 61 starts contraction and epochs after 120 remain at zero strength."""
    assigner = _assigner(
        candidate_expand_linear_decay=True,
        candidate_expand_full_epochs=60,
        candidate_expand_decay_epochs=60,
    )
    assert assigner.set_epoch(61) == pytest.approx(59 / 60)
    assert assigner.set_epoch(120) == pytest.approx(0.0)
    assert assigner.set_epoch(121) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_expand_0_8": 7}, "candidate_expand_0_8"),
        ({"candidate_expand_8_16": 15}, "candidate_expand_8_16"),
        ({"candidate_expand_0_8": -2}, "candidate_expand_0_8"),
        ({"candidate_expand_full_epochs": -1}, "candidate_expand_full_epochs"),
        ({"candidate_expand_decay_epochs": -1}, "candidate_expand_decay_epochs"),
        (
            {"candidate_expand_linear_decay": True, "candidate_expand_decay_epochs": 0},
            "candidate_expand_decay_epochs",
        ),
    ],
)
def test_candidate_expansion_targets_reject_shrinking_or_invalid_values(kwargs, message):
    """Targets are either disabled (-1) or no smaller than their interval upper bound."""
    with pytest.raises(ValueError, match=message):
        _assigner(**kwargs)
