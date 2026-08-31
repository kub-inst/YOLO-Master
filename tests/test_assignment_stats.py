"""Tests for non-invasive TAL/STAL positive-assignment telemetry."""

import torch

from ultralytics import YOLO  # noqa: F401 - initialize task/loss imports in package order
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


def _bare_loss(enabled: bool = True) -> v8DetectionLoss:
    """Construct only the telemetry state without building a detection model."""
    loss = object.__new__(v8DetectionLoss)
    loss.assignment_stats_enabled = enabled
    loss.assignment_small_area = 32.0**2
    loss.assignment_medium_area = 96.0**2
    loss._assignment_stats = torch.zeros(len(loss._ASSIGNMENT_STAT_NAMES), dtype=torch.long)
    return loss


def test_assignment_stats_are_binned_and_reset_without_changing_inputs():
    """Counters should preserve inputs and report post-conflict positives by GT area."""
    loss = _bare_loss()
    gt_bboxes = torch.tensor(
        [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 40.0, 40.0], [0.0, 0.0, 100.0, 100.0], [0, 0, 120, 120]]]
    )
    mask_gt = torch.ones(1, 4, 1, dtype=torch.bool)
    fg_mask = torch.tensor([[True, True, True, True, False]])
    target_gt_idx = torch.tensor([[0, 0, 1, 2, 0]])
    originals = tuple(x.clone() for x in (gt_bboxes, mask_gt, fg_mask, target_gt_idx))

    loss._update_assignment_stats(gt_bboxes, mask_gt, fg_mask, target_gt_idx)
    stats = {key: int(value) for key, value in loss.assignment_stats().items()}

    assert stats == {
        "gt_total": 4,
        "pos_total": 4,
        "zero_gt": 1,
        "gt_small": 1,
        "gt_medium": 1,
        "gt_large": 2,
        "pos_small": 2,
        "pos_medium": 1,
        "pos_large": 1,
        "zero_small": 0,
        "zero_medium": 0,
        "zero_large": 1,
    }
    for actual, original in zip((gt_bboxes, mask_gt, fg_mask, target_gt_idx), originals):
        torch.testing.assert_close(actual, original)

    loss.reset_assignment_stats()
    assert all(int(value) == 0 for value in loss.assignment_stats().values())


def test_assignment_stats_disabled_is_a_noop():
    """The default-off path should expose no metrics and perform no counter work."""
    loss = _bare_loss(enabled=False)
    loss._update_assignment_stats(
        torch.tensor([[[0.0, 0.0, 10.0, 10.0]]]),
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.tensor([[True]]),
        torch.tensor([[0]]),
    )
    assert loss.assignment_stats() == {}
    assert not loss._assignment_stats.any()


def test_dynamic_topk_changes_only_small_gt_candidate_count():
    """Dynamic TopK should use ceil(lambda*x) for small GTs and retain fixed K for larger GTs."""
    assigner = TaskAlignedAssigner(topk=3, small_area_threshold=32.0**2, dynamic_topk_small=True, dynamic_topk_lambda=0.8)
    metrics = torch.tensor([[[9.0, 8.0, 7.0, 6.0, 5.0, 0.0], [0.0, 9.0, 8.0, 7.0, 6.0, 5.0]]])
    candidates = torch.tensor([[[True, True, True, True, True, False], [False, True, True, True, True, True]]])
    gt_bboxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 40.0, 40.0]]])
    valid_gts = torch.ones(1, 2, 1, dtype=torch.bool)

    selected = assigner.select_topk_candidates(
        metrics,
        topk_mask=valid_gts.expand(-1, -1, 3),
        candidate_mask=candidates,
        gt_bboxes=gt_bboxes,
        mask_gt=valid_gts,
    )

    assert selected[0, 0].sum() == 4  # ceil(0.8 * 5)
    assert selected[0, 1].sum() == 3  # unchanged fixed K
    assert not selected.bool().logical_and(~candidates).any()
