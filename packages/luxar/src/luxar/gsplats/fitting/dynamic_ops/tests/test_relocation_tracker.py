"""Tests for RecentlyRelocatedTracker cooldown mechanism."""

import pytest
import torch

from luxar.gsplats.fitting.dynamic_ops.operations import RecentlyRelocatedTracker


def test_tracker_initialization():
    """Test tracker initializes correctly."""
    tracker = RecentlyRelocatedTracker(n_splats=100, cooldown_steps=3, device="cpu")

    assert tracker.n_splats == 100
    assert tracker.cooldown_steps == 3
    assert tracker.current_step == 0
    assert tracker.total_relocations == 0
    assert tracker.unique_splats_relocated == 0

    # All splats should be eligible initially
    candidates = torch.arange(100, dtype=torch.long)
    eligible = tracker.filter_eligible_splats(candidates)
    assert len(eligible) == 100


def test_tracker_marks_relocated():
    """Test marking splats as relocated."""
    tracker = RecentlyRelocatedTracker(n_splats=100, cooldown_steps=3, device="cpu")

    # Relocate splats 10, 20, 30
    relocated = torch.tensor([10, 20, 30], dtype=torch.long)
    tracker.mark_relocated_batch(relocated)

    assert tracker.total_relocations == 3
    assert tracker.unique_splats_relocated == 3

    stats = tracker.get_statistics()
    assert stats["total_relocations"] == 3
    assert stats["unique_splats"] == 3
    assert stats["currently_on_cooldown"] == 3


def test_tracker_cooldown_filtering():
    """Test that recently relocated splats are filtered out."""
    tracker = RecentlyRelocatedTracker(n_splats=100, cooldown_steps=3, device="cpu")

    # Relocate splats 10, 20, 30 at step 0
    relocated = torch.tensor([10, 20, 30], dtype=torch.long)
    tracker.mark_relocated_batch(relocated)

    # Immediately check eligibility
    candidates = torch.tensor([5, 10, 15, 20, 25, 30], dtype=torch.long)
    eligible = tracker.filter_eligible_splats(candidates)

    # 10, 20, 30 should be filtered out (on cooldown)
    # 5, 15, 25 should remain
    assert len(eligible) == 3
    assert 5 in eligible
    assert 15 in eligible
    assert 25 in eligible
    assert 10 not in eligible
    assert 20 not in eligible
    assert 30 not in eligible


def test_tracker_cooldown_expiration():
    """Test that cooldown expires after specified steps."""
    tracker = RecentlyRelocatedTracker(n_splats=100, cooldown_steps=3, device="cpu")

    # Relocate splat 10 at step 0
    tracker.mark_relocated_batch(torch.tensor([10], dtype=torch.long))
    assert tracker.current_step == 0

    candidates = torch.tensor([10], dtype=torch.long)

    # Step 0: Just relocated, should be filtered
    eligible = tracker.filter_eligible_splats(candidates)
    assert len(eligible) == 0

    # Advance to step 1: Still on cooldown
    tracker.advance_step()
    eligible = tracker.filter_eligible_splats(candidates)
    assert len(eligible) == 0

    # Advance to step 2: Still on cooldown
    tracker.advance_step()
    eligible = tracker.filter_eligible_splats(candidates)
    assert len(eligible) == 0

    # Advance to step 3: Cooldown should expire (3 steps have passed)
    tracker.advance_step()
    eligible = tracker.filter_eligible_splats(candidates)
    assert len(eligible) == 1
    assert 10 in eligible


def test_tracker_multiple_relocations():
    """Test tracking multiple relocations of the same splat."""
    tracker = RecentlyRelocatedTracker(n_splats=100, cooldown_steps=2, device="cpu")

    # First relocation
    tracker.mark_relocated_batch(torch.tensor([10], dtype=torch.long))
    assert tracker.total_relocations == 1
    assert tracker.unique_splats_relocated == 1

    # Advance past cooldown
    tracker.advance_step()
    tracker.advance_step()

    # Second relocation of same splat
    tracker.mark_relocated_batch(torch.tensor([10], dtype=torch.long))
    assert tracker.total_relocations == 2
    assert tracker.unique_splats_relocated == 1  # Still only 1 unique

    stats = tracker.get_statistics()
    assert stats["total_relocations"] == 2
    assert stats["unique_splats"] == 1


def test_tracker_prevents_immediate_rerelocation():
    """Test that tracker prevents the critical bug of immediate re-relocation."""
    tracker = RecentlyRelocatedTracker(n_splats=1000, cooldown_steps=3, device="cpu")

    # Simulate weak splat selection: indices 0-99 are weakest
    weak_candidates = torch.arange(100, dtype=torch.long)

    # Step 0: Relocate first 10 weakest
    to_relocate = weak_candidates[:10]
    tracker.mark_relocated_batch(to_relocate)

    # Step 1: Try to select weak splats again
    tracker.advance_step()
    eligible = tracker.filter_eligible_splats(weak_candidates)

    # The 10 just relocated should NOT be eligible
    assert len(eligible) == 90  # 100 - 10 on cooldown
    for idx in to_relocate:
        assert idx not in eligible

    # But other weak splats should be eligible
    assert 10 in eligible  # Not relocated yet
    assert 50 in eligible


def test_tracker_statistics():
    """Test statistics reporting."""
    tracker = RecentlyRelocatedTracker(n_splats=1000, cooldown_steps=3, device="cpu")

    # Step 0: Relocate splats 1-5 (all on cooldown at step 0)
    tracker.mark_relocated_batch(torch.tensor([1, 2, 3, 4, 5], dtype=torch.long))
    tracker.advance_step()  # Now at step 1

    # Step 1: Relocate splats 6-8 (all on cooldown at step 1)
    tracker.mark_relocated_batch(torch.tensor([6, 7, 8], dtype=torch.long))
    tracker.advance_step()  # Now at step 2

    # Step 2: Relocate splats 1 (repeat) and 9
    tracker.mark_relocated_batch(torch.tensor([1, 9], dtype=torch.long))

    # At step 2, all 9 unique splats are within cooldown (cooldown_steps=3):
    # - Splats 2-5: relocated at step 0, steps_since = 2, < 3 → on cooldown
    # - Splats 6-8: relocated at step 1, steps_since = 1, < 3 → on cooldown
    # - Splats 1, 9: relocated at step 2, steps_since = 0, < 3 → on cooldown
    stats = tracker.get_statistics()
    assert stats["total_relocations"] == 10  # 5 + 3 + 2
    assert stats["unique_splats"] == 9  # 1-9 (1 appears twice)
    assert stats["currently_on_cooldown"] == 9  # All 9 unique splats still on cooldown


@pytest.mark.parametrize("device", ["cpu"])
def test_tracker_gpu_compatibility(device):
    """Test tracker works with different devices."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tracker = RecentlyRelocatedTracker(n_splats=100, cooldown_steps=3, device=device)

    candidates = torch.tensor([10, 20, 30], dtype=torch.long, device=device)
    tracker.mark_relocated_batch(candidates)

    eligible = tracker.filter_eligible_splats(
        torch.arange(100, dtype=torch.long, device=device)
    )

    assert eligible.device.type == device
    assert len(eligible) == 97  # 100 - 3 on cooldown
