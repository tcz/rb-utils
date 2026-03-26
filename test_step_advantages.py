"""
Tests for step_advantages.py -- step-level advantage computation for Tree-VPO.

Run: python -m pytest training/utils/test_step_advantages.py -v
"""

import os
import sys

import pytest

# Ensure training/ is on path so `from utils.X` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.step_advantages import (
    compute_step_delta_advantages,
    compute_combined_advantages,
    compute_inter_beam_advantages,
    compute_intra_beam_advantages,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_entry(sibling_group_id: int, step_reward: float, beam_id: int = 0) -> dict:
    """Create a minimal training data entry for testing."""
    return {
        "sibling_group_id": sibling_group_id,
        "step_reward": step_reward,
        "beam_id": beam_id,
    }


def make_step_entry(step_reward: float, **extra) -> dict:
    """Create a minimal entry for step-delta advantage testing."""
    entry = {"step_reward": step_reward}
    entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# Step-delta advantage tests (Tree-VPO primary)
# ---------------------------------------------------------------------------

class TestStepDeltaAdvantages:

    def test_basic_positive_negative(self):
        """Positive step_reward -> positive advantage, negative -> negative."""
        data = [
            make_step_entry(0.3),
            make_step_entry(-0.1),
            make_step_entry(0.5),
        ]
        results = compute_step_delta_advantages(data, normalize=False)

        assert len(results) == 3
        assert results[0]["step_advantage"] == 0.3
        assert results[1]["step_advantage"] == -0.1
        assert results[2]["step_advantage"] == 0.5

    def test_normalized_zero_mean(self):
        """With normalize=True, advantages should be zero-mean."""
        data = [
            make_step_entry(0.1),
            make_step_entry(0.3),
            make_step_entry(0.5),
            make_step_entry(0.7),
        ]
        results = compute_step_delta_advantages(data, normalize=True)

        advantages = [r["step_advantage"] for r in results]
        assert sum(advantages) == pytest.approx(0.0, abs=1e-6)

    def test_normalized_unit_variance(self):
        """With normalize=True, advantages should have unit variance."""
        data = [
            make_step_entry(0.1),
            make_step_entry(0.3),
            make_step_entry(0.5),
            make_step_entry(0.7),
        ]
        results = compute_step_delta_advantages(data, normalize=True)

        advantages = [r["step_advantage"] for r in results]
        n = len(advantages)
        mu = sum(advantages) / n
        variance = sum((a - mu) ** 2 for a in advantages) / n
        assert variance == pytest.approx(1.0, abs=1e-6)

    def test_normalized_ordering_preserved(self):
        """Normalization should preserve relative ordering."""
        data = [
            make_step_entry(-0.2),
            make_step_entry(0.1),
            make_step_entry(0.4),
        ]
        results = compute_step_delta_advantages(data, normalize=True)

        assert results[0]["step_advantage"] < results[1]["step_advantage"]
        assert results[1]["step_advantage"] < results[2]["step_advantage"]

    def test_constant_rewards_zero_advantage(self):
        """All same step_reward -> all zero advantage when normalized."""
        data = [
            make_step_entry(0.5),
            make_step_entry(0.5),
            make_step_entry(0.5),
        ]
        results = compute_step_delta_advantages(data, normalize=True)

        assert all(r["step_advantage"] == 0.0 for r in results)

    def test_empty_input(self):
        """Empty input returns empty output."""
        assert compute_step_delta_advantages([]) == []

    def test_single_entry_no_normalize(self):
        """Single entry without normalization returns raw step_reward."""
        data = [make_step_entry(0.42)]
        results = compute_step_delta_advantages(data, normalize=False)

        assert results[0]["step_advantage"] == 0.42

    def test_single_entry_normalize(self):
        """Single entry with normalize=True uses raw value (can't normalize n=1)."""
        data = [make_step_entry(0.42)]
        results = compute_step_delta_advantages(data, normalize=True)

        assert results[0]["step_advantage"] == 0.42

    def test_preserves_original_fields(self):
        """Original fields should be preserved in output."""
        data = [
            {
                "step_reward": 0.3,
                "prefix": "<html>",
                "continuation": "<div>",
                "step": 1,
                "hybrid_reward": 0.7,
            },
        ]
        results = compute_step_delta_advantages(data, normalize=False)

        r = results[0]
        assert r["prefix"] == "<html>"
        assert r["continuation"] == "<div>"
        assert r["step"] == 1
        assert r["hybrid_reward"] == 0.7
        assert r["step_advantage"] == 0.3

    def test_does_not_mutate_input(self):
        """Should not mutate the input data."""
        data = [
            make_step_entry(0.3),
            make_step_entry(0.7),
        ]
        original_keys = [set(d.keys()) for d in data]

        compute_step_delta_advantages(data, normalize=True)

        for d, orig_keys in zip(data, original_keys):
            assert set(d.keys()) == orig_keys

    def test_mixed_positive_negative_rewards(self):
        """Realistic scenario: some steps help, some hurt."""
        data = [
            make_step_entry(0.15),   # element improved render
            make_step_entry(-0.05),  # element hurt render
            make_step_entry(0.25),   # big improvement
            make_step_entry(0.0),    # no change
            make_step_entry(-0.10),  # hurt render
        ]
        results = compute_step_delta_advantages(data, normalize=True)

        advantages = [r["step_advantage"] for r in results]
        # The biggest improvement (0.25) should have the highest advantage
        assert advantages[2] == max(advantages)
        # The biggest hurt (-0.10) should have the lowest advantage
        assert advantages[4] == min(advantages)

    def test_normalize_false_returns_raw_rewards(self):
        """Without normalization, step_advantage == step_reward exactly."""
        data = [
            make_step_entry(0.1),
            make_step_entry(-0.3),
            make_step_entry(0.0),
        ]
        results = compute_step_delta_advantages(data, normalize=False)

        for entry, result in zip(data, results):
            assert result["step_advantage"] == entry["step_reward"]


# ---------------------------------------------------------------------------
# Legacy: Intra-beam advantage tests
# ---------------------------------------------------------------------------

class TestIntraBeamAdvantages:

    def test_intra_beam_normalizes_within_group(self):
        """3 siblings [0.1, 0.3, 0.5]: lowest should be negative, highest positive."""
        data = [
            make_entry(sibling_group_id=1, step_reward=0.1),
            make_entry(sibling_group_id=1, step_reward=0.3),
            make_entry(sibling_group_id=1, step_reward=0.5),
        ]

        advantages = compute_intra_beam_advantages(data)

        assert len(advantages) == 3
        # Lowest reward -> negative advantage
        assert advantages[0] < 0.0
        # Middle reward -> near zero (it is the mean)
        assert advantages[1] == pytest.approx(0.0, abs=1e-6)
        # Highest reward -> positive advantage
        assert advantages[2] > 0.0

    def test_intra_beam_zero_centered(self):
        """Advantages within each group should sum to approximately zero."""
        data = [
            make_entry(sibling_group_id=1, step_reward=0.1),
            make_entry(sibling_group_id=1, step_reward=0.3),
            make_entry(sibling_group_id=1, step_reward=0.5),
            make_entry(sibling_group_id=2, step_reward=0.7),
            make_entry(sibling_group_id=2, step_reward=0.2),
        ]

        advantages = compute_intra_beam_advantages(data)

        # Group 1: indices 0, 1, 2
        group1_sum = advantages[0] + advantages[1] + advantages[2]
        assert group1_sum == pytest.approx(0.0, abs=1e-6)

        # Group 2: indices 3, 4
        group2_sum = advantages[3] + advantages[4]
        assert group2_sum == pytest.approx(0.0, abs=1e-6)

    def test_intra_beam_separate_groups(self):
        """Two groups should be normalized independently.

        Group A: [0.1, 0.9] -- large spread
        Group B: [0.4, 0.5] -- small spread

        The best in group B (0.5) should NOT have a higher advantage than
        the best in group A (0.9) just because of their absolute values.
        Each group is normalized relative to its own statistics.
        """
        data = [
            make_entry(sibling_group_id=1, step_reward=0.1),  # group A low
            make_entry(sibling_group_id=1, step_reward=0.9),  # group A high
            make_entry(sibling_group_id=2, step_reward=0.4),  # group B low
            make_entry(sibling_group_id=2, step_reward=0.5),  # group B high
        ]

        advantages = compute_intra_beam_advantages(data)

        # Within each group of 2 symmetric around the mean:
        # advantage = (x - mu) / sigma
        # Group A: mu=0.5, sigma=0.4, advantages: -1.0, +1.0
        # Group B: mu=0.45, sigma=0.05, advantages: -1.0, +1.0
        # Both groups should produce the same magnitude advantages
        assert advantages[0] == pytest.approx(-advantages[1], abs=1e-6)
        assert advantages[2] == pytest.approx(-advantages[3], abs=1e-6)

        # The absolute advantage magnitudes should be identical for both groups
        # (both are 2-element groups with symmetric deviations)
        assert abs(advantages[0]) == pytest.approx(abs(advantages[2]), abs=1e-6)

    def test_constant_rewards_zero_advantage(self):
        """All same reward within a group -> all zero advantage (no signal)."""
        data = [
            make_entry(sibling_group_id=1, step_reward=0.5),
            make_entry(sibling_group_id=1, step_reward=0.5),
            make_entry(sibling_group_id=1, step_reward=0.5),
        ]

        advantages = compute_intra_beam_advantages(data)

        assert all(a == 0.0 for a in advantages)

    def test_single_entry_group(self):
        """A group with one entry has no variance -> advantage is 0.0."""
        data = [make_entry(sibling_group_id=1, step_reward=0.8)]

        advantages = compute_intra_beam_advantages(data)

        assert advantages[0] == 0.0

    def test_empty_input(self):
        """Empty input returns empty output."""
        assert compute_intra_beam_advantages([]) == []

    def test_preserves_input_order(self):
        """Advantages should correspond to the same positions as input entries."""
        data = [
            make_entry(sibling_group_id=1, step_reward=0.9),  # idx 0: high
            make_entry(sibling_group_id=2, step_reward=0.1),  # idx 1: low in group 2
            make_entry(sibling_group_id=1, step_reward=0.1),  # idx 2: low in group 1
            make_entry(sibling_group_id=2, step_reward=0.9),  # idx 3: high in group 2
        ]

        advantages = compute_intra_beam_advantages(data)

        # Group 1 (indices 0, 2): 0.9 should be positive, 0.1 negative
        assert advantages[0] > 0.0  # high reward in group 1
        assert advantages[2] < 0.0  # low reward in group 1

        # Group 2 (indices 1, 3): 0.1 should be negative, 0.9 positive
        assert advantages[1] < 0.0  # low reward in group 2
        assert advantages[3] > 0.0  # high reward in group 2


# ---------------------------------------------------------------------------
# Inter-beam advantage tests
# ---------------------------------------------------------------------------

class TestInterBeamAdvantages:

    def test_inter_beam_normalizes_terminals(self):
        """[0.5, 0.7, 0.8, 0.6]: highest (0.8) gets positive, lowest (0.5) negative."""
        rewards = [0.5, 0.7, 0.8, 0.6]

        advantages = compute_inter_beam_advantages(rewards)

        assert len(advantages) == 4
        # 0.8 is the highest -> largest positive advantage
        assert advantages[2] > 0.0
        assert advantages[2] == max(advantages)
        # 0.5 is the lowest -> most negative advantage
        assert advantages[0] < 0.0
        assert advantages[0] == min(advantages)

    def test_inter_beam_constant_zero(self):
        """All same terminal reward -> all zero advantage."""
        rewards = [0.6, 0.6, 0.6, 0.6]

        advantages = compute_inter_beam_advantages(rewards)

        assert all(a == 0.0 for a in advantages)

    def test_inter_beam_zero_centered(self):
        """Advantages across terminals should sum to approximately zero."""
        rewards = [0.5, 0.7, 0.8, 0.6]

        advantages = compute_inter_beam_advantages(rewards)

        assert sum(advantages) == pytest.approx(0.0, abs=1e-6)

    def test_inter_beam_empty(self):
        """Empty input returns empty output."""
        assert compute_inter_beam_advantages([]) == []

    def test_inter_beam_single(self):
        """Single terminal has no variance -> advantage is 0.0."""
        advantages = compute_inter_beam_advantages([0.75])
        assert advantages == [0.0]

    def test_inter_beam_two_terminals(self):
        """Two terminals should get symmetric advantages."""
        advantages = compute_inter_beam_advantages([0.3, 0.7])

        assert advantages[0] < 0.0
        assert advantages[1] > 0.0
        assert advantages[0] == pytest.approx(-advantages[1], abs=1e-6)


# ---------------------------------------------------------------------------
# Combined advantage tests
# ---------------------------------------------------------------------------

class TestCombinedAdvantages:

    def test_combined_has_all_fields(self):
        """Each result entry should have intra_advantage, inter_advantage, total_advantage."""
        data = [
            make_entry(sibling_group_id=1, step_reward=0.3, beam_id=0),
            make_entry(sibling_group_id=1, step_reward=0.5, beam_id=0),
            make_entry(sibling_group_id=2, step_reward=0.4, beam_id=1),
            make_entry(sibling_group_id=2, step_reward=0.6, beam_id=1),
        ]
        terminal_rewards = {0: 0.7, 1: 0.8}

        results = compute_combined_advantages(data, terminal_rewards)

        assert len(results) == 4
        for r in results:
            assert "intra_advantage" in r
            assert "inter_advantage" in r
            assert "total_advantage" in r

    def test_combined_total_is_sum(self):
        """total_advantage should equal intra_advantage + inter_advantage."""
        data = [
            make_entry(sibling_group_id=1, step_reward=0.2, beam_id=0),
            make_entry(sibling_group_id=1, step_reward=0.6, beam_id=0),
            make_entry(sibling_group_id=2, step_reward=0.3, beam_id=1),
            make_entry(sibling_group_id=2, step_reward=0.7, beam_id=1),
        ]
        terminal_rewards = {0: 0.5, 1: 0.9}

        results = compute_combined_advantages(data, terminal_rewards)

        for r in results:
            expected_total = r["intra_advantage"] + r["inter_advantage"]
            assert r["total_advantage"] == pytest.approx(expected_total, abs=1e-10)

    def test_inter_maps_to_beam_id(self):
        """Entries in the same beam should get the same inter_advantage."""
        data = [
            make_entry(sibling_group_id=1, step_reward=0.1, beam_id=0),
            make_entry(sibling_group_id=1, step_reward=0.3, beam_id=0),
            make_entry(sibling_group_id=2, step_reward=0.5, beam_id=0),
            make_entry(sibling_group_id=3, step_reward=0.2, beam_id=1),
            make_entry(sibling_group_id=3, step_reward=0.4, beam_id=1),
            make_entry(sibling_group_id=4, step_reward=0.6, beam_id=1),
        ]
        terminal_rewards = {0: 0.6, 1: 0.8}

        results = compute_combined_advantages(data, terminal_rewards)

        # All beam_id=0 entries should share the same inter_advantage
        beam0_inter = [r["inter_advantage"] for r in results if r["beam_id"] == 0]
        assert all(a == beam0_inter[0] for a in beam0_inter)

        # All beam_id=1 entries should share the same inter_advantage
        beam1_inter = [r["inter_advantage"] for r in results if r["beam_id"] == 1]
        assert all(a == beam1_inter[0] for a in beam1_inter)

        # The two beams should have different inter_advantages
        # (since 0.6 != 0.8)
        assert beam0_inter[0] != beam1_inter[0]

        # Beam 1 has higher terminal reward -> positive inter advantage
        assert beam1_inter[0] > 0.0
        assert beam0_inter[0] < 0.0

    def test_combined_preserves_original_fields(self):
        """Original fields from training_data should be preserved in output."""
        data = [
            {
                "sibling_group_id": 1,
                "step_reward": 0.3,
                "beam_id": 0,
                "prefix": "<html>",
                "continuation": "<div>",
                "step": 1,
                "hybrid_reward": 0.3,
                "is_terminal": False,
                "full_text": "<html><div>",
            },
        ]
        terminal_rewards = {0: 0.5}

        results = compute_combined_advantages(data, terminal_rewards)

        r = results[0]
        assert r["prefix"] == "<html>"
        assert r["continuation"] == "<div>"
        assert r["step"] == 1
        assert r["hybrid_reward"] == 0.3
        assert r["is_terminal"] is False
        assert r["full_text"] == "<html><div>"

    def test_combined_does_not_mutate_input(self):
        """compute_combined_advantages should not mutate the input data."""
        data = [
            make_entry(sibling_group_id=1, step_reward=0.3, beam_id=0),
            make_entry(sibling_group_id=1, step_reward=0.7, beam_id=0),
        ]
        terminal_rewards = {0: 0.8}

        # Save original keys
        original_keys = [set(d.keys()) for d in data]

        compute_combined_advantages(data, terminal_rewards)

        # Input entries should not have new keys added
        for d, orig_keys in zip(data, original_keys):
            assert set(d.keys()) == orig_keys

    def test_combined_multiple_beams_multiple_groups(self):
        """Realistic scenario: 2 beams, each with 2 steps of 3 siblings."""
        data = [
            # Beam 0, step 1: 3 siblings
            make_entry(sibling_group_id=1, step_reward=0.1, beam_id=0),
            make_entry(sibling_group_id=1, step_reward=0.3, beam_id=0),
            make_entry(sibling_group_id=1, step_reward=0.5, beam_id=0),
            # Beam 0, step 2: 3 siblings (from surviving parent)
            make_entry(sibling_group_id=2, step_reward=0.2, beam_id=0),
            make_entry(sibling_group_id=2, step_reward=0.4, beam_id=0),
            make_entry(sibling_group_id=2, step_reward=0.6, beam_id=0),
            # Beam 1, step 1: 3 siblings
            make_entry(sibling_group_id=3, step_reward=0.0, beam_id=1),
            make_entry(sibling_group_id=3, step_reward=0.2, beam_id=1),
            make_entry(sibling_group_id=3, step_reward=0.4, beam_id=1),
            # Beam 1, step 2: 3 siblings
            make_entry(sibling_group_id=4, step_reward=0.3, beam_id=1),
            make_entry(sibling_group_id=4, step_reward=0.5, beam_id=1),
            make_entry(sibling_group_id=4, step_reward=0.7, beam_id=1),
        ]
        terminal_rewards = {0: 0.6, 1: 0.9}

        results = compute_combined_advantages(data, terminal_rewards)

        assert len(results) == 12

        # All results should have the advantage fields
        for r in results:
            assert "intra_advantage" in r
            assert "inter_advantage" in r
            assert "total_advantage" in r
            assert r["total_advantage"] == pytest.approx(
                r["intra_advantage"] + r["inter_advantage"], abs=1e-10
            )

        # Intra-advantages within each group sum to ~0
        for gid in [1, 2, 3, 4]:
            group_intra = [r["intra_advantage"] for r in results
                           if r["sibling_group_id"] == gid]
            assert sum(group_intra) == pytest.approx(0.0, abs=1e-6)

        # Beam 1 has higher terminal reward -> positive inter advantage
        beam1_inter = results[6]["inter_advantage"]
        beam0_inter = results[0]["inter_advantage"]
        assert beam1_inter > 0.0
        assert beam0_inter < 0.0
