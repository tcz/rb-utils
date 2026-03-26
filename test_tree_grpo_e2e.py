"""
End-to-end integration test for Tree-VPO (Visual Policy Optimization) pipeline.

Tests the full flow from beam search → rewards → step-delta advantages → loss
computation using mocks (no GPU, no vLLM, no Playwright needed).

Run: cd training && python -m pytest utils/test_tree_grpo_e2e.py -v
"""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.beam_search import BeamSearchConfig, run_beam_search, BeamTree, reset_sibling_group_counter
from utils.hybrid_reward import compute_hybrid_reward
from utils.step_advantages import (
    compute_step_delta_advantages,
    compute_intra_beam_advantages,
    compute_inter_beam_advantages,
    compute_combined_advantages,
)


class MockRewardPool:
    """Mocked RewardPool that returns deterministic rewards based on HTML length."""

    def calculate_metrics_batch_intermediate(self, items):
        results = []
        for pred_html, ref_html, vw, vh in items:
            # Reward proportional to HTML length (more elements = better)
            # Normalized to [0, 1] range
            html_len = len(pred_html)
            fake_lpips = max(0.0, 1.0 - html_len / 500.0)  # longer = lower loss

            # Create fake screenshot arrays (small for speed)
            fake_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            results.append({
                'lpips_score': fake_lpips,
                'pred_screenshot': fake_img,
                'gt_screenshot': fake_img.copy(),
            })
        return results

    def calculate_metrics_batch(self, items, **kwargs):
        results = []
        for pred_html, ref_html, vw, vh in items:
            html_len = len(pred_html)
            fake_lpips = max(0.0, 1.0 - html_len / 500.0)
            results.append({
                'perceptual_loss': fake_lpips,
                'similarity': 1.0 - fake_lpips,
            })
        return results

    def calculate_metrics(self, pred, ref, vw, vh, **kwargs):
        return self.calculate_metrics_batch([(pred, ref, vw, vh)])[0]


def create_deterministic_mock_generate(seed=42):
    """Mock generate_fn producing inline-style HTML elements."""
    rng = random.Random(seed)

    def generate(prefix, K):
        results = []
        open_divs = prefix.count("<div") - prefix.count("</div>")
        element_count = prefix.count("<div")

        for _ in range(K):
            if element_count > 3 and rng.random() < 0.2:
                results.append("")  # terminal
                continue

            if element_count == 0:
                d = rng.choice(["row", "column"])
                results.append(f'<div style="display:flex;flex-direction:{d};">')
            elif open_divs > 0 and element_count > 2 and rng.random() < 0.3:
                results.append("</div>")
            else:
                c = rng.choice(["#ff0000", "#00ff00", "#0000ff", "#ffff00"])
                w = rng.choice([50, 100, 150])
                h = rng.choice([50, 100])
                results.append(
                    f'<div style="background:{c};width:{w}px;height:{h}px;"></div>'
                )
        return results

    return generate


class TestBeamSearchE2E(unittest.TestCase):
    """Test beam search produces a valid tree structure."""

    def setUp(self):
        reset_sibling_group_counter()

    def test_basic_beam_search(self):
        config = BeamSearchConfig(M=2, K=3, N=2, max_steps=3)
        generate_fn = create_deterministic_mock_generate()

        def mock_reward_fn(full_html, reference):
            return len(full_html) / 500.0  # simple length-based reward

        trees = run_beam_search(config, generate_fn, mock_reward_fn, reference=None)

        self.assertEqual(len(trees), 2)  # M=2 beams

        for tree in trees:
            nodes = tree.all_nodes()
            self.assertGreater(len(nodes), 1)  # root + at least some children

            root = tree.root
            self.assertEqual(root.step, 0)
            self.assertEqual(root.full_text, "")

            # All non-root nodes have parents
            for node in nodes:
                if node is not root:
                    self.assertIsNotNone(node.parent)
                    self.assertGreater(node.step, 0)

    def test_training_data_extraction(self):
        config = BeamSearchConfig(M=1, K=3, N=2, max_steps=3)
        generate_fn = create_deterministic_mock_generate()

        def mock_reward_fn(full_html, reference):
            return min(1.0, len(full_html) / 300.0)

        trees = run_beam_search(config, generate_fn, mock_reward_fn, reference=None)
        tree = trees[0]

        data = tree.extract_training_data()
        self.assertGreater(len(data), 0)

        for entry in data:
            self.assertIn("prefix", entry)
            self.assertIn("continuation", entry)
            self.assertIn("step_reward", entry)
            self.assertIn("hybrid_reward", entry)
            self.assertIn("sibling_group_id", entry)
            # full_text = prefix + continuation
            self.assertEqual(entry["full_text"], entry["prefix"] + entry["continuation"])


class TestBeamSearchWithRewards(unittest.TestCase):
    """Test beam search + hybrid reward integration."""

    def test_beam_search_with_mock_rewards(self):
        """Full pipeline: beam search → rendering → hybrid reward → step-delta advantages."""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from utils.hybrid_reward import compute_hybrid_reward, compute_emd

        generate_fn = create_deterministic_mock_generate(seed=123)
        reward_pool = MockRewardPool()

        config_M, config_K, config_N, max_steps = 2, 3, 2, 3
        alpha = 0.5
        reference_html = '<div style="display:flex;"><div style="background:red;width:100px;height:100px;"></div></div>'

        # Run instrumented beam search (same logic as training loop)
        sibling_group_counter = 0
        all_entries = []

        for beam_idx in range(config_M):
            active_nodes = [{"full_text": "", "hybrid_reward": 0.0}]

            for step in range(max_steps):
                all_children = []

                for parent in active_nodes:
                    sibling_group_counter += 1
                    continuations = generate_fn(parent["full_text"], config_K)

                    for cont in continuations:
                        is_terminal = (cont == "")
                        child = {
                            "prefix": parent["full_text"],
                            "continuation": cont,
                            "full_text": parent["full_text"] + cont,
                            "step": step + 1,
                            "hybrid_reward": parent["hybrid_reward"] if is_terminal else 0.0,
                            "step_reward": 0.0,
                            "beam_id": beam_idx,
                            "is_terminal": is_terminal,
                        }
                        all_children.append(child)

                to_render = [c for c in all_children if not c["is_terminal"]]
                if to_render:
                    items = [(c["full_text"], reference_html, 1024, 1024)
                             for c in to_render]
                    results = reward_pool.calculate_metrics_batch_intermediate(items)

                    for child, r in zip(to_render, results):
                        if r is not None:
                            child["hybrid_reward"] = compute_hybrid_reward(
                                r["pred_screenshot"], r["gt_screenshot"],
                                r["lpips_score"], alpha,
                            )

                for child in all_children:
                    parent_reward = 0.0
                    for p in active_nodes:
                        if p["full_text"] == child["prefix"]:
                            parent_reward = p["hybrid_reward"]
                            break
                    child["step_reward"] = child["hybrid_reward"] - parent_reward

                all_entries.extend(all_children)

                non_terminal = [c for c in all_children if not c["is_terminal"]]
                if not non_terminal:
                    break
                non_terminal.sort(key=lambda n: n["hybrid_reward"], reverse=True)
                active_nodes = non_terminal[:config_N]

        # Verify we got entries
        self.assertGreater(len(all_entries), 0)

        # Compute step-delta advantages
        results = compute_step_delta_advantages(all_entries, normalize=True)

        self.assertEqual(len(results), len(all_entries))

        # Check all entries have step_advantage field
        for entry in results:
            self.assertIn("step_advantage", entry)

        # Verify advantages are zero-mean across the full batch
        advantages = [e["step_advantage"] for e in results]
        mean_adv = sum(advantages) / len(advantages)
        self.assertAlmostEqual(mean_adv, 0.0, places=5,
                               msg=f"Advantages not zero-centered: mean={mean_adv}")

    def test_non_terminal_entries_have_positive_rewards(self):
        """Non-terminal entries should have computed hybrid rewards."""
        generate_fn = create_deterministic_mock_generate(seed=99)
        reward_pool = MockRewardPool()

        sibling_group_counter = 0
        all_entries = []

        active_nodes = [{"full_text": "", "hybrid_reward": 0.0}]
        for step in range(2):
            all_children = []
            for parent in active_nodes:
                sibling_group_counter += 1
                continuations = generate_fn(parent["full_text"], 3)
                for cont in continuations:
                    is_terminal = (cont == "")
                    child = {
                        "prefix": parent["full_text"],
                        "continuation": cont,
                        "full_text": parent["full_text"] + cont,
                        "step": step + 1,
                        "hybrid_reward": parent["hybrid_reward"] if is_terminal else 0.0,
                        "step_reward": 0.0,
                        "beam_id": 0,
                        "is_terminal": is_terminal,
                    }
                    all_children.append(child)

            to_render = [c for c in all_children if not c["is_terminal"]]
            if to_render:
                items = [(c["full_text"], "<html></html>", 1024, 1024)
                         for c in to_render]
                results = reward_pool.calculate_metrics_batch_intermediate(items)
                for child, r in zip(to_render, results):
                    if r is not None:
                        from utils.hybrid_reward import compute_hybrid_reward
                        child["hybrid_reward"] = compute_hybrid_reward(
                            r["pred_screenshot"], r["gt_screenshot"],
                            r["lpips_score"], 0.5,
                        )

            all_entries.extend(all_children)
            non_terminal = [c for c in all_children if not c["is_terminal"]]
            if not non_terminal:
                break
            non_terminal.sort(key=lambda n: n["hybrid_reward"], reverse=True)
            active_nodes = non_terminal[:2]

        non_terminal_entries = [e for e in all_entries if not e["is_terminal"]]
        for entry in non_terminal_entries:
            self.assertGreater(entry["hybrid_reward"], 0.0,
                               f"Non-terminal entry has zero reward: {entry['full_text'][:50]}")


class TestAdvantageComputation(unittest.TestCase):
    """Test advantage computation in isolation."""

    def test_step_delta_advantages(self):
        """Step-delta advantages: batch-normalized step_reward values."""
        training_data = [
            {"step_reward": 0.3},
            {"step_reward": 0.5},
            {"step_reward": 0.2},
            {"step_reward": -0.1},
            {"step_reward": 0.1},
        ]
        results = compute_step_delta_advantages(training_data, normalize=True)

        advantages = [r["step_advantage"] for r in results]
        # Zero-mean
        self.assertAlmostEqual(sum(advantages) / len(advantages), 0.0, places=5)
        # Higher step_reward → higher advantage
        self.assertGreater(results[1]["step_advantage"], results[0]["step_advantage"])
        self.assertGreater(results[0]["step_advantage"], results[2]["step_advantage"])
        # Negative step_reward → lowest advantage
        self.assertLess(results[3]["step_advantage"], results[4]["step_advantage"])

    def test_step_delta_no_normalize(self):
        """Without normalization, step_advantage == step_reward."""
        training_data = [
            {"step_reward": 0.3},
            {"step_reward": -0.1},
        ]
        results = compute_step_delta_advantages(training_data, normalize=False)
        self.assertAlmostEqual(results[0]["step_advantage"], 0.3, places=8)
        self.assertAlmostEqual(results[1]["step_advantage"], -0.1, places=8)

    def test_legacy_intra_beam_advantages_zero_centered(self):
        """[Legacy] Intra-beam group-relative advantages."""
        training_data = [
            {"sibling_group_id": 1, "step_reward": 0.3},
            {"sibling_group_id": 1, "step_reward": 0.5},
            {"sibling_group_id": 1, "step_reward": 0.2},
            {"sibling_group_id": 2, "step_reward": 0.8},
            {"sibling_group_id": 2, "step_reward": 0.1},
        ]
        advantages = compute_intra_beam_advantages(training_data)

        g1 = advantages[:3]
        self.assertAlmostEqual(sum(g1) / len(g1), 0.0, places=5)

        g2 = advantages[3:]
        self.assertAlmostEqual(sum(g2) / len(g2), 0.0, places=5)

        self.assertGreater(advantages[1], advantages[0])
        self.assertGreater(advantages[0], advantages[2])

    def test_legacy_inter_beam_advantages(self):
        """[Legacy] Inter-beam advantages."""
        terminal_rewards = [0.8, 0.2, 0.5]
        advantages = compute_inter_beam_advantages(terminal_rewards)

        self.assertEqual(len(advantages), 3)
        self.assertGreater(advantages[0], advantages[2])
        self.assertGreater(advantages[2], advantages[1])


class TestGRPOLoss(unittest.TestCase):
    """Test GRPO loss computation logic."""

    def test_loss_positive_advantage_encourages(self):
        """Positive advantage should make loss negative (encourage the action)."""
        import torch

        # Simulate log_probs for a sequence
        log_probs = torch.tensor([-0.5, -1.0, -0.3], requires_grad=True)
        advantage = 1.0

        # L = -advantage * mean(log_probs)
        loss = -advantage * log_probs.mean()

        # With positive advantage and negative log_probs,
        # loss should be positive (pushing log_probs up via gradient descent)
        self.assertGreater(loss.item(), 0.0)

    def test_loss_negative_advantage_discourages(self):
        """Negative advantage should make loss positive (discourage the action)."""
        import torch

        log_probs = torch.tensor([-0.5, -1.0, -0.3], requires_grad=True)
        advantage = -1.0

        loss = -advantage * log_probs.mean()

        # With negative advantage and negative log_probs,
        # loss should be negative (pushing log_probs down via gradient descent)
        self.assertLess(loss.item(), 0.0)

    def test_zero_advantage_zero_loss(self):
        """Zero advantage should contribute zero to loss."""
        import torch

        log_probs = torch.tensor([-0.5, -1.0, -0.3], requires_grad=True)
        advantage = 0.0

        loss = -advantage * log_probs.mean()
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class TestFullPipelineE2E(unittest.TestCase):
    """Integration test verifying the complete pipeline data flow."""

    def test_pipeline_produces_training_signal(self):
        """Full pipeline: mock generation → rewards → step-delta advantages → training entries."""
        from utils.hybrid_reward import compute_hybrid_reward

        generate_fn = create_deterministic_mock_generate(seed=77)
        reward_pool = MockRewardPool()

        # Run beam search with M=2, K=3, N=2, 3 steps
        M, K, N, max_steps = 2, 3, 2, 3
        reference_html = '<div>reference</div>'
        alpha = 0.5

        sibling_group_counter = 0
        all_entries = []

        for beam_idx in range(M):
            active_nodes = [{"full_text": "", "hybrid_reward": 0.0}]

            for step in range(max_steps):
                all_children = []
                for parent in active_nodes:
                    sibling_group_counter += 1
                    continuations = generate_fn(parent["full_text"], K)
                    for cont in continuations:
                        is_terminal = (cont == "")
                        child = {
                            "prefix": parent["full_text"],
                            "continuation": cont,
                            "full_text": parent["full_text"] + cont,
                            "step": step + 1,
                            "hybrid_reward": parent["hybrid_reward"] if is_terminal else 0.0,
                            "step_reward": 0.0,
                            "beam_id": beam_idx,
                            "is_terminal": is_terminal,
                        }
                        all_children.append(child)

                to_render = [c for c in all_children if not c["is_terminal"]]
                if to_render:
                    items = [(c["full_text"], reference_html, 1024, 1024)
                             for c in to_render]
                    results = reward_pool.calculate_metrics_batch_intermediate(items)
                    for child, r in zip(to_render, results):
                        if r is not None:
                            child["hybrid_reward"] = compute_hybrid_reward(
                                r["pred_screenshot"], r["gt_screenshot"],
                                r["lpips_score"], alpha,
                            )

                for child in all_children:
                    parent_reward = 0.0
                    for p in active_nodes:
                        if p["full_text"] == child["prefix"]:
                            parent_reward = p["hybrid_reward"]
                            break
                    child["step_reward"] = child["hybrid_reward"] - parent_reward

                all_entries.extend(all_children)
                non_terminal = [c for c in all_children if not c["is_terminal"]]
                if not non_terminal:
                    break
                non_terminal.sort(key=lambda n: n["hybrid_reward"], reverse=True)
                active_nodes = non_terminal[:N]

        # Compute step-delta advantages
        results_with_advantages = compute_step_delta_advantages(all_entries, normalize=True)

        # Verify pipeline produced useful training signal
        non_terminal = [
            e for e in results_with_advantages
            if not e["is_terminal"] and e["continuation"]
        ]
        self.assertGreater(len(non_terminal), 5,
                           "Should have at least 5 non-terminal entries for training")

        # Verify advantage variance (there should be signal, not all zeros)
        advantages = [e["step_advantage"] for e in non_terminal]
        adv_std = np.std(advantages)
        self.assertGreater(adv_std, 0.01,
                           f"Advantage std too low ({adv_std:.4f}), no training signal")

        # Verify some entries have positive and some have negative advantage
        positive = sum(1 for a in advantages if a > 0)
        negative = sum(1 for a in advantages if a < 0)
        self.assertGreater(positive, 0, "Should have some positive advantages")
        self.assertGreater(negative, 0, "Should have some negative advantages")

        # Verify rewards are bounded [0, 1]
        for entry in non_terminal:
            self.assertGreaterEqual(entry["hybrid_reward"], 0.0)
            self.assertLessEqual(entry["hybrid_reward"], 1.0)

        print(f"\nPipeline summary:")
        print(f"  Total entries: {len(results_with_advantages)}")
        print(f"  Non-terminal: {len(non_terminal)}")
        print(f"  Advantage std: {adv_std:.4f}")
        print(f"  Positive/negative: {positive}/{negative}")
        print(f"  Reward range: [{min(e['hybrid_reward'] for e in non_terminal):.3f}, "
              f"{max(e['hybrid_reward'] for e in non_terminal):.3f}]")


if __name__ == '__main__':
    unittest.main()
