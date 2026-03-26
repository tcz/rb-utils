"""
Tests for per-token GRPO advantage computation and loss.

These test the core primitives for token-level credit assignment in GRPO:
- compute_per_token_advantages: group-relative normalization at token level
- per_token_grpo_loss: PPO-clip loss with (B,T) advantages
- PerTokenGRPOTrainer: TRL GRPOTrainer subclass with per-token advantage support
"""
import pytest
import torch
import math


# ── TestPerTokenAdvantages ──────────────────────────────────────────────


class TestPerTokenAdvantages:
    """Test compute_per_token_advantages: group-relative normalization at the token level."""

    def test_basic_normalization(self):
        """Two completions: one with high rewards, one with low.
        High-reward tokens should get positive advantages, low-reward negative."""
        from utils.per_token_grpo import compute_per_token_advantages

        # Completion 0: high rewards (0.9 each)
        # Completion 1: low rewards (0.1 each)
        token_rewards = [
            [0.9, 0.9, 0.9],
            [0.1, 0.1, 0.1],
        ]
        completion_lengths = [3, 3]
        max_len = 3

        advantages = compute_per_token_advantages(token_rewards, completion_lengths, max_len)

        assert advantages.shape == (2, 3)
        # All of completion 0 should be positive
        assert (advantages[0] > 0).all(), f"Expected all positive, got {advantages[0]}"
        # All of completion 1 should be negative
        assert (advantages[1] < 0).all(), f"Expected all negative, got {advantages[1]}"
        # Mean should be ~0 (group-relative)
        assert abs(advantages.mean().item()) < 1e-5

    def test_variable_length_completions(self):
        """Padding tokens must get advantage = 0."""
        from utils.per_token_grpo import compute_per_token_advantages

        token_rewards = [
            [0.8, 0.7, 0.9],       # length 3
            [0.2, 0.3],             # length 2 (will be padded to 4)
        ]
        completion_lengths = [3, 2]
        max_len = 4

        advantages = compute_per_token_advantages(token_rewards, completion_lengths, max_len)

        assert advantages.shape == (2, 4)
        # Padding positions should be exactly 0
        assert advantages[0, 3] == 0.0, "Padding at [0,3] should be 0"
        assert advantages[1, 2] == 0.0, "Padding at [1,2] should be 0"
        assert advantages[1, 3] == 0.0, "Padding at [1,3] should be 0"
        # Non-padding positions should be non-zero (rewards differ enough)
        assert advantages[0, 0] != 0.0
        assert advantages[1, 0] != 0.0

    def test_within_completion_variance(self):
        """Different tokens within the same completion should get different advantages
        when they have different rewards."""
        from utils.per_token_grpo import compute_per_token_advantages

        # Completion 0: token 0 renders well (0.9), token 1 renders badly (0.1)
        # Completion 1: uniform medium (0.5, 0.5)
        token_rewards = [
            [0.9, 0.1],
            [0.5, 0.5],
        ]
        completion_lengths = [2, 2]
        max_len = 2

        advantages = compute_per_token_advantages(token_rewards, completion_lengths, max_len)

        assert advantages.shape == (2, 2)
        # Within completion 0, token 0 should have higher advantage than token 1
        assert advantages[0, 0] > advantages[0, 1], (
            f"Token 0 (reward=0.9) should have higher advantage than token 1 (reward=0.1), "
            f"got {advantages[0, 0]:.4f} vs {advantages[0, 1]:.4f}"
        )

    def test_all_same_rewards_zero_advantage(self):
        """When all rewards are identical, std ~ 0, so all advantages should be 0."""
        from utils.per_token_grpo import compute_per_token_advantages

        token_rewards = [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]
        completion_lengths = [3, 3]
        max_len = 3

        advantages = compute_per_token_advantages(token_rewards, completion_lengths, max_len)

        assert advantages.shape == (2, 3)
        # All should be 0 (no variance)
        assert (advantages.abs() < 1e-6).all(), f"Expected all ~0, got {advantages}"

    def test_six_completions_group(self):
        """Standard GRPO group of 6 completions with varying quality."""
        from utils.per_token_grpo import compute_per_token_advantages

        # 6 completions of varying length and quality
        token_rewards = [
            [0.95, 0.90, 0.85, 0.92],       # good
            [0.10, 0.15, 0.20],              # bad
            [0.50, 0.55, 0.60, 0.45, 0.50],  # medium
            [0.80, 0.75],                     # decent
            [0.30, 0.25, 0.35],              # poor
            [0.70, 0.65, 0.60, 0.55],        # ok
        ]
        completion_lengths = [4, 3, 5, 2, 3, 4]
        max_len = 5

        advantages = compute_per_token_advantages(token_rewards, completion_lengths, max_len)

        assert advantages.shape == (6, 5)

        # Check padding is zero
        assert advantages[0, 4] == 0.0  # comp 0 has length 4
        assert advantages[1, 3] == 0.0  # comp 1 has length 3
        assert advantages[1, 4] == 0.0
        assert advantages[3, 2] == 0.0  # comp 3 has length 2
        assert advantages[3, 3] == 0.0
        assert advantages[3, 4] == 0.0
        assert advantages[4, 3] == 0.0  # comp 4 has length 3
        assert advantages[4, 4] == 0.0

        # Mean of non-padded advantages should be ~0
        mask = torch.zeros(6, 5, dtype=torch.bool)
        for i, l in enumerate(completion_lengths):
            mask[i, :l] = True
        non_padded = advantages[mask]
        assert abs(non_padded.mean().item()) < 1e-5, (
            f"Mean of non-padded advantages should be ~0, got {non_padded.mean().item()}"
        )

        # Best completion (0) should have mostly positive advantages
        assert (advantages[0, :4] > 0).sum() >= 3, "Best completion should be mostly positive"
        # Worst completion (1) should have mostly negative advantages
        assert (advantages[1, :3] < 0).sum() >= 2, "Worst completion should be mostly negative"


# ── TestPerTokenLoss ────────────────────────────────────────────────────


class TestPerTokenLoss:
    """Test per_token_grpo_loss: PPO-clip with (B,T) advantages."""

    def test_per_token_loss_shape(self):
        """Loss should be a scalar, not NaN or inf."""
        from utils.per_token_grpo import per_token_grpo_loss

        B, T = 4, 10
        per_token_logps = torch.randn(B, T)
        old_per_token_logps = per_token_logps.detach() + torch.randn(B, T) * 0.1
        advantages = torch.randn(B, T)
        completion_mask = torch.ones(B, T)
        # Mask out last 2 tokens of each sequence
        completion_mask[:, -2:] = 0.0

        loss = per_token_grpo_loss(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            completion_mask=completion_mask,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )

        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be inf"

    def test_positive_advantages_reduce_loss(self):
        """When advantages are all positive and the model increases probability
        of those tokens (ratio > 1), the loss should be more negative (better)
        than when ratio = 1."""
        from utils.per_token_grpo import per_token_grpo_loss

        B, T = 2, 5
        completion_mask = torch.ones(B, T)

        # Case 1: ratio = 1 (old and new logps identical)
        old_logps = torch.zeros(B, T)
        new_logps_same = torch.zeros(B, T)
        advantages = torch.ones(B, T)  # all positive

        loss_same = per_token_grpo_loss(
            per_token_logps=new_logps_same,
            old_per_token_logps=old_logps,
            advantages=advantages,
            completion_mask=completion_mask,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )

        # Case 2: model increases probability (logps go up -> ratio > 1)
        new_logps_better = old_logps + 0.1  # slightly higher log-prob
        loss_better = per_token_grpo_loss(
            per_token_logps=new_logps_better,
            old_per_token_logps=old_logps,
            advantages=advantages,
            completion_mask=completion_mask,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )

        # With positive advantages, increasing probability should make loss more negative
        # (because loss = -min(ratio*adv, clip(ratio)*adv) and ratio > 1 with positive adv)
        assert loss_better < loss_same, (
            f"Increasing prob with positive advantages should reduce loss, "
            f"got loss_better={loss_better:.6f} vs loss_same={loss_same:.6f}"
        )

    def test_masked_tokens_dont_contribute(self):
        """Tokens with mask=0 should not affect the loss."""
        from utils.per_token_grpo import per_token_grpo_loss

        B, T = 2, 5
        old_logps = torch.zeros(B, T)
        new_logps = torch.zeros(B, T)
        advantages = torch.ones(B, T)

        # Full mask
        full_mask = torch.ones(B, T)
        loss_full = per_token_grpo_loss(
            per_token_logps=new_logps,
            old_per_token_logps=old_logps,
            advantages=advantages,
            completion_mask=full_mask,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )

        # Partial mask: mask out last 3 tokens, but set huge advantages there
        partial_mask = torch.ones(B, T)
        partial_mask[:, 2:] = 0.0
        advantages_with_noise = advantages.clone()
        advantages_with_noise[:, 2:] = 1000.0  # Should not affect loss

        loss_partial = per_token_grpo_loss(
            per_token_logps=new_logps,
            old_per_token_logps=old_logps,
            advantages=advantages_with_noise,
            completion_mask=partial_mask,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )

        # Both should give the same loss for the unmasked tokens
        # (ratio=1, adv=1 for both, but different number of tokens)
        # The loss value per token is -min(1*1, 1*1) = -1 for unmasked tokens
        # full: mean over 5 tokens then mean over batch
        # partial: mean over 2 tokens then mean over batch
        # Both should be -1.0 since per-token loss is the same
        assert abs(loss_full.item() - loss_partial.item()) < 1e-5, (
            f"Masked tokens should not change per-token loss value, "
            f"got full={loss_full:.6f} vs partial={loss_partial:.6f}"
        )


# ── TestPerTokenGRPOTrainerOverrides ────────────────────────────────────


class TestPerTokenGRPOTrainerOverrides:
    """Test that PerTokenGRPOTrainer properly overrides TRL's _compute_loss."""

    def test_compute_loss_uses_per_token_advantages(self):
        """Verify the class has the _compute_loss override."""
        from utils.per_token_grpo import PerTokenGRPOTrainer
        from trl.trainer.grpo_trainer import GRPOTrainer

        # PerTokenGRPOTrainer should be a subclass of GRPOTrainer
        assert issubclass(PerTokenGRPOTrainer, GRPOTrainer)

        # It should override _compute_loss
        assert PerTokenGRPOTrainer._compute_loss is not GRPOTrainer._compute_loss, (
            "PerTokenGRPOTrainer must override _compute_loss"
        )

    def test_loss_with_2d_advantages(self):
        """per_token_grpo_loss should work correctly with 2D advantages (B, T)
        instead of the scalar advantages that TRL's default uses."""
        from utils.per_token_grpo import per_token_grpo_loss

        B, T = 3, 8
        # Simulate different advantages per token
        advantages = torch.randn(B, T)
        per_token_logps = torch.randn(B, T, requires_grad=True)
        old_per_token_logps = per_token_logps.detach()
        completion_mask = torch.ones(B, T)

        loss = per_token_grpo_loss(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            completion_mask=completion_mask,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )

        # Should be differentiable
        loss.backward()
        assert per_token_logps.grad is not None, "Loss should be differentiable w.r.t. logps"
        assert per_token_logps.grad.shape == (B, T)

    def test_advantage_variance_affects_gradient(self):
        """With per-token advantages, different tokens should get different gradients.
        This is the key difference from scalar advantages where all tokens in a
        completion get the same gradient direction."""
        from utils.per_token_grpo import per_token_grpo_loss

        B, T = 1, 4
        completion_mask = torch.ones(B, T)

        # Advantages: first 2 tokens positive, last 2 negative
        advantages = torch.tensor([[1.0, 1.0, -1.0, -1.0]])

        per_token_logps = torch.zeros(B, T, requires_grad=True)
        old_per_token_logps = torch.zeros(B, T)

        loss = per_token_grpo_loss(
            per_token_logps=per_token_logps,
            old_per_token_logps=old_per_token_logps,
            advantages=advantages,
            completion_mask=completion_mask,
            epsilon_low=0.2,
            epsilon_high=0.2,
        )
        loss.backward()

        grad = per_token_logps.grad[0]
        # Positive advantage tokens: gradient should push to increase logp (negative grad for loss min)
        # Negative advantage tokens: gradient should push to decrease logp
        # The sign of gradients should differ between positive and negative advantage tokens
        assert grad[0] * grad[2] < 0, (
            f"Gradients for positive and negative advantage tokens should have opposite signs, "
            f"got grad[0]={grad[0]:.4f}, grad[2]={grad[2]:.4f}"
        )


# ── TestElementMappedRewardComputation ─────────────────────────────────


class TestElementMappedRewardComputation:

    def test_compute_char_rewards_from_elements(self):
        """Given element scores and alignment, compute per-character rewards."""
        from utils.per_token_grpo import compute_char_rewards

        model_output = '<div class="a">text</div><div class="b">more</div>'
        overall_loss = 0.3
        alpha = 0.5

        element_losses = [
            (0, 24, 0.1, 10000),   # first div: low loss (good)
            (24, 49, 0.5, 10000),  # second div: high loss (bad)
        ]
        css_mappings = []

        char_rewards = compute_char_rewards(
            model_output, overall_loss, alpha, element_losses, css_mappings)

        assert len(char_rewards) == len(model_output)
        avg_first = sum(char_rewards[0:24]) / 24
        avg_second = sum(char_rewards[24:49]) / 25
        assert avg_first > avg_second

    def test_unmapped_chars_get_overall_only(self):
        """Characters not mapped to any element get reward based on overall loss only."""
        from utils.per_token_grpo import compute_char_rewards

        model_output = 'abc'
        overall_loss = 0.4
        alpha = 0.5
        element_losses = []  # no elements mapped
        css_mappings = []

        char_rewards = compute_char_rewards(
            model_output, overall_loss, alpha, element_losses, css_mappings)

        expected = 1.0 - 0.5 * 0.4  # = 0.8
        for r in char_rewards:
            assert abs(r - expected) < 1e-6

    def test_css_mappings_apply_element_loss(self):
        """CSS character ranges should get the corresponding element's LPIPS."""
        from utils.per_token_grpo import compute_char_rewards

        model_output = '.red { color: red; }'  # 20 chars
        overall_loss = 0.2
        alpha = 0.5
        element_losses = [
            (0, 0, 0.3, 5000),  # element 0, not mapped via HTML
        ]
        css_mappings = [
            (0, 20, 0),  # chars 0-20 are CSS for element 0
        ]

        char_rewards = compute_char_rewards(
            model_output, overall_loss, alpha, element_losses, css_mappings)

        # All chars should have element loss of 0.3
        expected = 1.0 - (0.5 * 0.2 + 0.3)  # = 0.6
        for r in char_rewards:
            assert abs(r - expected) < 1e-6, f"Expected {expected}, got {r}"

    def test_char_to_token_rewards(self):
        """Char rewards aggregate correctly to token rewards."""
        from utils.per_token_grpo import char_rewards_to_token_rewards

        char_rewards = [0.8, 0.8, 0.2, 0.2, 0.5, 0.5]
        offsets = [(0, 2), (2, 4), (4, 6)]

        token_rewards = char_rewards_to_token_rewards(char_rewards, offsets)

        assert len(token_rewards) == 3
        assert abs(token_rewards[0] - 0.8) < 1e-6
        assert abs(token_rewards[1] - 0.2) < 1e-6
        assert abs(token_rewards[2] - 0.5) < 1e-6

    def test_char_to_token_empty_span(self):
        """Empty span tokens get global average reward."""
        from utils.per_token_grpo import char_rewards_to_token_rewards

        char_rewards = [0.6, 0.4]
        offsets = [(0, 0), (0, 1), (1, 2)]  # first is empty span

        token_rewards = char_rewards_to_token_rewards(char_rewards, offsets)

        assert len(token_rewards) == 3
        assert abs(token_rewards[0] - 0.5) < 1e-6  # global avg of [0.6, 0.4]
        assert abs(token_rewards[1] - 0.6) < 1e-6
        assert abs(token_rewards[2] - 0.4) < 1e-6

    def test_nested_elements_innermost_wins(self):
        """Parent element should NOT dilute child's LPIPS for HTML ranges.

        For chars covered by both parent and child, only the child's (innermost)
        LPIPS should apply. Parent's LPIPS applies only to its own markup
        (opening/closing tags, whitespace between children).
        """
        from utils.per_token_grpo import compute_char_rewards

        # Parent spans 0-50, child spans 20-40
        # Parent opening tag: 0-20, child: 20-40, parent closing: 40-50
        model_output = '<div class="parent"><div class="child">x</div></div>'
        overall_loss = 0.2
        alpha = 0.5

        parent_loss = 0.05  # parent looks fine
        child_loss = 0.8    # child looks terrible
        element_losses = [
            (0, len(model_output), parent_loss, 20000),  # parent: full range
            (20, 41, child_loss, 5000),                    # child: inner range
        ]
        css_mappings = []

        char_rewards = compute_char_rewards(
            model_output, overall_loss, alpha, element_losses, css_mappings)

        # Child chars (20-41) should get child_loss, NOT blended with parent
        child_expected = 1.0 - (alpha * overall_loss + child_loss)
        for c in range(20, 41):
            assert abs(char_rewards[c] - child_expected) < 1e-6, \
                f"Char {c}: expected {child_expected}, got {char_rewards[c]}"

        # Parent chars (0-20 and 41-end) should get parent_loss
        parent_expected = 1.0 - (alpha * overall_loss + parent_loss)
        for c in list(range(0, 20)) + list(range(41, len(model_output))):
            assert abs(char_rewards[c] - parent_expected) < 1e-6, \
                f"Char {c}: expected {parent_expected}, got {char_rewards[c]}"

    def test_css_still_area_weighted_with_html(self):
        """CSS mappings should still area-weight-average with the HTML element entry."""
        from utils.per_token_grpo import compute_char_rewards

        model_output = '.box { color: red; }'  # 20 chars
        overall_loss = 0.2
        alpha = 0.5

        # One element mapped via both HTML range and CSS
        element_losses = [
            (0, 20, 0.3, 5000),   # element 0 via HTML
        ]
        css_mappings = [
            (0, 20, 0),  # same element via CSS
        ]

        char_rewards = compute_char_rewards(
            model_output, overall_loss, alpha, element_losses, css_mappings)

        # Both sources point to same element (same loss/area), so result is 0.3
        expected = 1.0 - (alpha * overall_loss + 0.3)
        for r in char_rewards:
            assert abs(r - expected) < 1e-6


# ── TestGenerateAndScoreOverride ────────────────────────────────────


class TestGenerateAndScoreOverride:
    """Tests for ElementMappedGRPOTrainer._generate_and_score_completions override."""

    def test_element_mapped_trainer_has_override(self):
        """ElementMappedGRPOTrainer overrides _generate_and_score_completions."""
        from utils.per_token_grpo import ElementMappedGRPOTrainer
        assert '_generate_and_score_completions' in ElementMappedGRPOTrainer.__dict__

    def test_end_to_end_token_reward_pipeline(self):
        """Test the full pipeline: element extraction -> alignment -> char rewards -> token rewards."""
        from utils.per_token_grpo import compute_char_rewards, char_rewards_to_token_rewards, compute_per_token_advantages

        # Simulate two completions for the same prompt
        model_outputs = [
            '<div style="background:red;width:200px;height:200px;"></div>',
            '<div style="background:blue;width:200px;height:200px;"></div>',
        ]

        # Simulate element losses (from element extraction)
        element_losses_per_completion = [
            [(0, len(model_outputs[0]), 0.1, 40000)],  # good rendering
            [(0, len(model_outputs[1]), 0.5, 40000)],  # bad rendering
        ]
        overall_losses = [0.1, 0.5]
        alpha = 0.5

        # Compute per-token rewards for each completion
        all_token_rewards = []
        for i, text in enumerate(model_outputs):
            char_rewards = compute_char_rewards(
                text, overall_losses[i], alpha,
                element_losses_per_completion[i], [])
            # Simple tokenization: one char per token
            offsets = [(j, j+1) for j in range(len(text))]
            token_rewards = char_rewards_to_token_rewards(char_rewards, offsets)
            all_token_rewards.append(token_rewards)

        # Compute per-token advantages
        lengths = [len(tr) for tr in all_token_rewards]
        max_len = max(lengths)
        advantages = compute_per_token_advantages(all_token_rewards, lengths, max_len)

        assert advantages.shape[0] == 2
        assert advantages.shape[1] == max_len
        # Good completion should have positive mean advantage
        assert advantages[0, :lengths[0]].mean() > 0
        # Bad completion should have negative mean advantage
        assert advantages[1, :lengths[1]].mean() < 0

    def test_override_replaces_advantages_shape(self):
        """The override must produce (B, T) advantages, not (B,) scalar advantages.

        We test this by verifying the method's logic: given completion_ids of shape
        (B, T), the resulting advantages should also be (B, T).
        """
        from utils.per_token_grpo import compute_per_token_advantages

        # 6 completions (typical GRPO group), varying length
        all_token_rewards = [
            [0.9] * 100,  # good: 100 tokens
            [0.1] * 80,   # bad: 80 tokens
            [0.5] * 90,   # medium: 90 tokens
            [0.8] * 110,  # decent: 110 tokens
            [0.3] * 95,   # poor: 95 tokens
            [0.6] * 85,   # ok: 85 tokens
        ]
        lengths = [len(tr) for tr in all_token_rewards]
        max_len = max(lengths)

        advantages = compute_per_token_advantages(all_token_rewards, lengths, max_len)

        # Shape should be (6, max_len) = (6, 110), NOT (6,)
        assert advantages.shape == (6, max_len)
        assert advantages.dim() == 2, "Advantages must be 2D (B, T)"

        # Padding must be zero
        for i, length in enumerate(lengths):
            if length < max_len:
                assert (advantages[i, length:] == 0).all(), (
                    f"Padding positions must be zero for completion {i}"
                )

    def test_fallback_to_scalar_on_none_result(self):
        """If token_level results are None, fallback uses uniform advantage."""
        from utils.per_token_grpo import compute_per_token_advantages

        # Simulate: token rewards derived from scalar fallback (uniform per completion)
        scalar_reward_good = 0.8
        scalar_reward_bad = 0.2
        len_good, len_bad = 50, 60

        all_token_rewards = [
            [scalar_reward_good] * len_good,
            [scalar_reward_bad] * len_bad,
        ]
        lengths = [len_good, len_bad]
        max_len = max(lengths)

        advantages = compute_per_token_advantages(all_token_rewards, lengths, max_len)

        # Within each completion, all non-padded advantages should be identical
        # (because all token rewards are the same within each completion)
        adv_good = advantages[0, :len_good]
        adv_bad = advantages[1, :len_bad]
        assert torch.allclose(adv_good, adv_good[0].expand_as(adv_good)), (
            "Uniform token rewards should produce uniform advantages"
        )
        assert torch.allclose(adv_bad, adv_bad[0].expand_as(adv_bad)), (
            "Uniform token rewards should produce uniform advantages"
        )
        # But good should be positive, bad negative
        assert adv_good[0] > 0
        assert adv_bad[0] < 0

    def test_viewport_dimensions_stored(self):
        """ElementMappedGRPOTrainer stores viewport_width and viewport_height."""
        from utils.per_token_grpo import ElementMappedGRPOTrainer
        # Cannot instantiate without full GRPOTrainer setup, but verify the class
        # accepts viewport params by checking __init__ signature
        import inspect
        sig = inspect.signature(ElementMappedGRPOTrainer.__init__)
        params = list(sig.parameters.keys())
        assert 'viewport_width' in params, "Should accept viewport_width"
        assert 'viewport_height' in params, "Should accept viewport_height"
        assert 'alpha' in params, "Should accept alpha"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
