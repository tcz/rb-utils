"""
Tests for hybrid_reward.py — hybrid LPIPS + EMD visual reward.

Run: python -m pytest training/utils/test_hybrid_reward.py -v
"""

import os
import sys

import numpy as np
import pytest

# Ensure training/ is on path so `from utils.X` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.hybrid_reward import compute_color_histogram, compute_emd, compute_hybrid_reward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_solid_image(color, height=64, width=64):
    """Create a solid-color (H, W, 3) uint8 image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img


# ---------------------------------------------------------------------------
# EMD tests
# ---------------------------------------------------------------------------

class TestEMD:

    def test_identical_images_emd_zero(self):
        """EMD should be 0.0 when both images are identical."""
        img = make_solid_image((128, 64, 200))
        emd = compute_emd(img, img)
        assert emd == pytest.approx(0.0, abs=1e-10)

    def test_different_images_emd_positive(self):
        """EMD should be > 0 for black vs white images."""
        black = make_solid_image((0, 0, 0))
        white = make_solid_image((255, 255, 255))
        emd = compute_emd(black, white)
        assert emd > 0.0

    def test_similar_images_lower_emd(self):
        """Adding some white pixels to a black image should yield lower EMD
        vs all-white than pure black vs all-white."""
        black = make_solid_image((0, 0, 0))
        white = make_solid_image((255, 255, 255))

        # Partially white: top half white, bottom half black
        partial = black.copy()
        partial[: partial.shape[0] // 2, :] = (255, 255, 255)

        emd_full = compute_emd(black, white)
        emd_partial = compute_emd(partial, white)

        assert emd_partial < emd_full, (
            f"Partial white->white EMD ({emd_partial:.4f}) should be less than "
            f"black->white EMD ({emd_full:.4f})"
        )


# ---------------------------------------------------------------------------
# Histogram tests
# ---------------------------------------------------------------------------

class TestColorHistogram:

    def test_color_histogram_shape(self):
        """Histogram should have bins^3 entries and sum to 1.0."""
        img = make_solid_image((100, 150, 200))
        for bins in [4, 8, 16]:
            hist = compute_color_histogram(img, bins=bins)
            assert hist.shape == (bins ** 3,), (
                f"Expected shape ({bins**3},), got {hist.shape}"
            )
            assert hist.sum() == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Hybrid reward tests
# ---------------------------------------------------------------------------

class TestHybridReward:

    def test_hybrid_reward_range(self):
        """Hybrid reward should always be in [0, 1]."""
        img1 = make_solid_image((50, 100, 150))
        img2 = make_solid_image((200, 50, 100))

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for lpips_score in [0.0, 0.3, 0.7, 1.0]:
                reward = compute_hybrid_reward(img1, img2, lpips_score, alpha=alpha)
                assert 0.0 <= reward <= 1.0, (
                    f"Reward {reward} out of [0,1] for alpha={alpha}, "
                    f"lpips={lpips_score}"
                )

    def test_hybrid_reward_alpha_1(self):
        """alpha=1.0 should return pure LPIPS similarity (1 - lpips_score)."""
        img1 = make_solid_image((0, 0, 0))
        img2 = make_solid_image((255, 255, 255))

        for lpips_score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            reward = compute_hybrid_reward(img1, img2, lpips_score, alpha=1.0)
            expected = 1.0 - lpips_score
            assert reward == pytest.approx(expected, abs=1e-10), (
                f"alpha=1.0: expected {expected}, got {reward}"
            )

    def test_hybrid_reward_alpha_0(self):
        """alpha=0.0 with identical images should return 1.0 (pure EMD similarity,
        EMD=0 for identical images => sim_color=1.0)."""
        img = make_solid_image((100, 200, 50))
        reward = compute_hybrid_reward(img, img, lpips_score=0.5, alpha=0.0)
        assert reward == pytest.approx(1.0, abs=1e-10), (
            f"alpha=0, identical images: expected 1.0, got {reward}"
        )
