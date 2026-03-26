"""
Hybrid visual reward combining LPIPS (perceptual similarity) and EMD
(Earth Mover's Distance on color histograms).

LPIPS captures spatial/structural similarity but gives poor signal for early
generation steps where elements are present but mispositioned. EMD on color
histograms gives "partial credit" for having the right visual content
regardless of position.

Usage:
    from utils.hybrid_reward import compute_hybrid_reward

    reward = compute_hybrid_reward(
        rendered_img=rendered_array,    # (H, W, 3) uint8
        reference_img=reference_array,  # (H, W, 3) uint8
        lpips_score=0.35,               # precomputed LPIPS
        alpha=0.5,                      # weight: 1.0=pure LPIPS, 0.0=pure EMD
    )
    # reward in [0, 1], higher = better
"""

import numpy as np
from scipy.stats import wasserstein_distance


def compute_color_histogram(img: np.ndarray, bins: int = 8) -> np.ndarray:
    """Compute a normalized 3D color histogram from an RGB image.

    Args:
        img: RGB image as (H, W, 3) uint8 numpy array.
        bins: Number of bins per channel. Total bins = bins^3.

    Returns:
        Flattened, normalized histogram with bins^3 entries summing to 1.0.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got shape {img.shape}")

    # np.histogramdd expects (N, 3) array of pixel values
    pixels = img.reshape(-1, 3).astype(np.float64)

    hist, _ = np.histogramdd(
        pixels,
        bins=bins,
        range=[(0, 256), (0, 256), (0, 256)],
    )

    # Normalize so histogram sums to 1.0
    total = hist.sum()
    if total > 0:
        hist = hist / total

    return hist.ravel()


def compute_emd(img1: np.ndarray, img2: np.ndarray, bins: int = 8) -> float:
    """Earth Mover's Distance between color histograms of two images.

    Uses scipy.stats.wasserstein_distance with bin indices as positions,
    operating on flattened 3D color histograms.

    Args:
        img1: First RGB image as (H, W, 3) uint8.
        img2: Second RGB image as (H, W, 3) uint8.
        bins: Number of bins per channel.

    Returns:
        EMD value. 0.0 for identical images, positive for different images.
    """
    hist1 = compute_color_histogram(img1, bins=bins)
    hist2 = compute_color_histogram(img2, bins=bins)

    n_bins = bins ** 3
    positions = np.arange(n_bins, dtype=np.float64)

    return float(wasserstein_distance(positions, positions, hist1, hist2))


def compute_hybrid_reward(
    rendered_img: np.ndarray,
    reference_img: np.ndarray,
    lpips_score: float,
    alpha: float = 0.5,
) -> float:
    """Combine LPIPS and EMD into a single reward in [0, 1].

    Args:
        rendered_img: Rendered (predicted) image as (H, W, 3) uint8.
        reference_img: Ground truth image as (H, W, 3) uint8.
        lpips_score: Precomputed LPIPS score (0 = identical, higher = worse).
        alpha: Blending weight. 1.0 = pure LPIPS similarity, 0.0 = pure EMD
               similarity.

    Returns:
        Reward in [0, 1], higher = better visual fidelity.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # Perceptual similarity from LPIPS
    sim_perceptual = 1.0 - float(lpips_score)

    # Color distribution similarity from EMD
    bins = 8
    emd_raw = compute_emd(rendered_img, reference_img, bins=bins)
    max_emd = bins ** 3 - 1  # maximum possible EMD for bin indices [0, N-1]
    emd_normalized = min(emd_raw / max_emd, 1.0)
    sim_color = 1.0 - emd_normalized

    reward = alpha * sim_perceptual + (1.0 - alpha) * sim_color

    # Clamp to [0, 1] for safety (LPIPS can occasionally exceed 1.0)
    return float(max(0.0, min(1.0, reward)))
