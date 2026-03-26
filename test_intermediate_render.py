"""
Tests for intermediate rendering mode in similarity_parallel.py.

The intermediate mode returns LPIPS scores AND screenshot numpy arrays,
which are needed by the beam search module for EMD/color histogram
computation in the hybrid reward function.

Run: cd "/Users/tcz/Dropbox/Reverse Browser/V2" && python -m pytest training/utils/test_intermediate_render.py -v
"""

import os
import sys

import numpy as np
import pytest

# Ensure training/ is on path so `from utils.X` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VALIDATION_DATA_DIR = "data-rb-validate"
VIEWPORT = (1024, 1024)


# ---------------------------------------------------------------------------
# Pool fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pool():
    os.makedirs(VALIDATION_DATA_DIR, exist_ok=True)
    from utils.similarity_parallel import RewardPool
    p = RewardPool(num_workers=2, validation_data_dir=VALIDATION_DATA_DIR)
    yield p
    p.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_intermediate_render_returns_screenshots(pool):
    """Intermediate mode returns LPIPS score plus pred and gt screenshot arrays."""
    partial_html = '<div style="background:red;width:100px;height:100px;"></div>'
    reference_html = '<div style="background:red;width:100px;height:100px;position:absolute;top:50px;left:50px;"></div>'

    results = pool.calculate_metrics_batch_intermediate(
        items=[(partial_html, reference_html, 1024, 1024)],
    )
    assert len(results) == 1
    r = results[0]
    assert "lpips_score" in r
    assert "pred_screenshot" in r
    assert "gt_screenshot" in r
    assert isinstance(r["pred_screenshot"], np.ndarray)
    assert isinstance(r["gt_screenshot"], np.ndarray)
    assert r["pred_screenshot"].shape[2] == 3  # RGB
    assert r["gt_screenshot"].shape[2] == 3  # RGB
    assert r["pred_screenshot"].dtype == np.uint8
    assert r["gt_screenshot"].dtype == np.uint8


def test_batch_intermediate_render(pool):
    """Batch intermediate mode returns correct number of results with valid LPIPS."""
    items = [
        ('<div style="background:red;width:50px;height:50px;"></div>',
         '<div style="background:blue;width:100px;height:100px;"></div>', 1024, 1024),
        ('<div style="background:green;width:80px;height:80px;"></div>',
         '<div style="background:blue;width:100px;height:100px;"></div>', 1024, 1024),
    ]
    results = pool.calculate_metrics_batch_intermediate(items)
    assert len(results) == 2
    for r in results:
        assert 0.0 <= r["lpips_score"] <= 1.0
        assert isinstance(r["pred_screenshot"], np.ndarray)
        assert isinstance(r["gt_screenshot"], np.ndarray)


def test_intermediate_identical_markup_low_lpips(pool):
    """Identical markup should produce near-zero LPIPS in intermediate mode."""
    html = '<div style="background:red;width:200px;height:200px;margin:50px;"></div>'
    results = pool.calculate_metrics_batch_intermediate(
        items=[(html, html, 1024, 1024)],
    )
    r = results[0]
    assert r["lpips_score"] < 0.01, (
        f"Identical markup should have near-zero LPIPS, got {r['lpips_score']}"
    )


def test_intermediate_screenshot_dimensions(pool):
    """Screenshots should match the requested viewport dimensions."""
    vw, vh = 1024, 1024
    html = '<div style="background:blue;width:100px;height:100px;"></div>'
    results = pool.calculate_metrics_batch_intermediate(
        items=[(html, html, vw, vh)],
    )
    r = results[0]
    # Playwright screenshots match viewport
    assert r["pred_screenshot"].shape[0] == vh
    assert r["pred_screenshot"].shape[1] == vw
    assert r["gt_screenshot"].shape[0] == vh
    assert r["gt_screenshot"].shape[1] == vw


def test_intermediate_malformed_html(pool):
    """Partial/malformed HTML should still render and return results.

    During beam search, partial HTML (unclosed tags, incomplete attributes)
    is the norm. Browsers auto-close unclosed tags per HTML spec.
    """
    partial = '<div style="background:red;width:100px;height:100px">'  # unclosed
    reference = '<div style="background:red;width:100px;height:100px;"></div>'
    results = pool.calculate_metrics_batch_intermediate(
        items=[(partial, reference, 1024, 1024)],
    )
    assert results[0] is not None
    assert "lpips_score" in results[0]
    assert isinstance(results[0]["pred_screenshot"], np.ndarray)


def test_intermediate_does_not_break_standard_mode(pool):
    """Standard calculate_metrics_batch still works after intermediate calls."""
    # Run intermediate first
    pool.calculate_metrics_batch_intermediate(
        items=[('<div>test</div>', '<div>test</div>', 1024, 1024)],
    )
    # Standard mode should still work
    r = pool.calculate_metrics('<div>test</div>', '<div>test</div>', 1024, 1024)
    assert r is not None
    assert "perceptual_loss" in r
    assert "similarity" in r
