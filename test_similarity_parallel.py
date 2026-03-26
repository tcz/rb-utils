"""
Tests for similarity_parallel.py — parallel reward pipeline.

Run: python -m pytest training/utils/test_similarity_parallel.py -v
"""

import os
import sys
import time
import pytest

# Ensure training/ is on path so `from utils.X` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VALIDATION_DATA_DIR = "data-rb-validate"
VIEWPORT = (1024, 1024)

# ---------------------------------------------------------------------------
# Fixtures (inline HTML)
# ---------------------------------------------------------------------------

MARKUP_A = """<!DOCTYPE html>
<html><head><style>
  body { margin: 0; background: #3498db; }
  .box { width: 200px; height: 200px; background: #e74c3c; margin: 50px auto; }
</style></head><body><div class="box"></div></body></html>"""

MARKUP_B = """<!DOCTYPE html>
<html><head><style>
  body { margin: 0; background: #2ecc71; }
  .box { width: 300px; height: 100px; background: #f39c12; margin: 100px auto; border-radius: 20px; }
</style></head><body><div class="box"></div></body></html>"""

MARKUP_C = """<!DOCTYPE html>
<html><head><style>
  body { margin: 0; background: #000; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; padding: 10px; }
  .cell { height: 100px; }
  .c1 { background: red; } .c2 { background: green; } .c3 { background: blue; }
  .c4 { background: yellow; } .c5 { background: purple; } .c6 { background: cyan; }
</style></head><body><div class="grid">
  <div class="cell c1"></div><div class="cell c2"></div><div class="cell c3"></div>
  <div class="cell c4"></div><div class="cell c5"></div><div class="cell c6"></div>
</div></body></html>"""

MARKUP_MALFORMED = "<html><body><div style='color: invalid'>broken</div>"

MARKUP_EMPTY = ""

MARKUP_MINIMAL = "<html><body>test</body></html>"

MARKUP_TOKEN_LEVEL = """<!DOCTYPE html>
<html><head><style>
  body { margin: 0; background: white; }
  .container { display: flex; gap: 10px; padding: 20px; }
  .box { width: 200px; height: 200px; }
  .red { background: red; }
  .blue { background: blue; }
</style></head><body>
<div class="container">
  <div class="box red"></div>
  <div class="box blue"></div>
</div>
</body></html>"""

MARKUP_TOKEN_LEVEL_WRONG_COLOR = """<!DOCTYPE html>
<html><head><style>
  body { margin: 0; background: white; }
  .container { display: flex; gap: 10px; padding: 20px; }
  .box { width: 200px; height: 200px; }
  .red { background: green; }
  .blue { background: blue; }
</style></head><body>
<div class="container">
  <div class="box red"></div>
  <div class="box blue"></div>
</div>
</body></html>"""


# ---------------------------------------------------------------------------
# Pool fixture (shared across tests in module, torn down at end)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pool():
    os.makedirs(VALIDATION_DATA_DIR, exist_ok=True)
    from utils.similarity_parallel import RewardPool
    p = RewardPool(num_workers=4, validation_data_dir=VALIDATION_DATA_DIR)
    yield p
    p.shutdown()


# ---------------------------------------------------------------------------
# 1. Correctness — metrics match sequential baseline
# ---------------------------------------------------------------------------

class TestCorrectnessVsSequential:
    """Verify parallel pipeline produces same results as the original sequential one."""

    @pytest.fixture(scope="class")
    def sequential_results(self):
        """Compute reference results using the original calculate_metrics."""
        from utils.similarity import calculate_metrics
        pairs = [
            (MARKUP_A, MARKUP_A),   # identical
            (MARKUP_A, MARKUP_B),   # different
            (MARKUP_C, MARKUP_C),   # identical complex
        ]
        results = []
        for pred, exp in pairs:
            r = calculate_metrics(pred, exp, *VIEWPORT)
            results.append(r)
        return results

    def test_identical_markup(self, pool, sequential_results):
        r = pool.calculate_metrics(MARKUP_A, MARKUP_A, *VIEWPORT)
        ref = sequential_results[0]
        assert r is not None
        assert abs(r['perceptual_loss'] - ref['perceptual_loss']) < 1e-4
        assert abs(r['similarity'] - ref['similarity']) < 1e-4

    def test_different_markup(self, pool, sequential_results):
        r = pool.calculate_metrics(MARKUP_A, MARKUP_B, *VIEWPORT)
        ref = sequential_results[1]
        assert r is not None
        assert abs(r['perceptual_loss'] - ref['perceptual_loss']) < 1e-4
        assert abs(r['similarity'] - ref['similarity']) < 1e-4

    def test_complex_identical(self, pool, sequential_results):
        r = pool.calculate_metrics(MARKUP_C, MARKUP_C, *VIEWPORT)
        ref = sequential_results[2]
        assert r is not None
        assert abs(r['perceptual_loss'] - ref['perceptual_loss']) < 1e-4
        assert abs(r['similarity'] - ref['similarity']) < 1e-4


# ---------------------------------------------------------------------------
# 2. Batch correctness — order preserved, all results returned
# ---------------------------------------------------------------------------

class TestBatchCorrectness:

    def test_batch_order_and_completeness(self, pool):
        items = [
            (MARKUP_A, MARKUP_A, *VIEWPORT),  # identical → low loss
            (MARKUP_A, MARKUP_B, *VIEWPORT),  # different → higher loss
            (MARKUP_B, MARKUP_C, *VIEWPORT),  # very different
            (MARKUP_C, MARKUP_C, *VIEWPORT),  # identical complex
            (MARKUP_B, MARKUP_B, *VIEWPORT),  # identical
            (MARKUP_A, MARKUP_C, *VIEWPORT),  # different
        ]
        results = pool.calculate_metrics_batch(items)

        assert len(results) == 6
        for r in results:
            assert r is not None
            assert 'perceptual_loss' in r
            assert 'similarity' in r

        # Identical pairs should have near-zero loss
        assert results[0]['perceptual_loss'] < 0.01
        assert results[3]['perceptual_loss'] < 0.01
        assert results[4]['perceptual_loss'] < 0.01

        # Different pairs should have higher loss than identical
        assert results[1]['perceptual_loss'] > results[0]['perceptual_loss']


# ---------------------------------------------------------------------------
# 3. Error handling — malformed HTML doesn't crash the pool
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_malformed_html_returns_result(self, pool):
        """Malformed HTML should still render (browsers are lenient) and return metrics."""
        r = pool.calculate_metrics(MARKUP_MALFORMED, MARKUP_A, *VIEWPORT)
        # Should return a result (browser renders something), not crash
        assert r is not None
        assert 'perceptual_loss' in r

    def test_empty_string_returns_result(self, pool):
        """Empty markup should render a blank page and return metrics."""
        r = pool.calculate_metrics(MARKUP_EMPTY, MARKUP_A, *VIEWPORT)
        assert r is not None
        assert 'perceptual_loss' in r

    def test_mixed_batch_with_errors(self, pool):
        """Batch with mix of valid and edge-case markup should not crash."""
        items = [
            (MARKUP_A, MARKUP_A, *VIEWPORT),
            (MARKUP_MALFORMED, MARKUP_A, *VIEWPORT),
            (MARKUP_EMPTY, MARKUP_B, *VIEWPORT),
            (MARKUP_B, MARKUP_B, *VIEWPORT),
        ]
        results = pool.calculate_metrics_batch(items)
        assert len(results) == 4
        # Valid pairs should definitely succeed
        assert results[0] is not None
        assert results[3] is not None

    def test_pool_functional_after_errors(self, pool):
        """Pool should still work after processing edge cases."""
        r = pool.calculate_metrics(MARKUP_A, MARKUP_A, *VIEWPORT)
        assert r is not None
        assert r['perceptual_loss'] < 0.01


# ---------------------------------------------------------------------------
# 4. Parallelism — speedup is real
# ---------------------------------------------------------------------------

class TestParallelism:

    def test_batch_faster_than_sequential(self, pool):
        """Batch of 4 should be at least 2x faster than 4 sequential calls."""
        items = [
            (MARKUP_A, MARKUP_B, *VIEWPORT),
            (MARKUP_B, MARKUP_C, *VIEWPORT),
            (MARKUP_C, MARKUP_A, *VIEWPORT),
            (MARKUP_A, MARKUP_C, *VIEWPORT),
        ]

        # Sequential timing
        t0 = time.monotonic()
        for item in items:
            pool.calculate_metrics(*item)
        seq_time = time.monotonic() - t0

        # Batch timing
        t0 = time.monotonic()
        pool.calculate_metrics_batch(items)
        batch_time = time.monotonic() - t0

        print(f"Sequential: {seq_time:.2f}s, Batch: {batch_time:.2f}s, "
              f"Speedup: {seq_time / batch_time:.1f}x")
        assert batch_time < seq_time * 0.5, (
            f"Expected at least 2x speedup but got {seq_time / batch_time:.1f}x "
            f"(seq={seq_time:.2f}s, batch={batch_time:.2f}s)")


# ---------------------------------------------------------------------------
# 5. Screenshot cache — LRU works
# ---------------------------------------------------------------------------

class TestScreenshotCache:

    def test_expected_markup_cached(self, pool):
        """Same expected markup in two batch items should use cache (second is faster or same)."""
        import hashlib
        cache_key = hashlib.sha256((MARKUP_C + f"{VIEWPORT[0]}x{VIEWPORT[1]}").encode()).hexdigest()[:16]

        # First call populates cache
        pool.calculate_metrics(MARKUP_A, MARKUP_C, *VIEWPORT)
        assert pool._cache_get(cache_key) is not None

        # Second call with same expected markup should hit cache
        cached_path_before = pool._cache_get(cache_key)
        pool.calculate_metrics(MARKUP_B, MARKUP_C, *VIEWPORT)
        cached_path_after = pool._cache_get(cache_key)
        assert cached_path_before == cached_path_after

    def test_lru_eviction(self):
        """Cache evicts oldest entries when exceeding max size."""
        os.makedirs(VALIDATION_DATA_DIR, exist_ok=True)
        from utils.similarity_parallel import RewardPool
        small_pool = RewardPool(num_workers=2, validation_data_dir=VALIDATION_DATA_DIR,
                                cache_max_size=3)
        try:
            # Fill cache with 4 different expected markups (exceeds limit of 3)
            markups = [
                MARKUP_A,
                MARKUP_B,
                MARKUP_C,
                MARKUP_MINIMAL,
            ]
            for exp in markups:
                small_pool.calculate_metrics(MARKUP_A, exp, *VIEWPORT)

            # Cache should have at most 3 entries
            assert len(small_pool._screenshot_cache) <= 3
        finally:
            small_pool.shutdown()


# ---------------------------------------------------------------------------
# 6. Pool lifecycle — startup and shutdown
# ---------------------------------------------------------------------------

class TestPoolLifecycle:

    def test_shutdown_and_reinit(self):
        """Pool can be shut down and a new one created."""
        os.makedirs(VALIDATION_DATA_DIR, exist_ok=True)
        from utils.similarity_parallel import RewardPool

        pool1 = RewardPool(num_workers=2, validation_data_dir=VALIDATION_DATA_DIR)
        r = pool1.calculate_metrics(MARKUP_A, MARKUP_A, *VIEWPORT)
        assert r is not None
        pool1.shutdown()

        # Verify no worker threads still alive
        alive = [t for t in pool1._workers if t.is_alive()]
        assert len(alive) == 0

        # Create fresh pool
        pool2 = RewardPool(num_workers=2, validation_data_dir=VALIDATION_DATA_DIR)
        r = pool2.calculate_metrics(MARKUP_A, MARKUP_A, *VIEWPORT)
        assert r is not None
        pool2.shutdown()

    def test_singleton_lifecycle(self):
        """Singleton can be created, used, shut down, and recreated."""
        from utils.similarity_parallel import get_reward_pool, shutdown_reward_pool, _reward_pool_lock
        import utils.similarity_parallel as mod

        # Reset singleton state
        with _reward_pool_lock:
            mod._reward_pool = None

        p = get_reward_pool(num_workers=2, validation_data_dir=VALIDATION_DATA_DIR)
        r = p.calculate_metrics(MARKUP_A, MARKUP_A, *VIEWPORT)
        assert r is not None
        shutdown_reward_pool()

        # Should be able to get a new one
        p2 = get_reward_pool(num_workers=2, validation_data_dir=VALIDATION_DATA_DIR)
        r2 = p2.calculate_metrics(MARKUP_B, MARKUP_B, *VIEWPORT)
        assert r2 is not None
        shutdown_reward_pool()


# ---------------------------------------------------------------------------
# 7. Token-level mode — element-level extraction and per-element LPIPS
# ---------------------------------------------------------------------------

class TestTokenLevelMode:
    """Tests for token_level=True mode in RewardPool."""

    def test_token_level_returns_extended_result(self, pool):
        """token_level=True returns a TokenLevelResult with element_infos, browser_dom, screenshots."""
        from utils.similarity_parallel import TokenLevelResult
        results = pool.calculate_metrics_batch(
            [(MARKUP_TOKEN_LEVEL, MARKUP_TOKEN_LEVEL, *VIEWPORT)],
            token_level=True,
        )
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, TokenLevelResult)
        assert r.similarity > 0
        assert r.perceptual_loss >= 0
        assert isinstance(r.element_infos, list)
        assert len(r.element_infos) > 0
        assert isinstance(r.browser_dom, str)
        assert len(r.browser_dom) > 0
        assert r.pred_screenshot_array is not None
        assert r.gt_screenshot_array is not None
        assert r.pred_screenshot_array.ndim == 3
        assert r.pred_screenshot_array.shape[2] == 3

    def test_token_level_elements_have_valid_bounds(self, pool):
        """Extracted elements should have positive width and height."""
        results = pool.calculate_metrics_batch(
            [(MARKUP_TOKEN_LEVEL, MARKUP_TOKEN_LEVEL, *VIEWPORT)],
            token_level=True,
        )
        r = results[0]
        for el in r.element_infos:
            assert el.width > 0, f"Element {el.selector} has non-positive width: {el.width}"
            assert el.height > 0, f"Element {el.selector} has non-positive height: {el.height}"

    def test_token_level_element_lpips_identical(self, pool):
        """Identical markup should yield per-element LPIPS close to 0."""
        results = pool.calculate_metrics_batch(
            [(MARKUP_TOKEN_LEVEL, MARKUP_TOKEN_LEVEL, *VIEWPORT)],
            token_level=True,
        )
        r = results[0]
        for el in r.element_infos:
            assert el.lpips_score < 0.05, (
                f"Element {el.selector} LPIPS too high for identical markup: {el.lpips_score}"
            )

    def test_token_level_element_lpips_wrong_color(self, pool):
        """Changed element (red->green) should have higher LPIPS than unchanged (blue)."""
        results = pool.calculate_metrics_batch(
            [(MARKUP_TOKEN_LEVEL_WRONG_COLOR, MARKUP_TOKEN_LEVEL, *VIEWPORT)],
            token_level=True,
        )
        r = results[0]
        # Find the red/green box and the blue box by selector
        red_el = None
        blue_el = None
        for el in r.element_infos:
            if 'red' in el.selector:
                red_el = el
            elif 'blue' in el.selector:
                blue_el = el

        assert red_el is not None, f"Could not find red element. Elements: {[e.selector for e in r.element_infos]}"
        assert blue_el is not None, f"Could not find blue element. Elements: {[e.selector for e in r.element_infos]}"

        # The changed element (red->green) should have higher LPIPS
        assert red_el.lpips_score > blue_el.lpips_score, (
            f"Changed element LPIPS ({red_el.lpips_score:.4f}) should be > "
            f"unchanged element LPIPS ({blue_el.lpips_score:.4f})"
        )
        # Blue box is unchanged, should have very low LPIPS
        assert blue_el.lpips_score < 0.05

    def test_token_level_css_mappings_found(self, pool):
        """CDP should find CSS property mappings for the test markup."""
        results = pool.calculate_metrics_batch(
            [(MARKUP_TOKEN_LEVEL, MARKUP_TOKEN_LEVEL, *VIEWPORT)],
            token_level=True,
        )
        r = results[0]
        assert isinstance(r.css_mappings, list)
        assert len(r.css_mappings) > 0, "Expected CDP to find at least one CSS mapping"
        # Each mapping is (model_char_start, model_char_end, element_index)
        for mapping in r.css_mappings:
            assert len(mapping) == 3
            assert mapping[0] < mapping[1], "start should be < end"
            assert mapping[2] >= 0, "element_index should be >= 0"

    def test_token_level_dom_positions(self, pool):
        """Elements should have DOM character positions set."""
        results = pool.calculate_metrics_batch(
            [(MARKUP_TOKEN_LEVEL, MARKUP_TOKEN_LEVEL, *VIEWPORT)],
            token_level=True,
        )
        r = results[0]
        elements_with_dom_pos = [el for el in r.element_infos if el.dom_char_start is not None]
        assert len(elements_with_dom_pos) > 0, (
            "Expected at least some elements to have DOM character positions"
        )
        for el in elements_with_dom_pos:
            assert el.dom_char_start >= 0
            assert el.dom_char_end > el.dom_char_start
            # The DOM position should point to valid content in browser_dom
            assert el.dom_char_end <= len(r.browser_dom)

    def test_token_level_backward_compatible(self, pool):
        """Default mode (no token_level) still returns plain dict."""
        results = pool.calculate_metrics_batch(
            [(MARKUP_TOKEN_LEVEL, MARKUP_TOKEN_LEVEL, *VIEWPORT)],
        )
        r = results[0]
        assert isinstance(r, dict)
        assert 'perceptual_loss' in r
        assert 'similarity' in r

    def test_token_level_batch_mixed(self, pool):
        """Batch of multiple items all get token-level results."""
        items = [
            (MARKUP_TOKEN_LEVEL, MARKUP_TOKEN_LEVEL, *VIEWPORT),
            (MARKUP_TOKEN_LEVEL_WRONG_COLOR, MARKUP_TOKEN_LEVEL, *VIEWPORT),
            (MARKUP_A, MARKUP_A, *VIEWPORT),
        ]
        from utils.similarity_parallel import TokenLevelResult
        results = pool.calculate_metrics_batch(items, token_level=True)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, TokenLevelResult), f"Expected TokenLevelResult, got {type(r)}"
            assert r.similarity > 0
            assert r.perceptual_loss >= 0
