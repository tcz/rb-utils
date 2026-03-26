"""
Parallel reward pipeline for GRPO training.

RewardPool maintains a thread pool of Playwright browser workers and a shared
LPIPS model.  All completions in a GRPO step are scored concurrently, cutting
reward time from ~48s (sequential) to ~8-10s (6 workers).

Usage:
    pool = get_reward_pool(num_workers=6)
    results = pool.calculate_metrics_batch([
        (pred_html, exp_html, 1024, 1024),
        ...
    ])

    # Token-level mode: also extract element bounding boxes and per-element LPIPS
    results = pool.calculate_metrics_batch([
        (pred_html, exp_html, 1024, 1024),
        ...
    ], token_level=True)
    # Returns list of TokenLevelResult instead of dict
"""

import hashlib
import os
import queue
import threading
import uuid
import warnings
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional

import lpips
import numpy as np
import torch
from PIL import Image
import PIL
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import torchvision.transforms as transforms

from .similarity import (
    remove_alpha,
    calculate_similarity,
    VALIDATION_DATA_DIR,
    WEB_SERVER_PORT,
)
from .server import ImagePlaceholderHTTPRequestHandler

PIL.Image.MAX_IMAGE_PIXELS = 1_000_000_000

_transform = transforms.ToTensor()


# ------------------------------------------------------------------
# Data classes for token-level mode
# ------------------------------------------------------------------

@dataclass
class ElementInfo:
    """Per-element data extracted during token-level rendering."""
    selector: str
    tag: str
    x: int
    y: int
    width: int
    height: int
    outerHTML: str
    lpips_score: float
    raw_element_index: int = -1  # index into original elements_raw list from extract_elements()
    dom_char_start: Optional[int] = None
    dom_char_end: Optional[int] = None


@dataclass
class TokenLevelResult:
    """Extended result when token_level=True."""
    similarity: float
    perceptual_loss: float
    element_infos: list  # list of ElementInfo
    browser_dom: str  # full browser DOM text for alignment
    css_mappings: list  # list of (model_char_start, model_char_end, element_index)
    pred_screenshot_array: Optional[np.ndarray] = None  # HxWx3 uint8
    gt_screenshot_array: Optional[np.ndarray] = None


class RewardPool:
    """Thread pool of Playwright workers with shared LPIPS for parallel reward computation."""

    def __init__(self, num_workers=6, validation_data_dir=VALIDATION_DATA_DIR,
                 web_server_port=WEB_SERVER_PORT, cache_max_size=256):
        self.num_workers = num_workers
        self._validation_data_dir = validation_data_dir
        self._web_server_port = web_server_port

        os.makedirs(validation_data_dir, exist_ok=True)

        # Shared LPIPS model — serialized via lock for thread safety
        self._lpips_fn = lpips.LPIPS(net='alex', verbose=False)
        self._lpips_fn.eval()
        self._lpips_lock = threading.Lock()

        # Screenshot LRU cache
        self._screenshot_cache: OrderedDict[str, str] = OrderedDict()
        self._cache_max_size = cache_max_size
        self._cache_lock = threading.Lock()

        # HTTP server serving validation_data_dir on web_server_port
        self._server_thread = threading.Thread(
            target=self._start_server, daemon=True)
        self._server_thread.start()

        # Work queue and worker threads
        self._work_queue: queue.Queue = queue.Queue()
        self._workers: list[threading.Thread] = []
        for i in range(num_workers):
            t = threading.Thread(target=self._worker_loop, name=f"reward-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_metrics_batch(self, items, token_level=False):
        """Score a batch of (pred_markup, exp_markup, vw, vh) tuples in parallel.

        Args:
            items: List of (pred_markup, exp_markup, vw, vh) tuples.
            token_level: If True, also extract element bounding boxes, run CDP
                CSS mapping, compute per-element LPIPS, and return TokenLevelResult
                instead of plain dict.

        Returns a list of dict|TokenLevelResult|None in the same order as input.
        """
        futures = []
        for item in items:
            f = Future()
            self._work_queue.put((f, item, token_level))
            futures.append(f)
        return [f.result() for f in futures]

    def calculate_metrics(self, pred_markup, exp_markup, vw, vh, token_level=False):
        """Single-item convenience wrapper."""
        results = self.calculate_metrics_batch(
            [(pred_markup, exp_markup, vw, vh)], token_level=token_level)
        return results[0]

    def calculate_metrics_batch_intermediate(
        self,
        items: list,  # list of (partial_html, ref_html, vw, vh)
    ) -> list:
        """Render partial HTML and reference, return LPIPS + screenshot arrays.

        This is a lighter-weight alternative to token_level mode that skips
        element extraction and CDP CSS mapping. It returns just the LPIPS score
        and both screenshot arrays as numpy arrays, which are needed for
        EMD/color histogram computation in the hybrid reward function.

        Args:
            items: List of (partial_html, ref_html, vw, vh) tuples.

        Returns:
            List of dicts (or None on failure) with keys:
                lpips_score: float
                pred_screenshot: np.ndarray (H, W, 3) uint8
                gt_screenshot: np.ndarray (H, W, 3) uint8
        """
        futures = []
        for item in items:
            f = Future()
            self._work_queue.put((f, item, "intermediate"))
            futures.append(f)
        return [f.result() for f in futures]

    def shutdown(self):
        """Gracefully stop all workers and close browsers."""
        for _ in self._workers:
            self._work_queue.put(None)  # poison pill
        for t in self._workers:
            t.join(timeout=30)
        self._workers.clear()
        # Clean cached screenshot files
        with self._cache_lock:
            for path in self._screenshot_cache.values():
                try:
                    os.remove(path)
                except OSError:
                    pass
            self._screenshot_cache.clear()

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    def _start_server(self):
        """Start HTTP server on configured port serving validation_data_dir."""
        from functools import partial
        from pathlib import Path
        import http.server

        Handler = partial(
            ImagePlaceholderHTTPRequestHandler,
            directory=self._validation_data_dir,
            cache_dir='cache',
            image_source_dir=Path(__file__).resolve().parent / 'DIV2K_valid_HR',
            font_source_dir=Path(__file__).resolve().parent / 'fonts',
            image_cache_limit=500,
        )

        while True:
            try:
                with http.server.ThreadingHTTPServer(
                    ("0.0.0.0", self._web_server_port), Handler
                ) as httpd:
                    httpd.allow_reuse_address = True
                    httpd.serve_forever()
            except OSError as e:
                if e.errno in (98, 48):  # Address already in use
                    return  # another server is running, that's fine
                import time as _time
                _time.sleep(1)
            except Exception:
                import time as _time
                _time.sleep(1)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """Each worker owns one Playwright + Chromium instance, reused across items."""
        pw = sync_playwright().start()
        browser = pw.chromium.launch(args=["--disable-gpu"])

        try:
            while True:
                item = self._work_queue.get()
                if item is None:
                    break  # poison pill
                future, (pred_markup, exp_markup, vw, vh), mode = item
                # Dispatch to the appropriate rendering method
                if mode == "intermediate":
                    render_fn = self._render_and_score_intermediate
                elif mode:  # token_level=True
                    render_fn = self._render_and_score_token_level
                else:  # token_level=False (standard)
                    render_fn = self._render_and_score
                try:
                    result = render_fn(browser, pred_markup, exp_markup, vw, vh)
                    future.set_result(result)
                except Exception as e1:
                    # Browser may be in a bad state — restart and retry once
                    try:
                        browser.close()
                    except Exception:
                        pass
                    try:
                        browser = pw.chromium.launch(args=["--disable-gpu"])
                        result = render_fn(browser, pred_markup, exp_markup, vw, vh)
                        future.set_result(result)
                    except Exception as e2:
                        warnings.warn(f"Reward worker failed after retry: {e2}")
                        future.set_result(None)
        finally:
            try:
                browser.close()
            except Exception:
                pass
            try:
                pw.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Rendering + scoring
    # ------------------------------------------------------------------

    def _render_and_score(self, browser, pred_markup, exp_markup, vw, vh):
        """Render both markups, compute LPIPS + multi-scale similarity."""
        uid = uuid.uuid4().hex
        pred_html = os.path.join(self._validation_data_dir, f"pred_{uid}.html")
        exp_html = os.path.join(self._validation_data_dir, f"exp_{uid}.html")
        pred_png = os.path.join(self._validation_data_dir, f"pred_{uid}.png")
        exp_png = os.path.join(self._validation_data_dir, f"exp_{uid}.png")

        try:
            # Write HTML files
            with open(pred_html, 'w') as f:
                f.write(pred_markup)
            with open(exp_html, 'w') as f:
                f.write(exp_markup)

            # Screenshot predicted (always fresh)
            self._take_screenshot(browser, f"pred_{uid}.html", pred_png, vw, vh)

            # Screenshot expected (check cache first)
            cache_key = hashlib.sha256((exp_markup + f"{vw}x{vh}").encode()).hexdigest()[:16]
            cached_exp = self._cache_get(cache_key)
            if cached_exp and os.path.exists(cached_exp):
                exp_png = cached_exp
            else:
                self._take_screenshot(browser, f"exp_{uid}.html", exp_png, vw, vh)
                self._cache_put(cache_key, exp_png)
                # Another worker may have raced us — if our path wasn't stored, clean it up
                actual_cached = self._cache_get(cache_key)
                if actual_cached != exp_png:
                    try:
                        os.remove(exp_png)
                    except OSError:
                        pass
                    exp_png = actual_cached

            # Compute metrics (before cleanup removes pred_png)
            result = self._compute_metrics(pred_png, exp_png)
            return result

        finally:
            # Clean up HTML files and predicted PNG (not needed after metrics computed)
            for p in (pred_html, exp_html, pred_png):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def _render_and_score_intermediate(self, browser, pred_markup, exp_markup, vw, vh):
        """Render both markups, compute LPIPS, and return screenshot arrays.

        Like _render_and_score but also returns pred and gt screenshots as
        numpy arrays (H, W, 3) uint8. This is needed for EMD/color histogram
        computation in the hybrid reward function without the overhead of
        element extraction and CDP CSS mapping from token-level mode.
        """
        uid = uuid.uuid4().hex
        pred_html = os.path.join(self._validation_data_dir, f"pred_{uid}.html")
        exp_html = os.path.join(self._validation_data_dir, f"exp_{uid}.html")
        pred_png = os.path.join(self._validation_data_dir, f"pred_{uid}.png")
        exp_png = os.path.join(self._validation_data_dir, f"exp_{uid}.png")

        try:
            # Write HTML files
            with open(pred_html, 'w') as f:
                f.write(pred_markup)
            with open(exp_html, 'w') as f:
                f.write(exp_markup)

            # Screenshot predicted (always fresh)
            self._take_screenshot(browser, f"pred_{uid}.html", pred_png, vw, vh)

            # Screenshot expected (check cache first)
            cache_key = hashlib.sha256((exp_markup + f"{vw}x{vh}").encode()).hexdigest()[:16]
            cached_exp = self._cache_get(cache_key)
            if cached_exp and os.path.exists(cached_exp):
                exp_png = cached_exp
            else:
                self._take_screenshot(browser, f"exp_{uid}.html", exp_png, vw, vh)
                self._cache_put(cache_key, exp_png)
                actual_cached = self._cache_get(cache_key)
                if actual_cached != exp_png:
                    try:
                        os.remove(exp_png)
                    except OSError:
                        pass
                    exp_png = actual_cached

            # Load screenshots as numpy arrays BEFORE cleanup
            pred_img = np.array(Image.open(pred_png).convert('RGB'))
            gt_img = np.array(Image.open(exp_png).convert('RGB'))

            # Compute LPIPS (reuses _compute_metrics which also does similarity)
            metrics = self._compute_metrics(pred_png, exp_png)

            return {
                'lpips_score': metrics['perceptual_loss'],
                'pred_screenshot': pred_img,
                'gt_screenshot': gt_img,
            }

        finally:
            # Clean up HTML files and predicted PNG (not cached exp_png)
            for p in (pred_html, exp_html, pred_png):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def _take_screenshot(self, browser, html_filename, out_path, vw, vh, max_retries=3):
        """Navigate to HTML file via local server and take screenshot."""
        url = f"http://localhost:{self._web_server_port}/{html_filename}"
        for attempt in range(max_retries):
            context = None
            try:
                context = browser.new_context(
                    viewport={'width': vw, 'height': vh},
                    service_workers="block",
                )
                page = context.new_page()
                try:
                    page.goto(url, wait_until="networkidle", timeout=10000)
                    page.wait_for_timeout(1000)
                except PlaywrightTimeoutError:
                    warnings.warn(f"Timeout loading {url}, falling back to domcontentloaded")
                    page.goto(url, wait_until="domcontentloaded", timeout=10000)
                    page.wait_for_timeout(5000)
                page.screenshot(path=out_path)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    warnings.warn(f"Screenshot attempt {attempt + 1} failed: {e}. Retrying...")
                    import time
                    time.sleep(0.5)
                else:
                    raise
            finally:
                if context:
                    try:
                        context.close()
                    except Exception:
                        pass

    def _compute_metrics(self, img1_path, img2_path):
        """Compute LPIPS perceptual loss and multi-scale similarity."""
        image1 = Image.open(img1_path)
        image2 = Image.open(img2_path)

        if image1.size == image2.size:
            new_image2 = Image.new("RGB", image1.size, (255, 255, 255))
            new_image2.paste(image2, (0, 0))
            image2 = new_image2.crop((0, 0, image1.size[0], image1.size[1]))

        t1 = remove_alpha(_transform(image1))
        t2 = remove_alpha(_transform(image2))

        similarity = calculate_similarity(t1, t2)

        # LPIPS with shared model — serialized via lock for thread safety
        norm1 = (t1 - 0.5) * 2
        norm2 = (t2 - 0.5) * 2
        with self._lpips_lock, torch.no_grad():
            loss = self._lpips_fn(norm1.unsqueeze(0), norm2.unsqueeze(0))
        perceptual_loss = loss.squeeze().item()

        return {
            'similarity': similarity,
            'perceptual_loss': perceptual_loss,
        }

    # ------------------------------------------------------------------
    # Token-level rendering + scoring
    # ------------------------------------------------------------------

    def _render_and_score_token_level(self, browser, pred_markup, exp_markup, vw, vh):
        """Render both markups with element extraction, CDP CSS mapping, and per-element LPIPS.

        Returns a TokenLevelResult with full-image metrics plus element-level data.
        """
        from .token_rewards import extract_elements, get_css_mappings_via_cdp, find_element_in_dom, crop_element

        uid = uuid.uuid4().hex
        pred_html = os.path.join(self._validation_data_dir, f"pred_{uid}.html")
        exp_html_path = os.path.join(self._validation_data_dir, f"exp_{uid}.html")
        pred_png = os.path.join(self._validation_data_dir, f"pred_{uid}.png")
        exp_png = os.path.join(self._validation_data_dir, f"exp_{uid}.png")

        try:
            # Write HTML files
            with open(pred_html, 'w') as f:
                f.write(pred_markup)
            with open(exp_html_path, 'w') as f:
                f.write(exp_markup)

            # --- Render predicted HTML with full browser context ---
            pred_url = f"http://localhost:{self._web_server_port}/pred_{uid}.html"
            context = browser.new_context(
                viewport={'width': vw, 'height': vh},
                service_workers="block",
            )
            try:
                page = context.new_page()
                try:
                    page.goto(pred_url, wait_until="networkidle", timeout=10000)
                    page.wait_for_timeout(1000)
                except PlaywrightTimeoutError:
                    warnings.warn(f"Timeout loading {pred_url}, falling back to domcontentloaded")
                    page.goto(pred_url, wait_until="domcontentloaded", timeout=10000)
                    page.wait_for_timeout(5000)

                # Capture clean browser DOM BEFORE extract_elements modifies it
                browser_dom = page.content()

                # Take screenshot
                page.screenshot(path=pred_png)

                # Extract elements (also sets data-tr-idx attributes on DOM nodes)
                elements_raw = extract_elements(page)

                # Get CSS source range mappings via CDP while page is still open
                css_mappings = get_css_mappings_via_cdp(page, elements_raw, pred_markup)
            finally:
                context.close()

            # --- Render expected HTML (with screenshot caching) ---
            cache_key = hashlib.sha256((exp_markup + f"{vw}x{vh}").encode()).hexdigest()[:16]
            cached_exp = self._cache_get(cache_key)
            if cached_exp and os.path.exists(cached_exp):
                exp_png = cached_exp
            else:
                self._take_screenshot(browser, f"exp_{uid}.html", exp_png, vw, vh)
                self._cache_put(cache_key, exp_png)
                actual_cached = self._cache_get(cache_key)
                if actual_cached != exp_png:
                    try:
                        os.remove(exp_png)
                    except OSError:
                        pass
                    exp_png = actual_cached

            # --- Load screenshots as numpy arrays ---
            pred_img = np.array(Image.open(pred_png).convert('RGB'))
            gt_img = np.array(Image.open(exp_png).convert('RGB'))

            # Resize gt to match pred if shapes differ
            if pred_img.shape != gt_img.shape:
                gt_pil = Image.fromarray(gt_img).resize(
                    (pred_img.shape[1], pred_img.shape[0]), Image.BILINEAR
                )
                gt_img = np.array(gt_pil)

            # --- Full-image metrics (same as _compute_metrics) ---
            result_dict = self._compute_metrics(pred_png, exp_png)
            similarity = result_dict['similarity']
            perceptual_loss = result_dict['perceptual_loss']

            # --- Per-element LPIPS ---
            element_infos = []
            for raw_idx, el in enumerate(elements_raw):
                area = el['width'] * el['height']
                if area < 16:  # min_element_area
                    continue

                crop_pred = crop_element(pred_img, el['x'], el['y'], el['width'], el['height'])
                crop_gt = crop_element(gt_img, el['x'], el['y'], el['width'], el['height'])
                el_lpips = self._compute_element_lpips_shared(crop_pred, crop_gt)

                if el_lpips is not None:
                    ei = ElementInfo(
                        selector=el['selector'],
                        tag=el['tag'],
                        x=el['x'], y=el['y'],
                        width=el['width'], height=el['height'],
                        outerHTML=el.get('outerHTML', ''),
                        lpips_score=el_lpips,
                        raw_element_index=raw_idx,
                    )

                    # Find element position in browser DOM
                    dom_range = find_element_in_dom(browser_dom, el.get('outerHTML', ''))
                    if dom_range:
                        ei.dom_char_start, ei.dom_char_end = dom_range

                    element_infos.append(ei)

            return TokenLevelResult(
                similarity=similarity,
                perceptual_loss=perceptual_loss,
                element_infos=element_infos,
                browser_dom=browser_dom,
                css_mappings=css_mappings,
                pred_screenshot_array=pred_img,
                gt_screenshot_array=gt_img,
            )

        finally:
            # Clean up HTML files and predicted PNG
            for p in (pred_html, exp_html_path, pred_png):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def _compute_element_lpips_shared(self, crop1, crop2):
        """Compute LPIPS between two image crops using the pool's shared LPIPS model.

        Same logic as compute_element_lpips in token_rewards.py but uses
        self._lpips_fn with self._lpips_lock for thread safety.

        Args:
            crop1: numpy array HxWx3 uint8 (predicted element crop)
            crop2: numpy array HxWx3 uint8 (ground truth element crop)

        Returns:
            LPIPS score as float, or None if crops are invalid.
        """
        if crop1 is None or crop2 is None:
            return None
        if crop1.shape[0] < 4 or crop1.shape[1] < 4:
            return None
        if crop2.shape[0] < 4 or crop2.shape[1] < 4:
            return None

        # Resize crop2 to match crop1 if needed
        if crop1.shape != crop2.shape:
            crop2_pil = Image.fromarray(crop2)
            crop2_pil = crop2_pil.resize((crop1.shape[1], crop1.shape[0]), Image.BILINEAR)
            crop2 = np.array(crop2_pil)

        # Convert to tensors [-1, 1]
        t1 = torch.from_numpy(crop1).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1
        t2 = torch.from_numpy(crop2).float().permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1

        # Upscale if too small for LPIPS
        if t1.shape[2] < 64 or t1.shape[3] < 64:
            scale = max(64 / t1.shape[2], 64 / t1.shape[3])
            new_h = max(64, int(t1.shape[2] * scale))
            new_w = max(64, int(t1.shape[3] * scale))
            t1 = torch.nn.functional.interpolate(t1, size=(new_h, new_w), mode='bilinear')
            t2 = torch.nn.functional.interpolate(t2, size=(new_h, new_w), mode='bilinear')

        # Use shared LPIPS with lock
        with self._lpips_lock, torch.no_grad():
            loss = self._lpips_fn(t1, t2)
        return loss.squeeze().item()

    # ------------------------------------------------------------------
    # Screenshot cache (LRU)
    # ------------------------------------------------------------------

    def _cache_get(self, key):
        with self._cache_lock:
            if key in self._screenshot_cache:
                self._screenshot_cache.move_to_end(key)
                return self._screenshot_cache[key]
        return None

    def _cache_put(self, key, path):
        with self._cache_lock:
            if key in self._screenshot_cache:
                self._screenshot_cache.move_to_end(key)
                return
            self._screenshot_cache[key] = path
            while len(self._screenshot_cache) > self._cache_max_size:
                _, evicted_path = self._screenshot_cache.popitem(last=False)
                try:
                    os.remove(evicted_path)
                except OSError:
                    pass


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_reward_pool = None
_reward_pool_lock = threading.Lock()


def get_reward_pool(num_workers=6, **kwargs):
    """Get or create the singleton RewardPool."""
    global _reward_pool
    with _reward_pool_lock:
        if _reward_pool is None:
            _reward_pool = RewardPool(num_workers=num_workers, **kwargs)
        return _reward_pool


def shutdown_reward_pool():
    """Shut down the singleton pool if it exists."""
    global _reward_pool
    with _reward_pool_lock:
        if _reward_pool is not None:
            _reward_pool.shutdown()
            _reward_pool = None
