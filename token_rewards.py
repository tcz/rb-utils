"""
Token-level reward computation for GRPO training.

Given a model-generated HTML string and ground truth HTML, computes per-token
rewards by mapping visual elements back to the tokens that generated them.

Pipeline:
  1. Render both HTMLs via Playwright, take screenshots
  2. Extract element bounding boxes from the rendered model output
  3. Compute per-element LPIPS (crop each element from both screenshots)
  4. Map elements to character positions in model output via rapidfuzz alignment
  5. Map character positions to token IDs via tokenizer offset mapping
  6. Compute per-token rewards: r_t = 1 - (alpha * OVERALL_LOSS + ELEMENT_LOSS_t)

Usage:
    from utils.token_rewards import compute_token_rewards

    result = compute_token_rewards(
        model_output="<html>...",
        ground_truth="<html>...",
        viewport_width=1024,
        viewport_height=1024,
        alpha=0.5,
    )
    # result.token_rewards: list of per-token reward floats
    # result.overall_loss: scalar LPIPS for the full image
    # result.element_scores: list of (selector, bounds, lpips) tuples
"""
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from rapidfuzz.distance import Levenshtein

from .similarity import start_server, WEB_SERVER_PORT, VALIDATION_DATA_DIR

import lpips
import threading
import uuid as uuid_module


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class ElementScore:
    """An element's bounding box and its LPIPS score."""
    selector: str
    tag: str
    x: int
    y: int
    width: int
    height: int
    lpips_score: float
    # Character range in the browser DOM that this element maps to
    dom_char_start: Optional[int] = None
    dom_char_end: Optional[int] = None
    # Character range in the model output (via alignment)
    model_char_start: Optional[int] = None
    model_char_end: Optional[int] = None


@dataclass
class TokenRewardResult:
    """Full result of token-level reward computation."""
    model_output: str
    overall_loss: float                      # full-image LPIPS
    overall_similarity: float                # full-image multi-scale similarity
    element_scores: list[ElementScore]       # per-element LPIPS
    token_rewards: Optional[list[float]]     # per-token rewards (if tokenizer provided)
    token_texts: Optional[list[str]]         # per-token text (if tokenizer provided)
    char_rewards: list[float]                # per-character rewards (before tokenization)
    alpha: float                             # alpha used for reward formula
    css_mappings_count: int = 0              # number of CSS property mappings found via CDP
    pred_screenshot_path: Optional[str] = None
    gt_screenshot_path: Optional[str] = None


# ── Element extraction via Playwright ────────────────────────────────

SKIP_TAGS = {'html', 'head', 'body', 'script', 'style', 'meta', 'link', 'br', 'title'}

# HTML void elements: self-closing, no closing tag needed
VOID_ELEMENTS = {
    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
    'link', 'meta', 'param', 'source', 'track', 'wbr',
}

EXTRACT_ELEMENTS_JS = """
() => {
    const results = [];
    const nodes = [];
    const skip = new Set(""" + json.dumps(list(SKIP_TAGS)) + """);
    const walker = document.createTreeWalker(
        document.body || document.documentElement,
        NodeFilter.SHOW_ELEMENT,
        null
    );
    let node = walker.currentNode;
    while (node) {
        const tag = node.tagName.toLowerCase();
        if (!skip.has(tag)) {
            const rect = node.getBoundingClientRect();
            const style = window.getComputedStyle(node);
            if (rect.width > 0 && rect.height > 0 &&
                style.display !== 'none' && style.visibility !== 'hidden') {
                let selector = tag;
                if (node.id) {
                    selector = '#' + node.id;
                } else if (node.className && typeof node.className === 'string') {
                    selector = tag + '.' + node.className.trim().split(/\\s+/).join('.');
                }
                // Capture outerHTML BEFORE we modify the DOM with data-tr-idx
                results.push({
                    selector: selector,
                    tag: tag,
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                    outerHTML: node.outerHTML.substring(0, 500),
                });
                nodes.push(node);
            }
        }
        node = walker.nextNode();
    }
    // Set unique data-tr-idx for CDP queries (after outerHTML capture)
    for (let i = 0; i < nodes.length; i++) {
        nodes[i].setAttribute('data-tr-idx', i.toString());
    }
    return results;
}
"""


def extract_elements(page):
    """Extract visible element bounding boxes from a Playwright page."""
    elements = page.evaluate(EXTRACT_ELEMENTS_JS)
    return elements


# ── Per-element LPIPS ────────────────────────────────────────────────

_lpips_fn = None

def get_lpips_fn():
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net='alex', verbose=False)
    return _lpips_fn


def crop_element(img_array, x, y, w, h):
    """Crop a region from an image array (H, W, 3)."""
    img_h, img_w = img_array.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_array[y1:y2, x1:x2]


def compute_element_lpips(crop1, crop2):
    """Compute LPIPS between two image crops (numpy HxWx3 uint8)."""
    if crop1 is None or crop2 is None:
        return None
    if crop1.shape[0] < 4 or crop1.shape[1] < 4:
        return None
    if crop2.shape[0] < 4 or crop2.shape[1] < 4:
        return None

    # Resize crop2 to match crop1 if needed
    if crop1.shape != crop2.shape:
        from PIL import Image as PILImage
        crop2_pil = PILImage.fromarray(crop2)
        crop2_pil = crop2_pil.resize((crop1.shape[1], crop1.shape[0]), PILImage.BILINEAR)
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

    fn = get_lpips_fn()
    with torch.no_grad():
        loss = fn(t1, t2)
    return loss.item()


# ── Alignment: model output <-> browser DOM ──────────────────────────

def align_texts(model_output: str, browser_dom: str):
    """
    Align model output to browser DOM using rapidfuzz Levenshtein opcodes.

    Returns a mapping: model_char_pos -> dom_char_pos (or -1 if deleted).
    Also returns the reverse: dom_char_pos -> model_char_pos (or -1 if inserted).
    """
    ops = Levenshtein.opcodes(model_output, browser_dom)

    model_to_dom = [-1] * len(model_output)
    dom_to_model = [-1] * len(browser_dom)

    for op in ops:
        if op.tag == 'equal':
            for offset in range(op.src_end - op.src_start):
                model_to_dom[op.src_start + offset] = op.dest_start + offset
                dom_to_model[op.dest_start + offset] = op.src_start + offset
        elif op.tag == 'replace':
            src_len = op.src_end - op.src_start
            dest_len = op.dest_end - op.dest_start
            for offset in range(src_len):
                dest_offset = int(offset * dest_len / src_len)
                model_to_dom[op.src_start + offset] = op.dest_start + dest_offset
            for offset in range(dest_len):
                src_offset = int(offset * src_len / dest_len)
                dom_to_model[op.dest_start + offset] = op.src_start + src_offset
        elif op.tag == 'delete':
            pass  # model chars with no DOM counterpart
        elif op.tag == 'insert':
            pass  # DOM chars with no model counterpart

    return model_to_dom, dom_to_model


def _find_closing_tag(dom: str, tag_name: str, after_opening_tag_end: int):
    """
    Find the matching closing tag for `tag_name` starting search from
    `after_opening_tag_end` (the position right after the opening tag's '>').

    Tracks nesting depth: nested opening tags of the same name increment depth,
    closing tags decrement it. Returns the end position (after '</tag>') or None.
    """
    # Regex matching opening or closing tags of the same name.
    # Captures: group(1) = '/' if closing tag, group(2) = tag name
    pattern = re.compile(
        r'<(/?)(' + re.escape(tag_name) + r')(?:\s[^>]*)?>',
        re.IGNORECASE,
    )
    depth = 1
    for m in pattern.finditer(dom, after_opening_tag_end):
        matched_tag = m.group(2).lower()
        if matched_tag != tag_name.lower():
            continue
        is_closing = m.group(1) == '/'
        # Check for self-closing syntax (/> at end)
        is_self_closing = m.group(0).endswith('/>')
        if is_closing:
            depth -= 1
            if depth == 0:
                return m.end()
        elif not is_self_closing and matched_tag not in VOID_ELEMENTS:
            depth += 1
    return None


def find_element_in_dom(browser_dom: str, element_html_prefix: str):
    """
    Find where an element's outer HTML appears in the browser DOM.

    Returns (start_char, end_char) covering the full element range: from the
    opening tag through the matching closing tag (inclusive of content and
    nested children). For void/self-closing elements, returns just the tag.

    Falls back to opening-tag-only range if the closing tag cannot be found.
    Returns None if the element is not found at all.
    """
    if not element_html_prefix or not browser_dom:
        return None

    # Try exact match first
    prefix = element_html_prefix[:200]  # Use first 200 chars of outerHTML
    idx = browser_dom.find(prefix)
    if idx >= 0:
        return _resolve_element_range(browser_dom, idx)

    # Try with just the opening tag pattern
    # e.g., <div class="box box-1">
    match = re.match(r'<(\w+)([^>]*)>', prefix)
    if match:
        tag_pattern = f'<{match.group(1)}{match.group(2)}'
        idx = browser_dom.find(tag_pattern)
        if idx >= 0:
            return _resolve_element_range(browser_dom, idx)

    return None


def _resolve_element_range(dom: str, start: int):
    """
    Given the start position of an element in the DOM, find its full range.

    For void elements or self-closing tags: returns (start, end_of_opening_tag).
    For normal elements: finds matching closing tag and returns (start, end_of_closing_tag).
    Falls back to opening-tag-only range if closing tag not found.
    """
    # Find the end of the opening tag
    tag_end = dom.find('>', start)
    if tag_end < 0:
        return None

    opening_tag_end = tag_end + 1

    # Extract the tag name
    tag_match = re.match(r'<(\w+)', dom[start:])
    if not tag_match:
        return start, opening_tag_end
    tag_name = tag_match.group(1).lower()

    # Self-closing syntax: <tag ... />
    opening_tag_text = dom[start:opening_tag_end]
    if opening_tag_text.endswith('/>'):
        return start, opening_tag_end

    # Void element: no closing tag
    if tag_name in VOID_ELEMENTS:
        return start, opening_tag_end

    # Find the matching closing tag
    closing_end = _find_closing_tag(dom, tag_name, opening_tag_end)
    if closing_end is not None:
        return start, closing_end

    # Fallback: return opening tag only
    return start, opening_tag_end


# ── CDP CSS source range mapping ──────────────────────────────────────

def _source_range_to_offsets(css_text, sr):
    """Convert CDP SourceRange {startLine, startColumn, endLine, endColumn}
    to (start_char, end_char) offsets within css_text."""
    lines = css_text.split('\n')
    start = sum(len(lines[i]) + 1 for i in range(min(sr['startLine'], len(lines))))
    start += sr['startColumn']
    end = sum(len(lines[i]) + 1 for i in range(min(sr['endLine'], len(lines))))
    end += sr['endColumn']
    return start, end


def get_css_mappings_via_cdp(page, elements_raw, model_output):
    """
    Use CDP CSS.getMatchedStylesForNode to map CSS properties to character
    positions in model_output.

    Requires elements to have data-tr-idx attributes set (done by extract_elements).

    Returns list of (model_char_start, model_char_end, raw_element_index) tuples.
    Each tuple means: characters [start, end) in model_output are CSS that
    applies to elements_raw[raw_element_index].
    """
    cdp = page.context.new_cdp_session(page)
    mappings = []

    try:
        cdp.send("DOM.enable")
        cdp.send("CSS.enable")

        doc = cdp.send("DOM.getDocument")
        root_id = doc["root"]["nodeId"]

        # Cache: stylesheet_id -> (css_text, offset_in_model_output) or None
        _sheet_cache = {}

        def _get_sheet_info(sheet_id):
            if sheet_id in _sheet_cache:
                return _sheet_cache[sheet_id]
            try:
                text = cdp.send("CSS.getStyleSheetText", {"styleSheetId": sheet_id})["text"]
            except Exception:
                _sheet_cache[sheet_id] = None
                return None

            # Find where this CSS text appears in the model output
            idx = model_output.find(text)
            if idx < 0:
                # Browser may strip/add trailing whitespace
                stripped = text.strip()
                idx = model_output.find(stripped)
                if idx >= 0:
                    text = stripped
            if idx < 0:
                _sheet_cache[sheet_id] = None
                return None

            _sheet_cache[sheet_id] = (text, idx)
            return text, idx

        for el_idx in range(len(elements_raw)):
            try:
                resp = cdp.send("DOM.querySelector", {
                    "nodeId": root_id,
                    "selector": f'[data-tr-idx="{el_idx}"]'
                })
                node_id = resp.get("nodeId", 0)
                if node_id == 0:
                    continue
            except Exception:
                continue

            try:
                styles = cdp.send("CSS.getMatchedStylesForNode", {"nodeId": node_id})
            except Exception:
                continue

            for rule_match in styles.get("matchedCSSRules", []):
                rule = rule_match.get("rule", {})
                sheet_id = rule.get("styleSheetId")
                if not sheet_id:
                    continue

                info = _get_sheet_info(sheet_id)
                if info is None:
                    continue
                css_text, sheet_offset = info

                # Map the selector range
                sel_range = rule.get("selectorList", {}).get("range")
                if sel_range:
                    s, e = _source_range_to_offsets(css_text, sel_range)
                    s = max(0, min(s, len(css_text)))
                    e = max(s, min(e, len(css_text)))
                    if e > s:
                        mappings.append((sheet_offset + s, sheet_offset + e, el_idx))

                # Map each CSS property range
                style = rule.get("style", {})
                for prop in style.get("cssProperties", []):
                    prop_range = prop.get("range")
                    if prop_range and not prop.get("disabled", False):
                        ps, pe = _source_range_to_offsets(css_text, prop_range)
                        ps = max(0, min(ps, len(css_text)))
                        pe = max(ps, min(pe, len(css_text)))
                        if pe > ps:
                            mappings.append((sheet_offset + ps, sheet_offset + pe, el_idx))

    except Exception as e:
        warnings.warn(f"CDP CSS mapping failed: {e}")
    finally:
        try:
            cdp.detach()
        except Exception:
            pass

    return mappings


# ── Main computation ─────────────────────────────────────────────────

def compute_token_rewards(
    model_output: str,
    ground_truth: str,
    viewport_width: int = 1024,
    viewport_height: int = 1024,
    alpha: float = 0.5,
    tokenizer=None,
    min_element_area: int = 16,
) -> TokenRewardResult:
    """
    Compute per-token rewards for a model-generated HTML string.

    Args:
        model_output: The HTML string generated by the model.
        ground_truth: The ground truth HTML string.
        viewport_width: Viewport width for rendering.
        viewport_height: Viewport height for rendering.
        alpha: Weight for overall loss in per-token reward formula.
                r_t = 1 - (alpha * OVERALL_LOSS + ELEMENT_LOSS_t)
        tokenizer: Optional HuggingFace tokenizer for token-level mapping.
        min_element_area: Minimum element area (px^2) for LPIPS computation.

    Returns:
        TokenRewardResult with per-character and optionally per-token rewards.
    """
    from playwright.sync_api import sync_playwright

    uid = uuid_module.uuid4().hex[:8]
    os.makedirs(VALIDATION_DATA_DIR, exist_ok=True)

    # Write HTML files
    pred_html_path = os.path.join(VALIDATION_DATA_DIR, f'pred_{uid}.html')
    gt_html_path = os.path.join(VALIDATION_DATA_DIR, f'gt_{uid}.html')
    pred_screenshot_path = os.path.join(VALIDATION_DATA_DIR, f'pred_{uid}.png')
    gt_screenshot_path = os.path.join(VALIDATION_DATA_DIR, f'gt_{uid}.png')

    with open(pred_html_path, 'w') as f:
        f.write(model_output)
    with open(gt_html_path, 'w') as f:
        f.write(ground_truth)

    # Ensure server is running (same pattern as similarity.calculate_metrics)
    from . import similarity as _sim_module
    if _sim_module.server_thread is None or not _sim_module.server_thread.is_alive():
        _sim_module.server_thread = threading.Thread(target=start_server, daemon=True)
        _sim_module.server_thread.start()
        import time
        time.sleep(0.5)

    try:
        os.environ["PW_TEST_SCREENSHOT_NO_FONTS_READY"] = "1"
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--disable-gpu"])

            # Render model output, extract elements, get CSS mappings via CDP
            ctx_pred = browser.new_context(
                viewport={'width': viewport_width, 'height': viewport_height},
                service_workers="block",
            )
            page_pred = ctx_pred.new_page()
            pred_url = f'http://localhost:{WEB_SERVER_PORT}/{os.path.basename(pred_html_path)}'
            page_pred.goto(pred_url, wait_until="domcontentloaded", timeout=10000)
            page_pred.wait_for_timeout(1000)
            page_pred.screenshot(path=pred_screenshot_path)
            # Capture clean DOM before extract_elements adds data-tr-idx attributes
            browser_dom = page_pred.content()
            # Extract elements (also sets data-tr-idx on DOM nodes for CDP)
            elements_raw = extract_elements(page_pred)
            # Get CSS source range mappings via CDP while page is still open
            css_mappings = get_css_mappings_via_cdp(page_pred, elements_raw, model_output)
            ctx_pred.close()

            # Render ground truth
            ctx_gt = browser.new_context(
                viewport={'width': viewport_width, 'height': viewport_height},
                service_workers="block",
            )
            page_gt = ctx_gt.new_page()
            gt_url = f'http://localhost:{WEB_SERVER_PORT}/{os.path.basename(gt_html_path)}'
            page_gt.goto(gt_url, wait_until="domcontentloaded", timeout=10000)
            page_gt.wait_for_timeout(1000)
            page_gt.screenshot(path=gt_screenshot_path)
            ctx_gt.close()

            browser.close()

        # Load screenshots as numpy arrays
        pred_img = np.array(Image.open(pred_screenshot_path).convert('RGB'))
        gt_img = np.array(Image.open(gt_screenshot_path).convert('RGB'))

        # Resize gt to match pred if needed
        if pred_img.shape != gt_img.shape:
            gt_pil = Image.fromarray(gt_img).resize(
                (pred_img.shape[1], pred_img.shape[0]), Image.BILINEAR
            )
            gt_img = np.array(gt_pil)

        # Overall LPIPS
        overall_loss = compute_element_lpips(pred_img, gt_img)
        if overall_loss is None:
            overall_loss = 1.0

        # Overall similarity (multi-scale MSE)
        from .similarity import calculate_similarity, remove_alpha
        transform = transforms.ToTensor()
        t_pred = remove_alpha(transform(Image.open(pred_screenshot_path)))
        t_gt = remove_alpha(transform(Image.open(gt_screenshot_path)))
        if t_pred.shape == t_gt.shape:
            overall_similarity = calculate_similarity(t_pred, t_gt)
        else:
            overall_similarity = 0.0

        # Per-element LPIPS (track raw index for CSS mapping)
        element_scores = []
        raw_idx_to_score = {}  # raw_element_index -> (lpips_score, area)
        for raw_idx, el in enumerate(elements_raw):
            area = el['width'] * el['height']
            if area < min_element_area:
                continue

            crop_pred = crop_element(pred_img, el['x'], el['y'], el['width'], el['height'])
            crop_gt = crop_element(gt_img, el['x'], el['y'], el['width'], el['height'])
            el_lpips = compute_element_lpips(crop_pred, crop_gt)

            if el_lpips is not None:
                raw_idx_to_score[raw_idx] = (el_lpips, area)
                es = ElementScore(
                    selector=el['selector'],
                    tag=el['tag'],
                    x=el['x'], y=el['y'],
                    width=el['width'], height=el['height'],
                    lpips_score=el_lpips,
                )

                # Find element in browser DOM
                dom_range = find_element_in_dom(browser_dom, el['outerHTML'])
                if dom_range:
                    es.dom_char_start, es.dom_char_end = dom_range

                element_scores.append(es)

        # Align model output to browser DOM
        model_to_dom, dom_to_model = align_texts(model_output, browser_dom)

        # Map DOM positions back to model positions for each element
        for es in element_scores:
            if es.dom_char_start is not None:
                # Find the model character range that maps to this DOM range
                model_starts = []
                model_ends = []
                for dom_pos in range(es.dom_char_start, es.dom_char_end):
                    if dom_pos < len(dom_to_model) and dom_to_model[dom_pos] >= 0:
                        model_starts.append(dom_to_model[dom_pos])
                        model_ends.append(dom_to_model[dom_pos])
                if model_starts:
                    es.model_char_start = min(model_starts)
                    es.model_char_end = max(model_ends) + 1

        # Compute per-character rewards
        # For each character in model output, find the element(s) it maps to
        n_chars = len(model_output)
        char_element_losses = [[] for _ in range(n_chars)]

        # HTML element ranges: innermost element wins.
        # Overlapping HTML ranges always mean ancestor-descendant, so the
        # innermost element has the smallest character span.
        # We track (lpips, area, span) per char, then pick min-span.
        char_html_candidates = [[] for _ in range(n_chars)]
        for es in element_scores:
            if es.model_char_start is not None:
                area = es.width * es.height
                span = es.model_char_end - es.model_char_start
                for c in range(es.model_char_start, min(es.model_char_end, n_chars)):
                    char_html_candidates[c].append((es.lpips_score, area, span))

        for c in range(n_chars):
            if char_html_candidates[c]:
                # Innermost = smallest character span
                innermost = min(char_html_candidates[c], key=lambda x: x[2])
                char_element_losses[c].append((innermost[0], innermost[1]))

        # CSS source range mappings: area-weighted average (a CSS rule can
        # match multiple unrelated elements, so averaging is appropriate)
        for css_start, css_end, raw_idx in css_mappings:
            if raw_idx in raw_idx_to_score:
                lpips_val, area = raw_idx_to_score[raw_idx]
                for c in range(max(0, css_start), min(css_end, n_chars)):
                    char_element_losses[c].append((lpips_val, area))

        # Per-character reward: r_c = 1 - (alpha * overall + element_loss)
        char_rewards = []
        for c in range(n_chars):
            if char_element_losses[c]:
                # Area-weighted average (HTML contributes one entry, CSS may add more)
                total_area = sum(area for _, area in char_element_losses[c])
                weighted_loss = sum(loss * area for loss, area in char_element_losses[c]) / total_area
                reward = 1.0 - (alpha * overall_loss + weighted_loss)
            else:
                # Unmapped character: only overall loss
                reward = 1.0 - alpha * overall_loss
            char_rewards.append(reward)

        # Token-level rewards (if tokenizer provided)
        token_rewards = None
        token_texts = None
        if tokenizer is not None:
            encoding = tokenizer(model_output, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding['offset_mapping']
            token_ids = encoding['input_ids']

            token_rewards = []
            token_texts = []
            for i, (start, end) in enumerate(offsets):
                if start == end:
                    # Special token or empty
                    token_rewards.append(1.0 - alpha * overall_loss)
                    token_texts.append(tokenizer.decode([token_ids[i]]))
                else:
                    # Average char rewards for this token's span
                    span_rewards = char_rewards[start:end]
                    token_rewards.append(sum(span_rewards) / len(span_rewards))
                    token_texts.append(model_output[start:end])

        return TokenRewardResult(
            model_output=model_output,
            overall_loss=overall_loss,
            overall_similarity=overall_similarity,
            element_scores=element_scores,
            token_rewards=token_rewards,
            token_texts=token_texts,
            char_rewards=char_rewards,
            alpha=alpha,
            css_mappings_count=len(css_mappings),
            pred_screenshot_path=pred_screenshot_path,
            gt_screenshot_path=gt_screenshot_path,
        )

    finally:
        # Clean up temp files
        for p in [pred_html_path, gt_html_path]:
            try:
                os.remove(p)
            except OSError:
                pass
