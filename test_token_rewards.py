"""
Test and visualize token-level rewards.

Generates an HTML report showing per-token reward distribution with color coding.
Works with toy examples and can load real dataset samples.

Usage:
    python -m utils.test_token_rewards                    # toy examples only
    python -m utils.test_token_rewards --dataset           # include dataset samples
    python -m utils.test_token_rewards --alpha 0.0         # pure element-level
    python -m utils.test_token_rewards --alpha 1.0         # heavy overall weight

Output: token_rewards_report.html (open in browser)
"""
import argparse
import base64
import difflib
import html as html_module
import os
import sys
import time

# Add parent dir to path so we can import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_color(reward, min_r, max_r):
    """Map reward to a color: red (bad) -> yellow (neutral) -> green (good)."""
    if max_r == min_r:
        t = 0.5
    else:
        t = (reward - min_r) / (max_r - min_r)
    t = max(0, min(1, t))

    if t < 0.5:
        # Red to yellow
        r = 255
        g = int(255 * (t * 2))
        b = 0
    else:
        # Yellow to green
        r = int(255 * (1 - (t - 0.5) * 2))
        g = 200
        b = 0

    return f'rgb({r},{g},{b})'


def _mapped_preview(model_output, start, end, max_chars=30):
    """Return a tooltip-friendly preview of the mapped text range.

    Shows the first and last ~max_chars characters with ellipsis in between
    if the range is longer than 2*max_chars."""
    text = model_output[start:end]
    if len(text) <= max_chars * 2:
        return html_module.escape(text)
    head = html_module.escape(text[:max_chars])
    tail = html_module.escape(text[-max_chars:])
    return f'{head}...{tail}'


def _element_boundary_set(element_scores):
    """Build sets of character positions that are element range boundaries.

    Returns (starts, ends) where starts is a set of char indices where an
    element range begins and ends is a set where one ends."""
    starts = set()
    ends = set()
    for es in element_scores:
        if es.model_char_start is not None and es.model_char_end is not None:
            starts.add(es.model_char_start)
            ends.add(es.model_char_end)
    return starts, ends


def _boundary_style(chunk_start, chunk_end, boundary_starts, boundary_ends):
    """Return inline CSS border styles if the chunk overlaps element boundaries."""
    styles = []
    for pos in range(chunk_start, chunk_end):
        if pos in boundary_starts:
            styles.append('border-left:2px solid #1976d2')
            break
    for pos in range(chunk_start, chunk_end):
        if pos in boundary_ends:
            styles.append('border-right:2px solid #1976d2')
            break
    return ';'.join(styles)


def _render_diff_side(lines, opcodes, side):
    """Render one side of a side-by-side diff with inline highlights.
    side='left' highlights deletions (ground truth), side='right' highlights insertions (model output)."""
    parts = []
    for tag, i1, i2, j1, j2 in opcodes:
        if side == 'left':
            chunk = lines[0][i1:i2]
            if tag == 'equal':
                for line in chunk:
                    parts.append(html_module.escape(line))
            elif tag in ('replace', 'delete'):
                for line in chunk:
                    parts.append(f'<span class="diff-del">{html_module.escape(line)}</span>')
            # 'insert' — nothing on the left side
        else:
            chunk = lines[1][j1:j2]
            if tag == 'equal':
                for line in chunk:
                    parts.append(html_module.escape(line))
            elif tag in ('replace', 'insert'):
                for line in chunk:
                    parts.append(f'<span class="diff-ins">{html_module.escape(line)}</span>')
            # 'delete' — nothing on the right side
    return '\n'.join(parts)


def generate_bbox_overlay_html(result):
    """Generate an SVG overlay on the model output screenshot showing per-element advantage.

    Each element's bounding box is drawn with the same color scheme as the
    advantage column in the element scores table (red=negative, green=positive).
    """
    if not result.pred_screenshot_path or not os.path.exists(result.pred_screenshot_path):
        return ''
    if not result.element_scores:
        return ''

    with open(result.pred_screenshot_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode('ascii')

    # Get image dimensions
    from PIL import Image as PILImage
    img = PILImage.open(result.pred_screenshot_path)
    img_w, img_h = img.size

    # Compute per-element advantages
    el_rewards = [1.0 - (result.alpha * result.overall_loss + es.lpips_score)
                  for es in result.element_scores]
    el_mean = sum(el_rewards) / len(el_rewards) if el_rewards else 0
    el_std = (sum((r - el_mean) ** 2 for r in el_rewards) / len(el_rewards)) ** 0.5 if el_rewards else 0

    # Build SVG rectangles
    rects = []
    for es, el_r in zip(result.element_scores, el_rewards):
        adv = (el_r - el_mean) / el_std if el_std > 1e-8 else 0.0
        color = get_color(adv, -2, 2)
        selector_esc = html_module.escape(es.selector)
        rects.append(
            f'<rect x="{es.x}" y="{es.y}" width="{es.width}" height="{es.height}" '
            f'fill="{color}" fill-opacity="0.35" stroke="{color}" stroke-width="2" '
            f'stroke-opacity="0.8">'
            f'<title>{selector_esc}\nLPIPS: {es.lpips_score:.4f}\n'
            f'Advantage: {adv:+.3f}</title></rect>'
        )

    svg_content = '\n'.join(rects)

    return (
        f'<details open><summary>Advantage heatmap (bounding boxes)</summary>'
        f'<div style="position:relative;display:inline-block;margin:8px 0">'
        f'<img src="data:image/png;base64,{img_b64}" '
        f'style="display:block;max-width:100%;border:1px solid #ccc;border-radius:6px">'
        f'<svg viewBox="0 0 {img_w} {img_h}" '
        f'style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none">'
        f'<g style="pointer-events:all">{svg_content}</g>'
        f'</svg></div></details>'
    )


def generate_token_html(result, title, ground_truth=None):
    """Generate HTML for a single test case showing colored tokens."""
    parts = []
    parts.append(f'<h3>{html_module.escape(title)}</h3>')
    parts.append(f'<div class="stats">')
    parts.append(f'<span>Overall LPIPS: <b>{result.overall_loss:.4f}</b></span>')
    parts.append(f'<span>Overall Similarity: <b>{result.overall_similarity:.4f}</b></span>')
    parts.append(f'<span>Alpha: <b>{result.alpha}</b></span>')
    parts.append(f'<span>Elements found: <b>{len(result.element_scores)}</b></span>')
    mapped = sum(1 for es in result.element_scores if es.model_char_start is not None)
    parts.append(f'<span>Elements mapped: <b>{mapped}</b></span>')
    parts.append(f'<span>CSS mappings: <b>{result.css_mappings_count}</b></span>')
    parts.append(f'</div>')

    # Side-by-side screenshots: Ground Truth vs Model Output
    if result.gt_screenshot_path and result.pred_screenshot_path:
        imgs = {}
        for label, path in [('gt', result.gt_screenshot_path), ('pred', result.pred_screenshot_path)]:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    imgs[label] = base64.b64encode(f.read()).decode('ascii')
        if 'gt' in imgs and 'pred' in imgs:
            parts.append('<div class="screenshots">')
            parts.append('<div class="screenshot-pair">')
            parts.append(f'<div class="screenshot-col"><div class="screenshot-label">Ground Truth</div>'
                         f'<img src="data:image/png;base64,{imgs["gt"]}"></div>')
            parts.append(f'<div class="screenshot-col"><div class="screenshot-label">Model Output</div>'
                         f'<img src="data:image/png;base64,{imgs["pred"]}"></div>')
            parts.append('</div></div>')

    # Bounding box advantage overlay on model output
    bbox_html = generate_bbox_overlay_html(result)
    if bbox_html:
        parts.append(bbox_html)

    # Side-by-side markup diff: Ground Truth vs Model Output
    if ground_truth and ground_truth != result.model_output:
        gt_lines = ground_truth.splitlines(keepends=False)
        pred_lines = result.model_output.splitlines(keepends=False)
        sm = difflib.SequenceMatcher(None, gt_lines, pred_lines)
        opcodes = sm.get_opcodes()

        gt_html = _render_diff_side((gt_lines, pred_lines), opcodes, 'left')
        pred_html = _render_diff_side((gt_lines, pred_lines), opcodes, 'right')

        parts.append('<details open><summary>Markup diff (Ground Truth vs Model Output)</summary>')
        parts.append('<div class="diff-container">')
        parts.append(f'<div class="diff-col"><div class="diff-label">Ground Truth</div>'
                     f'<pre class="diff-code">{gt_html}</pre></div>')
        parts.append(f'<div class="diff-col"><div class="diff-label">Model Output</div>'
                     f'<pre class="diff-code">{pred_html}</pre></div>')
        parts.append('</div></details>')

    # Element scores table
    if result.element_scores:
        # Compute element-level reward and advantage for display
        el_rewards = [1.0 - (result.alpha * result.overall_loss + es.lpips_score)
                      for es in result.element_scores]
        el_mean = sum(el_rewards) / len(el_rewards) if el_rewards else 0
        el_std = (sum((r - el_mean) ** 2 for r in el_rewards) / len(el_rewards)) ** 0.5 if el_rewards else 0

        parts.append('<details open><summary>Element scores ({} elements)</summary>'.format(
            len(result.element_scores)))
        parts.append('<table class="el-table"><tr><th>Selector</th><th>Bounds</th>'
                     '<th>LPIPS</th><th>Reward</th><th>Advantage</th><th>Mapped</th></tr>')
        for i, es in enumerate(sorted(result.element_scores, key=lambda e: -e.lpips_score)):
            if es.model_char_start is not None and es.model_char_end is not None:
                preview = _mapped_preview(result.model_output, es.model_char_start, es.model_char_end)
                mapped_str = (f'<span title="{preview}" style="cursor:help;border-bottom:1px dotted #999">'
                              f'{es.model_char_start}-{es.model_char_end}</span>')
            else:
                mapped_str = 'N/A'
            el_reward = 1.0 - (result.alpha * result.overall_loss + es.lpips_score)
            el_adv = (el_reward - el_mean) / el_std if el_std > 1e-8 else 0.0
            color = get_color(1 - es.lpips_score, 0, 1)
            adv_color = get_color(el_adv, -2, 2)
            parts.append(
                f'<tr><td>{html_module.escape(es.selector)}</td>'
                f'<td>{es.x},{es.y} {es.width}x{es.height}</td>'
                f'<td style="background:{color};color:black;font-weight:bold">{es.lpips_score:.4f}</td>'
                f'<td style="font-weight:bold">{el_reward:.4f}</td>'
                f'<td style="background:{adv_color};color:black;font-weight:bold">{el_adv:+.3f}</td>'
                f'<td>{mapped_str}</td></tr>'
            )
        parts.append('</table></details>')

    # Per-character reward visualization
    char_rewards = result.char_rewards
    if char_rewards:
        min_r = min(char_rewards)
        max_r = max(char_rewards)
        mean_r = sum(char_rewards) / len(char_rewards)
        std_r = (sum((r - mean_r) ** 2 for r in char_rewards) / len(char_rewards)) ** 0.5

        # Compute per-character advantages: A_c = (r_c - mu) / sigma
        if std_r > 1e-8:
            char_advantages = [(r - mean_r) / std_r for r in char_rewards]
        else:
            char_advantages = [0.0] * len(char_rewards)

        parts.append('<h4>Per-Character Rewards &amp; Advantages</h4>')
        parts.append(f'<div class="stats">'
                     f'<span>Min reward: <b>{min_r:.4f}</b></span>'
                     f'<span>Max reward: <b>{max_r:.4f}</b></span>'
                     f'<span>Mean (μ): <b>{mean_r:.4f}</b></span>'
                     f'<span>Std (σ): <b>{std_r:.4f}</b></span>'
                     f'</div>')

        parts.append('<div class="token-vis">')
        # Build element boundary sets for visual markers
        el_starts, el_ends = _element_boundary_set(result.element_scores)
        # Group characters into chunks for readability
        text = result.model_output
        i = 0
        while i < len(text):
            # Find a run of characters with similar rewards (within 0.05)
            j = i + 1
            while j < len(text) and j - i < 80:
                if abs(char_rewards[j] - char_rewards[i]) > 0.05:
                    break
                # Break chunk at element boundaries so markers align
                if j in el_starts or j in el_ends:
                    break
                j += 1

            chunk_text = text[i:j]
            chunk_reward = sum(char_rewards[i:j]) / (j - i)
            chunk_adv = sum(char_advantages[i:j]) / (j - i)
            color = get_color(chunk_reward, min_r, max_r)
            boundary = _boundary_style(i, j, el_starts, el_ends)

            escaped = html_module.escape(chunk_text).replace('\n', '<br>')
            style = f'background:{color}'
            if boundary:
                style += f';{boundary}'
            parts.append(f'<span class="token" style="{style}" '
                         f'title="chars {i}-{j} | reward={chunk_reward:.4f} | advantage={chunk_adv:+.3f}">'
                         f'{escaped}</span>')
            i = j

        parts.append('</div>')

    # Per-token reward visualization (if available)
    if result.token_rewards and result.token_texts:
        min_r = min(result.token_rewards)
        max_r = max(result.token_rewards)
        mean_tr = sum(result.token_rewards) / len(result.token_rewards)
        std_tr = (sum((r - mean_tr) ** 2 for r in result.token_rewards) / len(result.token_rewards)) ** 0.5

        # Per-token advantages: A_t = (r_t - mu) / sigma
        if std_tr > 1e-8:
            token_advantages = [(r - mean_tr) / std_tr for r in result.token_rewards]
        else:
            token_advantages = [0.0] * len(result.token_rewards)

        parts.append('<h4>Per-Token Rewards &amp; Advantages</h4>')
        parts.append(f'<div class="stats">'
                     f'<span>Tokens: <b>{len(result.token_rewards)}</b></span>'
                     f'<span>Min: <b>{min_r:.4f}</b></span>'
                     f'<span>Max: <b>{max_r:.4f}</b></span>'
                     f'<span>Mean (μ): <b>{mean_tr:.4f}</b></span>'
                     f'<span>Std (σ): <b>{std_tr:.4f}</b></span>'
                     f'</div>')

        parts.append('<div class="token-vis">')
        for i, (text, reward, adv) in enumerate(zip(result.token_texts, result.token_rewards, token_advantages)):
            color = get_color(reward, min_r, max_r)
            escaped = html_module.escape(text).replace('\n', '<br>').replace(' ', '&nbsp;')
            if not escaped.strip():
                escaped = '&nbsp;'
            parts.append(f'<span class="token" style="background:{color}" '
                         f'title="token {i}: {html_module.escape(repr(text))} | reward={reward:.4f} | advantage={adv:+.3f}">'
                         f'{escaped}</span>')
        parts.append('</div>')

    # Reward distribution histogram (text-based)
    if result.token_rewards:
        rewards = result.token_rewards
        parts.append('<h4>Reward Distribution</h4>')
        parts.append('<div class="histogram">')
        n_bins = 20
        min_val = min(rewards)
        max_val = max(rewards)
        if max_val > min_val:
            bin_width = (max_val - min_val) / n_bins
            bins = [0] * n_bins
            for r in rewards:
                b = min(int((r - min_val) / bin_width), n_bins - 1)
                bins[b] += 1
            max_count = max(bins)
            for b in range(n_bins):
                bar_len = int(50 * bins[b] / max_count) if max_count > 0 else 0
                lo = min_val + b * bin_width
                hi = lo + bin_width
                color = get_color((lo + hi) / 2, min_val, max_val)
                parts.append(
                    f'<div style="font-family:monospace;font-size:11px">'
                    f'<span style="display:inline-block;width:120px">[{lo:.3f}, {hi:.3f})</span>'
                    f'<span style="display:inline-block;width:{bar_len * 6}px;height:14px;'
                    f'background:{color};margin-right:4px"></span>'
                    f'{bins[b]}</div>'
                )
        parts.append('</div>')

    return '\n'.join(parts)


def generate_group_html(group_case, group_results, group_mu, group_sigma):
    """Generate HTML for a G=2 group advantage comparison."""
    parts = []
    gt = group_case['ground_truth']
    parts.append(f'<h2>{html_module.escape(group_case["name"])}</h2>')
    parts.append(f'<div class="stats">'
                 f'<span>Group size: <b>G={len(group_results)}</b></span>'
                 f'<span>Group mean (μ): <b>{group_mu:.4f}</b></span>'
                 f'<span>Group std (σ): <b>{group_sigma:.4f}</b></span>'
                 f'</div>')

    for label, result, elapsed in group_results:
        parts.append(f'<div class="test-case">')
        parts.append(f'<h3>{html_module.escape(label)}</h3>')
        parts.append(f'<div class="stats">'
                     f'<span>Computation time: <b>{elapsed:.2f}s</b></span>'
                     f'<span>Overall LPIPS: <b>{result.overall_loss:.4f}</b></span>'
                     f'<span>Overall Similarity: <b>{result.overall_similarity:.4f}</b></span>'
                     f'</div>')

        # Screenshots
        if result.gt_screenshot_path and result.pred_screenshot_path:
            imgs = {}
            for k, path in [('gt', result.gt_screenshot_path), ('pred', result.pred_screenshot_path)]:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        imgs[k] = base64.b64encode(f.read()).decode('ascii')
            if 'gt' in imgs and 'pred' in imgs:
                parts.append('<div class="screenshots"><div class="screenshot-pair">')
                parts.append(f'<div class="screenshot-col"><div class="screenshot-label">Ground Truth</div>'
                             f'<img src="data:image/png;base64,{imgs["gt"]}"></div>')
                parts.append(f'<div class="screenshot-col"><div class="screenshot-label">Model Output</div>'
                             f'<img src="data:image/png;base64,{imgs["pred"]}"></div>')
                parts.append('</div></div>')

        # Bounding box advantage overlay
        bbox_html = generate_bbox_overlay_html(result)
        if bbox_html:
            parts.append(bbox_html)

        # Markup diff
        if gt != result.model_output:
            gt_lines = gt.splitlines(keepends=False)
            pred_lines = result.model_output.splitlines(keepends=False)
            sm = difflib.SequenceMatcher(None, gt_lines, pred_lines)
            opcodes = sm.get_opcodes()
            gt_html = _render_diff_side((gt_lines, pred_lines), opcodes, 'left')
            pred_html = _render_diff_side((gt_lines, pred_lines), opcodes, 'right')
            parts.append('<details open><summary>Markup diff</summary>')
            parts.append('<div class="diff-container">')
            parts.append(f'<div class="diff-col"><div class="diff-label">Ground Truth</div>'
                         f'<pre class="diff-code">{gt_html}</pre></div>')
            parts.append(f'<div class="diff-col"><div class="diff-label">Model Output</div>'
                         f'<pre class="diff-code">{pred_html}</pre></div>')
            parts.append('</div></details>')

        # Element scores with group advantage
        if result.element_scores:
            el_rewards = [1.0 - (result.alpha * result.overall_loss + es.lpips_score)
                          for es in result.element_scores]
            parts.append('<details open><summary>Element scores</summary>')
            parts.append('<table class="el-table"><tr><th>Selector</th><th>Bounds</th>'
                         '<th>LPIPS</th><th>Reward</th>'
                         '<th>Single Adv.</th><th>Group Adv.</th><th>Mapped</th></tr>')
            # Single-completion stats for comparison
            el_mean_s = sum(el_rewards) / len(el_rewards)
            el_std_s = (sum((r - el_mean_s) ** 2 for r in el_rewards) / len(el_rewards)) ** 0.5
            for es in sorted(result.element_scores, key=lambda e: -e.lpips_score):
                el_r = 1.0 - (result.alpha * result.overall_loss + es.lpips_score)
                single_adv = (el_r - el_mean_s) / el_std_s if el_std_s > 1e-8 else 0.0
                group_adv = (el_r - group_mu) / group_sigma if group_sigma > 1e-8 else 0.0
                if es.model_char_start is not None and es.model_char_end is not None:
                    preview = _mapped_preview(result.model_output, es.model_char_start, es.model_char_end)
                    mapped_str = (f'<span title="{preview}" style="cursor:help;border-bottom:1px dotted #999">'
                                  f'{es.model_char_start}-{es.model_char_end}</span>')
                else:
                    mapped_str = 'N/A'
                s_color = get_color(single_adv, -2, 2)
                g_color = get_color(group_adv, -2, 2)
                parts.append(
                    f'<tr><td>{html_module.escape(es.selector)}</td>'
                    f'<td>{es.x},{es.y} {es.width}x{es.height}</td>'
                    f'<td style="font-weight:bold">{es.lpips_score:.4f}</td>'
                    f'<td style="font-weight:bold">{el_r:.4f}</td>'
                    f'<td style="background:{s_color};color:black;font-weight:bold">{single_adv:+.3f}</td>'
                    f'<td style="background:{g_color};color:black;font-weight:bold">{group_adv:+.3f}</td>'
                    f'<td>{mapped_str}</td></tr>')
            parts.append('</table></details>')

        # Per-character rewards with BOTH single and group advantages
        char_rewards = result.char_rewards
        if char_rewards:
            min_r = min(char_rewards)
            max_r = max(char_rewards)
            mean_r = sum(char_rewards) / len(char_rewards)
            std_r = (sum((r - mean_r) ** 2 for r in char_rewards) / len(char_rewards)) ** 0.5

            parts.append('<h4>Per-Character Rewards (group advantage on hover)</h4>')
            parts.append(f'<div class="stats">'
                         f'<span>Min reward: <b>{min_r:.4f}</b></span>'
                         f'<span>Max reward: <b>{max_r:.4f}</b></span>'
                         f'<span>Single μ: <b>{mean_r:.4f}</b></span>'
                         f'<span>Group μ: <b>{group_mu:.4f}</b></span>'
                         f'</div>')

            parts.append('<div class="token-vis">')
            # Build element boundary sets for visual markers
            el_starts, el_ends = _element_boundary_set(result.element_scores)
            text = result.model_output
            i = 0
            while i < len(text):
                j = i + 1
                while j < len(text) and j - i < 80:
                    if abs(char_rewards[j] - char_rewards[i]) > 0.05:
                        break
                    # Break chunk at element boundaries so markers align
                    if j in el_starts or j in el_ends:
                        break
                    j += 1
                chunk_reward = sum(char_rewards[i:j]) / (j - i)
                single_a = (chunk_reward - mean_r) / std_r if std_r > 1e-8 else 0.0
                group_a = (chunk_reward - group_mu) / group_sigma if group_sigma > 1e-8 else 0.0
                color = get_color(chunk_reward, min_r, max_r)
                boundary = _boundary_style(i, j, el_starts, el_ends)
                escaped = html_module.escape(text[i:j]).replace('\n', '<br>')
                style = f'background:{color}'
                if boundary:
                    style += f';{boundary}'
                parts.append(f'<span class="token" style="{style}" '
                             f'title="chars {i}-{j} | reward={chunk_reward:.4f} | '
                             f'single={single_a:+.3f} | group={group_a:+.3f}">'
                             f'{escaped}</span>')
                i = j
            parts.append('</div>')

        parts.append('</div>')  # end test-case

    return '\n'.join(parts)


def generate_report(test_results, group_data=None):
    """Generate the full HTML report."""
    parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Token-Level Rewards Report</title>
<style>
body { font-family: system-ui, sans-serif; margin: 20px; background: #fafafa; }
h1 { color: #333; }
h2 { color: #555; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
h3 { color: #444; margin-top: 24px; }
h4 { color: #666; margin-top: 16px; font-size: 14px; }
.stats { background: white; padding: 12px 16px; border-radius: 8px; margin: 8px 0;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 13px; }
.stats span { margin-right: 20px; }
.stats b { color: #333; }
.token-vis { background: white; padding: 12px; border-radius: 8px; margin: 8px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-family: 'SF Mono', 'Fira Code', monospace;
             font-size: 12px; line-height: 1.8; white-space: pre-wrap; word-break: break-all; }
.token { padding: 1px 0; border-radius: 2px; cursor: help; border-bottom: 1px solid rgba(0,0,0,0.1); }
.el-table { border-collapse: collapse; margin: 8px 0; font-size: 12px; }
.el-table th { background: #37474f; color: white; padding: 6px 10px; text-align: left; }
.el-table td { padding: 4px 10px; border-bottom: 1px solid #eee; font-family: monospace; }
details { margin: 8px 0; }
summary { cursor: pointer; color: #1976d2; font-size: 13px; }
.histogram { background: white; padding: 12px; border-radius: 8px; margin: 8px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.test-case { margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #e0e0e0; }
.screenshots { margin: 12px 0; }
.screenshot-pair { display: flex; gap: 16px; }
.screenshot-col { flex: 1; min-width: 0; }
.screenshot-col img { width: 100%; border: 1px solid #ccc; border-radius: 6px; display: block; }
.screenshot-label { font-size: 12px; font-weight: 600; color: #555; margin-bottom: 4px; text-align: center; }
.diff-container { display: flex; gap: 8px; margin: 8px 0; }
.diff-col { flex: 1; min-width: 0; }
.diff-label { font-size: 12px; font-weight: 600; color: #555; margin-bottom: 4px; }
.diff-code { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; margin: 0;
             font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 11px;
             line-height: 1.5; overflow-x: auto; white-space: pre; }
.diff-del { background: rgba(255,80,80,0.3); color: #ff9999; text-decoration: line-through; }
.diff-ins { background: rgba(80,200,80,0.3); color: #99ff99; }
</style></head><body>
<h1>Token-Level Rewards Report</h1>
<p>Hover over tokens to see their reward values. Green = high reward (good), Red = low reward (bad).</p>
"""]

    for title, result, elapsed, ground_truth in test_results:
        parts.append(f'<div class="test-case">')
        parts.append(f'<div class="stats"><span>Computation time: <b>{elapsed:.2f}s</b></span></div>')
        parts.append(generate_token_html(result, title, ground_truth))
        parts.append(f'</div>')

    parts.append('</body></html>')
    return '\n'.join(parts)


# ── Test cases ────────────────────────────────────────────────────────

TOY_CASES = [
    {
        "name": "Identical HTML",
        "ground_truth": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 20px; background: #f0f0f0; }
.box { width: 100px; height: 100px; background: #ff6b6b; border-radius: 8px; }
</style></head><body>
<div class="box"></div>
</body></html>""",
        "model_output": None,  # Same as ground truth
    },
    {
        "name": "Wrong color (one element)",
        "ground_truth": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 20px; background: #f0f0f0; }
.container { display: flex; gap: 10px; }
.box { width: 100px; height: 100px; border-radius: 8px; }
.box-1 { background: #ff6b6b; }
.box-2 { background: #4ecdc4; }
.box-3 { background: #45b7d1; }
</style></head><body>
<div class="container">
  <div class="box box-1"></div>
  <div class="box box-2"></div>
  <div class="box box-3"></div>
</div>
</body></html>""",
        "model_output": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 20px; background: #f0f0f0; }
.container { display: flex; gap: 10px; }
.box { width: 100px; height: 100px; border-radius: 8px; }
.box-1 { background: #ff6b6b; }
.box-2 { background: #ff0000; }
.box-3 { background: #45b7d1; }
</style></head><body>
<div class="container">
  <div class="box box-1"></div>
  <div class="box box-2"></div>
  <div class="box box-3"></div>
</div>
</body></html>""",
    },
    {
        "name": "Wrong layout (flex-direction changed)",
        "ground_truth": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 20px; background: #ffffff; }
.container { display: flex; flex-direction: row; gap: 10px; }
.box { width: 80px; height: 80px; }
.a { background: #e74c3c; }
.b { background: #3498db; }
.c { background: #2ecc71; }
</style></head><body>
<div class="container">
  <div class="box a"></div>
  <div class="box b"></div>
  <div class="box c"></div>
</div>
</body></html>""",
        "model_output": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 20px; background: #ffffff; }
.container { display: flex; flex-direction: column; gap: 10px; }
.box { width: 80px; height: 80px; }
.a { background: #e74c3c; }
.b { background: #3498db; }
.c { background: #2ecc71; }
</style></head><body>
<div class="container">
  <div class="box a"></div>
  <div class="box b"></div>
  <div class="box c"></div>
</div>
</body></html>""",
    },
    {
        "name": "Multiple errors (color + size + missing element)",
        "ground_truth": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 20px; background: #1a1a2e; }
.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.card { background: #16213e; border-radius: 12px; padding: 24px; color: white; }
.card h2 { margin: 0; font-size: 18px; color: #e94560; }
.card p { margin: 8px 0 0; font-size: 14px; color: #a8a8b3; }
</style></head><body>
<div class="grid">
  <div class="card"><h2>Revenue</h2><p>$1,234,567</p></div>
  <div class="card"><h2>Users</h2><p>45,678</p></div>
  <div class="card"><h2>Orders</h2><p>234</p></div>
  <div class="card"><h2>Traffic</h2><p>1.2M views</p></div>
</div>
</body></html>""",
        "model_output": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 20px; background: #1a1a2e; }
.grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.card { background: #ff0000; border-radius: 4px; padding: 24px; color: white; }
.card h2 { margin: 0; font-size: 24px; color: #ffffff; }
.card p { margin: 8px 0 0; font-size: 14px; color: #a8a8b3; }
</style></head><body>
<div class="grid">
  <div class="card"><h2>Revenue</h2><p>$1,234,567</p></div>
  <div class="card"><h2>Users</h2><p>45,678</p></div>
  <div class="card"><h2>Orders</h2><p>234</p></div>
</div>
</body></html>""",
    },
]


# ── Group test: G=2 completions for group-relative advantage ──────────

GROUP_CASE = {
    "name": "Group advantage: Wrong layout (G=2)",
    "ground_truth": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 20px; background: #ffffff; }
.container { display: flex; flex-direction: row; gap: 10px; }
.box { width: 80px; height: 80px; }
.a { background: #e74c3c; }
.b { background: #3498db; }
.c { background: #2ecc71; }
</style></head><body>
<div class="container">
  <div class="box a"></div>
  <div class="box b"></div>
  <div class="box c"></div>
</div>
</body></html>""",
    "completions": [
        {
            "label": "Completion 1: wrong flex-direction",
            "output": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 20px; background: #ffffff; }
.container { display: flex; flex-direction: column; gap: 10px; }
.box { width: 80px; height: 80px; }
.a { background: #e74c3c; }
.b { background: #3498db; }
.c { background: #2ecc71; }
</style></head><body>
<div class="container">
  <div class="box a"></div>
  <div class="box b"></div>
  <div class="box c"></div>
</div>
</body></html>""",
        },
        {
            "label": "Completion 2: correct layout, wrong color on .b",
            "output": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 20px; background: #ffffff; }
.container { display: flex; flex-direction: row; gap: 10px; }
.box { width: 80px; height: 80px; }
.a { background: #e74c3c; }
.b { background: #ff0000; }
.c { background: #2ecc71; }
</style></head><body>
<div class="container">
  <div class="box a"></div>
  <div class="box b"></div>
  <div class="box c"></div>
</div>
</body></html>""",
        },
    ],
}


def main():
    parser = argparse.ArgumentParser(description='Test token-level rewards')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha for reward formula')
    parser.add_argument('--dataset', action='store_true', help='Include dataset samples')
    parser.add_argument('--n-dataset', type=int, default=3, help='Number of dataset samples')
    args = parser.parse_args()

    from utils.token_rewards import compute_token_rewards

    test_results = []

    # Run toy cases
    for case in TOY_CASES:
        gt = case['ground_truth']
        pred = case['model_output'] if case['model_output'] is not None else gt
        name = case['name']

        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")

        start = time.perf_counter()
        result = compute_token_rewards(
            model_output=pred,
            ground_truth=gt,
            alpha=args.alpha,
        )
        elapsed = time.perf_counter() - start

        print(f"  Overall LPIPS: {result.overall_loss:.4f}")
        print(f"  Elements found: {len(result.element_scores)}")
        mapped = sum(1 for es in result.element_scores if es.model_char_start is not None)
        print(f"  Elements mapped: {mapped}")
        print(f"  CSS mappings: {result.css_mappings_count}")
        print(f"  Char rewards: min={min(result.char_rewards):.4f}, "
              f"max={max(result.char_rewards):.4f}, "
              f"mean={sum(result.char_rewards)/len(result.char_rewards):.4f}")
        print(f"  Time: {elapsed:.2f}s")

        test_results.append((name, result, elapsed, gt))

    # Dataset samples
    if args.dataset:
        try:
            from datasets import load_dataset
            ds = load_dataset("tcz/rb-box-layouts", split="easy")
            ds = ds.shuffle(seed=42)

            for i in range(min(args.n_dataset, len(ds))):
                sample = ds[i]
                gt_markup = sample['markup']

                # Create a corrupted version (simulate model output)
                import random
                random.seed(i + 100)
                pred_markup = gt_markup
                # Swap a few colors
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ff0000', '#00ff00', '#0000ff',
                          '#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
                for _ in range(3):
                    old_color = random.choice(colors)
                    new_color = random.choice(colors)
                    pred_markup = pred_markup.replace(old_color, new_color, 1)

                name = f"Dataset sample {i+1} (easy, corrupted colors)"
                print(f"\n{'='*60}")
                print(f"Test: {name}")
                print(f"{'='*60}")

                start = time.perf_counter()
                result = compute_token_rewards(
                    model_output=pred_markup,
                    ground_truth=gt_markup,
                    alpha=args.alpha,
                )
                elapsed = time.perf_counter() - start

                print(f"  Overall LPIPS: {result.overall_loss:.4f}")
                print(f"  Elements: {len(result.element_scores)}, mapped: "
                      f"{sum(1 for es in result.element_scores if es.model_char_start is not None)}")
                print(f"  Time: {elapsed:.2f}s")

                test_results.append((name, result, elapsed, gt_markup))

        except Exception as e:
            print(f"Could not load dataset: {e}")

    # ── Group advantage test (G=2) ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"GROUP TEST: {GROUP_CASE['name']}")
    print(f"{'='*60}")

    gt = GROUP_CASE['ground_truth']
    group_results = []
    for comp in GROUP_CASE['completions']:
        print(f"\n  --- {comp['label']} ---")
        start = time.perf_counter()
        result = compute_token_rewards(
            model_output=comp['output'],
            ground_truth=gt,
            alpha=args.alpha,
        )
        elapsed = time.perf_counter() - start
        print(f"  Overall LPIPS: {result.overall_loss:.4f}")
        print(f"  Char rewards: min={min(result.char_rewards):.4f}, "
              f"max={max(result.char_rewards):.4f}, "
              f"mean={sum(result.char_rewards)/len(result.char_rewards):.4f}")
        print(f"  Time: {elapsed:.2f}s")
        group_results.append((comp['label'], result, elapsed))

    # Compute group advantage: pool all char rewards across completions
    all_rewards = []
    for _, res, _ in group_results:
        all_rewards.extend(res.char_rewards)
    group_mu = sum(all_rewards) / len(all_rewards)
    group_sigma = (sum((r - group_mu) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5

    print(f"\n  Group stats: mu={group_mu:.4f}, sigma={group_sigma:.4f}")
    for label, res, _ in group_results:
        rewards = res.char_rewards
        mean_adv = sum((r - group_mu) / group_sigma for r in rewards) / len(rewards) if group_sigma > 1e-8 else 0
        print(f"  {label}: mean group advantage = {mean_adv:+.4f}")

    # Generate report
    report = generate_report(test_results, group_data=(GROUP_CASE, group_results, group_mu, group_sigma))
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token_rewards_report.html')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
