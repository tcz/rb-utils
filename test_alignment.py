"""
Test rapidfuzz Levenshtein opcodes for HTML source mapping.

Generates pairs of HTML strings (original + corrupted) with increasing noise levels,
runs alignment, and produces a visual HTML report showing matched segments.

Usage: python test_alignment.py
Output: alignment_report.html (open in browser)
"""
import random
import re
import time
import html as html_module
from rapidfuzz.distance import Levenshtein


# ── Corruption functions ──────────────────────────────────────────────

def add_whitespace(text, intensity=0.1):
    """Insert random whitespace/newlines."""
    chars = list(text)
    n_insertions = int(len(chars) * intensity)
    for _ in range(n_insertions):
        pos = random.randint(0, len(chars))
        ws = random.choice([' ', '  ', '\n', '\n  ', '\t'])
        chars.insert(pos, ws)
    return ''.join(chars)


def remove_closing_tags(text, fraction=0.3):
    """Remove a fraction of closing tags."""
    closing_tags = list(re.finditer(r'</\w+>', text))
    if not closing_tags:
        return text
    n_remove = max(1, int(len(closing_tags) * fraction))
    to_remove = random.sample(closing_tags, min(n_remove, len(closing_tags)))
    to_remove.sort(key=lambda m: m.start(), reverse=True)
    result = text
    for match in to_remove:
        result = result[:match.start()] + result[match.end():]
    return result


def remove_quotes(text, fraction=0.5):
    """Remove quotes from attribute values (e.g., class="foo" -> class=foo)."""
    def strip_quotes(m):
        if random.random() < fraction:
            return f'{m.group(1)}={m.group(3)}'
        return m.group(0)
    return re.sub(r'(\w+)=((["\']))(.*?)\3', strip_quotes, text)


def change_case(text, fraction=0.2):
    """Randomly change case of tag names."""
    def case_swap(m):
        tag = m.group(1)
        if random.random() < fraction:
            return f'<{tag.upper() if random.random() > 0.5 else tag.lower()}'
        return m.group(0)
    return re.sub(r'<(/?\w+)', case_swap, text)


def add_syntax_errors(text, intensity=0.05):
    """Insert random characters (typos, stray brackets, etc.)."""
    chars = list(text)
    n_errors = max(1, int(len(chars) * intensity))
    junk = list('><;:!@#$%&*(){}[]|\\~`')
    for _ in range(n_errors):
        pos = random.randint(0, len(chars) - 1)
        action = random.choice(['insert', 'delete', 'replace'])
        if action == 'insert':
            chars.insert(pos, random.choice(junk))
        elif action == 'delete' and len(chars) > 10:
            chars.pop(pos)
        else:
            chars[pos] = random.choice(junk)
    return ''.join(chars)


def shuffle_attributes(text):
    """Reorder attributes within tags."""
    def reorder_attrs(m):
        tag_name = m.group(1)
        attrs_str = m.group(2)
        attrs = re.findall(r'(\w+(?:=["\']?[^"\'>\s]*["\']?)?)', attrs_str)
        if len(attrs) > 1:
            random.shuffle(attrs)
        return f'<{tag_name} {" ".join(attrs)}'
    return re.sub(r'<(\w+)((?:\s+\w+(?:=["\']?[^"\'>\s]*["\']?)?)+)', reorder_attrs, text)


# ── Noise levels ──────────────────────────────────────────────────────

NOISE_LEVELS = [
    {
        "name": "Level 0: Identical",
        "description": "No corruption — baseline",
        "corrupt": lambda t: t,
    },
    {
        "name": "Level 1: Whitespace only",
        "description": "Extra spaces, newlines, tabs inserted",
        "corrupt": lambda t: add_whitespace(t, 0.05),
    },
    {
        "name": "Level 2: Whitespace + case changes",
        "description": "Whitespace + random tag case swaps",
        "corrupt": lambda t: change_case(add_whitespace(t, 0.05), 0.3),
    },
    {
        "name": "Level 3: Missing closing tags",
        "description": "Remove ~30% of closing tags",
        "corrupt": lambda t: remove_closing_tags(add_whitespace(t, 0.03), 0.3),
    },
    {
        "name": "Level 4: Missing quotes + closing tags",
        "description": "Remove quotes from attributes + some closing tags",
        "corrupt": lambda t: remove_closing_tags(remove_quotes(add_whitespace(t, 0.03), 0.5), 0.2),
    },
    {
        "name": "Level 5: Light syntax errors",
        "description": "All above + 2% random character corruption",
        "corrupt": lambda t: add_syntax_errors(
            remove_closing_tags(remove_quotes(change_case(add_whitespace(t, 0.03), 0.2), 0.3), 0.2),
            0.02
        ),
    },
    {
        "name": "Level 6: Heavy syntax errors",
        "description": "All above + 5% random character corruption",
        "corrupt": lambda t: add_syntax_errors(
            remove_closing_tags(remove_quotes(change_case(add_whitespace(t, 0.05), 0.3), 0.5), 0.3),
            0.05
        ),
    },
    {
        "name": "Level 7: Extreme corruption",
        "description": "All above + 10% character corruption + attribute shuffle",
        "corrupt": lambda t: add_syntax_errors(
            shuffle_attributes(
                remove_closing_tags(remove_quotes(change_case(add_whitespace(t, 0.08), 0.4), 0.6), 0.4)
            ),
            0.10
        ),
    },
]


# ── Test HTML samples ─────────────────────────────────────────────────

SAMPLES = [
    {
        "name": "Simple page",
        "html": '<html><head><title>Hello</title></head><body><p>Hello world</p></body></html>',
    },
    {
        "name": "Box layout (small)",
        "html": """<!DOCTYPE html>
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
    },
    {
        "name": "Box layout (medium)",
        "html": """<!DOCTYPE html>
<html><head><style>
body { margin: 0; padding: 0; background-color: #1a1a2e; font-family: Arial, sans-serif; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding: 32px; }
.card { background: #16213e; border-radius: 12px; padding: 24px; color: white; }
.card h2 { margin: 0 0 12px; font-size: 18px; color: #e94560; }
.card p { margin: 0; font-size: 14px; line-height: 1.6; color: #a8a8b3; }
.card.featured { grid-column: span 2; background: linear-gradient(135deg, #0f3460, #16213e); border: 1px solid #e94560; }
.header { text-align: center; padding: 48px 32px 16px; color: white; }
.header h1 { font-size: 36px; margin: 0; }
.header p { color: #a8a8b3; margin: 8px 0 0; }
.footer { text-align: center; padding: 32px; color: #a8a8b3; font-size: 12px; }
</style></head><body>
<div class="header">
  <h1>Dashboard</h1>
  <p>Overview of your analytics</p>
</div>
<div class="grid">
  <div class="card featured">
    <h2>Revenue</h2>
    <p>Total revenue this quarter: $1,234,567. Growth rate: 12.5% compared to last quarter.</p>
  </div>
  <div class="card">
    <h2>Users</h2>
    <p>Active users: 45,678</p>
  </div>
  <div class="card">
    <h2>Orders</h2>
    <p>Pending orders: 234</p>
  </div>
  <div class="card">
    <h2>Traffic</h2>
    <p>Page views: 1.2M</p>
  </div>
  <div class="card">
    <h2>Conversion</h2>
    <p>Rate: 3.4%</p>
  </div>
  <div class="card featured">
    <h2>Performance</h2>
    <p>Server uptime: 99.97%. Average response time: 142ms. Error rate: 0.03%.</p>
  </div>
</div>
<div class="footer">© 2026 Analytics Corp. All rights reserved.</div>
</body></html>""",
    },
    {
        "name": "Nested layout (complex)",
        "html": """<!DOCTYPE html>
<html><head><style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #fafafa; font-family: 'Segoe UI', sans-serif; }
.sidebar { position: fixed; left: 0; top: 0; width: 240px; height: 100vh; background: #2d3436; padding: 20px; }
.sidebar .logo { color: #fff; font-size: 24px; font-weight: bold; margin-bottom: 40px; }
.sidebar .nav-item { display: block; color: #b2bec3; padding: 12px 16px; border-radius: 8px; margin-bottom: 4px; text-decoration: none; font-size: 14px; }
.sidebar .nav-item.active { background: #6c5ce7; color: white; }
.main { margin-left: 240px; padding: 32px; }
.topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 32px; }
.topbar h1 { font-size: 28px; color: #2d3436; }
.topbar .avatar { width: 40px; height: 40px; border-radius: 50%; background: #6c5ce7; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 32px; }
.stat-card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.stat-card .label { font-size: 12px; color: #636e72; text-transform: uppercase; letter-spacing: 1px; }
.stat-card .value { font-size: 32px; font-weight: bold; color: #2d3436; margin: 8px 0; }
.stat-card .change { font-size: 13px; color: #00b894; }
.chart-area { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); height: 300px; }
</style></head><body>
<div class="sidebar">
  <div class="logo">AppName</div>
  <a class="nav-item active" href="#">Dashboard</a>
  <a class="nav-item" href="#">Analytics</a>
  <a class="nav-item" href="#">Users</a>
  <a class="nav-item" href="#">Settings</a>
</div>
<div class="main">
  <div class="topbar">
    <h1>Dashboard</h1>
    <div class="avatar"></div>
  </div>
  <div class="stats">
    <div class="stat-card">
      <div class="label">Revenue</div>
      <div class="value">$48.2K</div>
      <div class="change">+12.5%</div>
    </div>
    <div class="stat-card">
      <div class="label">Users</div>
      <div class="value">2,847</div>
      <div class="change">+8.3%</div>
    </div>
    <div class="stat-card">
      <div class="label">Orders</div>
      <div class="value">1,234</div>
      <div class="change">+15.2%</div>
    </div>
  </div>
  <div class="chart-area"></div>
</div>
</body></html>""",
    },
]


# ── Alignment + Report ────────────────────────────────────────────────

def run_alignment(original: str, corrupted: str):
    """Run Levenshtein alignment and return opcodes + timing."""
    start = time.perf_counter()
    ops = Levenshtein.opcodes(original, corrupted)
    elapsed = time.perf_counter() - start
    return ops, elapsed


def compute_stats(ops, original, corrupted):
    """Compute alignment statistics from opcodes."""
    equal_chars = 0
    replaced_chars = 0
    deleted_chars = 0
    inserted_chars = 0

    for op in ops:
        src_len = op.src_end - op.src_start
        dest_len = op.dest_end - op.dest_start
        if op.tag == 'equal':
            equal_chars += src_len
        elif op.tag == 'replace':
            replaced_chars += max(src_len, dest_len)
        elif op.tag == 'delete':
            deleted_chars += src_len
        elif op.tag == 'insert':
            inserted_chars += dest_len

    total_original = len(original)
    coverage = (equal_chars / total_original * 100) if total_original > 0 else 0

    return {
        'equal': equal_chars,
        'replaced': replaced_chars,
        'deleted': deleted_chars,
        'inserted': inserted_chars,
        'total_original': total_original,
        'total_corrupted': len(corrupted),
        'coverage': coverage,
        'n_ops': len(ops),
    }


def opcodes_to_html_segments(ops, original, corrupted):
    """Convert opcodes to colored HTML segments showing the alignment."""
    rows = []
    for op in ops:
        src_text = original[op.src_start:op.src_end]
        dest_text = corrupted[op.dest_start:op.dest_end]

        if op.tag == 'equal':
            rows.append({
                'tag': 'equal',
                'original': html_module.escape(src_text),
                'corrupted': html_module.escape(dest_text),
            })
        elif op.tag == 'replace':
            rows.append({
                'tag': 'replace',
                'original': html_module.escape(src_text),
                'corrupted': html_module.escape(dest_text),
            })
        elif op.tag == 'delete':
            rows.append({
                'tag': 'delete',
                'original': html_module.escape(src_text),
                'corrupted': '',
            })
        elif op.tag == 'insert':
            rows.append({
                'tag': 'insert',
                'original': '',
                'corrupted': html_module.escape(dest_text),
            })

    return rows


def generate_report(results):
    """Generate an HTML report from alignment results."""
    tag_colors = {
        'equal': '#e8f5e9',
        'replace': '#fff3e0',
        'delete': '#ffebee',
        'insert': '#e3f2fd',
    }
    tag_labels = {
        'equal': 'MATCH',
        'replace': 'REPLACE',
        'delete': 'DELETE',
        'insert': 'INSERT',
    }

    html_parts = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Alignment Report</title>
<style>
body { font-family: 'SF Mono', 'Fira Code', monospace; margin: 20px; background: #fafafa; font-size: 13px; }
h1 { font-family: system-ui, sans-serif; color: #333; }
h2 { font-family: system-ui, sans-serif; color: #555; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
h3 { font-family: system-ui, sans-serif; color: #666; margin-top: 24px; }
.stats { font-family: system-ui, sans-serif; background: white; padding: 16px; border-radius: 8px;
         margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.stats span { margin-right: 20px; }
.stats .good { color: #2e7d32; font-weight: bold; }
.stats .warn { color: #e65100; font-weight: bold; }
.stats .bad { color: #c62828; font-weight: bold; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }
th { background: #37474f; color: white; padding: 10px 12px; text-align: left; font-family: system-ui, sans-serif; font-size: 12px; }
td { padding: 6px 12px; border-bottom: 1px solid #eee; vertical-align: top; white-space: pre-wrap;
     word-break: break-all; max-width: 600px; }
.tag { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px;
       font-weight: bold; font-family: system-ui, sans-serif; min-width: 55px; text-align: center; }
.tag-equal { background: #c8e6c9; color: #1b5e20; }
.tag-replace { background: #ffe0b2; color: #e65100; }
.tag-delete { background: #ffcdd2; color: #b71c1c; }
.tag-insert { background: #bbdefb; color: #0d47a1; }
.truncated { color: #999; font-style: italic; font-family: system-ui, sans-serif; }
.summary-table { margin: 20px 0; }
.summary-table td { font-family: system-ui, sans-serif; }
</style></head><body>
<h1>Rapidfuzz Levenshtein Alignment Report</h1>
<p style="font-family: system-ui, sans-serif; color: #666;">
Testing character-level alignment between original HTML and corrupted versions at increasing noise levels.
</p>
"""]

    # Summary table
    html_parts.append('<h2>Summary</h2>')
    html_parts.append('<table class="summary-table"><tr><th>Sample</th><th>Noise Level</th>'
                      '<th>Orig Len</th><th>Corrupt Len</th><th>Time</th>'
                      '<th>Coverage</th><th>Ops</th></tr>')

    for r in results:
        s = r['stats']
        cov = s['coverage']
        cov_class = 'good' if cov > 95 else ('warn' if cov > 80 else 'bad')
        html_parts.append(
            f'<tr><td>{r["sample_name"]}</td><td>{r["noise_name"]}</td>'
            f'<td>{s["total_original"]}</td><td>{s["total_corrupted"]}</td>'
            f'<td>{r["time_ms"]:.1f}ms</td>'
            f'<td class="{cov_class}">{cov:.1f}%</td>'
            f'<td>{s["n_ops"]}</td></tr>'
        )

    html_parts.append('</table>')

    # Detailed segments
    html_parts.append('<h2>Detailed Alignment Segments</h2>')

    for r in results:
        html_parts.append(f'<h3>{r["sample_name"]} — {r["noise_name"]}</h3>')

        s = r['stats']
        cov = s['coverage']
        cov_class = 'good' if cov > 95 else ('warn' if cov > 80 else 'bad')
        html_parts.append(
            f'<div class="stats">'
            f'<span>Coverage: <span class="{cov_class}">{cov:.1f}%</span></span>'
            f'<span>Equal: {s["equal"]}</span>'
            f'<span>Replaced: {s["replaced"]}</span>'
            f'<span>Deleted: {s["deleted"]}</span>'
            f'<span>Inserted: {s["inserted"]}</span>'
            f'<span>Time: {r["time_ms"]:.1f}ms</span>'
            f'</div>'
        )

        segments = r['segments']
        max_segments_to_show = 100

        html_parts.append('<table><tr><th style="width:60px">Type</th>'
                          '<th>Original</th><th>Corrupted</th></tr>')

        for i, seg in enumerate(segments[:max_segments_to_show]):
            tag = seg['tag']
            tag_class = f'tag-{tag}'
            bg = tag_colors[tag]
            label = tag_labels[tag]

            orig_text = seg['original'][:200]
            if len(seg['original']) > 200:
                orig_text += '<span class="truncated">... (truncated)</span>'

            corr_text = seg['corrupted'][:200]
            if len(seg['corrupted']) > 200:
                corr_text += '<span class="truncated">... (truncated)</span>'

            html_parts.append(
                f'<tr style="background:{bg}">'
                f'<td><span class="tag {tag_class}">{label}</span></td>'
                f'<td>{orig_text}</td>'
                f'<td>{corr_text}</td></tr>'
            )

        if len(segments) > max_segments_to_show:
            html_parts.append(
                f'<tr><td colspan="3" class="truncated">'
                f'... {len(segments) - max_segments_to_show} more segments</td></tr>'
            )

        html_parts.append('</table>')

    html_parts.append('</body></html>')
    return '\n'.join(html_parts)


def main():
    random.seed(42)
    results = []

    for sample in SAMPLES:
        original = sample['html']
        print(f"\n{'='*60}")
        print(f"Sample: {sample['name']} ({len(original)} chars)")
        print(f"{'='*60}")

        for level in NOISE_LEVELS:
            corrupted = level['corrupt'](original)

            ops, elapsed = run_alignment(original, corrupted)
            stats = compute_stats(ops, original, corrupted)
            segments = opcodes_to_html_segments(ops, original, corrupted)

            time_ms = elapsed * 1000

            cov = stats['coverage']
            indicator = '  ' if cov > 95 else (' !' if cov > 80 else ' !!')
            print(f"  {level['name']}: coverage={cov:.1f}%{indicator}  "
                  f"ops={stats['n_ops']}  time={time_ms:.1f}ms  "
                  f"eq={stats['equal']} rep={stats['replaced']} "
                  f"del={stats['deleted']} ins={stats['inserted']}")

            results.append({
                'sample_name': sample['name'],
                'noise_name': level['name'],
                'noise_desc': level['description'],
                'stats': stats,
                'segments': segments,
                'time_ms': time_ms,
            })

    # Generate report
    report_html = generate_report(results)
    report_path = 'alignment_report.html'
    with open(report_path, 'w') as f:
        f.write(report_html)

    print(f"\nReport saved to: {report_path}")
    print("Open in a browser to view the visual alignment.")


if __name__ == '__main__':
    main()
