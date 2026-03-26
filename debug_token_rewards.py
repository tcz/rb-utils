"""
Debug visualization for token-level rewards using the real model.

Connects to a remote vLLM server (via SSH) to generate markup from a dataset
sample, then runs token-level reward computation locally (Playwright + LPIPS +
CDP CSS mapping + rapidfuzz alignment) and produces an HTML report showing
per-character and per-element reward attribution.

Prerequisites:
    - SSH access to the remote server (key-based auth)
    - vLLM server running on the remote machine (default port 8888)
    - Local: playwright, lpips, rapidfuzz, datasets, PIL

Usage (run from training/ directory):
    # Basic: generate from hard split, sample at offset 5 in the RL reserve
    python -m utils.debug_token_rewards hard 5

    # Custom remote server
    python -m utils.debug_token_rewards hard 5 --host 174.78.228.101 --ssh-port 40132

    # Change alpha (element-to-overall loss blend)
    python -m utils.debug_token_rewards easy 0 --alpha 0.3

    # Generate multiple completions for group advantage comparison
    python -m utils.debug_token_rewards hard 5 --num-generations 3

    # Custom output path
    python -m utils.debug_token_rewards hard 5 --output /tmp/debug.html

The report includes:
    - Input image from the dataset
    - Side-by-side screenshots (ground truth rendering vs model output rendering)
    - Markup diff (ground truth vs model output)
    - Element scores table (selector, bounds, LPIPS, reward, advantage)
    - Color-coded per-character rewards (green=good, red=bad, blue borders=element boundaries)
    - Reward distribution histogram
    - With --num-generations >1: group-relative advantages across completions

Output: debug_token_rewards_report.html (open in browser)
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


INSTRUCTION = "Generate the HTML and CSS markup that produces this webpage layout."
DATASET_NAME = "tcz/rb-box-layouts"
SEED = 3407
VLLM_PORT = 8888
VIEWPORT_WIDTH = 1024
VIEWPORT_HEIGHT = 1024


def load_dataset_sample(split: str, offset: int):
    """Load a specific sample from the dataset (RL reserve = second 50%)."""
    from datasets import load_dataset
    raw = load_dataset(DATASET_NAME, split=split)
    ds = raw.shuffle(seed=SEED)
    n = len(ds)
    rl_boundary = n // 2
    # RL reserve is the second half
    rl_idx = rl_boundary + offset
    if rl_idx >= n:
        print(f"Warning: offset {offset} exceeds RL reserve size ({n - rl_boundary}). "
              f"Using offset {offset % (n - rl_boundary)} instead.")
        rl_idx = rl_boundary + (offset % (n - rl_boundary))
    sample = ds[rl_idx]
    return sample


def generate_on_remote(split: str, offset: int, ssh_host: str, ssh_port: int,
                       vllm_port: int, num_generations: int = 1) -> list[str]:
    """Generate markup by running a Python script on the remote server.

    TRL's vLLM server uses a custom API (not OpenAI-compatible), so we run
    a script on the remote that loads the dataset, tokenizes the prompt with
    the processor, and calls TRL's VLLMClient.generate() directly.

    Returns list of generated markup strings.
    """
    # Python script to run on remote
    remote_script = f'''
import json, sys, warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import AutoProcessor
from trl.generation.vllm_client import VLLMClient

SEED = {SEED}
INSTRUCTION = "{INSTRUCTION}"
split = "{split}"
offset = {offset}
num_gen = {num_generations}
vllm_port = {vllm_port}

# Load dataset sample (same logic as local)
raw = load_dataset("{DATASET_NAME}", split=split)
ds = raw.shuffle(seed=SEED)
n = len(ds)
rl_boundary = n // 2
rl_idx = rl_boundary + offset
if rl_idx >= n:
    rl_idx = rl_boundary + (offset % (n - rl_boundary))
sample = ds[rl_idx]

# Build prompt text using processor
processor = AutoProcessor.from_pretrained("tcz/qwen3-vl-8b-box-layouts-sft-v2-900")
messages = [{{"role": "user", "content": [
    {{"type": "image"}},
    {{"type": "text", "text": INSTRUCTION}},
]}}]
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

pil_image = sample["image"]

# Call TRL's vLLM server — prompts must be strings, not token IDs
client = VLLMClient(f"http://localhost:{{vllm_port}}")
result = client.generate(
    prompts=[prompt_text] * num_gen,
    n=1,
    max_tokens=8192,
    temperature=0.7,
    images=[pil_image] * num_gen,
)

# Decode completions
completion_ids = result["completion_ids"]
completions = processor.batch_decode(completion_ids, skip_special_tokens=True)

# Output as JSON
print(json.dumps(completions))
'''

    # Write script to temp file and SCP to remote
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(remote_script)
        script_path = f.name

    remote_script_path = f"/tmp/debug_gen_{os.getpid()}.py"

    try:
        ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30"]
        subprocess.run(
            ["scp", "-P", str(ssh_port)] + ssh_opts + [script_path,
             f"root@{ssh_host}:{remote_script_path}"],
            check=True, capture_output=True, timeout=60
        )

        result = subprocess.run(
            ["ssh", "-p", str(ssh_port)] + ssh_opts + [f"root@{ssh_host}",
             f"python {remote_script_path} && rm -f {remote_script_path}"],
            capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            # stderr may be flooded with token IDs from VLLMClient debug output;
            # show stdout too as it may contain the actual traceback
            raise RuntimeError(
                f"Remote generation failed (exit code {result.returncode}):\n"
                f"stdout (last 500): {result.stdout[-500:]}\n"
                f"stderr (last 500): {result.stderr[-500:]}"
            )

        # The last line of stdout should be the JSON array
        # (earlier lines may have warnings/prints from library imports)
        lines = result.stdout.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('['):
                try:
                    completions = json.loads(line)
                    return completions
                except json.JSONDecodeError:
                    continue

        raise RuntimeError(f"Could not parse completions from remote output:\n"
                           f"{result.stdout[-1000:]}")

    finally:
        os.unlink(script_path)


def run_token_rewards(model_output: str, ground_truth: str, alpha: float):
    """Run token-level reward computation locally."""
    from utils.token_rewards import compute_token_rewards
    return compute_token_rewards(
        model_output=model_output,
        ground_truth=ground_truth,
        viewport_width=VIEWPORT_WIDTH,
        viewport_height=VIEWPORT_HEIGHT,
        alpha=alpha,
    )


def generate_debug_report(results, split, offset, alpha):
    """Generate the full HTML debug report.

    results: list of (label, TokenRewardResult, elapsed, ground_truth) tuples
    """
    from utils.test_token_rewards import (
        generate_token_html, generate_group_html, generate_report
    )

    # If multiple completions, show group advantage
    if len(results) > 1:
        # Compute group stats
        all_rewards = []
        for _, result, _, _ in results:
            all_rewards.extend(result.char_rewards)
        group_mu = sum(all_rewards) / len(all_rewards)
        group_sigma = (sum((r - group_mu) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5

        # Build the group case structure for generate_group_html
        group_results = [(label, result, elapsed) for label, result, elapsed, _ in results]
        group_case = {"name": f"Token-Level Debug: {split}[{offset}] (G={len(results)})"}

        parts = [_report_header(split, offset, alpha)]
        parts.append(generate_group_html(group_case, group_results, group_mu, group_sigma))
        parts.append('</body></html>')
        return '\n'.join(parts)
    else:
        # Single completion: use generate_report
        return generate_report(results)


def _report_header(split, offset, alpha):
    """Generate the HTML header for the report."""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Token-Level Debug: {split}[{offset}]</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #fafafa; }}
h1 {{ color: #333; }}
h2 {{ color: #555; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }}
h3 {{ color: #444; margin-top: 24px; }}
h4 {{ color: #666; margin-top: 16px; font-size: 14px; }}
.stats {{ background: white; padding: 12px 16px; border-radius: 8px; margin: 8px 0;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 13px; }}
.stats span {{ margin-right: 20px; }}
.stats b {{ color: #333; }}
.token-vis {{ background: white; padding: 12px; border-radius: 8px; margin: 8px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-family: 'SF Mono', 'Fira Code', monospace;
             font-size: 12px; line-height: 1.8; white-space: pre-wrap; word-break: break-all; }}
.token {{ padding: 1px 0; border-radius: 2px; cursor: help; border-bottom: 1px solid rgba(0,0,0,0.1); }}
.el-table {{ border-collapse: collapse; margin: 8px 0; font-size: 12px; }}
.el-table th {{ background: #37474f; color: white; padding: 6px 10px; text-align: left; }}
.el-table td {{ padding: 4px 10px; border-bottom: 1px solid #eee; font-family: monospace; }}
details {{ margin: 8px 0; }}
summary {{ cursor: pointer; color: #1976d2; font-size: 13px; }}
.histogram {{ background: white; padding: 12px; border-radius: 8px; margin: 8px 0;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.test-case {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #e0e0e0; }}
.screenshots {{ margin: 12px 0; }}
.screenshot-pair {{ display: flex; gap: 16px; }}
.screenshot-col {{ flex: 1; min-width: 0; }}
.screenshot-col img {{ width: 100%; border: 1px solid #ccc; border-radius: 6px; display: block; }}
.screenshot-label {{ font-size: 12px; font-weight: 600; color: #555; margin-bottom: 4px; text-align: center; }}
.diff-container {{ display: flex; gap: 8px; margin: 8px 0; }}
.diff-col {{ flex: 1; min-width: 0; }}
.diff-label {{ font-size: 12px; font-weight: 600; color: #555; margin-bottom: 4px; }}
.diff-code {{ background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; margin: 0;
             font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 11px;
             line-height: 1.5; overflow-x: auto; white-space: pre; }}
.diff-del {{ background: rgba(255,80,80,0.3); color: #ff9999; text-decoration: line-through; }}
.diff-ins {{ background: rgba(80,200,80,0.3); color: #99ff99; }}
.input-image {{ margin: 12px 0; }}
.input-image img {{ max-width: 512px; border: 1px solid #ccc; border-radius: 6px; }}
</style></head><body>
<h1>Token-Level Debug: {split}[{offset}]</h1>
<p>Alpha={alpha}. Hover over tokens to see reward values. Green=good, Red=bad.
Blue borders mark element boundaries.</p>
"""


def main():
    parser = argparse.ArgumentParser(
        description='Debug token-level rewards with real model output')
    parser.add_argument('split', choices=['easy', 'medium', 'hard'],
                        help='Dataset split')
    parser.add_argument('offset', type=int,
                        help='Sample offset within RL reserve')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha for reward formula (default: 0.5)')
    parser.add_argument('--host', type=str, default='174.78.228.101',
                        help='SSH host for remote server')
    parser.add_argument('--ssh-port', type=int, default=40132,
                        help='SSH port')
    parser.add_argument('--vllm-port', type=int, default=VLLM_PORT,
                        help='vLLM server port on remote')
    parser.add_argument('--num-generations', type=int, default=1,
                        help='Number of completions to generate (for group comparison)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML path (default: debug_token_rewards_report.html)')
    args = parser.parse_args()

    print(f"Loading dataset sample: {args.split}[{args.offset}]...")
    sample = load_dataset_sample(args.split, args.offset)
    ground_truth = sample['markup']
    image = sample['image']  # PIL Image

    # Convert image to base64 for the report
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    input_image_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    print(f"Generating {args.num_generations} completion(s) via remote vLLM "
          f"({args.host}:{args.ssh_port} -> localhost:{args.vllm_port})...")
    t0 = time.perf_counter()
    completions = generate_on_remote(
        split=args.split,
        offset=args.offset,
        ssh_host=args.host,
        ssh_port=args.ssh_port,
        vllm_port=args.vllm_port,
        num_generations=args.num_generations,
    )
    gen_time = time.perf_counter() - t0
    print(f"Generation done in {gen_time:.1f}s. Got {len(completions)} completion(s).")
    for i, c in enumerate(completions):
        print(f"  Completion {i+1}: {len(c)} chars")

    # Run token-level rewards for each completion
    results = []
    for i, completion in enumerate(completions):
        label = f"Completion {i+1}" if len(completions) > 1 else f"{args.split}[{args.offset}]"
        print(f"\nComputing token rewards for {label}...")
        t0 = time.perf_counter()
        result = run_token_rewards(completion, ground_truth, args.alpha)
        elapsed = time.perf_counter() - t0
        print(f"  Overall LPIPS: {result.overall_loss:.4f}")
        print(f"  Overall Similarity: {result.overall_similarity:.4f}")
        print(f"  Elements: {len(result.element_scores)}, "
              f"mapped: {sum(1 for e in result.element_scores if e.model_char_start is not None)}")
        print(f"  CSS mappings: {result.css_mappings_count}")
        print(f"  Char rewards: min={min(result.char_rewards):.4f}, "
              f"max={max(result.char_rewards):.4f}")
        print(f"  Time: {elapsed:.1f}s")
        results.append((label, result, elapsed, ground_truth))

    # Generate report
    if len(results) > 1:
        report_html = generate_debug_report(results, args.split, args.offset, args.alpha)
    else:
        # Single completion: use generate_report from test_token_rewards
        from utils.test_token_rewards import generate_report
        report_html = generate_report(results)

    # Inject input image at the top of the report (after <body> and first <h1>)
    input_img_html = (
        f'<div class="input-image">'
        f'<h3>Input Image ({args.split}[{args.offset}])</h3>'
        f'<img src="data:image/png;base64,{input_image_b64}"></div>'
    )
    # Insert after the first </h1> or </p> tag
    insert_markers = ['</p>\n', '</h1>\n']
    for marker in insert_markers:
        idx = report_html.find(marker)
        if idx >= 0:
            insert_pos = idx + len(marker)
            report_html = report_html[:insert_pos] + input_img_html + report_html[insert_pos:]
            break

    # Write report
    output_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'debug_token_rewards_report.html'
    )
    with open(output_path, 'w') as f:
        f.write(report_html)
    print(f"\nReport saved to: {output_path}")
    print(f"Open in browser to inspect token-level rewards.")


if __name__ == '__main__':
    main()
