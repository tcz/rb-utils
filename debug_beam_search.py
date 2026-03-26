"""
Debug visualization for beam search generation.

Runs an instrumented beam search (mock or real vLLM) on a dataset sample,
captures screenshots + rewards at each step, and outputs a JSON file that
can be visualized with beam_search_viewer.html.

Prerequisites:
    - Local: playwright, lpips, scipy, datasets, PIL
    - For vLLM mode: SSH tunnel to remote vLLM server

Usage (run from training/ directory):
    # Mock mode (no GPU needed — produces synthetic HTML elements):
    python -m utils.debug_beam_search easy 0 --mock --output beam_debug.json

    # vLLM mode (set up SSH tunnel first):
    #   ssh -L 8888:localhost:8888 -p 11092 root@80.188.223.202
    python -m utils.debug_beam_search easy 0 --vllm-url http://localhost:8888 \\
        --output beam_debug.json

    # Custom beam parameters:
    python -m utils.debug_beam_search easy 0 --mock --M 1 --K 3 --N 2 --max-steps 3

Output: JSON file for use with beam_search_viewer.html
"""
import argparse
import base64
import io
import json
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_NAME = "tcz/rb-box-layouts"
SEED = 3407
VIEWPORT_WIDTH = 1024
VIEWPORT_HEIGHT = 1024


def load_dataset_sample(split: str, offset: int):
    """Load a specific sample from the dataset (RL reserve = second 50%)."""
    from datasets import load_dataset
    raw = load_dataset(DATASET_NAME, split=split)
    ds = raw.shuffle(seed=SEED)
    n = len(ds)
    rl_boundary = n // 2
    rl_idx = rl_boundary + offset
    if rl_idx >= n:
        print(f"Warning: offset {offset} exceeds RL reserve size ({n - rl_boundary}). "
              f"Using offset {offset % (n - rl_boundary)} instead.")
        rl_idx = rl_boundary + (offset % (n - rl_boundary))
    return ds[rl_idx]


def numpy_to_base64(arr: np.ndarray) -> str:
    """Convert a (H, W, 3) uint8 numpy array to a base64-encoded PNG string."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def create_mock_generate_fn(seed=42):
    """Create a mock generate_fn that produces plausible inline-style HTML.

    Generates diverse colored div elements so different branches get different
    rewards when compared against the real reference image.
    """
    rng = random.Random(seed)
    colors = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#16a085", "#2980b9", "#8e44ad",
        "#c0392b", "#27ae60", "#d35400", "#2c3e50", "#7f8c8d",
    ]
    widths = [40, 60, 80, 100, 120, 150, 200]
    heights = [40, 60, 80, 100, 120]

    def generate(prefix, K):
        results = []
        open_divs = prefix.count("<div") - prefix.count("</div>")
        element_count = prefix.count("<div")

        for _ in range(K):
            # Small chance of EOS after several elements
            if element_count > 3 and rng.random() < 0.15:
                results.append("")
                continue

            if element_count == 0:
                # First element: flex container
                direction = rng.choice(["row", "column"])
                gap = rng.choice([5, 10, 15, 20])
                bg = rng.choice(["#ffffff", "#f5f5f5", "#f0f0f0", "#fafafa"])
                results.append(
                    f'<div style="display:flex;flex-direction:{direction};'
                    f'gap:{gap}px;padding:10px;background:{bg};">'
                )
            elif open_divs > 1 and rng.random() < 0.2:
                # Close a container and open a new one
                direction = rng.choice(["row", "column"])
                gap = rng.choice([5, 10, 15])
                results.append(
                    f'</div><div style="display:flex;flex-direction:{direction};'
                    f'gap:{gap}px;padding:5px;">'
                )
            elif open_divs > 0 and element_count > 4 and rng.random() < 0.3:
                # Close container(s) to wrap up
                results.append("</div>" * open_divs)
            else:
                # Leaf element with random styling
                color = rng.choice(colors)
                w = rng.choice(widths)
                h = rng.choice(heights)
                radius = rng.choice([0, 0, 0, 5, 10, 15])
                style = f"background:{color};width:{w}px;height:{h}px"
                if radius:
                    style += f";border-radius:{radius}px"
                results.append(f'<div style="{style};"></div>')

        return results

    return generate


def run_instrumented_beam_search(
    config_M, config_K, config_N, max_steps,
    generate_fn, reward_pool, reference_html, alpha, vw, vh,
):
    """Run beam search with screenshot capture at each step.

    Returns:
        beams_data: list of beam dicts with nodes
        gt_screenshot_b64: base64 of the ground truth rendered screenshot
    """
    from utils.hybrid_reward import compute_emd, compute_hybrid_reward

    gt_screenshot_b64 = None
    all_beams_data = []
    node_counter = 0
    sibling_group_counter = 0

    for beam_idx in range(config_M):
        nodes_data = []

        # Root node
        root_id = node_counter
        node_counter += 1
        root = {
            'id': root_id,
            'parent_id': None,
            'step': 0,
            'continuation': '',
            'full_text': '',
            'hybrid_reward': 0.0,
            'step_reward': 0.0,
            'lpips_score': None,
            'emd_similarity': None,
            'sibling_group_id': 0,
            'is_terminal': False,
            'survived': True,
            'screenshot_base64': None,
            'children_ids': [],
        }
        nodes_data.append(root)
        active_nodes = [root]

        for step in range(max_steps):
            all_children = []

            for parent in active_nodes:
                sibling_group_counter += 1
                continuations = generate_fn(parent['full_text'], config_K)

                for cont in continuations:
                    is_terminal = (cont == "")
                    child_id = node_counter
                    node_counter += 1

                    full_text = parent['full_text'] + cont
                    child = {
                        'id': child_id,
                        'parent_id': parent['id'],
                        'step': step + 1,
                        'continuation': cont,
                        'full_text': full_text,
                        'hybrid_reward': parent['hybrid_reward'] if is_terminal else 0.0,
                        'step_reward': 0.0,
                        'lpips_score': None,
                        'emd_similarity': None,
                        'sibling_group_id': sibling_group_counter,
                        'is_terminal': is_terminal,
                        'survived': False,
                        'screenshot_base64': None,
                        'children_ids': [],
                    }
                    parent['children_ids'].append(child_id)
                    nodes_data.append(child)
                    all_children.append(child)

            # Batch render all non-terminal children
            to_render = [c for c in all_children if not c['is_terminal']]
            if to_render:
                items = [
                    (c['full_text'], reference_html, vw, vh) for c in to_render
                ]
                print(f"  Beam {beam_idx}, step {step + 1}: "
                      f"rendering {len(items)} branches...")
                t0 = time.perf_counter()
                results = reward_pool.calculate_metrics_batch_intermediate(items)
                render_time = time.perf_counter() - t0
                print(f"    Rendered in {render_time:.1f}s")

                for child, r in zip(to_render, results):
                    if r is None:
                        # Render failed — treat as terminal
                        child['is_terminal'] = True
                        child['hybrid_reward'] = 0.0
                        continue

                    child['screenshot_base64'] = numpy_to_base64(r['pred_screenshot'])
                    if gt_screenshot_b64 is None:
                        gt_screenshot_b64 = numpy_to_base64(r['gt_screenshot'])

                    child['lpips_score'] = r['lpips_score']
                    emd_raw = compute_emd(
                        r['pred_screenshot'], r['gt_screenshot']
                    )
                    max_emd = 8 ** 3 - 1
                    child['emd_similarity'] = 1.0 - min(emd_raw / max_emd, 1.0)

                    child['hybrid_reward'] = compute_hybrid_reward(
                        r['pred_screenshot'], r['gt_screenshot'],
                        r['lpips_score'], alpha,
                    )

            # Compute step_reward for all children
            node_map = {n['id']: n for n in nodes_data}
            for child in all_children:
                parent = node_map[child['parent_id']]
                child['step_reward'] = child['hybrid_reward'] - parent['hybrid_reward']

            # Select top-N non-terminal by hybrid_reward
            non_terminal = [c for c in all_children if not c['is_terminal']]
            if not non_terminal:
                print(f"  Beam {beam_idx}: all terminal at step {step + 1}")
                break

            non_terminal.sort(key=lambda n: n['hybrid_reward'], reverse=True)
            active_nodes = non_terminal[:config_N]

            for node in active_nodes:
                node['survived'] = True

        # Mark final active nodes as survived (in case loop ended normally)
        for node in active_nodes:
            node['survived'] = True

        all_beams_data.append({
            'beam_id': beam_idx,
            'nodes': nodes_data,
        })

    return all_beams_data, gt_screenshot_b64


def compute_advantages(beams_data):
    """Compute intra/inter-beam advantages and add to node dicts."""
    from utils.step_advantages import (
        compute_intra_beam_advantages,
        compute_inter_beam_advantages,
    )

    # Collect all non-root training data
    training_data = []
    node_refs = []  # parallel list of (beam_idx, node_dict)
    for beam in beams_data:
        for node in beam['nodes']:
            if node['step'] == 0:
                continue
            training_data.append({
                'sibling_group_id': node['sibling_group_id'],
                'step_reward': node['step_reward'],
            })
            node_refs.append((beam['beam_id'], node))

    if not training_data:
        return

    # Intra-beam advantages
    intra = compute_intra_beam_advantages(training_data)
    for i, (beam_id, node) in enumerate(node_refs):
        node['intra_advantage'] = round(intra[i], 4)

    # Inter-beam advantages: use best surviving leaf per beam
    terminal_rewards = []
    beam_ids = []
    for beam in beams_data:
        nodes = beam['nodes']
        surviving = [
            n for n in nodes
            if n['survived'] and not n['children_ids']
        ]
        if surviving:
            best = max(surviving, key=lambda n: n['hybrid_reward'])
            terminal_rewards.append(best['hybrid_reward'])
            beam_ids.append(beam['beam_id'])

    if len(terminal_rewards) > 1:
        inter = compute_inter_beam_advantages(terminal_rewards)
        inter_by_beam = dict(zip(beam_ids, inter))
    else:
        inter_by_beam = {bid: 0.0 for bid in beam_ids}

    for beam_id, node in node_refs:
        inter_adv = inter_by_beam.get(beam_id, 0.0)
        node['inter_advantage'] = round(inter_adv, 4)
        node['total_advantage'] = round(
            node.get('intra_advantage', 0.0) + inter_adv, 4
        )


def build_output_json(
    metadata, reference_image_b64, reference_html,
    gt_screenshot_b64, beams_data,
):
    """Build the final JSON output."""
    # Round floats and remove full_text from nodes to save space
    # (full_text can be reconstructed from prefix chain)
    for beam in beams_data:
        for node in beam['nodes']:
            for key in ['hybrid_reward', 'step_reward', 'lpips_score', 'emd_similarity']:
                if node.get(key) is not None:
                    node[key] = round(node[key], 4)

    return {
        'metadata': metadata,
        'reference_image_base64': reference_image_b64,
        'reference_html': reference_html,
        'gt_screenshot_base64': gt_screenshot_b64,
        'beams': beams_data,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Debug beam search with visualization output')
    parser.add_argument('split', choices=['easy', 'medium', 'hard'],
                        help='Dataset split')
    parser.add_argument('offset', type=int,
                        help='Sample offset within RL reserve')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock generator (no GPU needed)')
    parser.add_argument('--vllm-url', type=str, default=None,
                        help='vLLM OpenAI-compatible API URL')
    parser.add_argument('--M', type=int, default=2,
                        help='Number of independent beams (default: 2)')
    parser.add_argument('--K', type=int, default=3,
                        help='Branches per parent per step (default: 3)')
    parser.add_argument('--N', type=int, default=2,
                        help='Beam width / survivors per step (default: 2)')
    parser.add_argument('--max-steps', type=int, default=4,
                        help='Max generation steps (default: 4)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Hybrid reward balance (default: 0.5)')
    parser.add_argument('--output', type=str, default='beam_debug.json',
                        help='Output JSON path (default: beam_debug.json)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for mock generator (default: 42)')
    args = parser.parse_args()

    if not args.mock and not args.vllm_url:
        parser.error("Must specify either --mock or --vllm-url")

    # Load dataset sample
    print(f"Loading dataset sample: {args.split}[{args.offset}]...")
    sample = load_dataset_sample(args.split, args.offset)
    reference_html = sample['markup']
    reference_image = sample['image']
    reference_image_b64 = pil_to_base64(reference_image)
    print(f"  Reference HTML: {len(reference_html)} chars")

    # Create generate_fn
    if args.mock:
        print("Using mock generator")
        generate_fn = create_mock_generate_fn(seed=args.seed)
    else:
        from utils.vllm_generate import create_vllm_generate_fn_from_pil
        print(f"Using vLLM at {args.vllm_url}")
        # Auto-detect model name from vLLM server
        from openai import OpenAI
        client = OpenAI(base_url=f"{args.vllm_url}/v1", api_key="unused")
        models = client.models.list()
        model_name = models.data[0].id
        print(f"  Model: {model_name}")

        generate_fn = create_vllm_generate_fn_from_pil(
            base_url=args.vllm_url,
            model_name=model_name,
            image=reference_image,
            temperature=0.8,
            max_tokens=512,
        )

    # Create reward pool
    from utils.similarity_parallel import RewardPool
    print("Starting reward pool (2 workers)...")
    reward_pool = RewardPool(num_workers=2)

    # Run instrumented beam search
    print(f"\nRunning beam search: M={args.M}, K={args.K}, N={args.N}, "
          f"max_steps={args.max_steps}, alpha={args.alpha}")
    t0 = time.perf_counter()
    beams_data, gt_screenshot_b64 = run_instrumented_beam_search(
        config_M=args.M,
        config_K=args.K,
        config_N=args.N,
        max_steps=args.max_steps,
        generate_fn=generate_fn,
        reward_pool=reward_pool,
        reference_html=reference_html,
        alpha=args.alpha,
        vw=VIEWPORT_WIDTH,
        vh=VIEWPORT_HEIGHT,
    )
    total_time = time.perf_counter() - t0
    print(f"\nBeam search complete in {total_time:.1f}s")

    # Summary
    total_nodes = sum(len(b['nodes']) for b in beams_data)
    total_renders = sum(
        1 for b in beams_data for n in b['nodes']
        if n['screenshot_base64'] is not None
    )
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total renders: {total_renders}")

    # Print reward ranges
    for beam in beams_data:
        rewards = [
            n['hybrid_reward'] for n in beam['nodes']
            if n['step'] > 0 and not n['is_terminal']
        ]
        if rewards:
            print(f"  Beam {beam['beam_id']}: reward range "
                  f"[{min(rewards):.3f}, {max(rewards):.3f}]")

    # Compute advantages
    print("\nComputing advantages...")
    compute_advantages(beams_data)

    # Build and save JSON
    metadata = {
        'dataset': DATASET_NAME,
        'split': args.split,
        'offset': args.offset,
        'config': {
            'M': args.M, 'K': args.K, 'N': args.N,
            'max_steps': args.max_steps,
        },
        'alpha': args.alpha,
        'timestamp': datetime.now().isoformat(),
        'mode': 'mock' if args.mock else 'vllm',
        'seed': args.seed if args.mock else None,
    }

    output = build_output_json(
        metadata=metadata,
        reference_image_b64=reference_image_b64,
        reference_html=reference_html,
        gt_screenshot_b64=gt_screenshot_b64,
        beams_data=beams_data,
    )

    with open(args.output, 'w') as f:
        json.dump(output, f)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nJSON saved to: {args.output} ({file_size_mb:.1f} MB)")
    print(f"Open beam_search_viewer.html and load this JSON to visualize.")

    reward_pool.shutdown()


if __name__ == '__main__':
    main()
