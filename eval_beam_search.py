"""
Beam search evaluation for Tree-VPO.

Compares greedy generation vs beam search (with intermediate rendering rewards)
on the SFT model. Demonstrates that beam search improves output quality at
inference time even without RL training.

Evaluation modes:
  1. Greedy: standard autoregressive generation (baseline)
  2. Beam search: element-by-element generation with pruning by hybrid reward

Requires:
  - vLLM server running the model (SSH tunnel or local)
  - RewardPool for rendering + LPIPS scoring

Usage (run from training/ directory):
    # Evaluate on 10 samples from the easy split:
    python -m utils.eval_beam_search --vllm-url http://localhost:8888 \
        --split easy --num-samples 10

    # Evaluate all splits with custom beam parameters:
    python -m utils.eval_beam_search --vllm-url http://localhost:8888 \
        --split all --M 2 --K 3 --N 2 --max-steps 10

    # Save detailed results to JSON:
    python -m utils.eval_beam_search --vllm-url http://localhost:8888 \
        --split easy --num-samples 5 --output eval_results.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_NAME = "tcz/rb-box-layouts-inline"
SEED = 3407
VIEWPORT_WIDTH = 1024
VIEWPORT_HEIGHT = 1024
INSTRUCTION = "Generate the HTML markup with inline styles that produces this webpage layout."


def load_eval_samples(split, num_samples, seed=SEED):
    """Load test samples matching the exact SFT training test split.

    The SFT training script splits each difficulty as:
        ds = raw[split].shuffle(seed=3407)
        sft_ds = ds[:n//2]          # first 50% for SFT
        test_ds = sft_ds[:100]      # first 100 of SFT portion = test
        eval_ds = sft_ds[100:110]   # next 10 = eval during training
        train_ds = sft_ds[110:]     # rest = training

    We load the same test_ds (first `num_samples` of SFT portion).
    """
    from datasets import load_dataset
    raw = load_dataset(DATASET_NAME)

    splits = [split] if split != "all" else ["easy", "medium", "hard"]
    samples = []

    for s in splits:
        ds = raw[s].shuffle(seed=seed)
        n = len(ds)
        sft_boundary = n // 2
        sft_ds = ds.select(range(sft_boundary))
        count = min(num_samples, len(sft_ds))
        for i in range(count):
            sample = sft_ds[i]
            samples.append({
                "image": sample["image"],
                "markup": sample["markup"],
                "difficulty": s,
            })

    return samples


def generate_greedy(vllm_url, model_name, image, max_tokens=8192):
    """Generate HTML greedily (standard autoregressive) via vLLM."""
    import base64
    import io
    from openai import OpenAI

    # Encode image to base64
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    client = OpenAI(base_url=f"{vllm_url}/v1", api_key="unused")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{image_b64}"
                }},
                {"type": "text", "text": INSTRUCTION},
            ],
        }],
        max_tokens=max_tokens,
        temperature=0.0,  # greedy
    )

    return response.choices[0].message.content


def run_beam_search_eval(
    generate_fn, reward_pool, reference_html, alpha, vw, vh,
    config_M, config_K, config_N, max_steps,
):
    """Run beam search and return the best final HTML and its reward.

    Closing tags (starting with "</") are handled deterministically: they
    inherit the parent's reward without rendering (no visual change) and
    don't consume branching budget.

    Returns:
        best_html: The HTML from the best surviving leaf
        best_reward: The hybrid reward of the best leaf
        num_renders: Total number of render calls made
    """
    from utils.hybrid_reward import compute_hybrid_reward

    best_html = ""
    best_reward = 0.0
    num_renders = 0

    for beam_idx in range(config_M):
        active_nodes = [{"full_text": "", "hybrid_reward": 0.0}]

        for step in range(max_steps):
            all_children = []

            for parent in active_nodes:
                # Generate one continuation first to check if it's a closing tag
                continuations = generate_fn(parent["full_text"], 1)
                first = continuations[0] if continuations else ""

                if first.startswith("</"):
                    # Closing tag: append deterministically, no branching.
                    # Inherit parent reward (closing tags don't change visuals).
                    child = {
                        "full_text": parent["full_text"] + first,
                        "hybrid_reward": parent["hybrid_reward"],
                        "is_terminal": False,
                    }
                    all_children.append(child)
                    continue

                # Normal element: generate full K branches
                if first == "":
                    # Terminal from first sample
                    all_children.append({
                        "full_text": parent["full_text"],
                        "hybrid_reward": parent["hybrid_reward"],
                        "is_terminal": True,
                    })
                else:
                    all_children.append({
                        "full_text": parent["full_text"] + first,
                        "hybrid_reward": 0.0,
                        "is_terminal": False,
                    })

                # Generate remaining K-1 branches
                if config_K > 1:
                    extra = generate_fn(parent["full_text"], config_K - 1)
                    for cont in extra:
                        is_terminal = (cont == "")
                        child = {
                            "full_text": parent["full_text"] + cont,
                            "hybrid_reward": parent["hybrid_reward"] if is_terminal else 0.0,
                            "is_terminal": is_terminal,
                        }
                        all_children.append(child)

            # Batch render non-terminal children that need scoring
            to_render = [
                c for c in all_children
                if not c["is_terminal"] and c["hybrid_reward"] == 0.0
            ]
            if to_render:
                items = [
                    (c["full_text"], reference_html, vw, vh)
                    for c in to_render
                ]
                results = reward_pool.calculate_metrics_batch_intermediate(items)
                num_renders += len(items)

                for child, r in zip(to_render, results):
                    if r is None:
                        child["is_terminal"] = True
                        child["hybrid_reward"] = 0.0
                        continue

                    child["hybrid_reward"] = compute_hybrid_reward(
                        r["pred_screenshot"], r["gt_screenshot"],
                        r["lpips_score"], alpha,
                    )

            # Select top-N non-terminal by hybrid_reward
            non_terminal = [c for c in all_children if not c["is_terminal"]]
            if not non_terminal:
                break

            non_terminal.sort(key=lambda n: n["hybrid_reward"], reverse=True)
            active_nodes = non_terminal[:config_N]

        # Track best across all beams
        for node in active_nodes:
            if node["hybrid_reward"] > best_reward:
                best_reward = node["hybrid_reward"]
                best_html = node["full_text"]

    return best_html, best_reward, num_renders


def evaluate_sample(
    sample, vllm_url, model_name, reward_pool,
    alpha, config_M, config_K, config_N, max_steps,
):
    """Evaluate a single sample with both greedy and beam search.

    Returns dict with metrics for both approaches.
    """
    from utils.vllm_generate import create_vllm_generate_fn_from_pil

    image = sample["image"]
    reference_html = sample["markup"]
    difficulty = sample["difficulty"]

    # Greedy generation
    t0 = time.monotonic()
    greedy_html = generate_greedy(vllm_url, model_name, image)
    greedy_time = time.monotonic() - t0

    # Score greedy output
    greedy_metrics = reward_pool.calculate_metrics(
        greedy_html, reference_html, VIEWPORT_WIDTH, VIEWPORT_HEIGHT
    )
    greedy_reward = 1.0 - greedy_metrics["perceptual_loss"] if greedy_metrics else 0.0

    # Beam search generation
    generate_fn = create_vllm_generate_fn_from_pil(
        base_url=vllm_url,
        model_name=model_name,
        image=image,
        instruction=INSTRUCTION,
        temperature=0.8,
        max_tokens=256,
    )

    t0 = time.monotonic()
    beam_html, beam_reward, num_renders = run_beam_search_eval(
        generate_fn=generate_fn,
        reward_pool=reward_pool,
        reference_html=reference_html,
        alpha=alpha,
        vw=VIEWPORT_WIDTH, vh=VIEWPORT_HEIGHT,
        config_M=config_M,
        config_K=config_K,
        config_N=config_N,
        max_steps=max_steps,
    )
    beam_time = time.monotonic() - t0

    # Also score beam output with plain LPIPS (for fair comparison with greedy)
    beam_metrics = reward_pool.calculate_metrics(
        beam_html, reference_html, VIEWPORT_WIDTH, VIEWPORT_HEIGHT
    )
    beam_lpips_reward = 1.0 - beam_metrics["perceptual_loss"] if beam_metrics else 0.0

    return {
        "difficulty": difficulty,
        "greedy_reward": greedy_reward,
        "greedy_time": greedy_time,
        "greedy_html_len": len(greedy_html),
        "beam_reward": beam_lpips_reward,  # LPIPS-only for fair comparison
        "beam_hybrid_reward": beam_reward,  # hybrid reward used for search
        "beam_time": beam_time,
        "beam_html_len": len(beam_html),
        "beam_num_renders": num_renders,
        "improvement": beam_lpips_reward - greedy_reward,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate greedy vs beam search on SFT model")
    parser.add_argument("--vllm-url", type=str, required=True,
                        help="vLLM server URL (e.g., http://localhost:8888)")
    parser.add_argument("--split", type=str, default="easy",
                        choices=["easy", "medium", "hard", "all"],
                        help="Dataset split to evaluate (default: easy)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Samples per split (default: 10)")
    parser.add_argument("--M", type=int, default=2,
                        help="Independent beams (default: 2)")
    parser.add_argument("--K", type=int, default=3,
                        help="Branches per parent (default: 3)")
    parser.add_argument("--N", type=int, default=2,
                        help="Beam width (default: 2)")
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Max beam search steps (default: 10)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Hybrid reward alpha (default: 0.5)")
    parser.add_argument("--reward-workers", type=int, default=4,
                        help="Reward pool workers (default: 4)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save detailed results to JSON")
    args = parser.parse_args()

    # Detect model
    from openai import OpenAI
    client = OpenAI(base_url=f"{args.vllm_url}/v1", api_key="unused")
    models = client.models.list()
    model_name = models.data[0].id
    print(f"Model: {model_name}")

    # Load samples
    print(f"Loading {args.num_samples} samples from {args.split}...")
    samples = load_eval_samples(args.split, args.num_samples)
    print(f"  Loaded {len(samples)} samples")

    # Create reward pool
    from utils.similarity_parallel import get_reward_pool
    print(f"Starting reward pool ({args.reward_workers} workers)...")
    reward_pool = get_reward_pool(num_workers=args.reward_workers)

    # Warmup
    warmup_item = ("<html><body>test</body></html>",
                    "<html><body>test</body></html>",
                    VIEWPORT_WIDTH, VIEWPORT_HEIGHT)
    reward_pool.calculate_metrics_batch([warmup_item] * args.reward_workers)
    print("Reward pool ready.\n")

    print(f"Beam search config: M={args.M}, K={args.K}, N={args.N}, "
          f"max_steps={args.max_steps}, alpha={args.alpha}")
    print(f"{'='*70}")

    # Evaluate
    all_results = []
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}/{len(samples)} ({sample['difficulty']})...", end=" ", flush=True)

        result = evaluate_sample(
            sample, args.vllm_url, model_name, reward_pool,
            args.alpha, args.M, args.K, args.N, args.max_steps,
        )
        all_results.append(result)

        print(f"greedy={result['greedy_reward']:.3f} "
              f"beam={result['beam_reward']:.3f} "
              f"(+{result['improvement']:+.3f}) "
              f"[{result['greedy_time']:.1f}s vs {result['beam_time']:.1f}s, "
              f"{result['beam_num_renders']} renders]")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Overall
    greedy_rewards = [r["greedy_reward"] for r in all_results]
    beam_rewards = [r["beam_reward"] for r in all_results]
    improvements = [r["improvement"] for r in all_results]

    print(f"\nOverall ({len(all_results)} samples):")
    print(f"  Greedy reward:     {np.mean(greedy_rewards):.4f} +/- {np.std(greedy_rewards):.4f}")
    print(f"  Beam reward:       {np.mean(beam_rewards):.4f} +/- {np.std(beam_rewards):.4f}")
    print(f"  Improvement:       {np.mean(improvements):+.4f} +/- {np.std(improvements):.4f}")
    print(f"  Beam wins:         {sum(1 for i in improvements if i > 0)}/{len(improvements)}")

    # Per difficulty
    for diff in ["easy", "medium", "hard"]:
        diff_results = [r for r in all_results if r["difficulty"] == diff]
        if not diff_results:
            continue
        g = [r["greedy_reward"] for r in diff_results]
        b = [r["beam_reward"] for r in diff_results]
        imp = [r["improvement"] for r in diff_results]
        print(f"\n  {diff} ({len(diff_results)} samples):")
        print(f"    Greedy: {np.mean(g):.4f}  Beam: {np.mean(b):.4f}  "
              f"Improvement: {np.mean(imp):+.4f}  "
              f"Wins: {sum(1 for i in imp if i > 0)}/{len(imp)}")

    # Timing
    greedy_times = [r["greedy_time"] for r in all_results]
    beam_times = [r["beam_time"] for r in all_results]
    total_renders = sum(r["beam_num_renders"] for r in all_results)
    print(f"\nTiming:")
    print(f"  Greedy avg: {np.mean(greedy_times):.1f}s")
    print(f"  Beam avg:   {np.mean(beam_times):.1f}s ({np.mean(beam_times)/np.mean(greedy_times):.1f}x)")
    print(f"  Total renders: {total_renders}")

    # Save detailed results
    if args.output:
        output_data = {
            "metadata": {
                "model": model_name,
                "dataset": DATASET_NAME,
                "split": args.split,
                "num_samples": len(all_results),
                "beam_config": {
                    "M": args.M, "K": args.K, "N": args.N,
                    "max_steps": args.max_steps,
                },
                "alpha": args.alpha,
                "timestamp": datetime.now().isoformat(),
            },
            "summary": {
                "greedy_mean": float(np.mean(greedy_rewards)),
                "greedy_std": float(np.std(greedy_rewards)),
                "beam_mean": float(np.mean(beam_rewards)),
                "beam_std": float(np.std(beam_rewards)),
                "improvement_mean": float(np.mean(improvements)),
                "improvement_std": float(np.std(improvements)),
                "beam_win_rate": sum(1 for i in improvements if i > 0) / len(improvements),
            },
            "results": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")

    reward_pool.shutdown()


if __name__ == "__main__":
    main()
