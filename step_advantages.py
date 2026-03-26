"""
Step-level advantage computation for Tree-VPO (Visual Policy Optimization).

The primary advantage signal is the **step-delta reward**: did adding this
element improve the rendered output compared to before it was added?

    advantage = hybrid_reward(node) - hybrid_reward(parent)

Positive = this element helped. Negative = it hurt. This is a direct,
dense signal that doesn't require sibling comparison — even with K=1
(no branching), every node gets a meaningful reward.

Optional batch normalization across the full batch of entries provides
variance reduction (keeps gradients on a consistent scale across steps
and samples), similar to how standard GRPO normalizes across rollouts.

Usage:
    from utils.step_advantages import compute_step_delta_advantages

    results = compute_step_delta_advantages(training_data, normalize=True)
    # Each result has: step_advantage (= step_reward, optionally normalized)

Legacy group-relative functions are preserved for ablation experiments.
"""

from collections import defaultdict


def compute_step_delta_advantages(
    training_data: list[dict],
    normalize: bool = True,
) -> list[dict]:
    """Compute advantages from step-delta rewards (Tree-VPO).

    Each entry's advantage is its step_reward: hybrid_reward(node) - hybrid_reward(parent).
    If normalize=True, advantages are zero-centered and unit-variance across the batch
    for variance reduction.

    Args:
        training_data: List of dicts, each must have 'step_reward'.
        normalize: If True, standardize advantages across the batch
            (zero mean, unit variance). Recommended for training stability.

    Returns:
        List of dicts (same length/order as training_data), each containing
        all original fields plus 'step_advantage'.
    """
    step_rewards = [e["step_reward"] for e in training_data]
    n = len(step_rewards)

    if n == 0:
        return []

    if normalize and n > 1:
        mu = sum(step_rewards) / n
        variance = sum((r - mu) ** 2 for r in step_rewards) / n
        sigma = variance ** 0.5

        if sigma < 1e-8:
            advantages = [0.0] * n
        else:
            advantages = [(r - mu) / (sigma + 1e-8) for r in step_rewards]
    else:
        advantages = list(step_rewards)

    results = []
    for entry, adv in zip(training_data, advantages):
        result = dict(entry)
        result["step_advantage"] = adv
        results.append(result)

    return results


# ---------------------------------------------------------------
# Legacy functions (kept for ablation experiments)
# ---------------------------------------------------------------

def compute_intra_beam_advantages(training_data: list[dict]) -> list[float]:
    """[Legacy] Group-relative advantages within each sibling group.

    Used by Tree-GRPO-style sibling comparison. Kept for ablation:
    does group-relative add value over Tree-VPO's raw step-delta?

    Groups entries by sibling_group_id and normalizes step_reward values
    within each group.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for i, entry in enumerate(training_data):
        groups[entry["sibling_group_id"]].append(i)

    advantages = [0.0] * len(training_data)

    for group_indices in groups.values():
        rewards = [training_data[i]["step_reward"] for i in group_indices]
        n = len(rewards)

        mu = sum(rewards) / n
        variance = sum((r - mu) ** 2 for r in rewards) / n
        sigma = variance ** 0.5

        if sigma < 1e-8:
            for i in group_indices:
                advantages[i] = 0.0
        else:
            for i, r in zip(group_indices, rewards):
                advantages[i] = (r - mu) / (sigma + 1e-8)

    return advantages


def compute_inter_beam_advantages(terminal_rewards: list[float]) -> list[float]:
    """[Legacy] Group-relative advantages across terminal beam rewards."""
    n = len(terminal_rewards)
    if n == 0:
        return []

    mu = sum(terminal_rewards) / n
    variance = sum((r - mu) ** 2 for r in terminal_rewards) / n
    sigma = variance ** 0.5

    if sigma < 1e-8:
        return [0.0] * n

    return [(r - mu) / (sigma + 1e-8) for r in terminal_rewards]


def compute_combined_advantages(
    training_data: list[dict],
    terminal_rewards: dict[int, float],
) -> list[dict]:
    """[Legacy] Combined intra-beam + inter-beam advantages (group-relative style).

    Kept for ablation experiments comparing Tree-VPO (step-delta) vs
    group-relative approach.
    """
    intra_advantages = compute_intra_beam_advantages(training_data)

    beam_ids = sorted(terminal_rewards.keys())
    terminal_reward_list = [terminal_rewards[bid] for bid in beam_ids]
    inter_advantage_list = compute_inter_beam_advantages(terminal_reward_list)

    inter_advantage_map: dict[int, float] = {}
    for bid, adv in zip(beam_ids, inter_advantage_list):
        inter_advantage_map[bid] = adv

    results = []
    for i, entry in enumerate(training_data):
        result = dict(entry)
        result["intra_advantage"] = intra_advantages[i]
        result["inter_advantage"] = inter_advantage_map[entry["beam_id"]]
        result["total_advantage"] = result["intra_advantage"] + result["inter_advantage"]
        results.append(result)

    return results
