"""
Per-token GRPO: group-relative advantage normalization at the token level.

Standard GRPO assigns one scalar advantage per completion (all tokens in a
completion share the same advantage). This module implements per-token
advantages where each token gets an individual advantage based on the visual
quality of the element it generated.

Key components:
- compute_per_token_advantages(): Token-level group-relative normalization
- per_token_grpo_loss(): PPO-clip loss with (B,T) advantages
- PerTokenGRPOTrainer: GRPOTrainer subclass that uses per-token advantages
- compute_char_rewards(): Map element-level LPIPS to per-character rewards
- char_rewards_to_token_rewards(): Aggregate char rewards to token rewards
- ElementMappedGRPOTrainer: PerTokenGRPOTrainer with element-level reward mapping

The PerTokenGRPOTrainer is a base class. ElementMappedGRPOTrainer inherits
from it and will override _generate_and_score_completions() to compute per-token
rewards from element-level visual similarity.
"""
import logging
import warnings

import torch
from trl.trainer.grpo_trainer import GRPOTrainer

logger = logging.getLogger(__name__)


def compute_per_token_advantages(
    token_rewards: list[list[float]],
    completion_lengths: list[int],
    max_len: int,
    group_size: int = None,
    valid: list[bool] = None,
) -> torch.Tensor:
    """
    Group-relative normalization at the token level.

    Collects all token rewards across the G completions OF ONE PROMPT GROUP,
    computes group mean and std, then normalizes each token individually:
        A[k,t] = (r[k,t] - mean(group_rewards)) / std(group_rewards)

    Padding positions get advantage = 0.0.
    If std < 1e-8, all advantages in that group are set to 0.0.

    Args:
        token_rewards: list of B lists, each containing per-token reward floats.
        completion_lengths: list of B ints (actual non-padded lengths).
        max_len: int for padding dimension.
        group_size: completions per prompt group (num_generations). The batch
            is normalized in consecutive chunks of this size, matching TRL's
            RepeatSampler layout. If None, the whole batch is treated as one
            group (only correct when the batch contains a single prompt —
            normalizing across prompts reintroduces the prompt-difficulty
            bias that group-relative normalization exists to remove).
        valid: optional list of B bools. Invalid completions (failed render,
            failed alignment, failed tokenization) are excluded from the
            group statistics and receive advantage 0.0 everywhere — a true
            neutral, rather than a spuriously extreme reward.

    Returns:
        torch.Tensor of shape (B, max_len) with per-token advantages.
    """
    B = len(token_rewards)
    if valid is None:
        valid = [True] * B
    if group_size is None or group_size <= 0:
        group_size = B
    if B % group_size != 0:
        # Defensive: fall back to one group rather than silently misgrouping
        group_size = B

    advantages = torch.zeros(B, max_len, dtype=torch.float32)

    for g0 in range(0, B, group_size):
        idxs = range(g0, min(g0 + group_size, B))

        group_rewards = []
        for k in idxs:
            if not valid[k]:
                continue
            length = min(completion_lengths[k], len(token_rewards[k]))
            group_rewards.extend(token_rewards[k][:length])

        if not group_rewards:
            continue

        rewards_t = torch.tensor(group_rewards, dtype=torch.float32)
        mu = rewards_t.mean()
        # correction=0 (population std): the token pool across the group's
        # completions IS the full population for this prompt.
        sigma = rewards_t.std(correction=0)

        if sigma < 1e-8:
            continue  # no variance — advantages stay 0 for this group

        for k in idxs:
            if not valid[k]:
                continue
            length = min(completion_lengths[k], len(token_rewards[k]))
            for t in range(length):
                advantages[k, t] = (token_rewards[k][t] - mu) / sigma

    return advantages


def per_token_grpo_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
) -> torch.Tensor:
    """
    GRPO policy loss with per-token advantages.

    Same PPO-clip formula as TRL but advantages is (B, T) instead of (B,)
    so no unsqueeze is needed.

    loss = -mean_batch(mean_tokens(min(ratio * adv, clip(ratio) * adv) * mask))

    Args:
        per_token_logps: (B, T) log-probabilities under current policy.
        old_per_token_logps: (B, T) log-probabilities under old policy.
        advantages: (B, T) per-token advantages.
        completion_mask: (B, T) binary mask (1 for real tokens, 0 for padding).
        epsilon_low: lower clipping bound.
        epsilon_high: upper clipping bound.

    Returns:
        Scalar loss.
    """
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high)

    # advantages is already (B, T) -- no unsqueeze needed
    per_token_loss1 = ratio * advantages
    per_token_loss2 = clipped_ratio * advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    # Mean over tokens (masked), then mean over batch
    # This matches TRL's "grpo" loss_type formula
    loss = (
        (per_token_loss * completion_mask).sum(dim=-1)
        / completion_mask.sum(dim=-1).clamp(min=1.0)
    ).mean()

    return loss


class PerTokenGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer subclass for per-token advantages.

    TRL >=0.29.0 natively supports (B, T) advantages in _compute_loss via:
        if advantages.dim() == 1: advantages = advantages.unsqueeze(1)
    So no _compute_loss override is needed.

    This is the base class. ElementMappedGRPOTrainer inherits from this and
    overrides _generate_and_score_completions() to compute per-token rewards.
    """
    pass


def compute_char_rewards(
    model_output: str,
    overall_loss: float,
    alpha: float,
    element_losses: list,  # list of (model_char_start, model_char_end, lpips_score, area)
    css_mappings: list,    # list of (model_char_start, model_char_end, element_index)
) -> list[float]:
    """
    Map element-level LPIPS scores to per-character rewards.

    For each character in model_output:
    - If mapped to element(s) via HTML range or CSS mapping:
        r_c = 1 - (alpha * overall_loss + (1 - alpha) * weighted_element_loss)
      where weighted_element_loss is the area-weighted average of all mapped
      elements' LPIPS.
    - If unmapped: the element term falls back to the overall loss:
        r_c = 1 - (alpha * overall_loss + (1 - alpha) * overall_loss)
            = 1 - overall_loss

    Why the convex blend (this replaces the earlier additive formula
    `1 - (alpha*overall + element)` with unmapped `1 - alpha*overall`):
    the additive version put mapped and unmapped characters on different
    reward scales — any mapped character scored strictly below any unmapped
    character (element_loss > 0 always), so after group normalization
    boilerplate/unmapped tokens received systematically positive advantages
    relative to tokens that produced well-rendered elements. The convex
    blend keeps every character on the same [0, 1] scale: unmapped
    characters are exactly neutral (they carry the completion-level signal
    only), and mapped characters score above/below that baseline according
    to whether their element rendered better/worse than the page overall.
    At alpha = 1 the formula reduces to vanilla GRPO for all tokens.

    Element mapping comes from two sources:
    1. element_losses: direct HTML char ranges (start, end, lpips, area).
       A character at position c is covered by element i if
       element_losses[i][0] <= c < element_losses[i][1].
    2. css_mappings: CSS char ranges mapped to element indices (start, end, elem_idx).
       The elem_idx indexes into element_losses to get the LPIPS score and area.

    When a character maps to multiple elements, their losses are aggregated
    via area-weighted average.

    Args:
        model_output: The model's generated text.
        overall_loss: Full-image LPIPS loss for this completion.
        alpha: Blend weight (1 = pure overall/vanilla, 0 = pure element-level).
        element_losses: List of (char_start, char_end, lpips_score, area) tuples.
            Each defines an HTML element's character range and its visual quality.
        css_mappings: List of (char_start, char_end, element_index) tuples.
            Maps CSS character ranges to elements in element_losses.

    Returns:
        List of floats, one reward per character in model_output.
    """
    n = len(model_output)
    if n == 0:
        return []

    # For each character, accumulate (lpips_score, area) pairs.
    # HTML elements use innermost-wins; CSS mappings area-weight-average.
    char_elements: list[list[tuple[float, float]]] = [[] for _ in range(n)]

    # 1. HTML element ranges: innermost element wins.
    # Overlapping HTML ranges = ancestor-descendant; innermost has smallest span.
    char_html_candidates: list[list[tuple[float, float, int]]] = [[] for _ in range(n)]
    for start, end, lpips_score, area in element_losses:
        span = end - start
        clamped_start = max(0, start)
        clamped_end = min(n, end)
        for c in range(clamped_start, clamped_end):
            char_html_candidates[c].append((lpips_score, area, span))

    for c in range(n):
        if char_html_candidates[c]:
            innermost = min(char_html_candidates[c], key=lambda x: x[2])
            char_elements[c].append((innermost[0], innermost[1]))

    # 2. CSS mappings: area-weighted average (a rule can match multiple elements)
    for start, end, elem_idx in css_mappings:
        if elem_idx < 0 or elem_idx >= len(element_losses):
            continue
        _, _, lpips_score, area = element_losses[elem_idx]
        clamped_start = max(0, start)
        clamped_end = min(n, end)
        for c in range(clamped_start, clamped_end):
            char_elements[c].append((lpips_score, area))

    # Compute per-character rewards (convex blend, see docstring)
    char_rewards = []
    for c in range(n):
        mappings = char_elements[c]
        if not mappings:
            # Unmapped: element term falls back to overall loss (neutral)
            element_term = overall_loss
        else:
            # Area-weighted average (HTML contributes one entry, CSS may add more)
            total_area = sum(area for _, area in mappings)
            if total_area < 1e-12:
                # Degenerate case: zero-area elements, use simple average
                element_term = sum(lpips for lpips, _ in mappings) / len(mappings)
            else:
                element_term = (
                    sum(lpips * area for lpips, area in mappings) / total_area
                )
        reward = 1.0 - (alpha * overall_loss + (1.0 - alpha) * element_term)
        # LPIPS can occasionally exceed 1.0 — keep rewards in [0, 1]
        char_rewards.append(max(0.0, min(1.0, reward)))

    return char_rewards


def char_rewards_to_token_rewards(
    char_rewards: list[float],
    token_offsets: list[tuple[int, int]],
) -> list[float]:
    """
    Aggregate per-character rewards to per-token rewards.

    Each token spans a range of characters given by token_offsets[i] = (start, end).
    The token's reward is the average of char_rewards[start:end].

    Empty spans (start == end, typical for special tokens like BOS/EOS) get
    the global average of all char_rewards.

    Args:
        char_rewards: List of per-character reward floats.
        token_offsets: List of (start, end) tuples, one per token.
            start and end are character indices into the original text.

    Returns:
        List of per-token reward floats, same length as token_offsets.
    """
    if not char_rewards:
        return [0.0] * len(token_offsets)

    global_avg = sum(char_rewards) / len(char_rewards)

    token_rewards = []
    for start, end in token_offsets:
        if start >= end:
            # Empty span (special token): use global average
            token_rewards.append(global_avg)
        else:
            # Average char rewards in this token's span
            span_rewards = char_rewards[start:end]
            token_rewards.append(sum(span_rewards) / len(span_rewards))

    return token_rewards


class ElementMappedGRPOTrainer(PerTokenGRPOTrainer):
    """
    PerTokenGRPOTrainer with element-level visual reward mapping.

    Uses compute_char_rewards() and char_rewards_to_token_rewards() to convert
    element-level LPIPS scores into per-token rewards, which are then normalized
    into per-token advantages by the parent class's infrastructure.

    The alpha parameter controls the blend between overall image loss and
    element-specific loss in the per-character reward formula:
        r_c = 1 - (alpha * overall_loss + (1 - alpha) * element_term)
    where element_term is the area-weighted element LPIPS for mapped
    characters and falls back to overall_loss for unmapped characters.
    - alpha=0: pure element-level reward (unmapped chars are neutral at
      1 - overall_loss)
    - alpha=1: reduces exactly to vanilla GRPO (uniform 1 - overall_loss)

    Overrides _generate_and_score_completions() to:
    1. Call super() for vanilla generation and scoring (scalar advantages)
    2. Run RewardPool in token_level mode for element-level data
    3. Align elements to model output characters via rapidfuzz
    4. Compute per-character rewards from element LPIPS scores
    5. Aggregate to per-token rewards and compute per-token advantages
    6. Replace the scalar advantages with (B, T) per-token advantages
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.5,
        viewport_width: int = 1024,
        viewport_height: int = 1024,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

    def _generate_and_score_completions(self, inputs, **kwargs):
        """Override to compute per-token advantages from element-level rewards.

        Flow:
        1. Call super() to do generation, reward computation, and scalar advantages
        2. Get completion texts and ground truth
        3. Call RewardPool in token_level mode for element data
        4. Run alignment + char->token mapping for each completion
        5. Compute per-token advantages and replace scalar advantages in output
        """
        from .similarity_parallel import get_reward_pool
        from .token_rewards import align_texts, find_element_in_dom

        # Step 1: Let vanilla GRPO handle generation, reward scoring, and scalar advantages
        output = super()._generate_and_score_completions(inputs, **kwargs)

        device = output["advantages"].device
        completion_ids = output["completion_ids"]   # (B, T)
        completion_mask = output["completion_mask"]  # (B, T)
        B, T = completion_ids.shape

        # Step 2: Decode completion texts
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Step 3: Get ground truth answers
        # TRL's dataloader repeats each sample num_generations times via RepeatSampler,
        # so inputs already has repeated entries. Each input dict has an "answer" key.
        answers = [inp.get("answer", "") for inp in inputs]

        # Step 4: Call RewardPool in token_level mode
        pool = get_reward_pool()
        items = [
            (text, ans, self.viewport_width, self.viewport_height)
            for text, ans in zip(completions_text, answers)
        ]

        try:
            token_results = pool.calculate_metrics_batch(items, token_level=True)
        except Exception as e:
            warnings.warn(f"Token-level reward computation failed: {e}. "
                          f"Falling back to scalar advantages.")
            return output

        # Step 5: For each completion, compute per-token rewards.
        # Completions that fail anywhere in the pipeline (render, alignment,
        # tokenization, offset verification) are marked invalid: they are
        # excluded from the group statistics and get advantage 0.0 everywhere
        # (a true neutral — no gradient), instead of a spuriously extreme
        # reward that conflates infrastructure failures with bad output.
        all_token_rewards = []
        valid_flags = []
        n_offset_mismatch = 0
        completion_lengths = completion_mask.sum(dim=1).tolist()  # actual non-padded lengths

        for i in range(B):
            text = completions_text[i]
            token_result = token_results[i]
            actual_len = int(completion_lengths[i])

            if token_result is None or not text.strip():
                all_token_rewards.append([0.0] * actual_len)
                valid_flags.append(False)
                continue

            overall_loss = token_result.perceptual_loss
            browser_dom = token_result.browser_dom
            element_infos = token_result.element_infos
            css_mappings_raw = token_result.css_mappings

            # Align model output to browser DOM
            try:
                _model_to_dom, dom_to_model = align_texts(text, browser_dom)
            except Exception as e:
                warnings.warn(f"Alignment failed for completion {i}: {e}")
                all_token_rewards.append([0.0] * actual_len)
                valid_flags.append(False)
                continue

            # Build element_losses: list of (model_char_start, model_char_end, lpips, area)
            # Also build a mapping from raw_element_index -> element_losses index
            # so CSS mappings can reference the right element.
            element_losses = []
            raw_idx_to_elem_loss_idx = {}  # raw_element_index -> index in element_losses

            for ei in element_infos:
                area = ei.width * ei.height
                if area < 16:
                    continue

                # Map DOM position to model output position via alignment
                model_start = None
                model_end = None

                if ei.dom_char_start is not None and ei.dom_char_end is not None:
                    model_positions = []
                    for dom_pos in range(ei.dom_char_start, min(ei.dom_char_end, len(dom_to_model))):
                        if dom_to_model[dom_pos] >= 0:
                            model_positions.append(dom_to_model[dom_pos])
                    if model_positions:
                        model_start = min(model_positions)
                        model_end = max(model_positions) + 1

                if model_start is not None and model_end is not None:
                    if ei.raw_element_index >= 0:
                        raw_idx_to_elem_loss_idx[ei.raw_element_index] = len(element_losses)
                    element_losses.append((model_start, model_end, ei.lpips_score, area))

            # Build CSS mappings referencing element_losses indices.
            # css_mappings_raw contains (model_char_start, model_char_end, raw_element_index)
            # where raw_element_index indexes into the original elements_raw list from
            # extract_elements(). We use raw_idx_to_elem_loss_idx to remap to element_losses.
            css_mappings = []
            for css_start, css_end, raw_idx in css_mappings_raw:
                if raw_idx in raw_idx_to_elem_loss_idx:
                    css_mappings.append((css_start, css_end, raw_idx_to_elem_loss_idx[raw_idx]))

            # Compute per-character rewards
            char_rewards = compute_char_rewards(
                text, overall_loss, self.alpha, element_losses, css_mappings
            )

            # Tokenize to get offset mapping for char->token aggregation
            try:
                encoding = self.processing_class.tokenizer(
                    text, return_offsets_mapping=True, add_special_tokens=False
                )
                offsets = encoding['offset_mapping']
                reencoded_ids = encoding['input_ids']
            except Exception as e:
                warnings.warn(f"Tokenization failed for completion {i}: {e}")
                all_token_rewards.append([0.0] * actual_len)
                valid_flags.append(False)
                continue

            # Verify the decode→re-encode round trip actually reproduces the
            # generated token sequence. The offset mapping only means anything
            # if re-encoded token k IS completion token k. Decoding skipped
            # special tokens, so the re-encoded sequence should match the
            # completion ids up to a trailing EOS (at most 1 token shorter).
            actual_ids = completion_ids[i, :actual_len].tolist()
            n_re = len(reencoded_ids)
            round_trip_ok = (
                actual_len - 1 <= n_re <= actual_len
                and reencoded_ids == actual_ids[:n_re]
            )
            if not round_trip_ok:
                n_offset_mismatch += 1
                all_token_rewards.append([0.0] * actual_len)
                valid_flags.append(False)
                continue

            # Aggregate char rewards to token rewards
            token_rewards = char_rewards_to_token_rewards(char_rewards, offsets)

            # Pad the (verified) 0-or-1 token gap — the trailing EOS — with
            # this completion's average reward (neutral for that position).
            if len(token_rewards) < actual_len:
                avg = sum(token_rewards) / max(len(token_rewards), 1)
                token_rewards.extend([avg] * (actual_len - len(token_rewards)))

            all_token_rewards.append(token_rewards)
            valid_flags.append(True)

        # Step 6: Compute per-token advantages via group-relative
        # normalization — per prompt group (num_generations consecutive
        # completions), matching TRL's RepeatSampler layout. Normalizing
        # across the whole batch would reintroduce prompt-difficulty bias.
        max_len = T  # padded sequence length from completion_ids
        per_token_advantages = compute_per_token_advantages(
            all_token_rewards,
            [int(l) for l in completion_lengths],
            max_len,
            group_size=self.num_generations,
            valid=valid_flags,
        )

        # Replace scalar advantages with per-token advantages
        output["advantages"] = per_token_advantages.to(device)

        # Log token-level metrics
        mode = "train" if self.model.training else "eval"
        # Fraction of completions that got element-level rewards (vs fallback)
        n_element_mapped = sum(1 for v in valid_flags if v)
        self._metrics[mode]["token_rewards/element_mapped_ratio"].append(
            n_element_mapped / max(B, 1)
        )
        self._metrics[mode]["token_rewards/offset_mismatch_ratio"].append(
            n_offset_mismatch / max(B, 1)
        )
        self._metrics[mode]["token_rewards/render_failed_ratio"].append(
            sum(1 for i in range(B) if token_results[i] is None) / max(B, 1)
        )

        # Average number of elements per completion
        n_elements = [
            len(token_results[i].element_infos)
            for i in range(B)
            if token_results[i] is not None
        ]
        if n_elements:
            self._metrics[mode]["token_rewards/avg_elements"].append(
                sum(n_elements) / len(n_elements)
            )

        return output
