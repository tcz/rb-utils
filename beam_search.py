"""
Beam search generation for Tree-VPO (Visual Policy Optimization).

Generates HTML element-by-element, tracking a tree of branches with visual
rewards at each step. The model generates until the next '<' character (i.e.,
one HTML element at a time), the partial HTML is rendered and scored, and the
top-N branches survive to the next step.

All branches — including pruned losers — are retained in the tree because they
provide negative advantage signal for training.

Usage:
    from utils.beam_search import BeamSearchConfig, run_beam_search

    config = BeamSearchConfig(M=2, K=3, N=2, max_steps=15)
    trees = run_beam_search(config, generate_fn, reward_fn, reference)

    # Extract training data from all trees
    for tree in trees:
        data = tree.extract_training_data()
        # Each entry: prefix, continuation, step_reward, hybrid_reward, ...
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class BeamSearchConfig:
    """Configuration for element-level beam search.

    Args:
        M: Number of independent beams (trees). Each starts from an empty root
           and explores independently. More beams = more diverse final outputs.
        K: Branches per parent per step. At each step, each active node spawns
           K continuations via generate_fn.
        N: Beam width (survivors per step). After scoring all children at a step,
           only the top-N by cumulative hybrid_reward survive to the next step.
        max_steps: Maximum number of generation steps (HTML elements).
    """
    M: int = 2
    K: int = 3
    N: int = 2
    max_steps: int = 15


@dataclass
class BeamNode:
    """A single node in the beam search tree.

    Each node represents a state: the full HTML generated up to this point,
    the continuation added at this step, and the visual reward at this state.

    Args:
        prefix: Full HTML up to the parent's state (parent.full_text).
        continuation: Tokens added at this step by generate_fn.
        step: Step number (0 for root, 1+ for generated nodes).
        hybrid_reward: Cumulative reward at this state (from reward_fn on
            the full rendered HTML).
        step_reward: Reward improvement from parent: hybrid_reward - parent.hybrid_reward.
        parent: Reference to parent node (None for root).
        children: List of child nodes spawned from this node.
        sibling_group_id: Unique ID grouping siblings from the same parent at the
            same step. Used for group-relative advantage computation.
        is_terminal: True if generate_fn returned empty string (EOS) for this
            branch, meaning no further expansion is possible.
    """
    prefix: str = ""
    continuation: str = ""
    step: int = 0
    hybrid_reward: float = 0.0
    step_reward: float = 0.0
    parent: Optional["BeamNode"] = None
    children: list["BeamNode"] = field(default_factory=list)
    sibling_group_id: int = 0
    is_terminal: bool = False

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children (never expanded or pruned)."""
        return len(self.children) == 0

    @property
    def full_text(self) -> str:
        """Complete HTML text at this node: prefix + continuation."""
        return self.prefix + self.continuation


@dataclass
class BeamTree:
    """Container for a complete beam search tree from one independent beam.

    Holds the root node and a flat list of all nodes for efficient traversal.
    """
    root: BeamNode
    _all_nodes: list[BeamNode] = field(default_factory=list)

    def all_nodes(self) -> list[BeamNode]:
        """Return all nodes in the tree (including root)."""
        return list(self._all_nodes)

    def leaves(self) -> list[BeamNode]:
        """Return all leaf nodes (nodes with no children)."""
        return [n for n in self._all_nodes if n.is_leaf]

    def surviving_leaves(self) -> list[BeamNode]:
        """Return leaf nodes that reached EOS or were active at max_steps.

        A surviving leaf is one that is terminal (hit EOS) or was among the
        deepest non-terminal leaves (survived to the final step of the search).
        Pruned leaves (eliminated mid-search) are excluded.
        """
        if not self._all_nodes:
            return []
        max_step = max(n.step for n in self._all_nodes)
        return [
            n for n in self._all_nodes
            if n.is_leaf and (n.is_terminal or n.step == max_step)
        ]

    def extract_training_data(self) -> list[dict]:
        """Extract training data from all non-root nodes in the tree.

        Every branch — including pruned losers — is included because pruned
        branches provide negative advantage signal for training.

        Returns:
            List of dicts, one per non-root node, each containing:
                prefix: Parent's full_text (the prompt for this generation step)
                continuation: What was generated at this step
                step_reward: hybrid_reward improvement over parent
                hybrid_reward: Cumulative reward at this state
                step: Step number (1-indexed)
                sibling_group_id: For grouping in advantage computation
                is_terminal: Whether this was a terminal node
                full_text: prefix + continuation
        """
        data = []
        for node in self._all_nodes:
            if node is self.root:
                continue
            data.append({
                "prefix": node.prefix,
                "continuation": node.continuation,
                "step_reward": node.step_reward,
                "hybrid_reward": node.hybrid_reward,
                "step": node.step,
                "sibling_group_id": node.sibling_group_id,
                "is_terminal": node.is_terminal,
                "full_text": node.full_text,
            })
        return data


# Counter for generating unique sibling group IDs across all trees and steps
_sibling_group_counter = 0


def _next_sibling_group_id() -> int:
    """Generate a globally unique sibling group ID."""
    global _sibling_group_counter
    _sibling_group_counter += 1
    return _sibling_group_counter


def reset_sibling_group_counter() -> None:
    """Reset the sibling group counter. Useful for deterministic testing."""
    global _sibling_group_counter
    _sibling_group_counter = 0


def run_beam_search(
    config: BeamSearchConfig,
    generate_fn: Callable[[str, int], list[str]],
    reward_fn: Callable[[str, object], float],
    reference: object,
) -> list[BeamTree]:
    """Run element-level beam search to generate HTML with visual feedback.

    For each of M independent beams, starts from an empty root and iteratively:
    1. For each active node, calls generate_fn to get K continuations
    2. Scores each continuation with reward_fn
    3. Keeps top-N non-terminal children as active nodes for the next step

    All branches (including pruned ones) are retained in the tree for training.

    Args:
        config: BeamSearchConfig with M, K, N, max_steps.
        generate_fn: Callable(prefix: str, k: int) -> list[str].
            Given the current HTML prefix and number of branches K, returns
            K continuation strings. An empty string signals EOS (terminal).
        reward_fn: Callable(full_html: str, reference: object) -> float.
            Scores the full rendered HTML against the reference. Returns a
            float reward (higher = better).
        reference: Reference object passed to reward_fn (e.g., reference image
            path, ground truth HTML, etc.).

    Returns:
        List of M BeamTree objects, one per independent beam.
    """
    trees = []

    for _beam_idx in range(config.M):
        root = BeamNode(prefix="", continuation="", step=0, hybrid_reward=0.0)
        all_nodes = [root]
        active_nodes = [root]

        for step in range(config.max_steps):
            all_children = []

            for parent in active_nodes:
                group_id = _next_sibling_group_id()
                continuations = generate_fn(parent.full_text, config.K)

                for cont in continuations:
                    is_terminal = (cont == "")
                    child = BeamNode(
                        prefix=parent.full_text,
                        continuation=cont,
                        step=step + 1,
                        parent=parent,
                        sibling_group_id=group_id,
                        is_terminal=is_terminal,
                    )

                    # Score the full HTML at this state
                    if is_terminal:
                        # Terminal nodes inherit parent's reward (no new content)
                        child.hybrid_reward = parent.hybrid_reward
                    else:
                        child.hybrid_reward = reward_fn(child.full_text, reference)

                    child.step_reward = child.hybrid_reward - parent.hybrid_reward
                    parent.children.append(child)
                    all_nodes.append(child)
                    all_children.append(child)

            # Select top-N non-terminal children by cumulative hybrid_reward
            non_terminal = [c for c in all_children if not c.is_terminal]

            if not non_terminal:
                # All children are terminal — beam search is done
                break

            # Sort by hybrid_reward descending, keep top N
            non_terminal.sort(key=lambda n: n.hybrid_reward, reverse=True)
            active_nodes = non_terminal[:config.N]

        tree = BeamTree(root=root, _all_nodes=all_nodes)
        trees.append(tree)

    return trees
