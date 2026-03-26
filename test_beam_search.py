"""
Tests for beam_search.py — element-level beam search for Tree-VPO.

Run: python -m pytest training/utils/test_beam_search.py -v
"""

import os
import sys

import pytest

# Ensure training/ is on path so `from utils.X` works
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.beam_search import (
    BeamNode,
    BeamSearchConfig,
    BeamTree,
    reset_sibling_group_counter,
    run_beam_search,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_counter():
    """Reset sibling group counter before each test for determinism."""
    reset_sibling_group_counter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_generate_fn(continuations_by_prefix=None, default_cont="<div>"):
    """Create a mock generate_fn.

    Args:
        continuations_by_prefix: Optional dict mapping prefix -> list of
            continuations. If the prefix is not found, uses default_cont
            repeated K times.
        default_cont: Default continuation string when no mapping exists.

    Returns:
        Callable(prefix: str, k: int) -> list[str]
    """
    call_count = [0]

    def generate_fn(prefix, k):
        call_count[0] += 1
        if continuations_by_prefix and prefix in continuations_by_prefix:
            conts = continuations_by_prefix[prefix]
            # Return up to k, padding with default if needed
            return (conts + [default_cont] * k)[:k]
        return [f"{default_cont}{call_count[0]}-{i}>" for i in range(k)]

    return generate_fn


def make_reward_fn(reward_map=None, default_reward=0.5):
    """Create a mock reward_fn.

    Args:
        reward_map: Optional dict mapping full_text -> reward. If the text
            is not found, returns default_reward.
        default_reward: Default reward when no mapping exists.

    Returns:
        Callable(full_text: str, reference: object) -> float
    """
    def reward_fn(full_text, reference):
        if reward_map and full_text in reward_map:
            return reward_map[full_text]
        return default_reward

    return reward_fn


def make_length_reward_fn():
    """Create a reward_fn that returns reward proportional to text length.

    This makes rewards deterministic and predictable for testing: longer
    completions = higher reward.
    """
    def reward_fn(full_text, reference):
        return len(full_text) / 100.0

    return reward_fn


# ---------------------------------------------------------------------------
# Test 1: Node structure
# ---------------------------------------------------------------------------

class TestBeamNodeStructure:

    def test_root_node_defaults(self):
        """Root node should have sensible defaults."""
        root = BeamNode()
        assert root.prefix == ""
        assert root.continuation == ""
        assert root.step == 0
        assert root.hybrid_reward == 0.0
        assert root.step_reward == 0.0
        assert root.parent is None
        assert root.children == []
        assert root.is_leaf is True
        assert root.full_text == ""
        assert root.is_terminal is False

    def test_parent_child_links(self):
        """Parent-child relationships should be properly linked."""
        root = BeamNode(prefix="", continuation="", step=0)
        child = BeamNode(
            prefix="",
            continuation="<div>hello</div>",
            step=1,
            parent=root,
            hybrid_reward=0.6,
            step_reward=0.6,
        )
        root.children.append(child)

        assert child.parent is root
        assert child in root.children
        assert root.is_leaf is False
        assert child.is_leaf is True

    def test_full_text_concatenation(self):
        """full_text should be prefix + continuation."""
        node = BeamNode(prefix="<html>", continuation="<body>")
        assert node.full_text == "<html><body>"

    def test_full_text_empty(self):
        """full_text of root with no content should be empty."""
        root = BeamNode()
        assert root.full_text == ""


# ---------------------------------------------------------------------------
# Test 2: Beam search produces trees
# ---------------------------------------------------------------------------

class TestBeamSearchProducesTree:

    def test_produces_m_trees(self):
        """run_beam_search should return exactly M trees."""
        config = BeamSearchConfig(M=2, K=3, N=2, max_steps=3)
        generate_fn = make_generate_fn()
        reward_fn = make_reward_fn(default_reward=0.5)

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)

        assert len(trees) == 2

    def test_each_tree_has_nodes(self):
        """Each tree should contain nodes beyond just the root."""
        config = BeamSearchConfig(M=2, K=3, N=2, max_steps=3)
        generate_fn = make_generate_fn()
        reward_fn = make_reward_fn(default_reward=0.5)

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)

        for tree in trees:
            assert len(tree.all_nodes()) > 1, "Tree should have more than just root"
            assert tree.root.step == 0
            assert tree.root.parent is None

    def test_tree_has_root(self):
        """Each tree's root should be in the all_nodes list."""
        config = BeamSearchConfig(M=1, K=2, N=1, max_steps=2)
        generate_fn = make_generate_fn()
        reward_fn = make_reward_fn(default_reward=0.5)

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)

        assert trees[0].root in trees[0].all_nodes()

    def test_tree_depth(self):
        """Nodes should have step values from 0 to max_steps."""
        config = BeamSearchConfig(M=1, K=2, N=1, max_steps=3)
        generate_fn = make_generate_fn()
        reward_fn = make_reward_fn(default_reward=0.5)

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        tree = trees[0]

        steps = set(n.step for n in tree.all_nodes())
        assert 0 in steps
        # With N=1 active node and K=2, each step generates K children
        # and keeps N=1 survivor. So we should reach max_steps.
        assert max(steps) == config.max_steps


# ---------------------------------------------------------------------------
# Test 3: Beam search keeps top N
# ---------------------------------------------------------------------------

class TestBeamSearchKeepsTopN:

    def test_only_n_survive_to_next_step(self):
        """After each step, only N nodes should survive to the next step.

        With M=1, K=4, N=2, max_steps=2:
        - Step 1: root spawns 4 children, top 2 survive
        - Step 2: 2 survivors each spawn 4, total 8 children, top 2 survive
        """
        # Reward = text length / 100 -> deterministic ordering
        config = BeamSearchConfig(M=1, K=4, N=2, max_steps=2)
        generate_fn = make_generate_fn()
        reward_fn = make_length_reward_fn()

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        tree = trees[0]

        # Step 1 nodes: root has K=4 children
        step1_nodes = [n for n in tree.all_nodes() if n.step == 1]
        assert len(step1_nodes) == 4

        # Step 2 nodes: only N=2 of the step-1 nodes were active parents
        step2_nodes = [n for n in tree.all_nodes() if n.step == 2]
        # N=2 parents * K=4 children each = 8 children at step 2
        assert len(step2_nodes) == 2 * 4

        # Verify that step-2 nodes come from exactly N=2 distinct parents
        step2_parents = set(id(n.parent) for n in step2_nodes)
        assert len(step2_parents) == 2

    def test_pruned_nodes_still_in_tree(self):
        """Pruned (non-surviving) nodes should still be in the tree for training."""
        config = BeamSearchConfig(M=1, K=4, N=2, max_steps=2)
        generate_fn = make_generate_fn()
        reward_fn = make_length_reward_fn()

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        tree = trees[0]

        step1_nodes = [n for n in tree.all_nodes() if n.step == 1]
        # All 4 are in the tree even though only 2 survived
        assert len(step1_nodes) == 4

        # The 2 pruned nodes at step 1 should be leaves (no children)
        step1_leaves = [n for n in step1_nodes if n.is_leaf]
        assert len(step1_leaves) == 2  # 4 total - 2 survivors = 2 pruned


# ---------------------------------------------------------------------------
# Test 4: Training data extraction
# ---------------------------------------------------------------------------

class TestTrainingDataExtraction:

    def test_training_data_count(self):
        """All non-root nodes should appear in training data.

        With M=1, K=3, N=2, max_steps=2:
        - Step 1: 3 children from root
        - Step 2: 2 survivors * 3 children = 6 children
        - Total: 9 non-root nodes
        """
        config = BeamSearchConfig(M=1, K=3, N=2, max_steps=2)
        generate_fn = make_generate_fn()
        reward_fn = make_length_reward_fn()

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        data = trees[0].extract_training_data()

        assert len(data) == 9  # 3 + 2*3 = 9

    def test_training_data_fields(self):
        """Each training data entry should have all required fields."""
        config = BeamSearchConfig(M=1, K=2, N=1, max_steps=1)
        generate_fn = make_generate_fn()
        reward_fn = make_reward_fn(default_reward=0.7)

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        data = trees[0].extract_training_data()

        assert len(data) > 0
        required_fields = {
            "prefix", "continuation", "step_reward", "hybrid_reward",
            "step", "sibling_group_id", "is_terminal", "full_text",
        }
        for entry in data:
            assert set(entry.keys()) == required_fields, (
                f"Missing fields: {required_fields - set(entry.keys())}"
            )

    def test_training_data_prefix_matches_parent(self):
        """Each entry's prefix should be the parent node's full_text."""
        config = BeamSearchConfig(M=1, K=2, N=1, max_steps=2)
        generate_fn = make_generate_fn()
        reward_fn = make_length_reward_fn()

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        data = trees[0].extract_training_data()

        # Step-1 entries should have prefix == "" (root's full_text)
        step1_entries = [d for d in data if d["step"] == 1]
        for entry in step1_entries:
            assert entry["prefix"] == ""

        # Step-2 entries should have non-empty prefix
        step2_entries = [d for d in data if d["step"] == 2]
        for entry in step2_entries:
            assert entry["prefix"] != ""
            assert entry["full_text"] == entry["prefix"] + entry["continuation"]

    def test_training_data_excludes_root(self):
        """Root node should not appear in training data."""
        config = BeamSearchConfig(M=1, K=2, N=1, max_steps=1)
        generate_fn = make_generate_fn()
        reward_fn = make_reward_fn(default_reward=0.5)

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        data = trees[0].extract_training_data()

        # No entry should have step=0
        steps = [d["step"] for d in data]
        assert 0 not in steps


# ---------------------------------------------------------------------------
# Test 5: Sibling group IDs
# ---------------------------------------------------------------------------

class TestSiblingGroupIds:

    def test_siblings_share_group_id(self):
        """Children from the same parent at the same step share sibling_group_id."""
        config = BeamSearchConfig(M=1, K=3, N=2, max_steps=1)
        generate_fn = make_generate_fn()
        reward_fn = make_reward_fn(default_reward=0.5)

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        tree = trees[0]

        # Root's children should all share the same group ID
        step1_nodes = [n for n in tree.all_nodes() if n.step == 1]
        assert len(step1_nodes) == 3  # K=3
        group_ids = set(n.sibling_group_id for n in step1_nodes)
        assert len(group_ids) == 1, (
            f"All siblings should share one group ID, got {group_ids}"
        )

    def test_different_parents_different_group_ids(self):
        """Children from different parents should have different sibling_group_ids."""
        config = BeamSearchConfig(M=1, K=3, N=2, max_steps=2)
        generate_fn = make_generate_fn()
        reward_fn = make_length_reward_fn()

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        tree = trees[0]

        # Step 2 nodes come from N=2 different parents
        step2_nodes = [n for n in tree.all_nodes() if n.step == 2]
        assert len(step2_nodes) == 6  # 2 parents * 3 children

        # Group by parent
        parent_groups = {}
        for n in step2_nodes:
            pid = id(n.parent)
            if pid not in parent_groups:
                parent_groups[pid] = set()
            parent_groups[pid].add(n.sibling_group_id)

        # Each parent's children should share one group ID
        for pid, gids in parent_groups.items():
            assert len(gids) == 1, f"Parent {pid} children have multiple group IDs: {gids}"

        # Different parents should have different group IDs
        all_group_ids = [list(gids)[0] for gids in parent_groups.values()]
        assert len(set(all_group_ids)) == 2, (
            f"Expected 2 distinct group IDs for 2 parents, got {set(all_group_ids)}"
        )

    def test_group_ids_in_training_data(self):
        """Training data should reflect the sibling_group_id structure."""
        config = BeamSearchConfig(M=1, K=3, N=2, max_steps=2)
        generate_fn = make_generate_fn()
        reward_fn = make_length_reward_fn()

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        data = trees[0].extract_training_data()

        # Step 1: all 3 entries share one group ID
        step1 = [d for d in data if d["step"] == 1]
        step1_gids = set(d["sibling_group_id"] for d in step1)
        assert len(step1_gids) == 1

        # Step 2: 6 entries split into 2 groups of 3
        step2 = [d for d in data if d["step"] == 2]
        step2_gids = set(d["sibling_group_id"] for d in step2)
        assert len(step2_gids) == 2
        for gid in step2_gids:
            count = sum(1 for d in step2 if d["sibling_group_id"] == gid)
            assert count == 3, f"Group {gid} has {count} entries, expected 3"


# ---------------------------------------------------------------------------
# Test 6: Terminal detection
# ---------------------------------------------------------------------------

class TestTerminalDetection:

    def test_empty_continuation_is_terminal(self):
        """If generate_fn returns empty string, the branch should be terminal."""
        # Generate fn that returns empty strings for some branches
        call_count = [0]

        def gen_fn(prefix, k):
            call_count[0] += 1
            results = []
            for i in range(k):
                if i == 0:
                    results.append("")  # terminal
                else:
                    results.append(f"<div>{call_count[0]}-{i}</div>")
            return results

        config = BeamSearchConfig(M=1, K=3, N=2, max_steps=3)
        reward_fn = make_reward_fn(default_reward=0.5)

        trees = run_beam_search(config, gen_fn, reward_fn, reference=None)
        tree = trees[0]

        # At step 1: one terminal child, two non-terminal
        step1_nodes = [n for n in tree.all_nodes() if n.step == 1]
        terminal_nodes = [n for n in step1_nodes if n.is_terminal]
        non_terminal_nodes = [n for n in step1_nodes if not n.is_terminal]

        assert len(terminal_nodes) == 1
        assert len(non_terminal_nodes) == 2
        assert terminal_nodes[0].continuation == ""

    def test_terminal_nodes_not_expanded(self):
        """Terminal nodes should not be selected as active parents."""
        # Generate fn where first branch is always terminal
        def gen_fn(prefix, k):
            results = []
            for i in range(k):
                if i == 0:
                    results.append("")
                else:
                    results.append(f"<div>{len(prefix)}-{i}</div>")
            return results

        config = BeamSearchConfig(M=1, K=3, N=2, max_steps=3)
        reward_fn = make_reward_fn(default_reward=0.5)

        trees = run_beam_search(config, gen_fn, reward_fn, reference=None)
        tree = trees[0]

        # Terminal nodes at step 1 should have no children
        step1_terminal = [n for n in tree.all_nodes()
                          if n.step == 1 and n.is_terminal]
        for node in step1_terminal:
            assert node.is_leaf, "Terminal nodes should not be expanded"

    def test_all_terminal_stops_search(self):
        """If all generated continuations are terminal, search should stop."""
        def gen_fn(prefix, k):
            return [""] * k  # all terminal

        config = BeamSearchConfig(M=1, K=3, N=2, max_steps=10)
        reward_fn = make_reward_fn(default_reward=0.5)

        trees = run_beam_search(config, gen_fn, reward_fn, reference=None)
        tree = trees[0]

        # Should have root + K terminal children at step 1, then stop
        assert len(tree.all_nodes()) == 1 + config.K  # root + 3 terminal
        max_step = max(n.step for n in tree.all_nodes())
        assert max_step == 1  # stopped after step 1

    def test_terminal_inherits_parent_reward(self):
        """Terminal nodes should inherit parent's hybrid_reward."""
        def gen_fn(prefix, k):
            return [""] * k

        parent_reward = 0.42

        def reward_fn(full_text, reference):
            # Should not be called for terminal nodes
            return parent_reward

        config = BeamSearchConfig(M=1, K=2, N=1, max_steps=1)

        # Set up so root has a known reward (0.0 by default)
        trees = run_beam_search(config, gen_fn, reward_fn, reference=None)
        tree = trees[0]

        for node in tree.all_nodes():
            if node.is_terminal:
                assert node.hybrid_reward == node.parent.hybrid_reward


# ---------------------------------------------------------------------------
# Test 7: Step rewards are deltas
# ---------------------------------------------------------------------------

class TestStepRewardsAreDeltas:

    def test_step_reward_is_delta_from_parent(self):
        """step_reward should equal child.hybrid_reward - parent.hybrid_reward."""
        # Use length-based reward for determinism
        config = BeamSearchConfig(M=1, K=3, N=2, max_steps=3)
        generate_fn = make_generate_fn()
        reward_fn = make_length_reward_fn()

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        tree = trees[0]

        for node in tree.all_nodes():
            if node.parent is not None:
                expected_delta = node.hybrid_reward - node.parent.hybrid_reward
                assert node.step_reward == pytest.approx(expected_delta, abs=1e-10), (
                    f"Step {node.step}: step_reward={node.step_reward:.6f} "
                    f"!= delta={expected_delta:.6f} "
                    f"(child_reward={node.hybrid_reward:.6f}, "
                    f"parent_reward={node.parent.hybrid_reward:.6f})"
                )

    def test_step_reward_for_root_children(self):
        """Root children's step_reward should equal their hybrid_reward (root reward is 0)."""
        config = BeamSearchConfig(M=1, K=3, N=1, max_steps=1)
        generate_fn = make_generate_fn()
        reward_fn = make_reward_fn(default_reward=0.7)

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        tree = trees[0]

        for node in tree.all_nodes():
            if node.step == 1 and not node.is_terminal:
                assert node.step_reward == pytest.approx(0.7, abs=1e-10), (
                    f"Root child step_reward should be 0.7, got {node.step_reward}"
                )

    def test_step_reward_can_be_negative(self):
        """Step reward can be negative if child scores worse than parent."""
        # Reward decreases with text length
        def decreasing_reward_fn(full_text, reference):
            return max(0, 1.0 - len(full_text) / 50.0)

        config = BeamSearchConfig(M=1, K=2, N=1, max_steps=2)
        generate_fn = make_generate_fn()

        trees = run_beam_search(config, generate_fn, decreasing_reward_fn, reference=None)
        tree = trees[0]

        # With decreasing reward, step_rewards at deeper steps should be negative
        for node in tree.all_nodes():
            if node.parent is not None:
                expected = node.hybrid_reward - node.parent.hybrid_reward
                assert node.step_reward == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Test: BeamTree.leaves() and surviving_leaves()
# ---------------------------------------------------------------------------

class TestBeamTreeMethods:

    def test_leaves_returns_only_leaves(self):
        """leaves() should return only nodes with no children."""
        config = BeamSearchConfig(M=1, K=2, N=1, max_steps=2)
        generate_fn = make_generate_fn()
        reward_fn = make_reward_fn(default_reward=0.5)

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        tree = trees[0]

        leaves = tree.leaves()
        for leaf in leaves:
            assert leaf.is_leaf
            assert len(leaf.children) == 0

        # Non-leaf nodes should not be in leaves
        non_leaves = [n for n in tree.all_nodes() if not n.is_leaf]
        for n in non_leaves:
            assert n not in leaves

    def test_extract_training_data_includes_all_branches(self):
        """Training data should include pruned branches too."""
        config = BeamSearchConfig(M=1, K=4, N=1, max_steps=2)
        generate_fn = make_generate_fn()
        reward_fn = make_length_reward_fn()

        trees = run_beam_search(config, generate_fn, reward_fn, reference=None)
        data = trees[0].extract_training_data()

        # Step 1: 4 branches, step 2: 1 survivor * 4 branches = 4
        # Total: 8 non-root nodes
        assert len(data) == 8

        # Both pruned and surviving branches should be present
        step1 = [d for d in data if d["step"] == 1]
        assert len(step1) == 4  # All K=4 branches, not just N=1 survivor
