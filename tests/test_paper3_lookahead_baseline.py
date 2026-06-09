import unittest
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from paper3_lookahead_baseline import (
    build_output_paths,
    interpret_township_b_result,
    select_lookahead_action,
)


class TreeEnv:
    """Small deterministic environment for testing lookahead action selection."""

    def __init__(self):
        self.state = ()
        self.transitions = {
            (): {0: 1.0, 1: 2.0},
            (0,): {2: 100.0},
            (1,): {3: 0.0},
        }

    def action_masks(self):
        import numpy as np

        actions = self.transitions.get(self.state, {})
        if not actions:
            return np.zeros(4, dtype=bool)
        mask = np.zeros(4, dtype=bool)
        for action in actions:
            mask[action] = True
        return mask

    def step(self, action):
        reward = self.transitions[self.state][int(action)]
        self.state = self.state + (int(action),)
        terminated = self.state not in self.transitions
        info = {"action": int(action)}
        return None, reward, terminated, False, info


class PruningRiskEnv:
    """Environment where immediate-reward pruning hides the best depth-2 action."""

    def __init__(self):
        self.state = ()
        self.transitions = {
            (): {0: 0.0, 1: 5.0, 2: 6.0},
            (0,): {3: 100.0},
            (1,): {3: 0.0},
            (2,): {3: 0.0},
        }

    def action_masks(self):
        import numpy as np

        actions = self.transitions.get(self.state, {})
        mask = np.zeros(4, dtype=bool)
        for action in actions:
            mask[action] = True
        return mask

    def step(self, action):
        reward = self.transitions[self.state][int(action)]
        self.state = self.state + (int(action),)
        terminated = self.state not in self.transitions
        info = {"action": int(action)}
        return None, reward, terminated, False, info


def snapshot_tree(env):
    return env.state


def restore_tree(env, snapshot):
    env.state = snapshot


class LookaheadBaselineTests(unittest.TestCase):
    def test_depth_one_selects_best_immediate_reward(self):
        env = TreeEnv()

        decision = select_lookahead_action(
            env,
            depth=1,
            beam_width=2,
            snapshot_fn=snapshot_tree,
            restore_fn=restore_tree,
        )

        self.assertEqual(decision.action, 1)
        self.assertEqual(decision.sequence, (1,))
        self.assertEqual(decision.score, 2.0)

    def test_depth_two_selects_best_cumulative_reward(self):
        env = TreeEnv()

        decision = select_lookahead_action(
            env,
            depth=2,
            beam_width=2,
            snapshot_fn=snapshot_tree,
            restore_fn=restore_tree,
        )

        self.assertEqual(decision.action, 0)
        self.assertEqual(decision.sequence, (0, 2))
        self.assertEqual(decision.score, 101.0)

    def test_selection_restores_environment_state(self):
        env = TreeEnv()

        select_lookahead_action(
            env,
            depth=2,
            beam_width=2,
            snapshot_fn=snapshot_tree,
            restore_fn=restore_tree,
        )

        self.assertEqual(env.state, ())

    def test_zero_beam_width_evaluates_all_valid_actions(self):
        env = PruningRiskEnv()

        decision = select_lookahead_action(
            env,
            depth=2,
            beam_width=0,
            snapshot_fn=snapshot_tree,
            restore_fn=restore_tree,
        )

        self.assertEqual(decision.action, 0)
        self.assertEqual(decision.sequence, (0, 3))
        self.assertEqual(decision.score, 100.0)

    def test_default_output_paths_encode_depth_and_beam(self):
        out_json, out_tex = build_output_paths(depth=2, beam_width=0, suffix="")

        self.assertEqual(out_json.name, "paper3_lookahead_d2_ball_results.json")
        self.assertEqual(out_tex.name, "paper3_lookahead_d2_ball_table_fragment.tex")

    def test_custom_output_suffix_is_sanitized(self):
        out_json, out_tex = build_output_paths(
            depth=3,
            beam_width=8,
            suffix="Township B sensitivity",
        )

        self.assertEqual(
            out_json.name,
            "paper3_lookahead_d3_b8_township_b_sensitivity_results.json",
        )
        self.assertEqual(
            out_tex.name,
            "paper3_lookahead_d3_b8_township_b_sensitivity_table_fragment.tex",
        )

    def test_interpretation_marks_positive_baimu_area_as_search_substitute(self):
        result = {"B": {"baimu_area_change_ha": 8.5, "depth": 2, "beam_width": 0}}

        message = interpret_township_b_result(result)

        self.assertIn("finite-depth search can reproduce", message)
        self.assertIn("multi-step planning rather than RL-specific learning", message)

    def test_interpretation_marks_negative_baimu_area_as_drl_advantage(self):
        result = {"B": {"baimu_area_change_ha": -12.0, "depth": 2, "beam_width": 0}}

        message = interpret_township_b_result(result)

        self.assertIn("does not reproduce", message)
        self.assertIn("learned scheduler", message)

    def test_interpretation_requests_township_b_when_missing(self):
        result = {"A": {"baimu_area_change_ha": 1.0, "depth": 2, "beam_width": 0}}

        message = interpret_township_b_result(result)

        self.assertIn("Township B was not run", message)


if __name__ == "__main__":
    unittest.main()
