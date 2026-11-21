"""Deterministic customer-state, masked learning and frozen persistence checks."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cafe.config import config_hash, load_config
from cafe.contracts import ArtifactManifest, CustomerView, FEATURE_NAMES, TARGET_NAMES
from cafe.customers.policy import (
    CustomerPolicy, PendingTransition, encode_state, load_policy, save_policy, valid_actions,
)


class CustomerTests(unittest.TestCase):
    """Exercise policy rules using only constructed customer observations."""

    def setUp(self) -> None:
        """Build an affordable observation and explicit artifact identity."""
        self.config = load_config(Path("config/default.json"))
        self.view = CustomerView(
            "arrival", 0, (1, 1), np.array([[22., 30., 18.], [22., 30., 18.]]),
            (0, 0), np.ones((2, 3), dtype=bool), None, None, 0, 0, 15, False,
        )
        self.waiting = replace(self.view, phase="waiting", cafe_id=1, product_id=2)
        self.state = encode_state(self.view)
        self.wait_state = encode_state(self.waiting)
        self.manifest = ArtifactManifest(
            "1", "1", config_hash(self.config), "catalogue-test", {"scenario": "test"},
            ("customer-train-0",), {"customer": 2}, FEATURE_NAMES, TARGET_NAMES,
            {"numpy": np.__version__},
        )

    def test_arrival_state_bins_and_bits(self) -> None:
        """Arrival state contains A's price, queue bins and ordered stock bits."""
        for length, expected in ((0, 0), (1, 1), (3, 1), (4, 2), (7, 2), (8, 3), (99, 3)):
            view = replace(self.view, queue_lengths=(length, length),
                           availability=np.array([[1, 0, 1], [0, 1, 0]]))
            self.assertEqual(encode_state(view), ("arrival", 1, expected, expected, 5, 2))

        self.assertEqual(encode_state(replace(self.view, type_id=2, price_profile_ids=(1, 2))), self.state)

    def test_waiting_state_bins(self) -> None:
        """Waiting uses the chosen café's price, queue position and minute bins."""
        view = replace(self.waiting, price_profile_ids=(0, 2), queue_position=4,
                       elapsed_wait=6, remaining_patience=3, final_minute=True)
        self.assertEqual(encode_state(view), ("waiting", 1, 2, 2, 2, 2, 1, 1, 1))

        for minutes, expected in ((0, 0), (2, 0), (3, 1), (5, 1), (6, 2), (10, 2), (11, 3)):
            self.assertEqual(encode_state(replace(self.waiting, elapsed_wait=minutes))[5], expected)

        for minutes, expected in ((0, 0), (1, 0), (2, 0), (3, 1), (5, 1), (6, 2), (15, 2)):
            self.assertEqual(encode_state(replace(self.waiting, remaining_patience=minutes))[6], expected)

    def test_arrival_mask(self) -> None:
        """Mask stock and cash budget, but allow negative utility and exact budget."""
        view = replace(self.view, prices=np.array([[35., 36., 34.], [22., 30., 18.]]),
                       availability=np.array([[1, 1, 1], [0, 1, 0]]))
        self.assertEqual(valid_actions(view, self.config), (0, 2, 4, 6))
        self.assertEqual(valid_actions(replace(view, availability=np.zeros((2, 3))), self.config), (6,))
        config = replace(self.config, customer_types=({**self.config.customer_types[0], "budget": 20},))
        self.assertEqual(valid_actions(self.view, config), (2, 5, 6))

    def test_waiting_mask(self) -> None:
        """Stock, budget and patience can force departure even before closing."""
        self.assertEqual(valid_actions(self.waiting, self.config), (0, 1))
        cases = (
            replace(self.waiting, remaining_patience=0),
            replace(self.waiting, elapsed_wait=15),
            replace(self.waiting, availability=np.zeros((2, 3))),
            replace(self.waiting, prices=np.full((2, 3), 36.)),
        )

        for view in cases:
            self.assertEqual(valid_actions(view, self.config), (1,))

        self.assertEqual(valid_actions(replace(self.waiting, elapsed_wait=14, remaining_patience=1), self.config), (0, 1))
        self.assertEqual(valid_actions(replace(self.waiting, type_id=1, elapsed_wait=5), self.config), (1,))

    def test_ties_exploration_and_masking(self) -> None:
        """Seeded ties replay and cover both cafés; masked maxima never win."""
        policy = CustomerPolicy({0: {self.state: [2, 100, 0, 2, 0, 0, 0]}})
        choices = [policy.choose(0, self.state, (0, 3, 6), epsilon=0,
                                 rng=np.random.default_rng(seed)) for seed in range(100)]
        replay = [policy.choose(0, self.state, (0, 3, 6), epsilon=0,
                                rng=np.random.default_rng(seed)) for seed in range(100)]
        self.assertEqual(choices, replay)
        self.assertEqual(set(choices), {0, 3})
        self.assertTrue(30 < choices.count(0) < 70)
        explored = {policy.choose(0, self.state, (0, 3, 6), epsilon=1,
                                  rng=np.random.default_rng(seed)) for seed in range(100)}
        self.assertEqual(explored, {0, 3, 6})
        self.assertEqual(CustomerPolicy().choose(0, self.state, (6,), epsilon=0,
                                                rng=np.random.default_rng(0)), 6)

    def test_q_updates(self) -> None:
        """Bootstrap only valid actions of the same type, never terminal states."""
        policy = CustomerPolicy({0: {self.wait_state: [100., 4.]},
                                 1: {self.wait_state: [500., 500.]}})
        pending = PendingTransition(0, self.state, 3)
        policy.update(pending, -0.4, self.wait_state, (1,), terminal=False, alpha=0.5, gamma=0.9)
        self.assertAlmostEqual(policy.tables[0][self.state][3], 1.6)
        policy.update(pending, 2., self.wait_state, (0, 1), terminal=True, alpha=0.5, gamma=1)
        self.assertAlmostEqual(policy.tables[0][self.state][3], 1.8)
        policy.update(PendingTransition(2, self.state, 6), 0, None, (), terminal=True, alpha=1, gamma=1)
        self.assertEqual(policy.tables[2][self.state][6], 0)
        policy.update(PendingTransition(2, self.state, 0), -0.7, self.wait_state, (0, 1),
                      terminal=False, alpha=1, gamma=1)
        self.assertEqual(policy.tables[2][self.state][0], -0.7)
        self.assertEqual(policy.tables[1][self.wait_state], [500., 500.])

    def test_freeze_is_independent(self) -> None:
        """Unseen choices, exposed rows and training cannot mutate a snapshot."""
        source = {0: {self.state: [0.] * 7}}
        policy = CustomerPolicy(source)
        frozen = policy.freeze()
        before = frozen.fingerprint()
        source[0][self.state][0] = 99
        frozen.tables[0][self.state][0] = 55
        policy.update(PendingTransition(0, self.state, 0), 10, None, (), terminal=True, alpha=1, gamma=1)
        frozen.choose(2, self.wait_state, (0, 1), epsilon=0, rng=np.random.default_rng(0))
        self.assertEqual(frozen.fingerprint(), before)
        self.assertNotEqual(policy.fingerprint(), before)

        with self.assertRaisesRegex(ValueError, "frozen"):
            frozen.update(PendingTransition(0, self.state, 0), 0, None, (), terminal=True, alpha=1, gamma=1)

        with self.assertRaisesRegex(ValueError, "epsilon"):
            frozen.choose(0, self.state, (6,), epsilon=0.1, rng=np.random.default_rng(0))

    def test_persistence_and_lineage(self) -> None:
        """Round trips retain decisions and reject changed lineage/schema/content."""
        policy = CustomerPolicy({2: {self.wait_state: [-2., 0.]}, 0: {self.state: [1.] * 7}})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "customers" / "policy.json"
            save_policy(policy, path, self.manifest)
            original = path.read_text()
            restored = load_policy(path, self.manifest)
            self.assertTrue(restored.frozen)
            self.assertEqual(restored.fingerprint(), policy.fingerprint())
            self.assertEqual(restored.choose(2, self.wait_state, (0, 1), epsilon=0,
                                             rng=np.random.default_rng(0)), 1)
            save_policy(restored, path, self.manifest)
            self.assertEqual(path.read_text(), original)

            for field, value in (("config_hash", "wrong"), ("parent_hashes", {}),
                                 ("feature_names", tuple(reversed(FEATURE_NAMES))),
                                 ("scenario_ids", ("other",)), ("simulator_version", "wrong")):
                with self.assertRaisesRegex(ValueError, field):
                    load_policy(path, replace(self.manifest, **{field: value}))

            for field, value in (("state_schema", "unknown"), ("fingerprint", "tampered")):
                changed = json.loads(original)
                changed[field] = value
                path.write_text(json.dumps(changed))

                with self.assertRaisesRegex(ValueError, field):
                    load_policy(path, self.manifest)

            changed = json.loads(original)
            changed["tables"][0]["states"][0]["values"] = [float("nan")] * 7
            path.write_text(json.dumps(changed))

            with self.assertRaisesRegex(ValueError, "finite"):
                load_policy(path, self.manifest)

    def test_invalid_inputs_fail_before_mutation(self) -> None:
        """Malformed observations and learning arguments produce useful errors."""
        for field, value in (("prices", np.ones((3, 2))), ("availability", np.full((2, 3), 2)),
                             ("phase", "unknown"), ("queue_position", -1)):
            with self.assertRaisesRegex(ValueError, field):
                encode_state(replace(self.view, **{field: value}))

        policy = CustomerPolicy()
        before = policy.fingerprint()

        for actions in ((), (7,), (0, 0)):
            with self.assertRaises(ValueError):
                policy.choose(0, self.state, actions, epsilon=0, rng=np.random.default_rng(0))

        with self.assertRaisesRegex(ValueError, "state"):
            policy.update(PendingTransition(0, self.state, 0), 1, None, (), terminal=False, alpha=1, gamma=1)

        with self.assertRaisesRegex(ValueError, "alpha"):
            policy.update(PendingTransition(0, self.state, 0), 1, None, (), terminal=True, alpha=2, gamma=1)

        self.assertEqual(policy.fingerprint(), before)
