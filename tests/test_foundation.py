"""Check the boundaries shared by the customer and simulation implementations."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cafe.cli import smoke
from cafe.config import config_hash, load_config, validate_config
from cafe.contracts import Action, FEATURE_NAMES, Observation, build_catalogue, make_history, validate_action
from cafe.managers.fixed import FixedManager


class FoundationTests(unittest.TestCase):
    """Exercise the fixed manager and the common record schema."""

    def setUp(self) -> None:
        """Load the same smoke configuration used by the CLI."""
        self.config = load_config(Path("config/smoke.json"))

    def test_catalogue(self) -> None:
        """Stable IDs must identify all numeric settings, including stock."""
        catalogue = build_catalogue(self.config)
        self.assertEqual(len(set(catalogue)), 27)
        self.assertEqual(catalogue[13], Action(13, (22, 30, 18), 2, (10, 10, 10)))
        with self.assertRaises(ValueError):
            validate_action(replace(catalogue[13], stock=(6, 6, 6)), catalogue)

        manager = FixedManager(catalogue[13])
        self.assertEqual(manager.choose(make_history([], 4)), catalogue[13])
        self.assertEqual(manager.last_decision.action_id, 13)
        manager.reset()
        self.assertIsNone(manager.last_decision)

    def test_history_snapshot(self) -> None:
        """Mutating simulator memory cannot change past observations or padding."""
        values = np.zeros(30)
        values[-1] = 1
        observation = Observation(0, values)
        values[3] = 999
        window = make_history([observation], 4)
        self.assertEqual(window.shape, (4, len(FEATURE_NAMES)))
        self.assertEqual(window[:, -1].tolist(), [0, 0, 0, 1])
        self.assertEqual(window[-1, 3], 0)
        with self.assertRaises(ValueError):
            make_history([observation, Observation(2, values)], 4)

    def test_configuration(self) -> None:
        """Smoke inherits costs but has a distinct duration and fingerprint."""
        full = load_config(Path("config/default.json"))
        self.assertEqual(full.products, self.config.products)
        self.assertNotEqual(config_hash(full), config_hash(self.config))
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_config(replace(full, simulation={**full.simulation, "day_minutes": 31}))

    def test_replay_contract_fixture(self) -> None:
        """Two runs must write identical, readable contract fixtures."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            smoke(self.config, output)
            saved = (output / "contracts.json").read_bytes()
            smoke(self.config, output)
            self.assertEqual(saved, (output / "contracts.json").read_bytes())
            self.assertEqual(json.loads(saved)["stage"], "contracts")
