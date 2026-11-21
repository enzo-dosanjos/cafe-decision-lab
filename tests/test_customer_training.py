"""Verify customer evaluation denominators, acceptance gates and checkpoint selection."""

from dataclasses import replace
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from cafe.config import config_hash, load_config
from cafe.contracts import Arrival, ArtifactManifest, Scenario, build_catalogue
from cafe.customers.policy import CustomerPolicy
from cafe.customers.training import check_behavior, evaluate_customers, train_customers
from cafe.managers.fixed import FixedManager


class CustomerTrainingTests(unittest.TestCase):
    """Use short independent days with deterministic validation outcomes."""

    def setUp(self) -> None:
        """Build a two-day training set and a separate validation schedule."""
        config = load_config(Path("config/smoke.json"))
        self.config = replace(config, customer_training={
            **config.customer_training, "days": 2, "validation_interval": 1, "validation_days": 1,
        })
        self.managers = [FixedManager(build_catalogue(config)[13])]
        arrivals = (Arrival(0, 0, 0, 31),)
        self.training = [Scenario(f"t-{day}", "training", day, arrivals) for day in range(2)]
        self.validation = [Scenario("v-0", "validation", 3, arrivals)]
        self.manifest = ArtifactManifest("1", "1", config_hash(self.config), "fixture", {},
                                         ("t-0", "t-1", "v-0"), {}, (), (), {})

    def test_evaluation_counts_only_served_waits(self) -> None:
        """An immediate purchase has valid zero wait; a missing type has no mean."""
        state = ("arrival", 1, 0, 0, 7, 7)
        policy = CustomerPolicy({0: {state: [1, 0, 0, 0, 0, 0, 0]}}, frozen=True)
        fingerprint = policy.fingerprint()
        rows = evaluate_customers(policy, self.validation, self.managers, self.config)
        self.assertEqual(rows[0]["mean_return"], 3)
        self.assertEqual(rows[1]["purchase_fraction"], 1)
        self.assertEqual(rows[1]["mean_served_wait"], 0)
        self.assertIsNone(rows[2]["mean_served_wait"])
        self.assertEqual(policy.fingerprint(), fingerprint)

    def test_missing_evidence_rejects_untrained_policy(self) -> None:
        """An unseen state or tied zero Q values cannot pass behavioral checks."""
        self.assertFalse(any(check_behavior(CustomerPolicy(), self.config).values()))

    def test_behavior_requires_strict_preferences_in_matched_contexts(self) -> None:
        """Observe two arrival preferences and compare waiting at the same queue position."""
        opening = ("arrival", 2, 0, 0, 7, 7)
        congestion = ("waiting", 0, 0, 2, 2, 0, 1, 1, 0)
        tables = {
            0: {opening: [1, -1, 0, 0, 0, 0, 0], congestion: [0.5, 0]},
            1: {congestion: [-0.5, 0]},
        }
        self.assertTrue(all(check_behavior(CustomerPolicy(tables), self.config).values()))

        # A different product does not provide a matched waiting comparison.
        other_product = ("waiting", 0, 1, 2, 2, 0, 1, 1, 0)
        tables[1] = {other_product: [-0.5, 0]}
        self.assertFalse(check_behavior(CustomerPolicy(tables), self.config)["costly_long_wait"])

    def test_keep_earliest_tie_and_log_failed_checkpoints(self) -> None:
        """Isolate ranking from stochastic learning and check both output paths."""
        score = [{"type": "all", "mean_return": 3.0}]

        with TemporaryDirectory() as tmp, patch(
            "cafe.customers.training.evaluate_customers", return_value=score
        ), patch("cafe.customers.training.check_behavior", return_value={"probe": True}):
            path = Path(tmp)
            policy = train_customers(self.config, self.training, self.validation, self.managers,
                                     path, self.manifest)
            self.assertTrue(policy.frozen)

            with (path / "validation.csv").open() as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual([row["new_best"] for row in rows], ["True", "False"])
            self.assertEqual([float(row["epsilon"]) for row in rows], [1.0, 0.05])

        with TemporaryDirectory() as tmp:
            path = Path(tmp)

            with self.assertRaisesRegex(ValueError, "No eligible"):
                train_customers(self.config, self.training, self.validation, self.managers,
                                path, self.manifest)

            self.assertTrue((path / "validation.csv").is_file())
            self.assertFalse((path / "policy.json").exists())

    def test_reject_reused_scenario_seed(self) -> None:
        """Renaming a reused exogenous seed must not bypass split validation."""
        overlap = [replace(self.validation[0], seed=self.training[0].seed)]

        with TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "distinct scenario"):
            train_customers(self.config, self.training, overlap, self.managers, Path(tmp), self.manifest)
