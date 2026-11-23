"""Check day-local adaptive rules and business ranking on matched validation days."""

from pathlib import Path
import unittest

import numpy as np

from cafe.config import load_config
from cafe.contracts import FEATURE_NAMES, Scenario, build_catalogue
from cafe.customers.policy import CustomerPolicy
from cafe.evaluate import select_managers
from cafe.managers.adaptive import AdaptiveManager
from cafe.managers.fixed import FixedManager


class ManagerTests(unittest.TestCase):
    """Exercise thresholds, state reset and validation accounting."""

    def setUp(self) -> None:
        """Use the smoke market with the full action catalogue."""
        self.config = load_config(Path("config/smoke.json"))
        self.catalogue = build_catalogue(self.config)

    def test_ema_pressure_and_reset(self) -> None:
        """Opening uses demand twenty, then updates once from each previous period."""
        manager = AdaptiveManager(self.catalogue, self.config, alpha=0.6, buffer=1.25, queue_threshold=3)
        h = np.zeros((4, 30))
        h[-1, -1] = 1
        self.assertEqual(manager.choose(h), self.catalogue[13])
        self.assertEqual(manager.choose(h), self.catalogue[0])
        h[-1, FEATURE_NAMES.index("current_queue_A")] = 3
        self.assertEqual(manager.choose(h), self.catalogue[18])
        h[-1, FEATURE_NAMES.index("current_queue_A")] = 0
        h[-1, FEATURE_NAMES.index("previous_A_abandonment")] = 6
        h[-1, FEATURE_NAMES.index("previous_A_entries")] = 200
        self.assertEqual(manager.choose(h), self.catalogue[-1])
        manager.reset()
        self.assertIsNone(manager.last_decision)
        self.assertEqual(manager.choose(np.zeros_like(h)), self.catalogue[13])

    def test_matched_validation_includes_empty_period_costs(self) -> None:
        """Both candidates face the same empty day and the cheaper one wins."""
        managers = [FixedManager(self.catalogue[-1]), FixedManager(self.catalogue[0])]
        days = [Scenario("empty", "management", 1, ())]
        policy = CustomerPolicy(frozen=True)
        selected, trials = select_managers({"fixed": managers}, days, policy, self.config)
        self.assertIs(selected["fixed"], managers[1])
        self.assertEqual(trials[1]["days"][0]["actions"], [0, 0])
        self.assertLess(trials[1]["mean_daily_profit"], 0)
        self.assertEqual(trials[1]["violation_rate"], 0)

    def test_validation_fallback_prioritizes_abandonment(self) -> None:
        """An infeasible family minimizes abandonment, then maximizes profit among ties."""
        from unittest.mock import patch

        managers = [FixedManager(action) for action in self.catalogue[:3]]
        trials = [{"family": "fixed", "candidate": i, "scenario_id": "v", "periods": 1,
                   "profit": profit, "abandonment": abandonment, "violations": 1, "fallbacks": 0}
                  for i, (profit, abandonment) in enumerate(((100, 8), (10, 6), (20, 6)))]
        scenarios = [Scenario("v", "management", 1, ())]
        policy = CustomerPolicy(frozen=True)

        with patch("cafe.evaluate._validation_days", return_value=trials):
            selected, _ = select_managers({"fixed": managers}, scenarios, policy, self.config)
            self.assertIs(selected["fixed"], managers[2])
            trials[1]["abandonment"] = 5
            selected, _ = select_managers({"fixed": managers}, scenarios, policy, self.config)
            self.assertIs(selected["fixed"], managers[1])

    def test_paired_days_average_seeds_and_reject_missing_day(self) -> None:
        """Seed replicates share one day; a constant paired gain has a point interval."""
        from cafe.evaluate import paired_profit_intervals

        rows = [{"manager": manager, "scenario_id": str(day), "profit": day * 100 + gain}
                for manager, gain in (("fixed", 0), ("lstm-seed-0", 2), ("lstm-seed-1", 4))
                for day in range(3)]
        result = paired_profit_intervals(rows, "fixed", seed=0, resamples=50)[1]
        self.assertEqual(result["days"], 3)
        self.assertEqual(result["seeds"], 2)
        self.assertEqual((result["mean_profit_difference"], result["ci_lower"], result["ci_upper"]), (3, 3, 3))

        with self.assertRaisesRegex(ValueError, "identical scenario"):
            paired_profit_intervals(rows[:-1], "fixed", seed=0)

    def test_evaluation_empty_day_preserves_costs_and_missing_denominators(self) -> None:
        """Held-out output includes zero-sales costs and undefined fraction/wait fields."""
        from tempfile import TemporaryDirectory
        from cafe.contracts import ArtifactManifest
        from cafe.evaluate import evaluate
        import csv

        manifest = ArtifactManifest("1", "1", "config", "catalogue", {}, ("empty",), {}, (), (), {})
        managers = {"fixed": FixedManager(self.catalogue[0]), "adaptive": FixedManager(self.catalogue[0])}

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            days = evaluate(managers, [Scenario("empty", "management", 1, ())], CustomerPolicy(frozen=True),
                            self.config, output_dir=output, manifest=manifest)
            self.assertEqual(days[0]["profit"], -356)
            self.assertEqual(days[0]["sales"], 0)
            self.assertIsNone(days[0]["mean_wait"])
            self.assertIsNone(days[0]["abandonment_fraction"])

            with (output / "summary.csv").open() as stream:
                summary = list(csv.DictReader(stream))

            self.assertEqual(summary[0]["served_mean_wait"], "")
            self.assertEqual(summary[0]["abandonment_fraction_valid_days"], "0")
