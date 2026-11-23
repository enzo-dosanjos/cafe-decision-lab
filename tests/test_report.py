"""Check scenario-paired forecast uncertainty and representative-day selection."""

import unittest

from cafe.report import _forecast_intervals, select_representative_day


class ReportTests(unittest.TestCase):
    """Keep repeated trajectories and seeds inside their independent scenario day."""

    def test_forecast_constant_paired_difference(self) -> None:
        """Exact predictions versus constant errors give a known interval for MAE and RMSE."""
        rows = [{"manager": f"{family}-seed-{seed}", "scenario_id": str(day),
                 "predicted_sales": str(day + error), "observed_sales": str(day)}
                for family, error in (("mlp", 2), ("lstm", 1)) for seed in range(3)
                for day in range(4) for trajectory in range(2)]
        results = _forecast_intervals(rows, resamples=20)

        for row in results:
            self.assertEqual((row["difference"], row["ci_lower"], row["ci_upper"]), (-1, -1, -1))
            self.assertEqual(row["days"], 4)

    def test_representative_day_tie_uses_id(self) -> None:
        """Selection uses adaptive profit with a stable tie, never the neural winner."""
        rows = [{"manager": "adaptive", "scenario_id": name, "profit": profit}
                for name, profit in (("z", 0), ("a", 2))]
        self.assertEqual(select_representative_day(rows), "a")

    def test_business_pairing_averages_seeds_within_days(self) -> None:
        """Repeated seed results do not create more independent bootstrap days."""
        from cafe.report import _business_intervals

        rows = []

        for label, gain in (("fixed", 0), ("adaptive", 1), ("mlp-seed-0", 2),
                            ("mlp-seed-1", 4), ("lstm-seed-0", 5)):
            for day in range(3):
                row = {"manager": label, "scenario_id": str(day)}
                row.update({field: day * 100 + gain for field in
                            ("profit", "sales", "abandonment", "waste_units", "waste_cost", "preparation_cost", "staff_cost")})
                rows.append(row)

        results = _business_intervals(rows, resamples=20)
        direct = next(r for r in results if r["manager"] == "lstm" and r["baseline"] == "mlp")
        self.assertEqual((direct["mean_daily_difference"], direct["ci_lower"], direct["ci_upper"]), (2, 2, 2))
        self.assertEqual(direct["days"], 3)
