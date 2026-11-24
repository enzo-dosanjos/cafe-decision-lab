"""Check time-block decisions and the ranking used by the service-limit experiment."""

from importlib.util import find_spec
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from cafe.config import load_config
from cafe.contracts import build_catalogue


@unittest.skipUnless(find_spec("torch"), "requires the neural extra")
class ServiceStudyTests(unittest.TestCase):
    """Cover block boundaries, independent-day resets and the profit-only endpoint."""

    def test_blocks_and_reset(self) -> None:
        """Switch exactly after four periods and replay the opening action on reset."""
        with patch.object(sys, "path", [str(Path("scripts").resolve()), *sys.path]):
            from service_study import ScheduledManager

        actions = list(build_catalogue(load_config(Path("config/default.json")))[:4])
        manager = ScheduledManager(actions)
        decisions = [manager.choose(None).action_id for _ in range(16)]
        self.assertEqual(decisions, [a.action_id for a in actions for _ in range(4)])
        manager.reset()
        self.assertEqual(manager.choose(None), actions[0])

    def test_constraint_and_unconstrained_ranking(self) -> None:
        """An infeasible best-profit action becomes eligible at the profit-only endpoint."""
        with patch.object(sys, "path", [str(Path("scripts").resolve()), *sys.path]):
            from service_study import rank

        rows = [{"mean_daily_profit": 9, "mean_period_abandonment": 3},
                {"mean_daily_profit": 20, "mean_period_abandonment": 7}]
        self.assertIs(rank(rows, 2), rows[0])
        self.assertIs(rank(rows, 5), rows[0])
        self.assertIs(rank(rows, float("inf")), rows[1])
