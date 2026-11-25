"""Deterministic checks for the shared scenario and market event loop."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from cafe.config import load_config
from cafe.contracts import Action, Arrival, DayResult, FEATURE_NAMES, Scenario, build_catalogue
from cafe.customers.policy import CustomerPolicy
from cafe.managers.fixed import FixedManager
from cafe.simulation.scenarios import customer_rng, generate_scenario, load_scenario, save_scenario
from cafe.simulation.simulator import build_observation, period_profit, run_day


class DeterministicPolicy:
    """Join the configured product/café and wait whenever allowed; record learning rewards."""

    frozen = False

    def __init__(self, arrival_action: int = 0) -> None:
        """Select one arrival action; the outside option remains the fallback."""
        self.arrival_action = arrival_action
        self.updates = []

    def choose(self, type_id: int, state: tuple, valid_actions: tuple[int, ...],
               *, epsilon: float, rng: np.random.Generator) -> int:
        """Make a deterministic masked choice without relying on Q-table training."""
        preferred = self.arrival_action if state[0] == "arrival" else 0

        return preferred if preferred in valid_actions else valid_actions[-1]

    def update(self, pending: object, reward: float, next_state: tuple | None,
               next_valid_actions: tuple[int, ...], *, terminal: bool, alpha: float, gamma: float) -> None:
        """Record each resolved transition so duplicated waiting penalties are visible."""
        self.updates.append((pending, reward, next_state, terminal))


class ChangingManager(FixedManager):
    """Change menu after the opening period to exercise carried queues."""

    def __init__(self, first: Action, later: Action) -> None:
        """Retain two valid catalogue entries."""
        super().__init__(first)
        self.first = first
        self.later = later
        self.calls = 0

    def reset(self) -> None:
        """Restart the action sequence at opening."""
        super().reset()
        self.calls = 0

    def choose(self, history: np.ndarray) -> Action:
        """Choose the first or later menu and retain ordinary decision diagnostics."""
        self.action = self.first if self.calls == 0 else self.later
        self.calls += 1

        return super().choose(history)


class SimulatorTests(unittest.TestCase):
    """Exercise conservation, costs, FIFO capacity and period/closing boundaries."""

    def setUp(self) -> None:
        """Use a short day with ordinary validated product and customer settings."""
        base = load_config(Path(__file__).resolve().parents[1] / "config/default.json")
        self.config = replace(base, simulation={**base.simulation, "day_minutes": 8, "period_minutes": 4})
        self.catalogue = build_catalogue(self.config)

    def scenario(self, minutes: tuple[int, ...], type_id: int = 0) -> Scenario:
        """Build ordered fixture arrivals with stable IDs and independent tie seeds."""
        return Scenario("fixture", "test", 11,
                        tuple(Arrival(i, minute, type_id, i + 100) for i, minute in enumerate(minutes)))

    def run_fixture(self, minutes: tuple[int, ...], action_id: int = 0,
                    policy: DeterministicPolicy | None = None, **kwargs: object) -> "DayResult":
        """Run the public entry point using a deterministic customer policy."""
        return run_day(self.scenario(minutes), policy or DeterministicPolicy(),
                       FixedManager(self.catalogue[action_id]), self.config,
                       trajectory_id="fixture", **kwargs)

    def test_minute_capture(self) -> None:
        """Check fractional service, closing and inventory without altering outcomes."""
        import pickle

        arrivals = (0,) * 5 + (7,) * 4
        rows = []
        traced = self.run_fixture(arrivals, snapshots=rows)
        reference = self.run_fixture(arrivals)
        self.assertEqual(pickle.dumps(traced), pickle.dumps(reference))
        self.assertEqual(rows[0]["stock"], [6, 6, 6])
        self.assertEqual(rows[1]["stock"], [5, 6, 6])
        self.assertEqual(rows[3]["stock"], [4, 6, 6])
        self.assertEqual(rows[4]["stock"], [6, 6, 6])
        self.assertEqual(rows[3]["queue"], rows[4]["queue"])
        self.assertGreater(rows[3]["queue"], 0)
        self.assertEqual(rows[-1]["queue"], 0)
        self.assertGreater(rows[-1]["left"], 0)

        for last, record in zip(rows[3::4], traced.periods):
            self.assertEqual(last["left"], record.abandonment)
            self.assertEqual(last["stock"], (record.action.stock - record.sales).tolist())

    def test_zero_arrival_accounting_and_observation(self) -> None:
        """No demand still incurs preparation and staffing costs in both cafés."""
        result = self.run_fixture(())

        for row in result.periods:
            np.testing.assert_array_equal(row.sales, [0, 0, 0])
            np.testing.assert_array_equal(row.waste, [6, 6, 6])
            self.assertEqual((row.preparation_cost, row.staff_cost, row.profit), (138, 40, -178))
            self.assertFalse(row.wait_valid)
            self.assertEqual(row.diagnostics["competitor"]["profit"], -310)

        opening = result.periods[0].observation.values
        np.testing.assert_array_equal(opening[2:22], np.zeros(20))
        self.assertEqual(opening[1], 1)
        self.assertEqual(opening[-1], 1)
        self.assertFalse(result.diagnostics["abandonment_fraction_valid"])
        np.testing.assert_array_equal(result.periods[1].observation.values[10:13], [6, 6, 6])
        self.assertEqual(len(FEATURE_NAMES), 30)

    def test_profit_example(self) -> None:
        """The specification's scalar cash example equals HKD 410."""
        action = Action(0, (30, 0, 0), 2, (40, 0, 0))
        self.assertEqual(period_profit(action, np.array([35, 0, 0]), np.array([10, 0, 0]), 120), 410)

    def test_fifo_capacity_and_waiting_rewards(self) -> None:
        """One employee serves at ticks 1 and 3, assigning each wait cost once."""
        policy = DeterministicPolicy()
        result = self.run_fixture((0, 0), policy=policy, learn=True, exploration_seed=12)
        row = result.periods[0]
        self.assertEqual(row.sales.tolist(), [2, 0, 0])
        self.assertEqual(row.mean_wait, 2)
        self.assertEqual(row.diagnostics["served_wait_total"], 4)
        np.testing.assert_allclose(result.diagnostics["episode_returns_by_type"][0], [4.8, 4.0])
        terminal = [update for update in policy.updates if update[3]]
        self.assertEqual(len(terminal), 2)
        np.testing.assert_allclose([update[1] for update in terminal], [5.2, 5.2])
        self.assertAlmostEqual(sum(update[1] for update in policy.updates), 8.8)

    def test_empty_time_does_not_bank_slots(self) -> None:
        """Unused whole capacity disappears instead of serving a later burst."""
        result = self.run_fixture((2, 2, 2), action_id=3)
        self.assertEqual(result.periods[0].sales.tolist(), [2, 0, 0])
        self.assertEqual(result.periods[0].diagnostics["end_queue"], 1)
        self.assertEqual(result.periods[1].sales.tolist(), [1, 0, 0])

    def test_fractional_credit_survives_period_boundary(self) -> None:
        """Odd-length periods preserve the fractional part of available capacity."""
        config = replace(self.config, simulation={**self.config.simulation, "day_minutes": 6, "period_minutes": 3})
        result = run_day(self.scenario((2,)), DeterministicPolicy(), FixedManager(self.catalogue[0]),
                         config, trajectory_id="fractional")
        self.assertEqual(result.periods[0].sales.sum(), 0)
        self.assertEqual(result.periods[1].sales.sum(), 1)
        self.assertEqual(result.periods[1].mean_wait, 1)

    def test_current_price_sales_across_boundary(self) -> None:
        """An earlier join consumes fresh stock and pays the new period's posted price."""
        manager = ChangingManager(self.catalogue[0], self.catalogue[18])
        result = run_day(self.scenario((2, 2)), DeterministicPolicy(), manager, self.config,
                         trajectory_id="changing")
        first, second = result.periods
        self.assertEqual(first.revenue, 19.8)
        self.assertEqual(second.revenue, 24.2)
        self.assertEqual(second.entries, 0)
        self.assertEqual(second.sales.tolist(), [1, 0, 0])
        self.assertEqual(second.mean_wait, 3)
        self.assertEqual(second.observation.values[13], 1)
        self.assertEqual(second.observation.values[2], 2)
        self.assertEqual(second.observation.values[15], 19.8)

    def test_last_slot_serves_existing_waiter_first(self) -> None:
        """The minute-six arrival gets the minute-seven slot ahead of the new arrival."""
        last = self.run_fixture((6, 7)).periods[-1]

        self.assertEqual(last.sales.tolist(), [1, 0, 0])
        self.assertEqual(last.mean_wait, 1)
        self.assertEqual(last.diagnostics["departure_reasons"], {"closing": 1})

    def test_closing_departures_and_terminal_wait(self) -> None:
        """Final-minute arrivals left after service depart before accounting is saved."""
        policy = DeterministicPolicy()
        result = self.run_fixture((7, 7, 7), policy=policy, learn=True, exploration_seed=2)
        last = result.periods[-1]
        self.assertEqual((last.entries, last.sales.sum(), last.abandonment), (3, 1, 2))
        self.assertEqual(last.diagnostics["departure_reasons"], {"closing": 2})
        self.assertEqual(last.diagnostics["end_queue"], 0)
        self.assertEqual(result.diagnostics["end_queue"], (0, 0))
        np.testing.assert_allclose([update[1] for update in policy.updates if update[3]], [5.2, -0.4, -0.4])
        self.assertEqual(result.diagnostics["abandonment_fraction"], 2 / 3)

    def test_stockouts_and_customer_conservation(self) -> None:
        """Selling out removes customers without consuming slots and never oversells."""
        config = replace(self.config, catalogue={**self.config.catalogue, "stock_profiles": [[1, 1, 1], [2, 2, 2], [3, 3, 3]]})
        result = run_day(self.scenario((0, 0, 0, 0)), DeterministicPolicy(),
                         FixedManager(build_catalogue(config)[6]), config, trajectory_id="stock")
        first = result.periods[0]
        self.assertEqual(first.sales.tolist(), [1, 0, 0])
        self.assertEqual(first.abandonment, 3)
        self.assertEqual(first.diagnostics["departure_reasons"], {"stockout": 3})
        self.assertEqual(sum(result.diagnostics["conservation"].values()), 4)

        for row in result.periods:
            np.testing.assert_array_equal(row.sales + row.waste, row.action.stock)
            self.assertTrue(np.all(row.waste >= 0))
            self.assertEqual(row.entries, int(row.sales.sum()) + row.abandonment)

    def test_fifo_order_and_departures_do_not_use_slots(self) -> None:
        """A stockout between two different products leaves the second service slot usable."""
        class ProductPolicy(DeterministicPolicy):
            """Use type IDs as chosen products to make FIFO purchase order observable."""

            def choose(self, type_id: int, state: tuple, valid_actions: tuple[int, ...],
                       *, epsilon: float, rng: np.random.Generator) -> int:
                """Choose the type-indexed product on arrival and wait while permitted."""
                preferred = type_id if state[0] == "arrival" else 0

                return preferred if preferred in valid_actions else valid_actions[-1]

        config = replace(self.config, catalogue={**self.config.catalogue,
                                                "stock_profiles": [[1, 1, 1], [2, 2, 2], [3, 3, 3]]})
        scenario = Scenario("fifo", "test", 0, (Arrival(0, 1, 2, 0), Arrival(1, 1, 2, 1), Arrival(2, 1, 0, 2)))
        policy = ProductPolicy()
        result = run_day(scenario, policy, FixedManager(build_catalogue(config)[6]), config,
                         trajectory_id="fifo", learn=True, exploration_seed=0)
        first = result.periods[0]
        self.assertEqual(first.sales.tolist(), [1, 0, 1])
        self.assertEqual(first.mean_wait, 0)
        self.assertEqual(first.abandonment, 1)
        self.assertEqual([pending.type_id for pending, _, _, terminal in policy.updates if terminal], [2, 2, 0])

    def test_price_and_patience_force_departures(self) -> None:
        """Queued customers may leave after a price rise or reaching their wait limit."""
        types = tuple({**profile, "budget": 21} for profile in self.config.customer_types)
        config = replace(self.config, customer_types=types)
        result = run_day(self.scenario((2, 2)), DeterministicPolicy(),
                         ChangingManager(self.catalogue[0], self.catalogue[18]), config,
                         trajectory_id="price")
        self.assertEqual(result.periods[1].diagnostics["departure_reasons"], {"price": 1})
        self.assertEqual(result.periods[1].sales.sum(), 0)
        types = tuple({**profile, "max_wait": 1} for profile in self.config.customer_types)
        config = replace(self.config, customer_types=types)
        policy = DeterministicPolicy()
        result = run_day(self.scenario((0,)), policy, FixedManager(self.catalogue[0]), config,
                         trajectory_id="patience", learn=True, exploration_seed=3)
        self.assertEqual(result.periods[0].diagnostics["departure_reasons"], {"patience": 1})
        self.assertEqual(policy.updates[-1][1], -0.4)
        self.assertTrue(policy.updates[-1][3])

    def test_competitor_and_outside_are_not_A_abandonment(self) -> None:
        """Outside choices and B sales conserve customers without affecting A entries."""
        outside = self.run_fixture((0, 1), policy=DeterministicPolicy(6))
        competitor = self.run_fixture((0, 1), policy=DeterministicPolicy(3))
        self.assertEqual(outside.diagnostics["outside_count"], 2)
        self.assertEqual(competitor.diagnostics["conservation"], {"purchased_B": 2})
        self.assertEqual(competitor.periods[0].diagnostics["competitor"]["sales"], [2, 0, 0])
        self.assertEqual(competitor.diagnostics["entries"], 0)
        self.assertEqual(competitor.diagnostics["abandonment"], 0)

    def test_replay_resets_state_and_preserves_frozen_policy(self) -> None:
        """A saved scenario replays identically and another manager cannot mutate it."""
        scenario = generate_scenario(self.config, "saved", "test", 5)
        policy = CustomerPolicy(frozen=True)
        fingerprint = policy.fingerprint()
        manager = ChangingManager(self.catalogue[0], self.catalogue[18])

        with TemporaryDirectory() as directory:
            path = Path(directory) / "day.npz"
            save_scenario(scenario, path)
            loaded = load_scenario(path)

        self.assertEqual(scenario, loaded)
        first = run_day(scenario, policy, manager, self.config, trajectory_id="replay")
        second = run_day(loaded, policy, manager, self.config, trajectory_id="replay")
        self.assertEqual(first.diagnostics, second.diagnostics)

        for before, after in zip(first.periods, second.periods):
            self.assertEqual(before.profit, after.profit)
            np.testing.assert_array_equal(before.sales, after.sales)
            np.testing.assert_array_equal(before.observation.values, after.observation.values)

        run_day(scenario, policy, FixedManager(self.catalogue[26]), self.config, trajectory_id="other")
        self.assertEqual(policy.fingerprint(), fingerprint)
        self.assertEqual(scenario, loaded)
        self.assertEqual(generate_scenario(self.config, "saved", "test", 5), scenario)

    def test_keyed_randomness_is_independent_of_extra_draws(self) -> None:
        """Extra policy calls never shift a later customer's decision stream."""
        scenario = self.scenario((0, 0))
        expected = customer_rng(scenario, 1, 0, "arrival").random(8)
        customer_rng(scenario, 0, 0, "arrival").random(1000)
        np.testing.assert_array_equal(customer_rng(scenario, 1, 0, "arrival").random(8), expected)
        self.assertFalse(np.array_equal(customer_rng(scenario, 1, 1, "waiting").random(8), expected))

    def test_reject_invalid_inputs(self) -> None:
        """Malformed schedules and unsupported learning settings fail at the boundary."""
        with self.assertRaisesRegex(ValueError, "epsilon"):
            self.run_fixture((), epsilon=0.1)

        with self.assertRaisesRegex(ValueError, "exploration_seed"):
            self.run_fixture((), learn=True)

        with self.assertRaisesRegex(ValueError, "arrival minute"):
            self.run_fixture((8,))

        with TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(path, arrivals=np.zeros((2, 3)), seed=0, scenario_id="bad", group="test")

            with self.assertRaisesRegex(ValueError, "bad.npz.*shape"):
                load_scenario(path)

        with self.assertRaisesRegex(ValueError, "previous"):
            build_observation(1, None, (0, 0), self.config)


if __name__ == "__main__":
    unittest.main()
