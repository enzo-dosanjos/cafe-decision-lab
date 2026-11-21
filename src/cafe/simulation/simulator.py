"""Run the shared FIFO market, charging current prices and all prepared stock."""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..config import Config, validate_config
from ..contracts import (Action, Array, Arrival, CustomerView, DayResult, FEATURE_NAMES,
                         Observation, PeriodRecord, Scenario, build_catalogue,
                         make_history, validate_action)
from ..managers.base import Manager
from .scenarios import _validate_scenario, customer_rng

if TYPE_CHECKING:
    from ..customers.policy import CustomerPolicy, PendingTransition


@dataclass
class CustomerInstance:
    """Own a customer's queue placement, accrued wait and unfinished Q transition."""

    arrival: Arrival
    cafe_id: int = -1
    product_id: int = -1
    elapsed_wait: int = 0
    pending: "PendingTransition | None" = None
    pending_reward: float = 0.0
    episode_return: float = 0.0


@dataclass
class CafeState:
    """Keep queue and fractional credit across periods, replacing inventory each period."""

    action: Action
    inventory: Array
    queue: deque[CustomerInstance] = field(default_factory=deque)
    service_credit: float = 0.0
    sales: Array = field(default_factory=lambda: np.zeros(3, dtype=int))
    entries: int = 0
    abandonment: int = 0
    served_wait: int = 0
    departed_wait: int = 0
    departure_reasons: Counter = field(default_factory=Counter)


def period_profit(action: Action, sales: Array, unit_costs: Array, staff_period_cost: float) -> float:
    """Return HKD revenue less all preparation and staff costs, without a waste penalty."""
    sales = np.asarray(sales, dtype=float)
    unit_costs = np.asarray(unit_costs, dtype=float)

    if (sales.shape != (3,) or unit_costs.shape != (3,) or not np.isfinite(sales).all()
            or not np.isfinite(unit_costs).all() or np.any(sales < 0)
            or np.any(sales > action.stock) or np.any(unit_costs < 0)
            or not np.isfinite(staff_period_cost) or staff_period_cost < 0):
        raise ValueError("sales/unit_costs must be finite nonnegative [3] arrays, sales <= stock; "
                         "staff_period_cost must be finite and nonnegative")

    return float(np.dot(action.prices, sales) - np.dot(unit_costs, action.stock)
                 - staff_period_cost * action.staff)


def build_observation(period_index: int, previous: PeriodRecord | None,
                      queue_lengths: tuple[int, int], config: Config) -> Observation:
    """Build the 30 causal features; opening previous metrics and action are zero."""
    period_count = config.simulation["day_minutes"] // config.simulation["period_minutes"]

    if type(period_index) is not int or not 0 <= period_index < period_count:
        raise ValueError("period_index must identify a period within the day")

    if len(queue_lengths) != 2 or any(type(q) is not int or q < 0 for q in queue_lengths):
        raise ValueError("queue_lengths must contain two nonnegative integers")

    if ((previous is None) != (period_index == 0)
            or (previous is not None and previous.period_index != period_index - 1)):
        raise ValueError("previous must be the preceding period record, or None at opening")

    values = dict.fromkeys(FEATURE_NAMES, 0.0)
    angle = 2 * np.pi * period_index / period_count
    values.update(time_sin=np.sin(angle), time_cos=np.cos(angle),
                  current_queue_A=queue_lengths[0], current_queue_B=queue_lengths[1],
                  observation_valid=1)
    competitor = build_catalogue(config)[config.simulation["competitor_action_id"]]
    values.update(zip(FEATURE_NAMES[-8:-1], (*competitor.prices, competitor.staff, *competitor.stock)))

    if previous is not None:
        values.update(previous_market_arrivals=previous.diagnostics["market_arrivals"],
                      previous_A_entries=previous.entries, previous_A_abandonment=previous.abandonment,
                      previous_A_mean_wait=previous.mean_wait, wait_valid=int(previous.wait_valid))
        values.update(zip(FEATURE_NAMES[4:7], previous.sales))
        values.update(zip(FEATURE_NAMES[10:13], previous.waste))
        values.update(zip(FEATURE_NAMES[15:22],
                          (*previous.action.prices, previous.action.staff, *previous.action.stock)))

    return Observation(period_index, np.array([values[name] for name in FEATURE_NAMES]))


def _accounting(cafe: CafeState, config: Config) -> dict:
    """Snapshot one café's period totals in JSON-compatible numeric units."""
    costs = np.array([product["unit_cost"] for product in config.products])
    sold = int(cafe.sales.sum())

    return {"sales": cafe.sales.tolist(), "entries": cafe.entries,
            "abandonment": cafe.abandonment, "mean_wait": cafe.served_wait / sold if sold else 0.0,
            "wait_valid": bool(sold), "served_wait_total": cafe.served_wait,
            "departed_wait_total": cafe.departed_wait, "end_queue": len(cafe.queue),
            "departure_reasons": dict(cafe.departure_reasons), "waste": cafe.inventory.tolist(),
            "revenue": float(np.dot(cafe.action.prices, cafe.sales)),
            "preparation_cost": float(np.dot(costs, cafe.action.stock)),
            "staff_cost": float(cafe.action.staff * config.simulation["staff_period_cost"]),
            "profit": period_profit(cafe.action, cafe.sales, costs, config.simulation["staff_period_cost"])}


def run_day(scenario: Scenario, customer_policy: "CustomerPolicy", manager: Manager,
            config: Config, *, trajectory_id: str, learn: bool = False,
            epsilon: float = 0.0, exploration_seed: int | None = None) -> DayResult:
    """Run one day; learning requires a mutable policy and explicit exploration seed.

    Queues and fractional credit cross period boundaries. Prices and inventory are
    never reserved: each purchase uses the current period's menu and stock.
    """
    from ..customers.policy import PendingTransition, encode_state, valid_actions

    validate_config(config)
    _validate_scenario(scenario)

    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id must be a nonempty string")

    if not np.isfinite(epsilon) or not 0 <= epsilon <= 1 or (not learn and epsilon != 0):
        raise ValueError("epsilon must be in [0,1], and zero for no-update runs")

    if learn and (customer_policy.frozen or type(exploration_seed) is not int or exploration_seed < 0):
        raise ValueError("learning requires a mutable policy and nonnegative exploration_seed")

    for arrival in scenario.arrivals:
        if arrival.minute >= config.simulation["day_minutes"] or arrival.type_id >= len(config.customer_types):
            raise ValueError("arrival minute/type_id lies outside configured day/customer types")

    catalogue = build_catalogue(config)
    competitor = catalogue[config.simulation["competitor_action_id"]]
    cafes = [CafeState(competitor, np.zeros(3, dtype=int)) for _ in range(2)]
    manager.reset()
    exploration_rng = np.random.default_rng(exploration_seed) if learn else None
    periods = []
    observations = []
    competitor_periods = []
    episodes = [[] for _ in config.customer_types]
    outcomes = Counter()
    arrival_index = 0
    market_arrivals = 0
    outside_count = 0
    decision = None
    day_minutes = config.simulation["day_minutes"]
    profile_width = len(config.catalogue["staff_counts"]) * len(config.catalogue["stock_profiles"])

    def view_for(customer: CustomerInstance, phase: str, minute: int, position: int = 0) -> CustomerView:
        """Snapshot only the observable market and this customer's own attributes."""
        profile = config.customer_types[customer.arrival.type_id]

        return CustomerView(phase, customer.arrival.type_id,
                            tuple(c.action.action_id // profile_width for c in cafes),
                            np.array([c.action.prices for c in cafes]),
                            tuple(len(c.queue) for c in cafes),
                            np.array([c.inventory > 0 for c in cafes]),
                            customer.cafe_id if phase == "waiting" else None,
                            customer.product_id if phase == "waiting" else None,
                            position, customer.elapsed_wait,
                            max(0, profile["max_wait"] - customer.elapsed_wait), minute == day_minutes - 1)

    def update(customer: CustomerInstance, reward: float, next_view: CustomerView | None) -> None:
        """Resolve the pending action once, including any accrued one-minute waiting cost."""
        customer.episode_return += reward

        if learn and customer.pending is not None:
            customer_policy.update(customer.pending, customer.pending_reward + reward,
                                   encode_state(next_view) if next_view is not None else None,
                                   valid_actions(next_view, config) if next_view is not None else (),
                                   terminal=next_view is None, alpha=config.customer_training["alpha"],
                                   gamma=config.customer_training["gamma"])

        customer.pending = None
        customer.pending_reward = 0.0

    def choose(customer: CustomerInstance, view: CustomerView, minute: int) -> int:
        """Resolve the prior transition, then retain this decision until its outcome."""
        update(customer, 0.0, view)
        state = encode_state(view)
        allowed = valid_actions(view, config)
        rng = exploration_rng if learn else customer_rng(scenario, customer.arrival.customer_id, minute, view.phase)
        action_id = customer_policy.choose(customer.arrival.type_id, state, allowed, epsilon=epsilon, rng=rng)

        if action_id not in allowed:
            raise ValueError(f"customer policy action {action_id!r} is not in valid_actions {allowed}")

        customer.pending = PendingTransition(customer.arrival.type_id, state, action_id)

        return action_id

    def finish(customer: CustomerInstance, outcome: str, reward: float = 0.0) -> None:
        """Complete an episode once and retain its type-specific return."""
        update(customer, reward, None)
        outcomes[outcome] += 1
        episodes[customer.arrival.type_id].append(customer.episode_return)

    def departure(customer: CustomerInstance, reason: str) -> None:
        """Count a queue departure in the period in which it happens."""
        cafe = cafes[customer.cafe_id]
        cafe.abandonment += 1
        cafe.departed_wait += customer.elapsed_wait
        cafe.departure_reasons[reason] += 1
        finish(customer, f"abandoned_{'AB'[customer.cafe_id]}")

    def forced_reason(customer: CustomerInstance) -> str | None:
        """Check current stock, budget and patience before decisions and sales."""
        cafe = cafes[customer.cafe_id]
        profile = config.customer_types[customer.arrival.type_id]

        if cafe.inventory[customer.product_id] == 0:
            return "stockout"

        if cafe.action.prices[customer.product_id] > profile["budget"]:
            return "price"

        if customer.elapsed_wait >= profile["max_wait"]:
            return "patience"

        return None

    def finalize() -> None:
        """Store both cafés' accounting and the causal observation for the chosen action."""
        cafe = cafes[0]
        totals = _accounting(cafe, config)
        competitor_totals = _accounting(cafes[1], config)
        competitor_periods.append(competitor_totals)
        diagnostics = {"market_arrivals": market_arrivals, "outside_count": outside_count,
                       "queue_lengths": tuple(len(c.queue) for c in cafes),
                       "end_queue": totals["end_queue"], "served_wait_total": cafe.served_wait,
                       "departed_wait_total": cafe.departed_wait,
                       "departure_reasons": dict(cafe.departure_reasons),
                       "decision": decision, "competitor": competitor_totals}
        periods.append(PeriodRecord(scenario.scenario_id, trajectory_id, len(periods), observations[-1],
                                    cafe.action, cafe.sales.copy(), cafe.entries, cafe.abandonment,
                                    totals["mean_wait"], totals["wait_valid"], cafe.inventory.copy(),
                                    totals["revenue"], totals["preparation_cost"], totals["staff_cost"],
                                    totals["profit"], diagnostics))

    for minute in range(day_minutes):
        if minute % config.simulation["period_minutes"] == 0:
            if minute:
                finalize()

            observation = build_observation(len(periods), periods[-1] if periods else None,
                                            tuple(len(c.queue) for c in cafes), config)
            observations.append(observation)
            action = manager.choose(make_history(observations, config.model["history_length"]))
            validate_action(action, catalogue)
            decision = manager.last_decision
            market_arrivals = outside_count = 0

            for cafe, selected in zip(cafes, (action, competitor)):
                cafe.action = selected
                cafe.inventory = np.array(selected.stock, dtype=int)
                cafe.sales = np.zeros(3, dtype=int)
                cafe.entries = cafe.abandonment = cafe.served_wait = cafe.departed_wait = 0
                cafe.departure_reasons = Counter()

        # Remove departures in FIFO order; later customers see the updated queue.
        for cafe in cafes:
            position = 0

            for _ in range(len(cafe.queue)):
                customer = cafe.queue[position]
                reason = forced_reason(customer)

                if reason is None and choose(customer, view_for(customer, "waiting", minute, position), minute) == 1:
                    reason = "voluntary"

                if reason is not None:
                    del cafe.queue[position]
                    departure(customer, reason)
                else:
                    position += 1

        while arrival_index < len(scenario.arrivals) and scenario.arrivals[arrival_index].minute == minute:
            customer = CustomerInstance(scenario.arrivals[arrival_index])
            arrival_index += 1
            market_arrivals += 1
            action_id = choose(customer, view_for(customer, "arrival", minute), minute)

            if action_id == 6:
                outside_count += 1
                finish(customer, "outside")
            else:
                customer.cafe_id, customer.product_id = divmod(action_id, 3)
                cafe = cafes[customer.cafe_id]
                cafe.entries += 1
                cafe.queue.append(customer)

        for cafe in cafes:
            credit = cafe.service_credit + cafe.action.staff / config.simulation["service_effort"]
            slots = int(np.floor(credit + 1e-12))
            cafe.service_credit = max(0.0, credit - slots)

            while cafe.queue:
                customer = cafe.queue[0]
                reason = forced_reason(customer)

                if reason is not None:
                    cafe.queue.popleft()
                    departure(customer, reason)
                    continue

                if slots == 0:
                    break

                cafe.queue.popleft()
                slots -= 1
                cafe.inventory[customer.product_id] -= 1
                cafe.sales[customer.product_id] += 1
                cafe.served_wait += customer.elapsed_wait
                profile = config.customer_types[customer.arrival.type_id]
                finish(customer, f"purchased_{'AB'[customer.cafe_id]}",
                       profile["values"][customer.product_id] - cafe.action.prices[customer.product_id])

            for customer in cafe.queue:
                waiting_cost = config.customer_types[customer.arrival.type_id]["waiting_cost"]
                customer.pending_reward -= waiting_cost
                customer.episode_return -= waiting_cost
                customer.elapsed_wait += 1

    # Closing departures belong to the last period, so count them before its record.
    for cafe in cafes:
        while cafe.queue:
            departure(cafe.queue.popleft(), "closing")

    finalize()
    entries = sum(row.entries for row in periods)
    abandonment = sum(row.abandonment for row in periods)
    reasons = Counter()

    for row in periods:
        reasons.update(row.diagnostics["departure_reasons"])

    diagnostics = {"market_arrivals": len(scenario.arrivals), "outside_count": outcomes["outside"],
                   "competitor": competitor_periods, "episode_returns_by_type": episodes,
                   "served_wait_total": sum(row.diagnostics["served_wait_total"] for row in periods),
                   "departure_reasons": dict(reasons), "conservation": dict(outcomes),
                   "entries": entries, "abandonment": abandonment,
                   "abandonment_fraction": abandonment / entries if entries else 0.0,
                   "abandonment_fraction_valid": bool(entries), "end_queue": (0, 0)}

    return DayResult(scenario.scenario_id, trajectory_id, periods, diagnostics)
