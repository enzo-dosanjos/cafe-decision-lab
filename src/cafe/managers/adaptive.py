"""Manage staffing, stock and prices using recent entries and queue pressure."""

from collections.abc import Sequence
from math import ceil, isfinite

from ..config import Config
from ..contracts import FEATURE_NAMES, Action, Array, DecisionInfo, build_catalogue
from .base import Manager


class AdaptiveManager(Manager):
    """Smooth café A entries and translate demand into one catalogue action."""

    def __init__(self, catalogue: Sequence[Action], config: Config, *, alpha: float,
                 buffer: float, queue_threshold: int) -> None:
        """Bind the demand update, stock buffer and high-price threshold."""
        if not 0 < alpha <= 1 or not isfinite(buffer) or buffer <= 0 or queue_threshold < 0:
            raise ValueError("expected 0 < alpha <= 1, positive buffer and nonnegative queue_threshold")

        if tuple(catalogue) != build_catalogue(config):
            raise ValueError("adaptive catalogue must match the configured actions")

        self.alpha = alpha
        self.buffer = buffer
        self.queue_threshold = queue_threshold
        self.config = config
        self.staff_levels = sorted(set(a.staff for a in catalogue))
        self.stock_levels = sorted({a.stock for a in catalogue}, key=lambda stock: (sum(stock), stock))
        prices = sorted({a.prices for a in catalogue})
        base = tuple(round(p["base_price"], 2) for p in config.products)

        if base not in prices:
            raise ValueError("adaptive manager requires a base-price catalogue profile")

        self.low, self.base, self.high = prices[0], base, prices[-1]
        self.actions = {(a.prices, a.staff, a.stock): a for a in catalogue}
        self.reset()

    def reset(self) -> None:
        """Start a fresh day with expected opening demand of twenty entries."""
        super().reset()
        self.demand = 20.0

    def choose(self, history: Array) -> Action:
        """Use only the current pre-decision observation and the day-local demand estimate."""
        current = history[-1]

        if self.last_decision is not None:
            entries = current[FEATURE_NAMES.index("previous_A_entries")]
            self.demand += self.alpha * (entries - self.demand)

        sim = self.config.simulation
        required = ceil(self.demand * sim["service_effort"] / (sim["period_minutes"] * 0.85))
        staff = next((n for n in self.staff_levels if n >= required), self.staff_levels[-1])
        stock = next((q for q in self.stock_levels if sum(q) >= self.buffer * self.demand), self.stock_levels[-1])
        queue = current[FEATURE_NAMES.index("current_queue_A")]
        departed = current[FEATURE_NAMES.index("previous_A_abandonment")]

        if queue >= self.queue_threshold or departed > self.config.evaluation["abandonment_limit"]:
            prices = self.high
        elif self.demand < 10:
            prices = self.low
        else:
            prices = self.base

        action = self.actions[prices, staff, stock]
        self.last_decision = DecisionInfo(action.action_id, None, None, None, False)

        return action
