"""Apply the common expected-profit selector to LSTM forecasts."""

from collections.abc import Sequence

import numpy as np

from ..config import Config
from ..contracts import Action, Array, build_catalogue
from ..models.lstm import LSTMPredictor
from .base import Manager
from .selection import select_action


class LSTMManager(Manager):
    """Choose a catalogue action from current observable management history."""

    def __init__(self, predictor: LSTMPredictor, catalogue: Sequence[Action], config: Config) -> None:
        """Bind trained weights, catalogue and known operating costs."""
        if tuple(catalogue) != build_catalogue(config):
            raise ValueError("LSTM manager catalogue must match configuration")

        self.predictor = predictor
        self.catalogue = tuple(catalogue)
        self.unit_costs = np.array([p["unit_cost"] for p in config.products])
        self.staff_cost = config.simulation["staff_period_cost"]
        self.limit = config.evaluation["abandonment_limit"]

    def choose(self, history: Array) -> Action:
        """Record feasibility and fallback diagnostics alongside the chosen action."""
        action, self.last_decision = select_action(self.predictor, history, self.catalogue,
                                                   self.unit_costs, self.staff_cost, self.limit)

        return action
