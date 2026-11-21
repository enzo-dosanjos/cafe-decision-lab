"""Constant-action management baseline."""

from ..contracts import Action, Array, DecisionInfo
from .base import Manager


class FixedManager(Manager):
    """Apply one configured action for every period."""

    def __init__(self, action: Action):
        """Keep the selected catalogue entry across days."""
        self.action = action

    def choose(self, history: Array) -> Action:
        """Return the fixed action and record its ID for the period log."""
        self.last_decision = DecisionInfo(self.action.action_id, None, None, None, False)

        return self.action
