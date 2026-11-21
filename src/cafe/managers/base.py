"""Common decision interface for café A managers."""

from abc import ABC, abstractmethod

from ..contracts import Action, Array, DecisionInfo


class Manager(ABC):
    """Select a catalogue entry from pre-decision history."""

    last_decision: DecisionInfo | None = None

    def reset(self) -> None:
        """Clear diagnostics when an independent day begins."""
        self.last_decision = None

    @abstractmethod
    def choose(self, history: Array) -> Action:
        """Return an action given raw observations in [history, feature] order."""
