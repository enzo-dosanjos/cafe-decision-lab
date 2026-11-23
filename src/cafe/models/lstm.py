"""Recurrent outcome predictor with independent state for every history window."""

import torch
from torch import Tensor, nn

from ..contracts import ScalerStats
from .base import OutcomePredictor


class LSTMPredictor(OutcomePredictor):
    """Encode the same causal history and candidate action used by the MLP."""

    def __init__(self, scalers: ScalerStats, *, F: int = 30, hidden_size: int = 32,
                 head_size: int = 32) -> None:
        """Create one recurrent layer and a ReLU prediction head."""
        super().__init__(scalers)

        if F != len(scalers.observation_mean) or min(hidden_size, head_size) < 1:
            raise ValueError("expected positive widths and F matching the scalers")

        self.F, self.hidden_size, self.head_size = F, hidden_size, head_size
        self.recurrent = nn.LSTM(F, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size + 7, head_size), nn.ReLU(), nn.Linear(head_size, 5))

    def forward_logits(self, scaled_history: Tensor, scaled_actions: Tensor) -> Tensor:
        """Start hidden/cell state at zero for each window, including batched inference."""
        if scaled_history.shape[1] < 1 or scaled_history.shape[2] != self.F:
            raise ValueError(f"expected nonempty history with {self.F} features")

        # Omitting hx creates fresh zeros; never retain state between calls or days.
        outputs, _ = self.recurrent(scaled_history)

        return self.head(torch.cat((outputs[:, -1], scaled_actions), dim=1))
