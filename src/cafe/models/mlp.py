"""Feed-forward outcome predictor for fixed-length management histories."""

import torch
from torch import Tensor, nn

from ..contracts import ScalerStats
from .base import OutcomePredictor


class MLPPredictor(OutcomePredictor):
    """Score a flattened causal history and the numeric action under consideration."""

    def __init__(self, scalers: ScalerStats, *, H: int = 4, F: int = 30,
                 hidden_sizes: tuple[int, int] = (64, 32)) -> None:
        """Create two ReLU layers followed by the five outcome logits."""
        super().__init__(scalers)

        if H < 1 or F != len(scalers.observation_mean) or len(hidden_sizes) != 2 or min(hidden_sizes) < 1:
            raise ValueError("expected positive history and widths and F matching the scalers")

        self.H, self.F, self.hidden_sizes = H, F, hidden_sizes
        first, second = hidden_sizes
        self.network = nn.Sequential(nn.Linear(H * F + 7, first), nn.ReLU(), nn.Linear(first, second),
                                     nn.ReLU(), nn.Linear(second, 5))

    def forward_logits(self, scaled_history: Tensor, scaled_actions: Tensor) -> Tensor:
        """Combine the history and candidate action in chronological field order."""
        if tuple(scaled_history.shape[1:]) != (self.H, self.F):
            raise ValueError(f"expected history dimensions {(self.H, self.F)}")

        return self.network(torch.cat((scaled_history.flatten(1), scaled_actions), dim=1))
