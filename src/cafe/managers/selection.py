"""Choose the most profitable catalogue action subject to predicted abandonment."""

from collections.abc import Sequence

import numpy as np
import torch

from ..contracts import Action, Array, DecisionInfo
from ..models.base import OutcomePredictor


def select_action(predictor: OutcomePredictor, history: Array, catalogue: Sequence[Action],
                  unit_costs: Array, staff_period_cost: float, abandonment_limit: float) -> tuple[Action, DecisionInfo]:
    """Score all candidates together and use stable action IDs for final ties."""
    device = predictor.action_mean.device
    numeric = np.array([(*a.prices, a.staff, *a.stock) for a in catalogue])
    actions = torch.as_tensor(numeric, dtype=torch.float32, device=device)
    histories = torch.as_tensor(history, dtype=torch.float32, device=device).expand(len(catalogue), -1, -1)
    predictions = predictor.predict(histories, actions).cpu().numpy()

    if predictions.shape != (len(catalogue), 5) or not np.isfinite(predictions).all():
        raise ValueError("predictor must return five finite outcomes per action")

    revenue = (numeric[:, :3] * predictions[:, :3]).sum(axis=1)
    profits = revenue - numeric[:, 4:] @ unit_costs - numeric[:, 3] * staff_period_cost
    feasible = np.flatnonzero(predictions[:, 3] <= abandonment_limit)
    ids = np.array([a.action_id for a in catalogue])

    if len(feasible):
        best = feasible[np.lexsort((ids[feasible], -profits[feasible]))[0]]
    else:
        best = np.lexsort((ids, -profits, predictions[:, 3]))[0]

    action = catalogue[best]
    decision = DecisionInfo(action.action_id, predictions[best].copy(), float(profits[best]),
                            len(feasible) > 0, len(feasible) == 0)

    return action, decision
