"""Scale raw model inputs once and decode physically bounded outcome predictions."""

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn
from torch.nn.functional import softplus

from ..contracts import ACTION_NAMES, FEATURE_NAMES, TARGET_NAMES, ScalerStats


class OutcomePredictor(nn.Module, ABC):
    """Return expected sales, abandonment and mean served wait in original units."""

    def __init__(self, scalers: ScalerStats) -> None:
        """Keep training statistics as buffers that move with the network."""
        super().__init__()
        self.training_scenario_ids = scalers.training_scenario_ids

        sizes = {"observation_mean": len(FEATURE_NAMES), "observation_std": len(FEATURE_NAMES),
                 "action_mean": len(ACTION_NAMES), "action_std": len(ACTION_NAMES), "target_std": len(TARGET_NAMES)}

        for name, size in sizes.items():
            values = torch.tensor(getattr(scalers, name), dtype=torch.float32)

            if values.shape != (size,) or not values.isfinite().all() or (name.endswith("_std") and (values < 1).any()):
                raise ValueError(f"{name}: expected {size} finite statistics and standard deviations >= 1")

            self.register_buffer(name, values)

        # The two validity flags retain their original binary values.
        flags = [FEATURE_NAMES.index(name) for name in ("wait_valid", "observation_valid")]
        self.observation_mean[flags] = 0
        self.observation_std[flags] = 1

    def forward(self, raw_history: Tensor, raw_actions: Tensor) -> Tensor:
        """Transform [N,H,F] histories and [N,7] actions into [N,5] expected outcomes."""
        if (raw_history.ndim != 3 or raw_history.shape[-1] != self.observation_mean.numel()
                or raw_actions.shape != (raw_history.shape[0], 7)):
            raise ValueError("expected raw_history [N,H,F] and raw_actions [N,7]")

        raw_history = raw_history.to(dtype=self.observation_mean.dtype)
        raw_actions = raw_actions.to(dtype=self.action_mean.dtype)
        history = (raw_history - self.observation_mean) / self.observation_std
        history = history * raw_history[..., -1:]
        actions = (raw_actions - self.action_mean) / self.action_std
        logits = self.forward_logits(history, actions)
        sales = raw_actions[:, 4:] * logits[:, :3].sigmoid()

        return torch.cat((sales, softplus(logits[:, 3:])), dim=1)

    @torch.inference_mode()
    def predict(self, raw_history: Tensor, raw_actions: Tensor) -> Tensor:
        """Evaluate frozen weights with the same units and shapes as forward."""
        self.eval()

        return self(raw_history, raw_actions)

    @abstractmethod
    def forward_logits(self, scaled_history: Tensor, scaled_actions: Tensor) -> Tensor:
        """Return one unconstrained logit for each target."""
