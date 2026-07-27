from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ConvergenceDecision:
    is_best: bool
    is_flat: bool
    relative_mse_improvement: float | None
    cosine_gain: float | None
    action: str


class ConvergenceMonitor:
    """Track a two-stage plateau: reduce LR once, then confirm convergence."""

    def __init__(
        self,
        *,
        min_relative_mse_improvement: float,
        min_cosine_gain: float,
        validations_before_lr_drop: int,
        validations_after_lr_drop: int,
    ) -> None:
        if min_relative_mse_improvement < 0 or min_cosine_gain < 0:
            raise ValueError("Convergence thresholds must be non-negative")
        if validations_before_lr_drop < 1 or validations_after_lr_drop < 1:
            raise ValueError("Plateau validation counts must be positive")
        self.min_relative_mse_improvement = min_relative_mse_improvement
        self.min_cosine_gain = min_cosine_gain
        self.validations_before_lr_drop = validations_before_lr_drop
        self.validations_after_lr_drop = validations_after_lr_drop
        self.best_mse = float("inf")
        self.best_cosine = float("-inf")
        self.best_step: int | None = None
        self.flat_validations = 0
        self.lr_reduced = False
        self.converged = False

    def update(self, *, step: int, mse: float, cosine: float) -> ConvergenceDecision:
        if self.converged:
            raise RuntimeError("Cannot update a monitor that has already converged")
        has_baseline = self.best_step is not None
        is_best = mse < self.best_mse
        if not has_baseline:
            relative_mse_improvement = None
            cosine_gain = None
            is_flat = False
        else:
            denominator = max(abs(self.best_mse), torch.finfo(torch.float64).eps)
            relative_mse_improvement = (self.best_mse - mse) / denominator
            cosine_gain = cosine - self.best_cosine
            is_flat = (
                relative_mse_improvement < self.min_relative_mse_improvement
                and cosine_gain < self.min_cosine_gain
            )

        if is_best:
            self.best_mse = mse
            self.best_step = step
        self.best_cosine = max(self.best_cosine, cosine)
        self.flat_validations = self.flat_validations + 1 if is_flat else 0

        action = "continue"
        if not self.lr_reduced and self.flat_validations >= self.validations_before_lr_drop:
            self.lr_reduced = True
            self.flat_validations = 0
            action = "reduce_lr"
        elif self.lr_reduced and self.flat_validations >= self.validations_after_lr_drop:
            self.converged = True
            action = "stop"

        return ConvergenceDecision(
            is_best=is_best,
            is_flat=is_flat,
            relative_mse_improvement=relative_mse_improvement,
            cosine_gain=cosine_gain,
            action=action,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_mse": self.best_mse,
            "best_cosine": self.best_cosine,
            "best_step": self.best_step,
            "flat_validations": self.flat_validations,
            "lr_reduced": self.lr_reduced,
            "converged": self.converged,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.best_mse = float(state["best_mse"])
        self.best_cosine = float(
            state.get("best_cosine", state.get("previous_cosine", float("-inf")))
        )
        self.best_step = state["best_step"]
        self.flat_validations = int(state["flat_validations"])
        self.lr_reduced = bool(state["lr_reduced"])
        self.converged = bool(state["converged"])


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
