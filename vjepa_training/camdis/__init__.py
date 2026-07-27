"""Camera and physical-motion disentanglement on frozen video features."""

from .losses import DisentanglementLoss, LossWeights

__all__ = ["DisentanglementLoss", "LossWeights"]
