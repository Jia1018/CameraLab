from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from camdis.models.disentangler import DisentangledFeatures, FeatureReconstructor


def token_relation_loss(reconstructed: Tensor, target: Tensor) -> Tensor:
    """Preserve spatial and temporal cosine geometry between patch tokens."""

    if reconstructed.shape != target.shape or reconstructed.ndim != 4:
        raise ValueError(
            "Expected matching [B, T, N, D] tensors, got "
            f"{tuple(reconstructed.shape)} and {tuple(target.shape)}"
        )

    reconstructed = F.normalize(reconstructed.float(), dim=-1)
    target = F.normalize(target.detach().float(), dim=-1)

    reconstructed_spatial = reconstructed @ reconstructed.transpose(-1, -2)
    target_spatial = target @ target.transpose(-1, -2)

    reconstructed_temporal = reconstructed.transpose(1, 2)
    target_temporal = target.transpose(1, 2)
    reconstructed_temporal = (
        reconstructed_temporal @ reconstructed_temporal.transpose(-1, -2)
    )
    target_temporal = target_temporal @ target_temporal.transpose(-1, -2)

    return 0.5 * (
        F.mse_loss(reconstructed_spatial, target_spatial)
        + F.mse_loss(reconstructed_temporal, target_temporal)
    )


@dataclass(frozen=True)
class LossWeights:
    reconstruction: float = 1.0
    cosine: float = 0.1
    camera_supervision: float = 0.2
    camera_invariance: float = 0.05
    content_invariance: float = 0.02
    identity_reconstruction: float = 1.0
    content_swap_reconstruction: float = 1.0
    camera_swap_reconstruction: float = 1.0
    double_swap_reconstruction: float = 1.0
    camera_factor_ranking: float = 0.0
    content_factor_ranking: float = 0.0
    factor_margin: float = 0.05
    camera_variance: float = 0.0
    camera_std_target: float = 0.1
    camera_delta_reconstruction: float = 0.0
    camera_delta_cosine: float = 0.0
    content_delta_reconstruction: float = 0.0


class DisentanglementLoss(nn.Module):
    """Exact crossed reconstruction losses for a 2x2 factor grid."""

    _RECONSTRUCTION_CATEGORIES = (
        "identity",
        "content_swap",
        "camera_swap",
        "double_swap",
    )

    def __init__(self, weights: LossWeights = LossWeights()) -> None:
        super().__init__()
        self.weights = weights

    @staticmethod
    def _reshape_features(
        features: DisentangledFeatures,
        batch_size: int,
    ) -> DisentangledFeatures:
        return DisentangledFeatures(
            content=features.content.reshape(batch_size, 2, 2, *features.content.shape[1:]),
            camera=features.camera.reshape(batch_size, 2, 2, *features.camera.shape[1:]),
            camera_prediction=features.camera_prediction.reshape(
                batch_size, 2, 2, *features.camera_prediction.shape[1:]
            ),
        )

    @staticmethod
    def _composition_category(content_swapped: bool, camera_swapped: bool) -> str:
        if content_swapped and camera_swapped:
            return "double_swap"
        if content_swapped:
            return "content_swap"
        if camera_swapped:
            return "camera_swap"
        return "identity"

    def forward(
        self,
        backbone_tokens: Tensor,
        features: DisentangledFeatures,
        reconstructor: FeatureReconstructor,
        camera_target: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute all 16 valid content/camera recombinations.

        ``backbone_tokens`` is ``[B, camera=2, physics=2, T, N, D]``.
        For target factor pair ``(c, p)``, content can come from either camera
        observation of physics ``p`` and camera can come from either physical
        observation along path ``c``. Every combination has an exact target in
        the input 2x2 grid.
        """

        if backbone_tokens.ndim != 6 or backbone_tokens.shape[1:3] != (2, 2):
            raise ValueError(
                "backbone_tokens must have shape [B, 2, 2, T, N, D], got "
                f"{tuple(backbone_tokens.shape)}"
            )
        batch_size = backbone_tokens.shape[0]
        features = self._reshape_features(features, batch_size)

        content_compositions = []
        camera_compositions = []
        targets = []
        composition_categories = []
        for target_camera in range(2):
            for target_physics in range(2):
                for content_source_camera in range(2):
                    for camera_source_physics in range(2):
                        content_compositions.append(
                            features.content[:, content_source_camera, target_physics]
                        )
                        camera_compositions.append(
                            features.camera[:, target_camera, camera_source_physics]
                        )
                        targets.append(backbone_tokens[:, target_camera, target_physics])
                        composition_categories.append(
                            self._composition_category(
                                content_source_camera != target_camera,
                                camera_source_physics != target_physics,
                            )
                        )

        composed_content = torch.cat(content_compositions, dim=0)
        composed_camera = torch.cat(camera_compositions, dim=0)
        target_tokens = torch.cat(targets, dim=0).detach()
        reconstructed = reconstructor(composed_content, composed_camera)

        reconstruction_grid = reconstructed.reshape(
            2, 2, 2, 2, batch_size, *reconstructed.shape[1:]
        )
        target_grid = target_tokens.reshape(
            2, 2, 2, 2, batch_size, *target_tokens.shape[1:]
        )
        reconstructed_camera_delta = (
            reconstruction_grid[1].float() - reconstruction_grid[0].float()
        )
        target_camera_delta = target_grid[1].float() - target_grid[0].float()
        camera_delta_reconstruction = F.mse_loss(
            reconstructed_camera_delta,
            target_camera_delta,
        )
        camera_delta_cosine = 1.0 - F.cosine_similarity(
            reconstructed_camera_delta.flatten(start_dim=4),
            target_camera_delta.flatten(start_dim=4),
            dim=-1,
        ).mean()
        content_delta_reconstruction = F.mse_loss(
            reconstruction_grid[:, 1].float() - reconstruction_grid[:, 0].float(),
            target_grid[:, 1].float() - target_grid[:, 0].float(),
        )

        category_weight_map = {
            "identity": self.weights.identity_reconstruction,
            "content_swap": self.weights.content_swap_reconstruction,
            "camera_swap": self.weights.camera_swap_reconstruction,
            "double_swap": self.weights.double_swap_reconstruction,
        }
        composition_weights = reconstructed.new_tensor(
            [category_weight_map[category] for category in composition_categories],
            dtype=torch.float32,
        ).repeat_interleave(batch_size)
        if composition_weights.sum() <= 0:
            raise ValueError("At least one reconstruction category weight must be positive")

        reconstruction_per_example = (
            (reconstructed.float() - target_tokens.float())
            .square()
            .flatten(1)
            .mean(dim=1)
        )
        cosine_per_example = 1.0 - F.cosine_similarity(
            reconstructed.float(), target_tokens.float(), dim=-1
        ).flatten(1).mean(dim=1)
        weight_sum = composition_weights.sum()
        reconstruction = (
            reconstruction_per_example * composition_weights
        ).sum() / weight_sum
        cosine = (cosine_per_example * composition_weights).sum() / weight_sum

        camera_invariance = 0.5 * (
            F.mse_loss(features.camera[:, 0, 0], features.camera[:, 0, 1])
            + F.mse_loss(features.camera[:, 1, 0], features.camera[:, 1, 1])
        )

        pooled_content = features.content.mean(dim=-2)
        content_invariance = 0.5 * (
            F.mse_loss(pooled_content[:, 0, 0], pooled_content[:, 1, 0])
            + F.mse_loss(pooled_content[:, 0, 1], pooled_content[:, 1, 1])
        )
        camera_path_separation = 0.5 * (
            F.mse_loss(features.camera[:, 0, 0], features.camera[:, 1, 0])
            + F.mse_loss(features.camera[:, 0, 1], features.camera[:, 1, 1])
        )
        content_physics_separation = 0.5 * (
            F.mse_loss(pooled_content[:, 0, 0], pooled_content[:, 0, 1])
            + F.mse_loss(pooled_content[:, 1, 0], pooled_content[:, 1, 1])
        )
        camera_factor_ranking = F.relu(
            camera_invariance
            - camera_path_separation
            + self.weights.factor_margin
        )
        content_factor_ranking = F.relu(
            content_invariance
            - content_physics_separation
            + self.weights.factor_margin
        )
        camera_feature_std = features.camera.float().reshape(
            -1, features.camera.shape[-1]
        ).std(dim=0, unbiased=False)
        camera_variance_regularization = F.relu(
            self.weights.camera_std_target - camera_feature_std
        ).mean()

        if self.weights.camera_supervision:
            if camera_target.shape[:2] != (batch_size, 2):
                raise ValueError(
                    f"camera_target must be [B, 2, T, K], got {tuple(camera_target.shape)}"
                )
            expanded_target = camera_target[:, :, None].expand_as(features.camera_prediction)
            camera_supervision = F.mse_loss(
                features.camera_prediction.float(), expanded_target.float()
            )
        else:
            camera_supervision = reconstruction.new_zeros(())

        total = (
            self.weights.reconstruction * reconstruction
            + self.weights.cosine * cosine
            + self.weights.camera_supervision * camera_supervision
            + self.weights.camera_invariance * camera_invariance
            + self.weights.content_invariance * content_invariance
            + self.weights.camera_factor_ranking * camera_factor_ranking
            + self.weights.content_factor_ranking * content_factor_ranking
            + self.weights.camera_variance * camera_variance_regularization
            + self.weights.camera_delta_reconstruction
            * camera_delta_reconstruction
            + self.weights.camera_delta_cosine * camera_delta_cosine
            + self.weights.content_delta_reconstruction
            * content_delta_reconstruction
        )

        with torch.no_grad():
            reconstruction_by_category: dict[str, list[Tensor]] = {
                category: [] for category in self._RECONSTRUCTION_CATEGORIES
            }
            for composition_index, category in enumerate(composition_categories):
                start = composition_index * batch_size
                end = start + batch_size
                reconstruction_by_category[category].append(
                    reconstruction_per_example[start:end].mean()
                )

            camera_feature_variance = camera_feature_std.square().mean()
            content_feature_variance = pooled_content.float().reshape(
                -1, pooled_content.shape[-1]
            ).var(dim=0, unbiased=False).mean()

        metrics = {
            "loss": total.detach(),
            "reconstruction": reconstruction.detach(),
            "cosine": cosine.detach(),
            "camera_supervision": camera_supervision.detach(),
            "camera_invariance": camera_invariance.detach(),
            "content_invariance": content_invariance.detach(),
            "camera_factor_ranking": camera_factor_ranking.detach(),
            "content_factor_ranking": content_factor_ranking.detach(),
            "camera_variance_regularization": camera_variance_regularization.detach(),
            "camera_path_separation": camera_path_separation.detach(),
            "content_physics_separation": content_physics_separation.detach(),
            "camera_feature_variance": camera_feature_variance.detach(),
            "content_feature_variance": content_feature_variance.detach(),
            "camera_delta_reconstruction": camera_delta_reconstruction.detach(),
            "camera_delta_cosine": camera_delta_cosine.detach(),
            "content_delta_reconstruction": content_delta_reconstruction.detach(),
        }
        metrics.update(
            {
                f"reconstruction_{category}": torch.stack(values).mean().detach()
                for category, values in reconstruction_by_category.items()
            }
        )
        return total, metrics
