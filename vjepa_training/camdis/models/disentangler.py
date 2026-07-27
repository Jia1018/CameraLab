from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class DisentangledFeatures:
    content: Tensor
    camera: Tensor
    camera_prediction: Tensor


def _encoder_layer(
    dim: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=num_heads,
        dim_feedforward=round(dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class _DisentanglingBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.local = _encoder_layer(dim, num_heads, mlp_ratio, dropout)
        self.camera_temporal = _encoder_layer(dim, num_heads, mlp_ratio, dropout)
        self.patch_temporal = _encoder_layer(dim, num_heads, mlp_ratio, dropout)

    def forward(self, patches: Tensor, camera: Tensor) -> tuple[Tensor, Tensor]:
        batch, time, spatial, dim = patches.shape
        local = torch.cat((camera.unsqueeze(2), patches), dim=2)
        local = self.local(local.reshape(batch * time, spatial + 1, dim))
        local = local.reshape(batch, time, spatial + 1, dim)
        camera, patches = local[:, :, 0], local[:, :, 1:]

        camera = self.camera_temporal(camera)
        patch_tracks = patches.permute(0, 2, 1, 3).reshape(batch * spatial, time, dim)
        patch_tracks = self.patch_temporal(patch_tracks)
        patches = patch_tracks.reshape(batch, spatial, time, dim).permute(0, 2, 1, 3)
        return patches, camera


class FeatureDisentangler(nn.Module):
    """VGGT-inspired adapter over frozen video patch tokens.

    Each tubelet gets one learnable camera query. The first query is distinct;
    later tubelets share a base query, as in VGGT, and receive learned temporal
    positions. Local camera-patch attention alternates with temporal camera and
    per-spatial-patch attention.
    """

    def __init__(
        self,
        backbone_dim: int,
        model_dim: int = 384,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_temporal_tokens: int = 64,
        camera_target_dim: int = 9,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.input_projection = nn.Linear(backbone_dim, model_dim)
        self.patch_type = nn.Parameter(torch.zeros(1, 1, 1, model_dim))
        self.camera_query_first = nn.Parameter(torch.empty(1, 1, model_dim))
        self.camera_query_rest = nn.Parameter(torch.empty(1, 1, model_dim))
        self.camera_temporal_position = nn.Parameter(
            torch.empty(1, max_temporal_tokens, model_dim)
        )
        self.blocks = nn.ModuleList(
            [
                _DisentanglingBlock(model_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.content_norm = nn.LayerNorm(model_dim)
        self.camera_norm = nn.LayerNorm(model_dim)
        self.camera_head = nn.Linear(model_dim, camera_target_dim)
        self.model_dim = model_dim
        self.max_temporal_tokens = max_temporal_tokens
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.camera_query_first, std=0.02)
        nn.init.normal_(self.camera_query_rest, std=0.02)
        nn.init.normal_(self.camera_temporal_position, std=0.02)
        nn.init.normal_(self.patch_type, std=0.02)

    def forward(
        self,
        tokens: Tensor,
        *,
        temporal_tokens: int | None = None,
        spatial_tokens: int | None = None,
    ) -> DisentangledFeatures:
        if tokens.ndim == 3:
            if temporal_tokens is None or spatial_tokens is None:
                raise ValueError("Flattened tokens require temporal_tokens and spatial_tokens")
            tokens = tokens.reshape(tokens.shape[0], temporal_tokens, spatial_tokens, -1)
        if tokens.ndim != 4:
            raise ValueError(f"Expected [B, T, N, D], got {tuple(tokens.shape)}")

        batch, time, _, _ = tokens.shape
        if time > self.max_temporal_tokens:
            raise ValueError(
                f"Got {time} temporal tokens, configured maximum is {self.max_temporal_tokens}"
            )
        patches = self.input_projection(tokens) + self.patch_type
        if time == 1:
            camera = self.camera_query_first
        else:
            camera = torch.cat(
                (self.camera_query_first, self.camera_query_rest.expand(1, time - 1, -1)),
                dim=1,
            )
        camera = camera.expand(batch, -1, -1)
        camera = camera + self.camera_temporal_position[:, :time]

        for block in self.blocks:
            patches, camera = block(patches, camera)
        content = self.content_norm(patches)
        camera = self.camera_norm(camera)
        return DisentangledFeatures(
            content=content,
            camera=camera,
            camera_prediction=self.camera_head(camera),
        )


class _ReconstructionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        camera_conditioning: str,
        max_spatial_tokens: int,
        spatial_basis_rank: int,
    ) -> None:
        super().__init__()
        self.patch_temporal = _encoder_layer(dim, num_heads, mlp_ratio, dropout)
        self.camera_temporal = _encoder_layer(dim, num_heads, mlp_ratio, dropout)
        self.local = _encoder_layer(dim, num_heads, mlp_ratio, dropout)
        self.camera_conditioning = camera_conditioning
        if camera_conditioning == "spatial_film":
            self.camera_affine = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 2 * dim),
            )
            nn.init.normal_(self.camera_affine[-1].weight, std=0.02)
            nn.init.zeros_(self.camera_affine[-1].bias)
        else:
            self.camera_affine = None
        if camera_conditioning in {"spatial_basis", "spatial_basis_film"}:
            if max_spatial_tokens < 1 or spatial_basis_rank < 1:
                raise ValueError(
                    "max_spatial_tokens and spatial_basis_rank must be positive"
                )
            self.camera_basis_coefficients = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, spatial_basis_rank, bias=False),
            )
            basis_dim = 2 * dim if camera_conditioning == "spatial_basis_film" else dim
            self.spatial_basis = nn.Parameter(
                torch.empty(max_spatial_tokens, spatial_basis_rank, basis_dim)
            )
            nn.init.normal_(self.spatial_basis, std=0.02)
            self.spatial_basis_rank = spatial_basis_rank
        else:
            self.camera_basis_coefficients = None
            self.register_parameter("spatial_basis", None)
            self.spatial_basis_rank = 0

    @staticmethod
    def _normalize_spatially(patches: Tensor) -> Tensor:
        moments = patches.float()
        mean = moments.mean(dim=2, keepdim=True)
        variance = moments.var(dim=2, keepdim=True, unbiased=False)
        return ((moments - mean) * torch.rsqrt(variance + 1e-6)).to(patches.dtype)

    def _condition_patches(self, patches: Tensor, camera: Tensor) -> Tensor:
        if self.camera_affine is not None:
            normalized = self._normalize_spatially(patches)
            scale, shift = self.camera_affine(camera).chunk(2, dim=-1)
            return normalized * (1.0 + torch.tanh(scale).unsqueeze(2)) + shift.unsqueeze(2)
        if self.camera_basis_coefficients is not None:
            spatial = patches.shape[2]
            if self.spatial_basis is None or spatial > self.spatial_basis.shape[0]:
                maximum = 0 if self.spatial_basis is None else self.spatial_basis.shape[0]
                raise ValueError(
                    f"Got {spatial} spatial tokens, configured maximum is {maximum}"
                )
            coefficients = self.camera_basis_coefficients(camera)
            basis = self.spatial_basis[:spatial].to(coefficients.dtype)
            residual = torch.einsum("btr,nrd->btnd", coefficients, basis)
            residual = residual / math.sqrt(self.spatial_basis_rank)
            if self.camera_conditioning == "spatial_basis_film":
                scale, shift = residual.chunk(2, dim=-1)
                normalized = self._normalize_spatially(patches)
                return normalized * (1.0 + torch.tanh(scale)) + shift
            return patches + residual.to(patches.dtype)
        return patches

    def forward(self, patches: Tensor, camera: Tensor) -> tuple[Tensor, Tensor]:
        batch, time, spatial, dim = patches.shape
        camera = self.camera_temporal(camera)
        patches = self._condition_patches(patches, camera)
        patch_tracks = patches.permute(0, 2, 1, 3).reshape(batch * spatial, time, dim)
        patch_tracks = self.patch_temporal(patch_tracks)
        patches = patch_tracks.reshape(batch, spatial, time, dim).permute(0, 2, 1, 3)

        local = torch.cat((camera.unsqueeze(2), patches), dim=2)
        local = self.local(local.reshape(batch * time, spatial + 1, dim))
        local = local.reshape(batch, time, spatial + 1, dim)
        return local[:, :, 1:], local[:, :, 0]


class FeatureReconstructor(nn.Module):
    """Symmetric decoder from content and camera tokens to backbone tokens."""

    def __init__(
        self,
        backbone_dim: int,
        model_dim: int = 384,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        camera_conditioning: str = "attention",
        max_spatial_tokens: int = 1024,
        spatial_basis_rank: int = 16,
    ) -> None:
        super().__init__()
        if camera_conditioning not in {
            "attention",
            "spatial_film",
            "spatial_basis",
            "spatial_basis_film",
        }:
            raise ValueError(
                "camera_conditioning must be 'attention', 'spatial_film', "
                "'spatial_basis', or 'spatial_basis_film', got "
                f"{camera_conditioning!r}"
            )
        self.blocks = nn.ModuleList(
            [
                _ReconstructionBlock(
                    model_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    camera_conditioning,
                    max_spatial_tokens,
                    spatial_basis_rank,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, backbone_dim)

    def forward(self, content: Tensor, camera: Tensor) -> Tensor:
        if content.ndim != 4 or camera.ndim != 3:
            raise ValueError("Expected content [B,T,N,D] and camera [B,T,D]")
        if content.shape[:2] != camera.shape[:2] or content.shape[-1] != camera.shape[-1]:
            raise ValueError(
                f"Incompatible content {tuple(content.shape)} and camera {tuple(camera.shape)}"
            )
        for block in self.blocks:
            content, camera = block(content, camera)
        return self.output_projection(self.output_norm(content))
