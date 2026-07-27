from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TokenGridSpec:
    num_frames: int
    height: int
    width: int
    tubelet_size: int = 2
    patch_size: int = 16

    @property
    def temporal_tokens(self) -> int:
        return self.num_frames // self.tubelet_size

    @property
    def height_tokens(self) -> int:
        return self.height // self.patch_size

    @property
    def width_tokens(self) -> int:
        return self.width // self.patch_size

    @property
    def spatial_tokens(self) -> int:
        return self.height_tokens * self.width_tokens

    @property
    def token_count(self) -> int:
        return self.temporal_tokens * self.spatial_tokens

    def validate(self) -> None:
        if self.num_frames % self.tubelet_size:
            raise ValueError("num_frames must be divisible by tubelet_size")
        if self.height % self.patch_size or self.width % self.patch_size:
            raise ValueError("height and width must be divisible by patch_size")


_MODEL_SPECS = {
    "vjepa2_1_vit_base_384": ("vit_base", "ema_encoder", 1),
    "vjepa2_1_vit_large_384": ("vit_large", "ema_encoder", 1),
    "vjepa2_1_vit_giant_384": ("vit_giant_xformers", "target_encoder", 4),
    "vjepa2_1_vit_gigantic_384": ("vit_gigantic_xformers", "target_encoder", 4),
}


def _clean_encoder_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "backbone."):
            key = key.removeprefix(prefix)
        cleaned[key] = value
    return cleaned


def _select_checkpoint_state(payload: Any, checkpoint_key: str) -> dict[str, Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("V-JEPA checkpoint must contain a dictionary")
    for key in (checkpoint_key, "target_encoder", "ema_encoder", "encoder", "state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return _clean_encoder_state_dict(value)
    if payload and all(isinstance(value, Tensor) for value in payload.values()):
        return _clean_encoder_state_dict(payload)
    raise KeyError(
        f"Could not find encoder weights. Tried {checkpoint_key}, target_encoder, "
        "ema_encoder, encoder, and state_dict."
    )


class FrozenVJEPA21(nn.Module):
    """Local-checkpoint wrapper that returns structured V-JEPA 2.1 tokens."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        model_name: str = "vjepa2_1_vit_base_384",
        grid: TokenGridSpec = TokenGridSpec(16, 384, 384),
        strict: bool = True,
    ) -> None:
        super().__init__()
        grid.validate()
        if model_name not in _MODEL_SPECS:
            raise ValueError(f"Unsupported V-JEPA 2.1 model: {model_name}")

        repo_path = Path(repo_path).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"V-JEPA checkpoint not found: {checkpoint_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from app.vjepa_2_1.models import vision_transformer

        architecture, checkpoint_key, distillation_outputs = _MODEL_SPECS[model_name]
        factory = getattr(vision_transformer, architecture)
        self.encoder = factory(
            patch_size=grid.patch_size,
            img_size=(grid.height, grid.width),
            num_frames=grid.num_frames,
            tubelet_size=grid.tubelet_size,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
            n_registers=0,
            n_output_distillation=distillation_outputs,
        )

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = _select_checkpoint_state(payload, checkpoint_key)
        incompatible = self.encoder.load_state_dict(state_dict, strict=strict)
        if not strict and incompatible.missing_keys:
            raise RuntimeError(
                "Non-strict checkpoint loading still missed encoder parameters: "
                + ", ".join(incompatible.missing_keys[:8])
            )

        self.grid = grid
        self.model_name = model_name
        self.embed_dim = self.encoder.embed_dim
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "FrozenVJEPA21":
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, video: Tensor) -> Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {tuple(video.shape)}")
        _, _, frames, height, width = video.shape
        actual = TokenGridSpec(
            num_frames=frames,
            height=height,
            width=width,
            tubelet_size=self.grid.tubelet_size,
            patch_size=self.grid.patch_size,
        )
        actual.validate()
        tokens = self.encoder(video)
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[-1]
        if tokens.shape[1] != actual.token_count:
            raise RuntimeError(
                f"Backbone returned {tokens.shape[1]} tokens, expected {actual.token_count} "
                f"for grid {actual.temporal_tokens}x{actual.height_tokens}x{actual.width_tokens}"
            )
        return tokens.reshape(
            video.shape[0], actual.temporal_tokens, actual.spatial_tokens, tokens.shape[-1]
        )
