from __future__ import annotations

import torch
from torch import Tensor


def blender_c2w_to_opencv(c2w: Tensor) -> Tensor:
    """Match SpaceTimePilot's Blender-to-OpenCV camera convention.

    ``c2w`` may have arbitrary leading dimensions followed by ``[4, 4]``.
    The right multiplication changes camera axes; the left multiplication
    changes the world-axis convention.
    """

    if c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected [..., 4, 4] c2w matrices, got {tuple(c2w.shape)}")

    camera_flip = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    camera_flip[1, 1] = -1
    camera_flip[2, 2] = -1
    world_flip = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    world_flip[0, 0] = -1
    return world_flip @ c2w @ camera_flip


def relative_transforms(c2w: Tensor, reference: str = "adjacent") -> Tensor:
    """Convert absolute camera-to-world poses to relative transforms.

    The returned transform at time ``t`` is expressed in the reference
    camera coordinate system. The first transform is identity.
    """

    if c2w.ndim < 3 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected [..., T, 4, 4], got {tuple(c2w.shape)}")
    if c2w.shape[-3] < 1:
        raise ValueError("At least one pose is required")

    if reference == "adjacent":
        previous = torch.cat((c2w[..., :1, :, :], c2w[..., :-1, :, :]), dim=-3)
    elif reference == "first":
        previous = c2w[..., :1, :, :].expand_as(c2w)
    else:
        raise ValueError(f"Unknown relative-pose reference: {reference}")

    relative = torch.linalg.inv(previous) @ c2w
    identity = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    relative[..., 0, :, :] = identity
    return relative


def rotation_matrix_to_6d(rotation: Tensor) -> Tensor:
    """Represent a rotation by its first two column vectors."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"Expected [..., 3, 3], got {tuple(rotation.shape)}")
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def transform_to_9d(transform: Tensor) -> Tensor:
    """Encode an SE(3) transform as rotation-6D followed by translation."""

    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"Expected [..., 4, 4], got {tuple(transform.shape)}")
    rotation = rotation_matrix_to_6d(transform[..., :3, :3])
    translation = transform[..., :3, 3]
    return torch.cat((rotation, translation), dim=-1)


def tubelet_motion_targets(
    c2w: Tensor,
    tubelet_size: int = 2,
    reference: str = "adjacent",
) -> Tensor:
    """Create one camera-motion target per video tubelet.

    A tubelet token sees ``tubelet_size`` input frames. Its target uses the
    final frame in that tubelet, then computes adjacent (cineGEN-style) or
    first-frame-relative motion between the selected poses.
    """

    if tubelet_size < 1:
        raise ValueError("tubelet_size must be positive")
    frame_count = c2w.shape[-3]
    token_count = frame_count // tubelet_size
    if token_count < 1:
        raise ValueError(
            f"Need at least {tubelet_size} frames for one tubelet, got {frame_count}"
        )
    indices = torch.arange(
        tubelet_size - 1,
        token_count * tubelet_size,
        tubelet_size,
        device=c2w.device,
    )
    sampled_c2w = c2w.index_select(-3, indices)
    return transform_to_9d(relative_transforms(sampled_c2w, reference=reference))
