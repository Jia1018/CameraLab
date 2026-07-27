import torch

from camdis.geometry.camera import (
    blender_c2w_to_opencv,
    relative_transforms,
    tubelet_motion_targets,
)


def test_adjacent_relative_translation_and_tubelet_alignment():
    c2w = torch.eye(4).repeat(4, 1, 1)
    c2w[:, 0, 3] = torch.tensor([0.0, 1.0, 3.0, 6.0])

    relative = relative_transforms(c2w, reference="adjacent")
    torch.testing.assert_close(relative[:, 0, 3], torch.tensor([0.0, 1.0, 2.0, 3.0]))

    target = tubelet_motion_targets(c2w, tubelet_size=2)
    assert target.shape == (2, 9)
    torch.testing.assert_close(target[:, -3], torch.tensor([0.0, 5.0]))


def test_blender_to_opencv_preserves_valid_homogeneous_transform():
    c2w = torch.eye(4)
    c2w[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    converted = blender_c2w_to_opencv(c2w)
    torch.testing.assert_close(converted[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(converted[:3, 3], torch.tensor([-1.0, 2.0, 3.0]))
