import torch

from camdis.losses import DisentanglementLoss, LossWeights, token_relation_loss
from camdis.models.disentangler import FeatureDisentangler, FeatureReconstructor


def test_token_relation_loss_preserves_spatial_and_temporal_geometry():
    torch.manual_seed(7)
    target = torch.randn(2, 3, 4, 8)
    identical = token_relation_loss(target, target)
    reconstructed = torch.randn(2, 3, 4, 8, requires_grad=True)
    different = token_relation_loss(reconstructed, target)

    assert identical.item() < 1e-10
    assert different.item() > identical.item()
    different.backward()
    assert reconstructed.grad is not None


def test_disentangler_and_all_crossed_reconstructions_backpropagate():
    torch.manual_seed(0)
    batch, time, spatial, backbone_dim = 2, 3, 4, 16
    token_grid = torch.randn(batch, 2, 2, time, spatial, backbone_dim)
    flat_tokens = token_grid.reshape(batch * 4, time, spatial, backbone_dim)

    disentangler = FeatureDisentangler(
        backbone_dim=backbone_dim,
        model_dim=12,
        depth=1,
        num_heads=3,
        max_temporal_tokens=4,
    )
    reconstructor = FeatureReconstructor(
        backbone_dim=backbone_dim,
        model_dim=12,
        depth=1,
        num_heads=3,
    )
    features = disentangler(flat_tokens)
    assert features.content.shape == (batch * 4, time, spatial, 12)
    assert features.camera.shape == (batch * 4, time, 12)
    assert features.camera_prediction.shape == (batch * 4, time, 9)

    camera_target = torch.randn(batch, 2, time, 9)
    loss, metrics = DisentanglementLoss(
        LossWeights(
            identity_reconstruction=0.25,
            content_swap_reconstruction=2.0,
            camera_swap_reconstruction=0.25,
            double_swap_reconstruction=2.0,
            camera_factor_ranking=0.1,
            content_factor_ranking=0.2,
            camera_variance=0.1,
            camera_delta_reconstruction=0.5,
            camera_delta_cosine=0.25,
            content_delta_reconstruction=0.5,
        )
    )(
        token_grid, features, reconstructor, camera_target
    )
    assert set(metrics) == {
        "loss",
        "reconstruction",
        "cosine",
        "camera_supervision",
        "camera_invariance",
        "content_invariance",
        "camera_factor_ranking",
        "content_factor_ranking",
        "camera_variance_regularization",
        "reconstruction_identity",
        "reconstruction_content_swap",
        "reconstruction_camera_swap",
        "reconstruction_double_swap",
        "camera_path_separation",
        "content_physics_separation",
        "camera_feature_variance",
        "content_feature_variance",
        "camera_delta_reconstruction",
        "camera_delta_cosine",
        "content_delta_reconstruction",
    }
    loss.backward()
    assert disentangler.camera_query_first.grad is not None
    assert reconstructor.output_projection.weight.grad is not None


def test_factor_delta_losses_contribute_without_pose_supervision():
    torch.manual_seed(3)
    batch, time, spatial, backbone_dim = 1, 2, 3, 8
    token_grid = torch.randn(batch, 2, 2, time, spatial, backbone_dim)
    flat_tokens = token_grid.reshape(batch * 4, time, spatial, backbone_dim)
    disentangler = FeatureDisentangler(
        backbone_dim=backbone_dim,
        model_dim=8,
        depth=1,
        num_heads=2,
        max_temporal_tokens=2,
    )
    reconstructor = FeatureReconstructor(
        backbone_dim=backbone_dim,
        model_dim=8,
        depth=1,
        num_heads=2,
        camera_conditioning="spatial_film",
    )
    features = disentangler(flat_tokens)
    criterion = DisentanglementLoss(
        LossWeights(
            reconstruction=0.0,
            cosine=0.0,
            camera_supervision=0.0,
            camera_invariance=0.0,
            content_invariance=0.0,
            camera_delta_reconstruction=1.5,
            camera_delta_cosine=0.25,
            content_delta_reconstruction=0.75,
        )
    )

    loss, metrics = criterion(token_grid, features, reconstructor, torch.empty(0))
    expected = (
        1.5 * metrics["camera_delta_reconstruction"]
        + 0.25 * metrics["camera_delta_cosine"]
        + 0.75 * metrics["content_delta_reconstruction"]
    )
    assert torch.allclose(loss.detach(), expected)
    assert metrics["camera_delta_reconstruction"].item() > 0.0
    assert metrics["content_delta_reconstruction"].item() > 0.0

    loss.backward()
    assert disentangler.camera_query_first.grad is not None
    assert reconstructor.output_projection.weight.grad is not None


def test_spatial_film_reconstructor_has_direct_camera_gradient():
    torch.manual_seed(2)
    reconstructor = FeatureReconstructor(
        backbone_dim=16,
        model_dim=12,
        depth=1,
        num_heads=3,
        camera_conditioning="spatial_film",
    )
    content = torch.randn(2, 3, 4, 12)
    camera = torch.randn(2, 3, 12, requires_grad=True)
    output = reconstructor(content, camera)
    output.square().mean().backward()

    affine = reconstructor.blocks[0].camera_affine
    assert affine is not None
    assert affine[-1].weight.grad is not None
    assert camera.grad is not None


def test_spatial_basis_reconstructor_has_position_dependent_camera_gradient():
    torch.manual_seed(4)
    reconstructor = FeatureReconstructor(
        backbone_dim=16,
        model_dim=12,
        depth=1,
        num_heads=3,
        camera_conditioning="spatial_basis",
        max_spatial_tokens=4,
        spatial_basis_rank=3,
    )
    content = torch.randn(2, 3, 4, 12)
    camera = torch.randn(2, 3, 12, requires_grad=True)
    output = reconstructor(content, camera)
    changed_output = reconstructor(content, camera.roll(1, dims=0))

    assert output.shape == (2, 3, 4, 16)
    assert not torch.allclose(output, changed_output)
    output.square().mean().backward()

    block = reconstructor.blocks[0]
    assert block.spatial_basis is not None
    assert block.spatial_basis.grad is not None
    assert block.camera_basis_coefficients is not None
    assert block.camera_basis_coefficients[-1].weight.grad is not None
    assert camera.grad is not None


def test_spatial_basis_film_has_position_dependent_camera_gradient():
    torch.manual_seed(5)
    reconstructor = FeatureReconstructor(
        backbone_dim=16,
        model_dim=12,
        depth=1,
        num_heads=3,
        camera_conditioning="spatial_basis_film",
        max_spatial_tokens=4,
        spatial_basis_rank=3,
    )
    content = torch.randn(2, 3, 4, 12)
    camera = torch.randn(2, 3, 12, requires_grad=True)
    output = reconstructor(content, camera)
    changed_output = reconstructor(content, camera.roll(1, dims=0))

    assert output.shape == (2, 3, 4, 16)
    assert not torch.allclose(output, changed_output)
    output.square().mean().backward()

    block = reconstructor.blocks[0]
    assert block.spatial_basis is not None
    assert block.spatial_basis.shape == (4, 3, 24)
    assert block.spatial_basis.grad is not None
    assert block.camera_basis_coefficients is not None
    assert block.camera_basis_coefficients[-1].weight.grad is not None
    assert camera.grad is not None


def test_pose_supervision_can_be_disabled_cleanly():
    torch.manual_seed(1)
    batch, time, spatial, backbone_dim = 1, 2, 3, 8
    token_grid = torch.randn(batch, 2, 2, time, spatial, backbone_dim)
    flat_tokens = token_grid.reshape(batch * 4, time, spatial, backbone_dim)
    disentangler = FeatureDisentangler(
        backbone_dim=backbone_dim,
        model_dim=8,
        depth=1,
        num_heads=2,
        max_temporal_tokens=2,
    )
    reconstructor = FeatureReconstructor(
        backbone_dim=backbone_dim,
        model_dim=8,
        depth=1,
        num_heads=2,
    )
    features = disentangler(flat_tokens)
    criterion = DisentanglementLoss()
    criterion.weights = criterion.weights.__class__(camera_supervision=0.0)
    loss, metrics = criterion(
        token_grid,
        features,
        reconstructor,
        torch.empty(0),
    )
    loss.backward()
    assert metrics["camera_supervision"].item() == 0.0
    assert disentangler.camera_head.weight.grad is None
    assert disentangler.camera_query_first.grad is not None
