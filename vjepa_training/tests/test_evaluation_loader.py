import torch
from torch.utils.data import TensorDataset

from scripts.evaluate_camera_interventions import build_evaluation_loader


def test_evaluation_loader_uses_spawn_and_preserves_subset_order():
    dataset = TensorDataset(torch.arange(5))
    loader = build_evaluation_loader(dataset, [3, 1], num_workers=1)

    assert loader.multiprocessing_context is not None
    assert loader.multiprocessing_context.get_start_method() == "spawn"
    assert [batch[0].item() for batch in loader] == [3, 1]
