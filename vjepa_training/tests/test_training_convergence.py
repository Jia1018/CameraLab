from pathlib import Path

import torch

from camdis.training import ConvergenceMonitor, atomic_json_dump, atomic_torch_save


def make_monitor() -> ConvergenceMonitor:
    return ConvergenceMonitor(
        min_relative_mse_improvement=0.01,
        min_cosine_gain=0.001,
        validations_before_lr_drop=3,
        validations_after_lr_drop=2,
    )


def test_monitor_reduces_lr_then_confirms_plateau() -> None:
    monitor = make_monitor()
    assert monitor.update(step=2000, mse=0.05, cosine=0.98).action == "continue"
    assert monitor.update(step=2500, mse=0.0498, cosine=0.9802).action == "continue"
    assert monitor.update(step=3000, mse=0.0497, cosine=0.9803).action == "continue"
    decision = monitor.update(step=3500, mse=0.0496, cosine=0.9804)
    assert decision.action == "reduce_lr"
    assert monitor.lr_reduced
    assert monitor.update(step=4000, mse=0.0495, cosine=0.9805).action == "continue"
    decision = monitor.update(step=4500, mse=0.0494, cosine=0.9806)
    assert decision.action == "stop"
    assert monitor.converged


def test_material_improvement_resets_flat_count_and_state_round_trips() -> None:
    monitor = make_monitor()
    monitor.update(step=2000, mse=0.05, cosine=0.98)
    monitor.update(step=2500, mse=0.0498, cosine=0.9802)
    decision = monitor.update(step=3000, mse=0.047, cosine=0.9815)
    assert not decision.is_flat
    assert decision.is_best
    assert monitor.flat_validations == 0

    restored = make_monitor()
    restored.load_state_dict(monitor.state_dict())
    assert restored.state_dict() == monitor.state_dict()


def test_recovery_below_historical_best_still_counts_as_flat() -> None:
    monitor = make_monitor()
    monitor.update(step=2000, mse=0.05, cosine=0.98)
    monitor.update(step=2500, mse=0.06, cosine=0.97)
    monitor.update(step=3000, mse=0.07, cosine=0.96)
    decision = monitor.update(step=3500, mse=0.055, cosine=0.975)

    assert decision.is_flat
    assert decision.relative_mse_improvement < 0
    assert decision.cosine_gain < 0
    assert decision.action == "reduce_lr"


def test_atomic_writers_leave_only_final_files(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "latest.pt"
    json_path = tmp_path / "history.json"
    atomic_torch_save({"tensor": torch.tensor([1, 2])}, checkpoint_path)
    atomic_json_dump({"step": 10}, json_path)

    assert torch.equal(
        torch.load(checkpoint_path, weights_only=True)["tensor"],
        torch.tensor([1, 2]),
    )
    assert json_path.read_text(encoding="utf-8").strip().endswith("}")
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "history.json",
        "latest.pt",
    ]
