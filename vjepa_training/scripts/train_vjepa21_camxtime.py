#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camdis.data import CamXTimeFactorGridDataset
from camdis.losses import DisentanglementLoss, LossWeights
from camdis.models import FeatureDisentangler, FeatureReconstructor, FrozenVJEPA21, TokenGridSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/vjepa21_camxtime_pilot.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--check-data-only",
        action="store_true",
        help="Decode one real 2x2 sample and print shapes without loading V-JEPA.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(config: dict) -> CamXTimeFactorGridDataset:
    return CamXTimeFactorGridDataset(**config["dataset"], seed=config["seed"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    dataset = build_dataset(config)

    if args.check_data_only:
        sample = dataset[0]
        print(
            {
                "video": tuple(sample["video"].shape),
                "camera_c2w": tuple(sample["camera_c2w"].shape),
                "camera_target": tuple(sample["camera_target"].shape),
                "scene_id": sample["scene_id"],
                "trajectory_ids": sample["trajectory_ids"],
                "camera_indices": sample["camera_indices"].tolist(),
                "time_indices": sample["time_indices"].tolist(),
            }
        )
        return

    checkpoint_path = config["backbone"].get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError(
            "Set backbone.checkpoint_path to a local V-JEPA 2.1 checkpoint. "
            "The repository's current hub URL points at a test localhost server."
        )

    device = torch.device(args.device)
    dataset_config = config["dataset"]
    grid = TokenGridSpec(
        num_frames=dataset_config["num_frames"],
        height=dataset_config["crop_size"],
        width=dataset_config["crop_size"],
        tubelet_size=dataset_config["tubelet_size"],
        patch_size=config["backbone"]["patch_size"],
    )
    backbone = FrozenVJEPA21(
        repo_path=config["backbone"]["repo_path"],
        checkpoint_path=checkpoint_path,
        model_name=config["backbone"]["model_name"],
        grid=grid,
    ).to(device)

    model_config = config["model"]
    disentangler = FeatureDisentangler(
        backbone_dim=backbone.embed_dim,
        **model_config,
    ).to(device)
    reconstructor = FeatureReconstructor(
        backbone_dim=backbone.embed_dim,
        **{key: value for key, value in model_config.items() if key != "max_temporal_tokens"},
        **config.get("reconstructor", {}),
    ).to(device)
    criterion = DisentanglementLoss(LossWeights(**config["loss"]))

    train_config = config["train"]
    num_workers = train_config["num_workers"]
    loader = DataLoader(
        dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        # Forking after CUDA and PyTorch thread pools are initialized can
        # deadlock workers during video preprocessing.
        multiprocessing_context="spawn" if num_workers else None,
    )
    optimizer = torch.optim.AdamW(
        list(disentangler.parameters()) + list(reconstructor.parameters()),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    dtype_name = train_config.get("dtype", "bfloat16")
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    use_autocast = device.type == "cuda"
    output_dir = Path(train_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = args.steps or train_config["steps"]

    disentangler.train()
    reconstructor.train()
    iterator = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            dataset.set_epoch(dataset.epoch + 1)
            iterator = iter(loader)
            batch = next(iterator)

        video = batch["video"].to(device, non_blocking=True)
        camera_target = batch["camera_target"].to(device, non_blocking=True)
        batch_size = video.shape[0]
        flat_video = video.reshape(-1, *video.shape[3:])

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_autocast,
        ):
            backbone_tokens = backbone(flat_video)
            features = disentangler(backbone_tokens)
            token_grid = backbone_tokens.reshape(batch_size, 2, 2, *backbone_tokens.shape[1:])
            loss, metrics = criterion(token_grid, features, reconstructor, camera_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(disentangler.parameters()) + list(reconstructor.parameters()), 1.0
        )
        optimizer.step()

        if step == 1 or step % train_config["log_every"] == 0:
            values = " ".join(f"{key}={value.item():.5f}" for key, value in metrics.items())
            print(f"step={step} {values}", flush=True)
        if step % train_config["save_every"] == 0 or step == steps:
            torch.save(
                {
                    "step": step,
                    "config": config,
                    "disentangler": disentangler.state_dict(),
                    "reconstructor": reconstructor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                output_dir / f"step_{step:07d}.pt",
            )


if __name__ == "__main__":
    main()
