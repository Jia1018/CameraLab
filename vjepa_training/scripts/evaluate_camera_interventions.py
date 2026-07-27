#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camdis.data import CamXTimeFactorGridDataset
from camdis.models import FeatureDisentangler, FeatureReconstructor, FrozenVJEPA21
from camdis.models import TokenGridSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Checkpoint label and path. May be passed more than once.",
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=10017)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must have NAME=PATH form, got {value!r}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not name or not path.is_file():
            raise FileNotFoundError(f"Invalid checkpoint specification: {value}")
        checkpoints[name] = path
    return checkpoints


def mse_per_example(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() - right.float()).square().flatten(1).mean(dim=1)


def cosine_per_example(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(
        left.float().flatten(1), right.float().flatten(1), dim=1
    )


def build_models(
    checkpoint_paths: dict[str, Path],
    config: dict,
    backbone_dim: int,
    device: torch.device,
) -> tuple[
    dict[str, tuple[FeatureDisentangler, FeatureReconstructor]],
    dict[str, str],
]:
    model_config = config["model"]
    models = {}
    weight_sources = {}
    for name, checkpoint_path in checkpoint_paths.items():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        disentangler = FeatureDisentangler(
            backbone_dim=backbone_dim,
            **model_config,
        )
        reconstructor = FeatureReconstructor(
            backbone_dim=backbone_dim,
            **{
                key: value
                for key, value in model_config.items()
                if key != "max_temporal_tokens"
            },
            **config.get("reconstructor", {}),
        )
        if "disentangler_ema" in payload and "reconstructor_ema" in payload:
            disentangler_key = "disentangler_ema"
            reconstructor_key = "reconstructor_ema"
            weight_sources[name] = "ema"
        else:
            disentangler_key = "disentangler"
            reconstructor_key = "reconstructor"
            weight_sources[name] = "raw"
        disentangler.load_state_dict(payload[disentangler_key], strict=True)
        reconstructor.load_state_dict(payload[reconstructor_key], strict=True)
        disentangler.eval().to(device)
        reconstructor.eval().to(device)
        models[name] = (disentangler, reconstructor)
    return models, weight_sources


def intervention_metrics(
    token_grid: torch.Tensor,
    disentangler: FeatureDisentangler,
    reconstructor: FeatureReconstructor,
) -> dict[str, torch.Tensor]:
    flat_tokens = token_grid.reshape(-1, *token_grid.shape[3:])
    features = disentangler(flat_tokens)
    content = features.content.reshape(1, 2, 2, *features.content.shape[1:])
    camera = features.camera.reshape(1, 2, 2, *features.camera.shape[1:])

    content_inputs = []
    right_cameras = []
    same_path_cameras = []
    wrong_cameras = []
    original_targets = []
    crossed_targets = []
    for camera_factor in range(2):
        for physics_factor in range(2):
            content_inputs.append(content[:, camera_factor, physics_factor])
            right_cameras.append(camera[:, camera_factor, physics_factor])
            same_path_cameras.append(camera[:, camera_factor, 1 - physics_factor])
            wrong_cameras.append(camera[:, 1 - camera_factor, physics_factor])
            original_targets.append(token_grid[:, camera_factor, physics_factor])
            crossed_targets.append(token_grid[:, 1 - camera_factor, physics_factor])

    composed_content = torch.cat(content_inputs, dim=0)
    right_camera = torch.cat(right_cameras, dim=0)
    same_path_camera = torch.cat(same_path_cameras, dim=0)
    wrong_camera = torch.cat(wrong_cameras, dim=0)
    zero_camera = torch.zeros_like(right_camera)
    reversed_camera = right_camera.flip(dims=(1,))
    original_target = torch.cat(original_targets, dim=0)
    crossed_target = torch.cat(crossed_targets, dim=0)

    all_cameras = torch.cat(
        (
            right_camera,
            same_path_camera,
            wrong_camera,
            zero_camera,
            reversed_camera,
        ),
        dim=0,
    )
    all_content = composed_content.repeat(5, 1, 1, 1)
    outputs = reconstructor(all_content, all_cameras)
    right, same_path, wrong, zero, reversed_output = outputs.chunk(5, dim=0)

    target_camera_delta = mse_per_example(original_target, crossed_target)
    epsilon = torch.finfo(torch.float32).eps
    identity_error = mse_per_example(right, original_target)
    wrong_output_delta = mse_per_example(wrong, right)
    wrong_camera_delta_cosine = cosine_per_example(
        wrong - right,
        crossed_target - original_target,
    )
    pairwise_reconstruction_error = (
        (right[:, None].float() - original_target[None].float())
        .square()
        .flatten(2)
        .mean(dim=2)
    )
    target_separation = (
        (original_target[:, None].float() - original_target[None].float())
        .square()
        .flatten(2)
        .mean(dim=2)
    )
    target_separation.fill_diagonal_(float("inf"))
    nearest_negative_distance = target_separation.min(dim=1).values
    return {
        "identity_to_original": identity_error,
        "identity_cosine": cosine_per_example(right, original_target),
        "identity_retrieval_top1": (
            pairwise_reconstruction_error.argmin(dim=1)
            == torch.arange(right.shape[0], device=right.device)
        ).float(),
        "identity_to_nearest_negative_ratio": identity_error
        / nearest_negative_distance.clamp_min(epsilon),
        "same_path_to_original": mse_per_example(same_path, original_target),
        "wrong_camera_to_original": mse_per_example(wrong, original_target),
        "wrong_camera_to_crossed": mse_per_example(wrong, crossed_target),
        "zero_camera_to_original": mse_per_example(zero, original_target),
        "reversed_camera_to_original": mse_per_example(reversed_output, original_target),
        "same_path_output_delta": mse_per_example(same_path, right),
        "wrong_camera_output_delta": wrong_output_delta,
        "zero_camera_output_delta": mse_per_example(zero, right),
        "reversed_camera_output_delta": mse_per_example(reversed_output, right),
        "target_camera_delta": target_camera_delta,
        "wrong_camera_normalized_delta": wrong_output_delta
        / target_camera_delta.clamp_min(epsilon),
        "wrong_camera_delta_cosine": wrong_camera_delta_cosine,
        "wrong_camera_target_preference": (
            mse_per_example(wrong, original_target)
            - mse_per_example(wrong, crossed_target)
        ),
        "wrong_camera_crossed_gain": (
            mse_per_example(right, crossed_target)
            - mse_per_example(wrong, crossed_target)
        ),
        "camera_token_wrong_path_delta": mse_per_example(wrong_camera, right_camera),
        "camera_token_same_path_delta": mse_per_example(
            same_path_camera, right_camera
        ),
    }


def summarize(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": statistics.fmean(items),
            "std": statistics.pstdev(items),
        }
        for key, items in sorted(values.items())
    }


def build_evaluation_loader(
    dataset: CamXTimeFactorGridDataset,
    indices: list[int],
    *,
    num_workers: int,
) -> DataLoader:
    """Decode videos outside the CUDA process to avoid PyAV thread deadlocks."""

    return DataLoader(
        Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        multiprocessing_context="spawn" if num_workers else None,
    )


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    checkpoint_paths = parse_checkpoints(args.checkpoint)
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    torch.manual_seed(args.seed)
    dataset_config = dict(config["dataset"])
    dataset_config["samples_per_epoch"] = max(
        dataset_config["samples_per_epoch"], 417
    )
    dataset = CamXTimeFactorGridDataset(
        **dataset_config,
        seed=args.seed,
    )
    sample_count = min(args.samples, len(dataset.scene_ids))
    if sample_count == 1:
        indices = [0]
    else:
        indices = [
            round(index * (len(dataset.scene_ids) - 1) / (sample_count - 1))
            for index in range(sample_count)
        ]
    loader = build_evaluation_loader(
        dataset,
        indices,
        num_workers=config["train"].get("num_workers", 0),
    )

    device = torch.device(args.device)
    grid = TokenGridSpec(
        num_frames=dataset_config["num_frames"],
        height=dataset_config["crop_size"],
        width=dataset_config["crop_size"],
        tubelet_size=dataset_config["tubelet_size"],
        patch_size=config["backbone"]["patch_size"],
    )
    backbone = FrozenVJEPA21(
        repo_path=config["backbone"]["repo_path"],
        checkpoint_path=config["backbone"]["checkpoint_path"],
        model_name=config["backbone"]["model_name"],
        grid=grid,
    ).eval().to(device)
    models, weight_sources = build_models(
        checkpoint_paths, config, backbone.embed_dim, device
    )
    dtype_name = config["train"].get("dtype", "bfloat16")
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
        dtype_name
    ]

    collected: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in models
    }
    scene_ids = []
    with torch.inference_mode():
        for position, sample in enumerate(loader, start=1):
            scene_id = sample["scene_id"][0]
            scene_ids.append(scene_id)
            video = sample["video"].to(device, non_blocking=True)
            flat_video = video.reshape(-1, *video.shape[3:])
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                backbone_tokens = backbone(flat_video)
                token_grid = backbone_tokens.reshape(
                    1, 2, 2, *backbone_tokens.shape[1:]
                )
                for name, (disentangler, reconstructor) in models.items():
                    metrics = intervention_metrics(
                        token_grid, disentangler, reconstructor
                    )
                    for key, value in metrics.items():
                        collected[name][key].extend(value.float().cpu().tolist())
            print(
                f"sample={position}/{sample_count} scene={scene_id}",
                flush=True,
            )

    result = {
        "metadata": {
            "config": str(args.config.resolve()),
            "checkpoints": {
                name: str(path) for name, path in checkpoint_paths.items()
            },
            "weight_sources": weight_sources,
            "seed": args.seed,
            "sample_count": sample_count,
            "composition_count": sample_count * 4,
            "scene_ids": scene_ids,
            "dataset_indices": indices,
        },
        "models": {
            name: summarize(metric_values)
            for name, metric_values in collected.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for name, metrics in result["models"].items():
        print(f"\n{name}")
        for key in (
            "identity_to_original",
            "identity_cosine",
            "identity_retrieval_top1",
            "identity_to_nearest_negative_ratio",
            "wrong_camera_to_original",
            "wrong_camera_to_crossed",
            "wrong_camera_output_delta",
            "wrong_camera_normalized_delta",
            "wrong_camera_delta_cosine",
            "wrong_camera_target_preference",
            "zero_camera_output_delta",
            "reversed_camera_output_delta",
        ):
            print(f"{key}={metrics[key]['mean']:.6f}")
    print(f"\nwrote={args.output}")


if __name__ == "__main__":
    main()
