#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camdis.data import CamXTimeFactorGridDataset
from camdis.losses import token_relation_loss
from camdis.models import FeatureDisentangler, FeatureReconstructor, FrozenVJEPA21
from camdis.models import TokenGridSpec
from camdis.training import ConvergenceMonitor, atomic_json_dump, atomic_torch_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path, default=None)
    checkpoint_group.add_argument("--initialize", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(
        ema_model.parameters(), model.parameters(), strict=True
    ):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer.detach())


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    start_index: int = 0,
) -> DataLoader:
    source = (
        dataset
        if start_index == 0
        else Subset(dataset, range(start_index, len(dataset)))
    )
    return DataLoader(
        source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        multiprocessing_context="spawn" if num_workers else None,
    )


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length < 1 or count < 1:
        raise ValueError("Validation length and count must be positive")
    count = min(length, count)
    if count == 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


@torch.inference_mode()
def validate_identity(
    *,
    loader: DataLoader,
    backbone: FrozenVJEPA21,
    disentangler: FeatureDisentangler,
    reconstructor: FeatureReconstructor,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> dict[str, float]:
    disentangler.eval()
    reconstructor.eval()
    mse_values = []
    cosine_values = []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        flat_video = video.reshape(-1, *video.shape[3:])
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            target = backbone(flat_video)
            features = disentangler(target)
            reconstructed = reconstructor(features.content, features.camera)
        mse_values.extend(
            (reconstructed.float() - target.float())
            .square()
            .flatten(1)
            .mean(dim=1)
            .cpu()
            .tolist()
        )
        cosine_values.extend(
            F.cosine_similarity(
                reconstructed.float().flatten(1),
                target.float().flatten(1),
                dim=1,
            )
            .cpu()
            .tolist()
        )
    disentangler.train()
    reconstructor.train()
    return {
        "mse": float(np.mean(mse_values)),
        "cosine": float(np.mean(cosine_values)),
        "examples": len(mse_values),
    }


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def full_checkpoint(
    *,
    step: int,
    config: dict,
    disentangler: FeatureDisentangler,
    reconstructor: FeatureReconstructor,
    optimizer: torch.optim.Optimizer,
    ema_disentangler: FeatureDisentangler | None,
    ema_reconstructor: FeatureReconstructor | None,
    monitor: ConvergenceMonitor | None = None,
    validation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    checkpoint = {
        "step": step,
        "config": config,
        "disentangler": disentangler.state_dict(),
        "reconstructor": reconstructor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }
    if ema_disentangler is not None and ema_reconstructor is not None:
        checkpoint["disentangler_ema"] = ema_disentangler.state_dict()
        checkpoint["reconstructor_ema"] = ema_reconstructor.state_dict()
    if monitor is not None:
        checkpoint["convergence_monitor"] = monitor.state_dict()
        checkpoint["validation_history"] = validation_history or []
    return checkpoint


def best_checkpoint(
    *,
    step: int,
    config: dict,
    validation: dict[str, float],
    disentangler: FeatureDisentangler,
    reconstructor: FeatureReconstructor,
) -> dict[str, Any]:
    return {
        "step": step,
        "config": config,
        "validation": validation,
        "disentangler": disentangler.state_dict(),
        "reconstructor": reconstructor.state_dict(),
    }


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    set_seed(config["seed"])

    dataset = CamXTimeFactorGridDataset(
        **config["dataset"], seed=config["seed"]
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
        checkpoint_path=config["backbone"]["checkpoint_path"],
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
        **{
            key: value
            for key, value in model_config.items()
            if key != "max_temporal_tokens"
        },
        **config.get("reconstructor", {}),
    ).to(device)

    train_config = config["train"]
    num_workers = train_config["num_workers"]
    parameters = list(disentangler.parameters()) + list(reconstructor.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    loss_config = config["autoencoder_loss"]
    relation_weight = float(loss_config.get("token_relation", 0.0))
    ema_decay = train_config.get("ema_decay")
    if ema_decay is not None and not 0.0 <= float(ema_decay) < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if ema_decay is not None:
        ema_disentangler = copy.deepcopy(disentangler).eval().requires_grad_(False)
        ema_reconstructor = copy.deepcopy(reconstructor).eval().requires_grad_(False)
    else:
        ema_disentangler = None
        ema_reconstructor = None
    dtype_name = train_config.get("dtype", "bfloat16")
    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[
        dtype_name
    ]
    output_dir = args.output_dir or Path(train_config["output_dir"])
    output_dir = output_dir.expanduser().resolve()
    config["train"]["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = args.steps or train_config["steps"]

    resume_path = args.resume or train_config.get("resume_from")
    initialize_path = args.initialize or train_config.get("initialize_from")
    if resume_path is not None and initialize_path is not None:
        raise ValueError("Configure only one of resume_from and initialize_from")
    checkpoint_path = resume_path or initialize_path
    start_step = 0
    resume_payload = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        resume_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        disentangler.load_state_dict(resume_payload["disentangler"], strict=True)
        reconstructor.load_state_dict(resume_payload["reconstructor"], strict=True)
        start_step = int(resume_payload["step"])
        if resume_path is not None:
            if "optimizer" not in resume_payload:
                raise KeyError(
                    f"Resume checkpoint has no optimizer state: {checkpoint_path}"
                )
            optimizer.load_state_dict(resume_payload["optimizer"])
            if ema_disentangler is not None and ema_reconstructor is not None:
                ema_disentangler.load_state_dict(
                    resume_payload.get(
                        "disentangler_ema", resume_payload["disentangler"]
                    ),
                    strict=True,
                )
                ema_reconstructor.load_state_dict(
                    resume_payload.get(
                        "reconstructor_ema", resume_payload["reconstructor"]
                    ),
                    strict=True,
                )
            if "rng_state" in resume_payload:
                restore_rng_state(resume_payload["rng_state"])
            print(f"resumed={checkpoint_path} step={start_step}", flush=True)
        else:
            if ema_disentangler is not None and ema_reconstructor is not None:
                ema_disentangler.load_state_dict(
                    resume_payload["disentangler"], strict=True
                )
                ema_reconstructor.load_state_dict(
                    resume_payload["reconstructor"], strict=True
                )
            print(
                f"initialized={checkpoint_path} step={start_step} optimizer=reset",
                flush=True,
            )
    if start_step >= steps:
        raise ValueError(f"Resume step {start_step} must be smaller than target step {steps}")

    start_epoch, start_index = divmod(start_step, len(dataset))
    dataset.set_epoch(start_epoch)
    loader = build_loader(
        dataset,
        batch_size=train_config["batch_size"],
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        start_index=start_index,
    )

    convergence_config = config.get("convergence")
    if convergence_config:
        monitor = ConvergenceMonitor(
            min_relative_mse_improvement=float(
                convergence_config["min_relative_mse_improvement"]
            ),
            min_cosine_gain=float(convergence_config["min_cosine_gain"]),
            validations_before_lr_drop=int(
                convergence_config["validations_before_lr_drop"]
            ),
            validations_after_lr_drop=int(
                convergence_config["validations_after_lr_drop"]
            ),
        )
        validation_history = []
        if resume_path is not None and "convergence_monitor" in resume_payload:
            monitor.load_state_dict(resume_payload["convergence_monitor"])
            validation_history = list(resume_payload.get("validation_history", []))
        validation_dataset_config = dict(dataset_config)
        validation_dataset_config["samples_per_epoch"] = max(
            validation_dataset_config["samples_per_epoch"], 417
        )
        validation_dataset = CamXTimeFactorGridDataset(
            **validation_dataset_config,
            seed=int(convergence_config["validation_seed"]),
        )
        validation_indices = evenly_spaced_indices(
            len(validation_dataset.scene_ids),
            int(convergence_config["validation_samples"]),
        )
        validation_loader = build_loader(
            Subset(validation_dataset, validation_indices),
            batch_size=1,
            num_workers=int(
                convergence_config.get("validation_num_workers", num_workers)
            ),
            pin_memory=device.type == "cuda",
        )
    else:
        monitor = None
        validation_history = []
        validation_loader = None
    del resume_payload

    def validate_and_save(step: int) -> bool:
        assert monitor is not None and validation_loader is not None
        validation = validate_identity(
            loader=validation_loader,
            backbone=backbone,
            disentangler=disentangler,
            reconstructor=reconstructor,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        decision = monitor.update(
            step=step,
            mse=validation["mse"],
            cosine=validation["cosine"],
        )
        if decision.action == "reduce_lr":
            factor = float(convergence_config["lr_drop_factor"])
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= factor
        record = {
            "step": step,
            **validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "is_best": decision.is_best,
            "is_flat": decision.is_flat,
            "relative_mse_improvement": decision.relative_mse_improvement,
            "cosine_gain": decision.cosine_gain,
            "action": decision.action,
        }
        validation_history.append(record)
        if decision.is_best:
            atomic_torch_save(
                best_checkpoint(
                    step=step,
                    config=config,
                    validation=validation,
                    disentangler=disentangler,
                    reconstructor=reconstructor,
                ),
                output_dir / "best.pt",
            )
        atomic_json_dump(validation_history, output_dir / "validation_history.json")
        atomic_torch_save(
            full_checkpoint(
                step=step,
                config=config,
                disentangler=disentangler,
                reconstructor=reconstructor,
                optimizer=optimizer,
                ema_disentangler=ema_disentangler,
                ema_reconstructor=ema_reconstructor,
                monitor=monitor,
                validation_history=validation_history,
            ),
            output_dir / "latest.pt",
        )
        print(
            f"validation step={step} mse={validation['mse']:.6f} "
            f"cosine={validation['cosine']:.6f} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"flat={decision.is_flat} action={decision.action}",
            flush=True,
        )
        return decision.action == "stop"

    disentangler.train()
    reconstructor.train()
    if (
        monitor is not None
        and bool(convergence_config.get("validate_on_resume", True))
        and (not validation_history or validation_history[-1]["step"] != start_step)
    ):
        validate_and_save(start_step)
    iterator = iter(loader)
    final_step = start_step
    stopped_for_convergence = False
    for step in range(start_step + 1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            dataset.set_epoch(dataset.epoch + 1)
            loader = build_loader(
                dataset,
                batch_size=train_config["batch_size"],
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )
            iterator = iter(loader)
            batch = next(iterator)

        video = batch["video"].to(device, non_blocking=True)
        flat_video = video.reshape(-1, *video.shape[3:])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=device.type == "cuda",
        ):
            target = backbone(flat_video).detach()
            features = disentangler(target)
            reconstructed = reconstructor(features.content, features.camera)
            reconstruction = F.mse_loss(reconstructed.float(), target.float())
            cosine = 1.0 - F.cosine_similarity(
                reconstructed.float(), target.float(), dim=-1
            ).mean()
            if relation_weight:
                relation = token_relation_loss(reconstructed, target)
            else:
                relation = reconstruction.new_zeros(())
            loss = (
                loss_config["reconstruction"] * reconstruction
                + loss_config["cosine"] * cosine
                + relation_weight * relation
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if ema_disentangler is not None and ema_reconstructor is not None:
            update_ema(ema_disentangler, disentangler, float(ema_decay))
            update_ema(ema_reconstructor, reconstructor, float(ema_decay))

        if step == 1 or step % train_config["log_every"] == 0:
            print(
                f"step={step} loss={loss.item():.5f} "
                f"reconstruction={reconstruction.item():.5f} "
                f"cosine={cosine.item():.5f} relation={relation.item():.5f}",
                flush=True,
            )
        final_step = step
        if (
            monitor is not None
            and step % int(convergence_config["validation_every"]) == 0
        ):
            stopped_for_convergence = validate_and_save(step)
            if stopped_for_convergence:
                break
        elif monitor is None and (
            step % train_config["save_every"] == 0 or step == steps
        ):
            torch.save(
                full_checkpoint(
                    step=step,
                    config=config,
                    disentangler=disentangler,
                    reconstructor=reconstructor,
                    optimizer=optimizer,
                    ema_disentangler=ema_disentangler,
                    ema_reconstructor=ema_reconstructor,
                ),
                output_dir / f"step_{step:07d}.pt",
            )

    if (
        monitor is not None
        and not stopped_for_convergence
        and (not validation_history or validation_history[-1]["step"] != final_step)
    ):
        validate_and_save(final_step)


if __name__ == "__main__":
    main()
