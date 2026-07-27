from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import av
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from camdis.geometry.camera import blender_c2w_to_opencv, tubelet_motion_targets


_VIDEO_PATTERN = re.compile(r"cam(\d{3})_full_motion\.mp4$")
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrajectoryGrid:
    scene_id: str
    trajectory_id: str
    directory: Path
    json_path: Path
    videos: tuple[Path, ...]


@dataclass(frozen=True)
class FactorGridCoordinates:
    scene_id: str
    trajectories: tuple[TrajectoryGrid, TrajectoryGrid]
    camera_indices: tuple[tuple[int, ...], tuple[int, ...]]
    time_indices: tuple[tuple[int, ...], tuple[int, ...]]


@lru_cache(maxsize=64)
def _read_camera_json(json_path: str) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    trajectory = payload["trajectory"]
    frame_keys = tuple(sorted(int(key) for key in trajectory))
    c2w = torch.tensor(
        [trajectory[str(key)]["c2w"] for key in frame_keys], dtype=torch.float32
    )
    intrinsics = torch.tensor(payload["intrinsics"]["K"], dtype=torch.float32)
    return c2w, intrinsics, frame_keys


class PyAVFrameReader:
    """Decode requested zero-based video frames with a single sequential pass."""

    def read_many(self, path: str | Path, frame_indices: Iterable[int]) -> dict[int, np.ndarray]:
        requested = sorted(set(int(index) for index in frame_indices))
        if not requested:
            return {}
        if requested[0] < 0:
            raise ValueError(f"Frame indices must be non-negative, got {requested[0]}")

        frames: dict[int, np.ndarray] = {}
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "FRAME"
            stream.thread_count = 2
            wanted = set(requested)
            final_index = requested[-1]
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index in wanted:
                    frames[frame_index] = frame.to_ndarray(format="rgb24")
                    wanted.remove(frame_index)
                    if not wanted:
                        break
                if frame_index >= final_index:
                    break

        missing = sorted(set(requested) - set(frames))
        if missing:
            raise RuntimeError(f"Could not decode frames {missing} from {path}")
        return frames


def _discover_scene_roots(root: Path) -> list[Path]:
    candidates: list[Path] = []
    train_root = root / "train"
    if train_root.is_dir():
        candidates.extend(train_root.glob("Scene[0-9][0-9][0-9]"))
    candidates.extend(root.glob("Scene[0-9][0-9][0-9]"))
    return sorted(set(path.resolve() for path in candidates if path.is_dir()))


def _discover_trajectory(
    directory: Path,
    expected_grid_size: int,
    require_complete: bool,
) -> TrajectoryGrid | None:
    indexed_videos: list[tuple[int, Path]] = []
    for video_path in directory.glob("cam*_full_motion.mp4"):
        match = _VIDEO_PATTERN.match(video_path.name)
        if match:
            indexed_videos.append((int(match.group(1)), video_path))
    indexed_videos.sort()

    json_paths = sorted(directory.glob("*-camera.json"))
    if not json_paths:
        return None
    if require_complete and len(indexed_videos) != expected_grid_size:
        return None
    if not indexed_videos:
        return None

    expected_ids = list(range(1, len(indexed_videos) + 1))
    actual_ids = [index for index, _ in indexed_videos]
    if actual_ids != expected_ids:
        return None

    return TrajectoryGrid(
        scene_id=directory.parent.name,
        trajectory_id=directory.name,
        directory=directory,
        json_path=json_paths[0],
        videos=tuple(path for _, path in indexed_videos),
    )


def discover_camxtime_grids(
    root: str | Path,
    expected_grid_size: int = 120,
    require_complete: bool = True,
) -> dict[str, tuple[TrajectoryGrid, ...]]:
    """Find completed raw CamXTime camera-by-time grids.

    Both ``CamXTime/Scene*`` (during download) and
    ``CamXTime/train/Scene*`` (final organized layout) are supported.
    """

    root = Path(root).expanduser()
    scenes: dict[str, tuple[TrajectoryGrid, ...]] = {}
    for scene_path in _discover_scene_roots(root):
        trajectories = []
        for trajectory_path in sorted(scene_path.glob("camera-trajectory-*")):
            grid = _discover_trajectory(
                trajectory_path,
                expected_grid_size=expected_grid_size,
                require_complete=require_complete,
            )
            if grid is not None:
                trajectories.append(grid)
        if trajectories:
            scenes[scene_path.name] = tuple(trajectories)
    return scenes


def _sample_monotonic_path(
    rng: random.Random,
    length: int,
    stride: int,
    grid_size: int,
) -> tuple[int, ...]:
    span = (length - 1) * stride
    if span >= grid_size:
        raise ValueError(
            f"Path length {length} with stride {stride} does not fit grid size {grid_size}"
        )
    start = rng.randint(0, grid_size - span - 1)
    path = [start + offset * stride for offset in range(length)]
    if rng.random() < 0.5:
        path.reverse()
    return tuple(path)


def _sample_distinct_path(
    rng: random.Random,
    other: Sequence[int],
    length: int,
    stride: int,
    grid_size: int,
) -> tuple[int, ...]:
    for _ in range(16):
        candidate = _sample_monotonic_path(rng, length, stride, grid_size)
        if candidate != tuple(other):
            return candidate
    return tuple(reversed(other))


def _resize_center_crop(frames: Tensor, crop_size: int) -> Tensor:
    """Apply V-JEPA's eval resize-short-side and center-crop policy."""

    if frames.ndim != 4:
        raise ValueError(f"Expected [T, C, H, W], got {tuple(frames.shape)}")
    height, width = frames.shape[-2:]
    short_side = int(crop_size * 256 / 224)
    scale = short_side / min(height, width)
    resized_height = max(crop_size, round(height * scale))
    resized_width = max(crop_size, round(width * scale))
    frames = F.interpolate(
        frames,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (resized_height - crop_size) // 2
    left = (resized_width - crop_size) // 2
    return frames[..., top : top + crop_size, left : left + crop_size]


class CamXTimeFactorGridDataset(Dataset[dict[str, Tensor | str | list[str]]]):
    """Sample exact 2x2 camera-motion by physical-time factor grids.

    A CamXTime MP4 fixes one camera position and varies physical time. A
    moving-camera clip therefore uses frame ``time[t]`` from MP4
    ``camera[t]``. Each item contains four clips ``video[camera, physics]``:

    - the two clips in a row share the exact camera path;
    - the two clips in a column share the exact physical-time path;
    - all four clips come from one scene, so crossed swap targets exist.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        num_frames: int = 16,
        crop_size: int = 384,
        camera_stride: int = 2,
        time_stride: int = 2,
        tubelet_size: int = 2,
        samples_per_epoch: int = 10_000,
        expected_grid_size: int = 120,
        require_complete: bool = True,
        seed: int = 0,
        frame_reader: PyAVFrameReader | None = None,
    ) -> None:
        if num_frames < tubelet_size or num_frames % tubelet_size:
            raise ValueError("num_frames must be a positive multiple of tubelet_size")
        if crop_size < 1:
            raise ValueError("crop_size must be positive")
        if samples_per_epoch < 1:
            raise ValueError("samples_per_epoch must be positive")

        self.root = Path(root).expanduser()
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.camera_stride = camera_stride
        self.time_stride = time_stride
        self.tubelet_size = tubelet_size
        self.samples_per_epoch = samples_per_epoch
        self.expected_grid_size = expected_grid_size
        self.seed = seed
        self.epoch = 0
        self.frame_reader = frame_reader or PyAVFrameReader()

        scenes = discover_camxtime_grids(
            self.root,
            expected_grid_size=expected_grid_size,
            require_complete=require_complete,
        )
        self.scenes = {
            scene_id: grids for scene_id, grids in scenes.items() if len(grids) >= 1
        }
        self.scene_ids = tuple(sorted(self.scenes))
        if not self.scene_ids:
            raise RuntimeError(
                f"No complete CamXTime grids found below {self.root}. "
                "A valid trajectory needs its camera JSON and all MP4 files."
            )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def sample_coordinates(self, index: int) -> FactorGridCoordinates:
        rng = random.Random(self.seed + self.epoch * self.samples_per_epoch + index)
        scene_id = self.scene_ids[index % len(self.scene_ids)]
        grids = self.scenes[scene_id]
        if len(grids) >= 2:
            trajectory_a, trajectory_b = rng.sample(grids, 2)
        else:
            trajectory_a = trajectory_b = grids[0]

        camera_a = _sample_monotonic_path(
            rng,
            self.num_frames,
            self.camera_stride,
            len(trajectory_a.videos),
        )
        camera_b = _sample_distinct_path(
            rng,
            camera_a,
            self.num_frames,
            self.camera_stride,
            len(trajectory_b.videos),
        )
        time_a = _sample_monotonic_path(
            rng,
            self.num_frames,
            self.time_stride,
            self.expected_grid_size,
        )
        time_b = _sample_distinct_path(
            rng,
            time_a,
            self.num_frames,
            self.time_stride,
            self.expected_grid_size,
        )
        return FactorGridCoordinates(
            scene_id=scene_id,
            trajectories=(trajectory_a, trajectory_b),
            camera_indices=(camera_a, camera_b),
            time_indices=(time_a, time_b),
        )

    def _decode_factor_grid(self, coordinates: FactorGridCoordinates) -> Tensor:
        requests: dict[Path, set[int]] = defaultdict(set)
        for camera_factor in range(2):
            trajectory = coordinates.trajectories[camera_factor]
            camera_path = coordinates.camera_indices[camera_factor]
            for physics_factor in range(2):
                time_path = coordinates.time_indices[physics_factor]
                for camera_index, time_index in zip(camera_path, time_path):
                    requests[trajectory.videos[camera_index]].add(time_index)

        decoded = {
            video_path: self.frame_reader.read_many(video_path, indices)
            for video_path, indices in requests.items()
        }

        clips = []
        for camera_factor in range(2):
            trajectory = coordinates.trajectories[camera_factor]
            camera_path = coordinates.camera_indices[camera_factor]
            physics_clips = []
            for physics_factor in range(2):
                time_path = coordinates.time_indices[physics_factor]
                frames = [
                    decoded[trajectory.videos[camera_index]][time_index]
                    for camera_index, time_index in zip(camera_path, time_path)
                ]
                physics_clips.append(np.stack(frames))
            clips.append(np.stack(physics_clips))

        video = torch.from_numpy(np.stack(clips)).permute(0, 1, 2, 5, 3, 4)
        flat_video = video.reshape(-1, 3, video.shape[-2], video.shape[-1]).float().div_(255.0)
        flat_video = _resize_center_crop(flat_video, self.crop_size)
        mean = torch.tensor(_IMAGENET_MEAN, dtype=flat_video.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=flat_video.dtype).view(1, 3, 1, 1)
        flat_video = (flat_video - mean) / std
        return flat_video.reshape(2, 2, self.num_frames, 3, self.crop_size, self.crop_size).permute(
            0, 1, 3, 2, 4, 5
        )

    def _camera_data(self, coordinates: FactorGridCoordinates) -> tuple[Tensor, Tensor, Tensor]:
        c2w_paths = []
        intrinsics = []
        for camera_factor in range(2):
            trajectory = coordinates.trajectories[camera_factor]
            blender_c2w, intrinsic, _ = _read_camera_json(str(trajectory.json_path))
            camera_indices = torch.tensor(
                coordinates.camera_indices[camera_factor], dtype=torch.long
            )
            opencv_c2w = blender_c2w_to_opencv(blender_c2w.index_select(0, camera_indices))
            c2w_paths.append(opencv_c2w)
            intrinsics.append(intrinsic)

        c2w = torch.stack(c2w_paths)
        targets = tubelet_motion_targets(
            c2w,
            tubelet_size=self.tubelet_size,
            reference="adjacent",
        )
        return c2w, targets, torch.stack(intrinsics)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | list[str]]:
        coordinates = self.sample_coordinates(index)
        c2w, camera_target, intrinsics = self._camera_data(coordinates)
        return {
            "video": self._decode_factor_grid(coordinates),
            "camera_c2w": c2w,
            "camera_target": camera_target,
            "intrinsics": intrinsics,
            "camera_indices": torch.tensor(coordinates.camera_indices, dtype=torch.long),
            "time_indices": torch.tensor(coordinates.time_indices, dtype=torch.long),
            "scene_id": coordinates.scene_id,
            "trajectory_ids": [grid.trajectory_id for grid in coordinates.trajectories],
        }
