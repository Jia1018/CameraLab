import json

from camdis.data.camxtime import CamXTimeFactorGridDataset


def _make_grid(root, scene, trajectory, grid_size):
    directory = root / scene / trajectory
    directory.mkdir(parents=True)
    for index in range(1, grid_size + 1):
        (directory / f"cam{index:03d}_full_motion.mp4").touch()
    identity = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    payload = {
        "intrinsics": {"K": [[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]},
        "trajectory": {
            str(33 + index): {"c2w": identity, "w2c": identity}
            for index in range(grid_size)
        },
    }
    with open(directory / f"{trajectory}-camera.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_factor_coordinates_share_rows_and_columns(tmp_path):
    grid_size = 8
    _make_grid(tmp_path, "Scene001", "camera-trajectory-01", grid_size)
    _make_grid(tmp_path, "Scene001", "camera-trajectory-02", grid_size)
    dataset = CamXTimeFactorGridDataset(
        tmp_path,
        num_frames=4,
        crop_size=32,
        camera_stride=1,
        time_stride=1,
        expected_grid_size=grid_size,
        samples_per_epoch=2,
        seed=4,
    )
    coordinates = dataset.sample_coordinates(0)
    assert coordinates.scene_id == "Scene001"
    assert len(coordinates.camera_indices) == 2
    assert len(coordinates.time_indices) == 2
    assert coordinates.camera_indices[0] != coordinates.camera_indices[1]
    assert coordinates.time_indices[0] != coordinates.time_indices[1]
    assert all(0 <= index < grid_size for path in coordinates.camera_indices for index in path)
    assert all(0 <= index < grid_size for path in coordinates.time_indices for index in path)
