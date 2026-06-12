# camera_motion_disentangle

Workspace for generating and reviewing paired synthetic videos for camera-motion / physical-motion disentanglement.

The paired data contract has two controlled axes:

- same camera trajectory + different physics/scene
- different camera trajectory + same physics and same scene

In the current bank generator, `scene_id` is explicit. A "same physics" pair means `physics_id` and `scene_id` are both fixed, so object motion, static scene, wall/floor colors, wall structure, and decorative background elements are identical. Only the camera changes.

## Repository Layout

- `scripts/generate_mock_pairs.py`: 2D dependency-light preview generator.
- `scripts/generate_kubric_pairs.py`: older Kubric-oriented contract scaffold.
- `scripts/generate_kubric_review_bank.py`: current review-bank generator; uses PyBullet for rigid-body simulation and Blender for rendering, with Kubric-style camera/physics/scene factor contracts.
- `scripts/generate_blender_pairs.py`: small 3-clip Blender preview.
- `scripts/generate_blender_rich_pairs.py`: hand-authored richer 6-clip Blender preview.
- `scripts/generate_blender_bank_pairs.py`: balanced combinatorial Blender preview bank used before the PyBullet review bank.
- `scripts/make_run_previews.py`: creates preview contact sheets from MP4s.
- `scripts/sync_site_to_docs.sh`: copies `site/` to `docs/` for GitHub Pages while excluding raw `frames/` directories.
- `site/`: local/generated working site. It may contain heavy `frames/` folders.
- `docs/`: GitHub Pages deployment folder. Pages is configured to serve `main /docs`.

## Environment After Restart

The Blender binary used here is:

```bash
/workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender
```

On a fresh container, install the small system libraries Blender needs:

```bash
apt-get update
apt-get install -y libsm6 libice6 libxext6 libegl1 libgl1 libxi6 libxrender1 libxkbcommon0
```

Verify Blender can start:

```bash
/workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender --background --version
```

Expected version is Blender `3.6.5`. If `ldd` reports missing libraries, check with:

```bash
ldd /workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender | grep 'not found'
```

The Python environments live under `/workspace/writeable/environments/`. The review-bank environment used here is `/workspace/writeable/environments/kubric_review`; it contains `pybullet`, `opencv-python-headless`, `pillow`, `imageio`, and `numpy`. The WHAC environment is `/workspace/writeable/environments/whac/.venv`. Blender-only scripts are run by Blender's bundled Python.

## Generate The Kubric Review Bank

`kubric_review_v0` is the current human-review bank. It is intentionally small enough for GitHub Pages and manual inspection:

- 42 clips from 6 camera trajectories x 7 physics programs;
- 72 balanced pair groups: 36 same-camera/different-physics and 36 same-physics/same-scene/different-camera;
- one diagnostic ambiguity group for dolly-in vs object-moving-toward-camera;
- PyBullet simulates rigid bodies at 240 Hz, then Blender renders the cached trajectories.

The official PyPI `kubric` package was not used in this container because its dependency resolver failed on the current Python stack. This review bank still uses the Kubric-style split we need for data design: explicit `camera_id`, `physics_id`, and `scene_id`, real PyBullet contacts, and Blender-rendered videos.

Create or repair the lightweight Python environment:

```bash
python3 -m venv /workspace/writeable/environments/kubric_review
/workspace/writeable/environments/kubric_review/bin/python -m pip install --upgrade pip setuptools wheel
/workspace/writeable/environments/kubric_review/bin/python -m pip install \
  pybullet imageio imageio-ffmpeg opencv-python-headless pillow numpy
```

Generate the review bank and deployable previews:

```bash
cd /workspace/writeable/code/camera_motion_disentangle

/workspace/writeable/environments/kubric_review/bin/python \
  scripts/generate_kubric_review_bank.py \
  --run-id kubric_review_v1 \
  --camera-limit 6 \
  --physics-limit 9 \
  --pairs 90 \
  --overwrite

/workspace/writeable/environments/kubric_review/bin/python \
  scripts/make_run_previews.py \
  --run-dir site/assets/runs/kubric_review_v1 \
  --samples 8 \
  --thumb-width 180

bash scripts/sync_site_to_docs.sh
```

The generated run is small for review: `site/assets/runs/kubric_review_v1` and `docs/assets/runs/kubric_review_v1` are about 12M each after previews. The `docs` sync excludes raw `frames/` directories.

## Generate A Corrected Blender Bank

From this repository:

```bash
cd /workspace/writeable/code/camera_motion_disentangle

/workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender \
  --background \
  --python scripts/generate_blender_bank_pairs.py \
  -- --run-id blender_bank_v3 --clips 72 --pairs 180

python3 scripts/make_run_previews.py --run-dir site/assets/runs/blender_bank_v3
bash scripts/sync_site_to_docs.sh
```

The generator writes MP4s, per-clip metadata, full render frames, and a manifest under `site/assets/runs/<run-id>/`. The sync step copies deployable assets into `docs/` while excluding `frames/`, because GitHub Pages should not serve raw render frames.

## GitHub Pages Deployment

GitHub Pages should be configured as:

```text
Deploy from branch: main
Folder: /docs
```

After generating and syncing:

```bash
git status --short
git add README.md scripts site docs .gitignore
git commit -m "Add PyBullet review bank for camera motion disentanglement"
git push origin main
```

If push fails with HTTPS credential errors inside the container, push from an environment with GitHub credentials configured.

## Current Design Notes

The current review-bank generator addresses the earlier failure modes:

- background geometry is no longer randomly placed in the main object-motion region;
- same-physics pairs keep both `physics_id` and `scene_id` fixed;
- background diversity comes mainly from wall/floor colors, wall height, panels/stripes, floor marks, and occasional edge/side objects;
- camera specs include lateral/height/target offsets, nonzero roll, and top-down starts, so views are not always centered and level.
- camera trajectories are sampled from per-axis Gaussian motion profiles; same-camera pairs still share the exact sampled `camera_id` trajectory.
- dynamic roll and angle changes are mostly slow, small-amplitude camera motions for human review comfort; abrupt large rotations are rare hard cases, not the default.
- `kubric_review_v1` uses PyBullet rigid-body trajectories for contacts and collisions; the older Blender banks remain useful only as visual/contract previews.
