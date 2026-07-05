# camera_motion_disentangle

Workspace for generating and reviewing paired synthetic videos for camera-motion / physical-motion disentanglement.

The paired data contract has two controlled axes:

- same camera trajectory + different physics/scene
- different camera trajectory + same physics and same scene

In the current bank generator, `scene_id` is explicit. A "same physics" pair means `physics_id` and `scene_id` are both fixed, so object motion, static scene, wall/floor colors, wall structure, and decorative background elements are identical. Only the camera changes.

## Repository Layout

- `scripts/generate_mock_pairs.py`: 2D dependency-light preview generator.
- `scripts/generate_kubric_pairs.py`: older Kubric-oriented contract scaffold.
- `scripts/generate_kubric_review_bank.py`: previous review-bank generator; uses direct PyBullet for rigid-body simulation and Blender for rendering, with Kubric-style camera/physics/scene factor contracts.
- `scripts/generate_official_kubric_review_bank.py`: current official-Kubric review-bank generator; uses PyPI `kubric` scene objects, Kubric PyBullet, and Kubric Blender renderer.
- `scripts/kubric_official_smoke.py`: smoke tests for the official PyPI `kubric` package, covering Kubric PyBullet and Blender renderer.
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
apt-get install -y libsm6 libice6 libxext6 libegl1 libegl-mesa0 libgl1 libxi6 libxrender1 libxkbcommon0
```

Verify Blender can start:

```bash
/workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender --background --version
```

Expected version is Blender `3.6.5`. If `ldd` reports missing libraries, check with:

```bash
ldd /workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender | grep 'not found'
```

The Python environments live under `/workspace/writeable/environments/`. The review-bank environment used here is `/workspace/writeable/environments/kubric_review`; it contains `pybullet`, `opencv-python-headless`, `pillow`, `imageio`, and `numpy`. The official Kubric test environment is `/workspace/writeable/environments/kubric_official`. The WHAC environment is `/workspace/writeable/environments/whac/.venv`. Blender-only scripts are run by Blender's bundled Python.

## Official Kubric Smoke Tests

The current review-bank generator is still Kubric-style PyBullet plus Blender, but the official PyPI `kubric` package now runs in this container with a few compatibility constraints:

- use `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, because Kubric `0.1.1` needs older TFDS/protobuf behavior;
- expose the official venv packages to Blender with `PYTHONPATH=/workspace/writeable/environments/kubric_official/lib/python3.10/site-packages`;
- Kubric `0.1.1` expects Blender's old `NLM` denoiser enum, so `scripts/kubric_official_smoke.py` applies a small Blender 3.6 compatibility patch for `OPENIMAGEDENOISE`.

Create or repair the official Kubric environment:

```bash
python3 -m venv /workspace/writeable/environments/kubric_official
/workspace/writeable/environments/kubric_official/bin/python -m pip install --upgrade pip setuptools wheel
/workspace/writeable/environments/kubric_official/bin/python -m pip install --no-deps kubric
/workspace/writeable/environments/kubric_official/bin/python -m pip install \
  traitlets numpy pyquaternion trimesh imageio pypng pandas munch bidict \
  singledispatchmethod pybullet scipy scikit-learn tensorflow==2.20.0 \
  tensorflow-datasets==4.4.0 OpenEXR Imath
```

Run the official Kubric physics smoke test:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/workspace/writeable/environments/kubric_official/bin/python \
  scripts/kubric_official_smoke.py \
  --mode physics \
  --out-dir /tmp/kubric_official_smoke
```

Run the official Kubric Blender renderer smoke test:

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
PYTHONPATH=/workspace/writeable/environments/kubric_official/lib/python3.10/site-packages \
/workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender \
  --background \
  --python scripts/kubric_official_smoke.py \
  -- \
  --mode render \
  --out-dir /tmp/kubric_official_smoke
```

The render smoke test should write `/tmp/kubric_official_smoke/frame_0001.png`.

## Generate The Official Kubric Review Bank

`kubric_review_v4_official` is the current official-Kubric human-review bank:

- 15 clips from 3 camera trajectories x 5 physics programs;
- 20 pair groups across same-camera/different-physics and same-physics/same-scene/different-camera comparisons;
- 96 frames at 24 fps, 640x480 review resolution;
- official `kubric.simulator.PyBullet` runs the rigid-body simulation;
- official `kubric.renderer.Blender` renders denoised Cycles PNG frames with 16 samples, then system `ffmpeg` encodes MP4 for GitHub Pages;
- each physics program audits expected contacts before rendering, using Kubric collision logs plus visible geometry contact checks.

Generate the official bank, previews, and deployable docs copy:

```bash
cd /workspace/writeable/code/camera_motion_disentangle

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/workspace/writeable/environments/kubric_official/bin/python \
  scripts/generate_official_kubric_review_bank.py \
  --run-id kubric_review_v4_official \
  --camera-limit 3 \
  --physics-limit 5 \
  --pairs 20 \
  --frames 96 \
  --width 640 \
  --height 480 \
  --overwrite

/workspace/writeable/environments/kubric_review/bin/python \
  scripts/make_run_previews.py \
  --run-dir site/assets/runs/kubric_review_v4_official \
  --samples 8 \
  --thumb-width 220

bash scripts/sync_site_to_docs.sh
```

Rendering may create temporary `site/.../frames/` directories; `scripts/sync_site_to_docs.sh` excludes them from `docs/`, and they can be deleted after previews/MP4s are generated to save disk space.

`kubric_review_v2_official` was the first official-Kubric smoke-sized web run. It used 30 frames at 160x120 and looked blocky when upscaled. `kubric_review_v3_official` improved length/resolution but still used very low Cycles sampling, which produced visible speckle/noise. Use v4 for visual review.

After v4, `scripts/generate_official_kubric_review_bank.py` is prepared for the next diversity pass: visible object colors are sampled in HSV space, visual material profiles are sampled from matte/satin/glossy/rubber/brushed-metal templates, and the sampled appearance plus physical parameters are written into per-clip metadata. Use a new run id such as `kubric_review_v5_official` when generating that diversity pass; `--physics-limit 6` includes the three-body scatter program.

## Generate The Kubric-Style Review Bank

`kubric_review_v1` is the previous Kubric-style human-review bank. It is intentionally small enough for GitHub Pages and manual inspection:

- 54 clips from 6 camera trajectories x 9 physics programs;
- 90 balanced pair groups across same-camera/different-physics and same-physics/same-scene/different-camera comparisons;
- one diagnostic ambiguity group for dolly-in vs object-moving-toward-camera;
- PyBullet simulates rigid bodies at 240 Hz, then Blender renders the cached trajectories.

The official PyPI `kubric` package was not used for `kubric_review_v1`; that run uses a local Kubric-style split with explicit `camera_id`, `physics_id`, and `scene_id`, real PyBullet contacts, and Blender-rendered videos. Official Kubric is now available through `scripts/generate_official_kubric_review_bank.py`; use `kubric_review_v4_official` for the current review pass.

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
- only part of the current PyBullet review-bank parameter space is Gaussian-sampled. Camera velocities and several object speed scalars use clipped Gaussians, but object size, mass, friction, restitution, many start positions, and some velocity components are still fixed hand-tuned constants. The next official-Kubric iteration should move those physical/material parameters to clipped Gaussian or mixture distributions with collision-quality filters.
