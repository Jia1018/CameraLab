# camera_motion_disentangle

Workspace for generating and reviewing paired synthetic videos for camera-motion / physical-motion disentanglement.

The paired data contract has two controlled axes:

- same camera trajectory + different physics/scene
- different camera trajectory + same physics and same scene

In the current bank generator, `scene_id` is explicit. A "same physics" pair means `physics_id` and `scene_id` are both fixed, so object motion, static scene, wall/floor colors, wall structure, and decorative background elements are identical. Only the camera changes.

## Repository Layout

- `scripts/generate_mock_pairs.py`: 2D dependency-light preview generator.
- `scripts/generate_kubric_pairs.py`: Kubric-oriented contract scaffold; real Kubric rendering is still a future step.
- `scripts/generate_blender_pairs.py`: small 3-clip Blender preview.
- `scripts/generate_blender_rich_pairs.py`: hand-authored richer 6-clip Blender preview.
- `scripts/generate_blender_bank_pairs.py`: combinatorial Blender bank used for the current review examples.
- `scripts/make_run_previews.py`: creates preview contact sheets from MP4s.
- `scripts/sync_site_to_docs.sh`: copies `site/` to `docs/` and removes frame directories for GitHub Pages.
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
apt-get install -y libsm6 libice6
```

Verify Blender can start:

```bash
/workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender --background --version
```

Expected version is Blender `3.6.5`. If `ldd` reports missing libraries, check with:

```bash
ldd /workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender | grep 'not found'
```

The Python environments live under `/workspace/writeable/environments/` if other project dependencies are needed. The WHAC environment is `/workspace/writeable/environments/whac/.venv`, but the Blender bank script itself is run by Blender's bundled Python.

## Generate A Corrected Blender Bank

From this repository:

```bash
cd /workspace/writeable/code/camera_motion_disentangle

/workspace/writeable/code/WHAC/blender-3.6.5-linux-x64/blender \
  --background \
  --python scripts/generate_blender_bank_pairs.py \
  -- --run-id blender_bank_v1 --clips 40 --pairs 120

python3 scripts/make_run_previews.py --run-dir site/assets/runs/blender_bank_v1
bash scripts/sync_site_to_docs.sh
```

The generator writes MP4s, per-clip metadata, full render frames, and a manifest under `site/assets/runs/<run-id>/`. The sync step copies deployable assets into `docs/` and removes `frames/`, because GitHub Pages should not serve the raw render frames.

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
git commit -m "Update corrected Blender bank examples"
git push origin main
```

If push fails with HTTPS credential errors inside the container, push from an environment with GitHub credentials configured.

## Current Design Notes

The current bank generator avoids the earlier failure modes:

- background geometry is no longer randomly placed in the main object-motion region;
- same-physics pairs keep both `physics_id` and `scene_id` fixed;
- background diversity comes mainly from wall/floor colors, wall height, panels/stripes, floor marks, and occasional edge/side objects;
- camera specs include lateral/height/target offsets and nonzero roll, so starts are not always centered and level.
