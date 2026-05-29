# camera_motion_disentangle

Small workspace for generating and reviewing paired synthetic videos for camera-motion / physical-motion disentanglement.

The paired design has two axes:

- same camera trajectory + different physical trajectories
- different camera trajectories + same physical trajectory

Current contents:

- `scripts/generate_mock_pairs.py`: dependency-light preview generator. It creates MP4s and metadata without Kubric, so the pairing protocol and review site can be tested immediately.
- `scripts/generate_kubric_pairs.py`: Kubric-oriented scaffold for the real generator.
- `site/`: static result browser that can later be deployed with GitHub Pages.

## Quick local preview

```bash
cd /workspace/writeable/code/camera_motion_disentangle
python3 scripts/generate_mock_pairs.py
python3 -m http.server 8008
```

Open `http://localhost:8008/site/`.

## GitHub Pages idea

This project can become a lightweight experiment workstation:

1. Commit `site/` plus selected generated runs under `site/assets/runs/`.
2. Push to a GitHub repo named `camera_motion_disentangle`.
3. Enable GitHub Pages from the repo settings.

For larger MP4s, use Git LFS or upload only compressed previews to Pages and keep full metadata/results elsewhere.
