# CamXTime + V-JEPA 2.1 Pilot

## Raw-grid contract

`camXXX_full_motion.mp4` fixes camera position `XXX`; its 120 video frames
vary physical time. A moving-camera clip is assembled as:

```text
output[t] = decode(video=camera_index[t], frame=time_index[t])
```

`CamXTimeFactorGridDataset` returns an exact 2x2 factor grid:

```text
                         physical path A     physical path B
camera trajectory A          video[0,0]          video[0,1]
camera trajectory B          video[1,0]          video[1,1]
```

Rows share camera motion; columns share the scene's physical-time path. The
loader accepts both the temporary `CamXTime/Scene*` download layout and the
final `CamXTime/train/Scene*` layout. It ignores incomplete trajectories.

## Representation model

The frozen V-JEPA output is reshaped from `[B, T'*H'*W', D]` to
`[B, T', H'*W', D]`. `FeatureDisentangler` then:

1. projects the frozen tokens into the adapter dimension;
2. adds one learnable camera query per tubelet;
3. uses a distinct first query and a shared later query, following VGGT;
4. alternates local camera-patch attention, camera temporal attention, and
   temporal attention along each spatial patch index;
5. returns transformed content patch tokens and camera tokens.

`FeatureReconstructor` maps any content/camera combination back to the frozen
V-JEPA token space. For each target `(camera=c, physics=p)`, content may be
taken from either observation of physics `p`, and camera may be taken from
either observation along camera path `c`. This gives 16 supervised
recombinations per 2x2 item.

Camera supervision uses adjacent tubelet transforms (`inv(c2w[t-1]) @ c2w[t]`)
encoded as rotation-6D plus translation. Blender poses use the SpaceTimePilot
Blender-to-OpenCV axis conversion.

## Commands

Install the package in the intended training environment:

```bash
cd /workspace/writeable/code/camera_motion_disentangle
python3 -m pip install -e '.[test]'
```

Validate a full-size real sample without loading V-JEPA:

```bash
PYTHONPATH=. python3 scripts/train_vjepa21_camxtime.py \
  --config configs/vjepa21_camxtime_pilot.yaml \
  --check-data-only
```

Run tests:

```bash
PYTHONPATH=. python3 -m pytest -q
```

For training, set `backbone.checkpoint_path` in
`configs/vjepa21_camxtime_pilot.yaml` to a local official checkpoint, then run:

```bash
PYTHONPATH=. python3 scripts/train_vjepa21_camxtime.py \
  --config configs/vjepa21_camxtime_pilot.yaml
```

The initial config uses the distilled V-JEPA 2.1 ViT-B/384 encoder. The local
wrapper also supports official large, giant, and gigantic V-JEPA 2.1 encoder
checkpoints. It does not call the repository hub download because this checkout
currently points its hub base URL to a localhost test server.
