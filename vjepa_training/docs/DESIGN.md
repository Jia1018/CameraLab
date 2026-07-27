# Camera and Physical Motion Disentanglement: Design Notes

Last updated: 2026-07-23

## Research objective

Learn a finer-grained representation on top of a frozen pretrained video
encoder. The representation should separate:

- content tokens: scene content and physical-world motion;
- camera tokens: camera viewpoint and camera motion.

The intended downstream evaluation covers content understanding, object-motion
understanding, camera-motion understanding, and embodied spatial reasoning.

## Current representation contract

The first backbone is V-JEPA 2.1 ViT-B/384. For a 16-frame 384x384 video it
returns a frozen feature grid with shape `[8, 576, 768]`:

- 8 temporal tokens from tubelet size 2;
- 24x24 spatial patch tokens from patch size 16;
- 768 feature dimensions.

The disentangler is attached after the frozen backbone. It does not modify or
insert tokens into V-JEPA itself. It produces:

- content: `[8, 576, 384]`;
- camera: `[8, 384]`.

Each tubelet receives a learnable camera query. The first query is distinct,
later queries share a base embedding, and all queries receive learned temporal
positions. A block performs local camera-patch attention, temporal attention
over camera tokens, and temporal attention over each spatial patch track.

## CamXTime factor grid

A raw CamXTime MP4 fixes a camera position and varies physical time. A moving
clip is synthesized as:

```text
video[t] = frame time_path[t] from MP4 camera_path[t]
```

Each training item contains an exact 2x2 grid:

```text
                     physics path P0   physics path P1
camera path C0             x00               x01
camera path C1             x10               x11
```

Rows share camera motion and columns share physical-time evolution. The four
clips come from one scene, so crossed feature reconstruction has a real target.
In CamXTime, different physics paths are different temporal samplings of one
scene simulation, not independent physical rollouts.

## Main self-supervised objective

The main method will not use numeric camera poses during training. It uses the
known factor pairing but no semantic camera labels, so it is more precisely
pair-structured self-supervision rather than fully unsupervised learning.

For target `(camera=c, physics=p)`, content can come from either camera
observation of `p`, and camera can come from either physics observation along
`c`:

```text
R(content[x[c_source,p]], camera[x[c,p_source]]) ~= VJEPA(x[c,p])
```

There are four source combinations for each of four targets, giving 16 exact
crossed reconstructions per factor grid.

The initial self-supervised loss is:

```text
L_ssl = 1.00 L_reconstruction
      + 0.10 L_cosine
      + 0.05 L_camera_invariance
      + 0.02 L_content_invariance
```

- reconstruction: MSE to frozen V-JEPA patch tokens;
- cosine: cosine distance to frozen V-JEPA patch tokens;
- camera invariance: same camera path should yield the same camera tokens for
  both physics paths;
- content invariance: spatially pooled content should agree across the two
  camera observations of one physics path.

Crossed reconstruction prevents complete branch collapse: if all camera tokens
were identical, the same content input would be required to reconstruct two
different camera targets.

## Pose supervision and ablations

Adjacent relative pose follows the cineGEN idea:

```text
delta_T[t] = inverse(c2w[t-1]) @ c2w[t]
```

It is encoded as rotation-6D plus translation-3D. This target will be excluded
from the main training objective and retained for evaluation and ablation.

Planned ablations:

| ID | Objective | Purpose |
| --- | --- | --- |
| A0 | crossed reconstruction + cosine | minimal factorization baseline |
| A1 | A0 + camera/content invariance | primary self-supervised method |
| A2 | A1 + independence or variance regularization | reduce factor leakage |
| A3 | A1 + relative-pose supervision | supervised ablation / upper bound |

For pose-free runs, pose metadata must not affect gradients. A clean
implementation should make the camera head and camera-target loading optional,
while retaining pose metadata for held-out evaluation.

## Evaluation plan

After training, freeze the backbone and disentangler. Evaluate camera tokens
with linear or small-MLP probes for relative rotation, translation direction,
translation magnitude, trajectory retrieval, and motion classification.
Evaluate content tokens on content and object-motion tasks. Evaluate all tokens
together on embodied spatial reasoning tasks.

## Open risks

- The latent factorization is identifiable only up to an arbitrary nonlinear
  coordinate system without pose supervision.
- Camera tokens may encode a scene-specific view sequence rather than a
  cross-scene camera coordinate system.
- Content invariance currently constrains only spatially pooled features, so
  patch-level camera information may remain.
- Reconstruction needs the initial viewpoint, while adjacent relative motion
  alone does not specify it. Camera tokens may reasonably encode both initial
  viewpoint and relative trajectory.
- Intrinsics are loaded but not modeled; this is acceptable only while they are
  constant or nearly constant in the pilot data.
- A held-out scene split and explicit leakage probes are required before a long
  training run can support scientific conclusions.

## Decision log

- 2026-07-23: Select frozen V-JEPA 2.1 as the first backbone.
- 2026-07-23: Attach VGGT-style camera queries after frozen backbone features.
- 2026-07-23: Use CamXTime to construct exact 2x2 camera-by-physics grids.
- 2026-07-23: Use adjacent relative camera motion for evaluation and optional
  supervision, inspired by cineGEN.
- 2026-07-23: Make pose-free pair-structured self-supervision the primary
  method; move pose supervision to ablation A3.

