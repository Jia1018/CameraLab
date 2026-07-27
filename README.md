# Camera Motion Disentanglement

This repository contains two related but independently runnable tasks:

- [`video_pair_generation/`](video_pair_generation/): Blender/Kubric paired-video
  generation, review, and batch planning.
- [`vjepa_training/`](vjepa_training/): CamXTime loading and camera/physical-motion
  disentanglement over frozen V-JEPA 2.1 features.

Generated review assets remain at the repository root:

- `site/`: local review site and generated assets.
- `docs/`: GitHub Pages deployment directory.

Large datasets, model checkpoints, and Python environments live outside this
Git repository under `/workspace/writeable`.
