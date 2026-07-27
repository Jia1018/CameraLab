# Second-Wave B1 Camera-Conditioned Pilot

Date: 2026-07-24
Status: completed; partial success, decision criteria not met

Launch:

- tmux session: `vjepa_second_wave_b1`;
- start: 2026-07-24 15:07:58 UTC;
- device: physical GPU 1 through `CUDA_VISIBLE_DEVICES=1`;
- training completed at step 500;
- the repaired 16-scene camera intervention test completed on the same GPU.

## Motivation

First-wave A1 and A0 reconstructors ignored camera tokens under wrong-camera,
zero-camera, and temporal-reversal interventions. A1 also compressed the
camera-token scale. B1 tests whether stronger crossed objectives plus an
explicit camera-conditioned decoder remove both failure modes.

## Changes From A1

- Spatial-FiLM at every reconstruction block:
  - normalize content channels over the spatial patch grid per tubelet;
  - derive channel scale and shift from the corresponding camera token;
  - apply the modulation to every patch before temporal and local attention.
- Reconstruction category weights:
  - identity: 0.25;
  - content swap: 2.0;
  - same-camera cross-physics swap: 0.25;
  - double swap: 2.0.
- Factor-ranking losses with margin 0.05:
  - camera paths should be farther apart than physics observations sharing a
    camera path;
  - physics paths should be farther apart than camera observations sharing
    physical motion.
- Camera standard-deviation hinge with target 0.1.
- Pose supervision remains disabled.

Shared V-JEPA, CamXTime, seed, optimizer, and 500-step schedule are unchanged.

## Validation

- unit tests: 6 passed;
- Python compile: passed;
- shell syntax: passed;
- two real GPU optimization steps and checkpoint serialization: passed;
- temporary smoke checkpoint removed before formal launch.

Formal observations:

~~~text
step 1:  loss=1.77349  reconstruction=1.64847  cosine=1.01482
step 10: loss=1.01640  reconstruction=0.94450  cosine=0.49665
step 20: loss=0.74035  reconstruction=0.69097  cosine=0.27091
step 500: loss=0.46419 reconstruction=0.43487 cosine=0.19142
          camera_path_separation=0.75011 camera_feature_variance=0.35990
~~~

## Decision Criteria

B1 is successful only if the post-training intervention test shows all of:

- wrong-camera output moves materially relative to the natural target
  camera-path distance;
- wrong-camera output becomes closer to the crossed-camera target than the
  original-camera target;
- zeroing or reversing camera tokens changes reconstruction measurably;
- camera-path separation and feature scale remain non-collapsed;
- crossed reconstruction improves without numerical instability.

Output:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b1_spatial_film_cross_rank_seed17
~~~

## Results

The deterministic 16-scene diagnostic used the same scene indices and seed as
A0/A1.

~~~text
                                      B1             A1
identity to original              0.411380       0.323888
target camera-path distance       0.690751       0.690699
wrong-camera output delta         0.001580       0.00000162
wrong-camera normalized delta     0.002160       0.00000254
wrong-camera crossed gain         0.002126       0.00000021
wrong-camera target preference   -0.025909      -0.125111
zero-camera output delta          0.013255       0.00000243
reversed-camera output delta      0.001013       0.00000162
camera-token wrong-path delta     0.235601       0.000733
~~~

B1 increases wrong-camera output sensitivity by about 975x over A1, prevents
camera-token scale collapse, and moves the intervention in the correct crossed
target direction. It nevertheless explains only about 0.22% of the true
camera-path feature difference. The wrong-camera output remains closer to the
original target, while identity reconstruction is worse than A1. B1 therefore
proves that direct conditioning can remove decoder bypass, but does not yet
achieve useful factor transfer.

The first automatic evaluation deadlocked after scene 4 because PyAV decoded
videos in the CUDA-owning main process. Scene 5 decoded normally in a CPU-only
process. Evaluation now uses an ordered `Subset` and spawned DataLoader workers,
matching the training loader's CUDA isolation. The regression suite passes
7 tests, and the repaired evaluation completed all 16 scenes in under two
minutes.

Artifact:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b1_spatial_film_cross_rank_seed17/camera_interventions_seed10017_n16.json
~~~

Next action: add symmetric camera and content intervention-delta reconstruction
losses. These losses use only the exact 2x2 feature targets and directly align
the direction and magnitude induced by changing one factor.
