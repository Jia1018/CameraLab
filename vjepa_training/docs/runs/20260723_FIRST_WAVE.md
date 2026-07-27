# First-Wave Pose-Free Pilot

Date: 2026-07-23
Status: completed

Launch:

- tmux session: `vjepa_first_wave`;
- A1 start: 2026-07-23 23:08:42 UTC;
- A1 completed: 2026-07-23 23:23 UTC;
- A0 completed: 2026-07-23 23:35 UTC.

## Purpose

Test whether crossed feature reconstruction learns stable camera/content
factors without numeric camera-pose supervision. This is an optimization pilot,
not a downstream or held-out generalization result.

## Shared setup

- frozen backbone: V-JEPA 2.1 ViT-B/384;
- input: 16 frames at 384x384;
- token grid: `[8, 576, 768]`;
- disentangler/reconstructor: 2 blocks, model dimension 384;
- CamXTime scenes: all 417 training scenes;
- seed: 17;
- batch size: 1;
- optimizer: AdamW, learning rate 1e-4, weight decay 0.05;
- precision: bfloat16;
- steps: 500 per run;
- checkpoint interval: 100 steps;
- pose loss: disabled (`camera_supervision=0.0`).

Pose metadata may remain present in the data batch for later evaluation, but it
is not an input to the model. With zero pose weight, pose loss computation is
skipped and the camera prediction head receives no gradient.

## Runs

### A1 - swap reconstruction plus invariance

Status: completed (500/500)

Config: `configs/vjepa21_camxtime_a1_500.yaml`

```text
L = 1.00 reconstruction
  + 0.10 cosine
  + 0.05 camera invariance
  + 0.02 content invariance
```

Output:

```text
/workspace/writeable/checkpoints/camera_motion_disentangle/first_wave/a1_swap_invariance_seed17
```

### A0 - swap reconstruction only

Status: completed (500/500)

Config: `configs/vjepa21_camxtime_a0_500.yaml`

```text
L = 1.00 reconstruction + 0.10 cosine
```

Output:

```text
/workspace/writeable/checkpoints/camera_motion_disentangle/first_wave/a0_swap_only_seed17
```

## Pose-free GPU smoke result

One real A1 step completed before launch:

```text
loss:                         1.74911
reconstruction:               1.64649
cosine:                       1.01902
camera supervision:           0.00000
camera invariance:            0.00047
content invariance:           0.03444
camera path separation:       0.03465
content physics separation:   0.00047
camera feature variance:      0.01176
content feature variance:     0.01143
reconstruction identity:      1.64700
reconstruction content swap:  1.64599
reconstruction camera swap:   1.64700
reconstruction double swap:   1.64599
```

The near-equal reconstruction categories are expected at random
initialization. The experiment should reveal whether crossed categories improve
without either factor collapsing or leaking excessively.

## Launch validation

The first launch at 22:54:46 UTC exposed a DataLoader deadlock: workers were
forked after CUDA and PyTorch thread pools had been initialized. A standalone
real sample decoded successfully, isolating the issue from the dataset. The
training loader now uses the `spawn` multiprocessing context whenever workers
are enabled.

Validation after the fix:

- unit tests: 5 passed;
- two real GPU steps completed, including the step-2 checkpoint;
- temporary smoke checkpoint removed before the formal run;
- formal A1 passed step 100 without NaN or OOM;
- the 251 MB `step_0000100.pt` checkpoint was written successfully.

First formal observations:

```text
step 1:  loss=1.74911  reconstruction=1.64649  cosine=1.01902
step 10: loss=0.87465  reconstruction=0.83496  cosine=0.39566
step 100: loss=0.51405  reconstruction=0.49507  cosine=0.18807
```

At step 100 the identity and camera-swap reconstruction MSE are both 0.48703;
content-swap and double-swap are 0.50310. Per-sample factor separation and
variance diagnostics fluctuate substantially, so aggregate evaluation is
required before interpreting disentanglement.

## Completed results

Both runs completed all 500 steps and wrote checkpoints at steps 100, 200,
300, 400, and 500. The table reports means over the ten logged points from
steps 410 through 500.

| Metric | A1 | A0 |
| --- | ---: | ---: |
| total loss | 0.421775 | 0.418532 |
| reconstruction | 0.402571 | 0.401248 |
| identity reconstruction | 0.338920 | 0.336944 |
| content-swap reconstruction | 0.466220 | 0.465551 |
| camera-swap reconstruction | 0.338919 | 0.336944 |
| camera invariance | 0.000014 | 0.000387 |
| camera-path separation | 0.001478 | 0.053060 |
| camera feature variance | 0.000763 | 0.020123 |
| content invariance | 0.093286 | 0.125991 |
| content-physics separation | 0.000641 | 0.000785 |
| content feature variance | 0.056522 | 0.065474 |

Interpretation:

- A1 substantially reduces camera variation across physics observations, but
  it also reduces absolute camera variance by about 26x and camera-path
  separation by about 36x relative to A0. Without a positive variance or
  information constraint, this is a scale-collapse warning.
- Content-swap reconstruction remains about 38% worse than identity
  reconstruction in A1. The content path is therefore not sufficiently
  invariant to camera motion after 500 steps.
- A1 and A0 have nearly identical reconstruction quality. The current weak
  invariance terms do not yet produce a clear optimization benefit.

`camera_swap` here means swapping camera tokens between physics observations
that share the same camera path. Equality with identity reconstruction is
desirable, but does not establish that the decoder uses camera tokens. A
wrong-camera, zero-camera, and token-permutation sensitivity evaluation is
needed before diagnosing decoder bypass.

## Camera intervention evaluation

The final A1 and A0 checkpoints were evaluated on 16 deterministic samples
uniformly spread across the 417 training scenes. The evaluation seed was 10017,
so the sampled camera/time paths differ from those used during training. This
is a mechanism diagnostic, not a held-out generalization result.

For fixed content `C[c,p]`, the test compares:

- the correct camera `K[c,p]`;
- the same camera path observed with other physics `K[c,1-p]`;
- the other camera path `K[1-c,p]`;
- a zero camera tensor;
- the correct camera tokens in reversed temporal order.

Key means over 64 compositions:

| Metric | A1 | A0 |
| --- | ---: | ---: |
| identity to original target | 0.323887838 | 0.321402159 |
| wrong camera to original target | 0.323887943 | 0.321403869 |
| wrong camera to crossed target | 0.448998719 | 0.447809862 |
| target camera-path distance | 0.690698745 | 0.690698745 |
| wrong-camera output delta | 0.000001620 | 0.000003011 |
| normalized wrong-camera delta | 0.000002541 | 0.000004480 |
| zero-camera output delta | 0.000002427 | 0.000006506 |
| reversed-camera output delta | 0.000001621 | 0.000001908 |
| camera-token wrong-path delta | 0.000732935 | 0.085775432 |

The intervention resolves the ambiguity in the training diagnostics:

- both reconstructors effectively ignore the camera input; changing, zeroing,
  or temporally reversing it changes the output only at the `1e-6` scale;
- the wrong-camera output remains closer to the original target than the
  crossed target, rather than following the substituted camera path;
- A0 camera tokens distinguish paths, but its decoder still bypasses them;
- A1 additionally compresses the camera-token scale by more than two orders of
  magnitude relative to A0.

Result JSON:

```text
/workspace/writeable/checkpoints/camera_motion_disentangle/first_wave/camera_interventions_seed10017_n16.json
```

The next model revision must both prevent camera-token scale collapse and force
the decoder to depend on the camera branch. Variance regularization alone is
insufficient because A0 already has path-varying camera tokens that the decoder
does not use.

## Diagnostics

Track:

- identity, content-swap, camera-swap, and double-swap reconstruction MSE;
- camera invariance versus camera-path separation;
- content invariance versus content-physics separation;
- camera/content feature variance;
- finite gradients and checkpoint creation.

The next stage must add a held-out scene split and frozen linear probes before
making representation-quality claims.
