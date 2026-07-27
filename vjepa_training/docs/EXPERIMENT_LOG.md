# Experiment Log

Last updated: 2026-07-24

## Recording convention

Every run should record its experiment ID, hypothesis, code/config state, data
split, command, checkpoint path, metrics, interpretation, and next action.
Engineering smoke tests are explicitly separated from scientific experiments.

## Available assets

- Train data: `/workspace/writeable/datasets/CamXTime/train`
- Raw evaluation data: `/workspace/writeable/datasets/CamXTime/eval`
- V-JEPA checkpoint:
  `/workspace/writeable/checkpoints/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt`
- Python environment:
  `/workspace/writeable/environments/camera_motion_disentangle`

## D000 - CamXTime integrity audit

Date: 2026-07-23

Purpose: verify that the apparent size difference from the download guide does
not indicate missing data.

Results:

```text
complete train scenes:       417
complete camera trajectories: 1668
training videos:          200160
raw evaluation videos:      3840
train logical size:  42,457,501,413 bytes
eval logical size:    1,058,484,166 bytes
```

AWS `s3 sync --dryrun` was run separately for training videos, camera JSONs,
and raw evaluation videos. All three exited with code 0 and produced no pending
copy operations. The local data matches the current S3 release. The guide's
54/56 GB figure is an estimate or refers to an earlier encoding/release.

## T000 - Unit tests

Date: 2026-07-23

Command:

```bash
/workspace/writeable/environments/camera_motion_disentangle/bin/python -m pytest -q
```

Result after the pose-free diagnostics and DataLoader fix:
`5 passed in 19.72s`.

Coverage includes camera geometry, factor-grid coordinate sharing, model
shapes, all 16 crossed reconstructions, and gradient propagation.

## S000 - Real-data loader smoke test

Date: 2026-07-23

Purpose: discover all complete trajectories and decode one real full-resolution
2x2 factor grid.

Result:

```text
video:         [2, 2, 3, 16, 384, 384]
camera_c2w:    [2, 16, 4, 4]
camera_target: [2, 8, 9]
```

Status: passed.

## S001 - One-step end-to-end GPU smoke test

Date: 2026-07-23

Purpose: verify official checkpoint loading, frozen V-JEPA encoding,
disentangling, 16-way reconstruction, backpropagation, optimization, and
checkpoint serialization.

Important: this engineering smoke test used the current pose-supervised loss.
It is not a result for the pose-free main method.

Result:

```text
loss:                  1.83692
reconstruction:        1.64649
cosine:                1.01902
camera_supervision:    0.43907
camera_invariance:     0.00047
content_invariance:    0.03444
```

Temporary checkpoint:

```text
/tmp/camera_motion_disentangle_vjepa_smoke/step_0000001.pt
```

Status: passed. This validates integration only; one step does not measure
representation quality.

## S002 - Pose-free two-step loader regression test

Date: 2026-07-23

Purpose: verify that two spawned DataLoader workers can supply consecutive real
CamXTime batches after CUDA initialization. The initial formal launch used the
default fork context and deadlocked before step 1.

Result:

- two pose-free A1 optimization steps completed in about 34 seconds including
  model loading and worker startup;
- the second-step checkpoint loaded successfully;
- the temporary checkpoint was removed before the formal run;
- exit code was 0.

Status: passed. The loader now selects multiprocessing context `spawn` when
`num_workers > 0`.

## First-wave runs

| ID | Status | Description |
| --- | --- | --- |
| A0 | completed (500/500) | pose-free crossed reconstruction + cosine |
| A1 | completed (500/500) | A0 + camera/content invariance; primary method |
| A2 | pending | A1 + factor-independence or variance regularization |
| A3 | pending | A1 + adjacent relative-pose supervision |

A1 started at 2026-07-23 23:08:42 UTC and both sequential runs completed by
23:35 UTC. Each run produced five scheduled checkpoints. Detailed
configuration, late-training aggregates, and interpretation are in
`docs/runs/20260723_FIRST_WAVE.md`.

Late-training reconstruction is essentially unchanged by A1 versus A0
(0.402571 versus 0.401248). A1 reduces content invariance error modestly, but
content-swap reconstruction remains substantially worse than identity. A1 also
reduces camera feature variance by about 26x, so positive variance/information
constraints and decoder-sensitivity diagnostics are required before A2.

A0/A1 are optimization pilots over all 417 training scenes, not
representation-quality experiments. Add a held-out scene split and frozen
linear probes before making generalization claims.

## D001 - Camera-token intervention diagnostic

Date: 2026-07-24

Purpose: determine whether the final A1/A0 reconstructors actually use camera
tokens. Sixteen deterministic samples, uniformly spread across the training
scenes with a new path-sampling seed, were evaluated with correct, wrong-path,
zero, and temporally reversed camera inputs.

~~~text
                                      A1            A0
target camera-path distance       0.690699      0.690699
wrong-camera output delta         0.00000162    0.00000301
zero-camera output delta          0.00000243    0.00000651
reversed-camera output delta      0.00000162    0.00000191
camera-token wrong-path delta     0.00073294    0.08577543
~~~

The wrong-camera output remains closer to the original-camera target than the
crossed-camera target for both models. Both reconstructors therefore bypass the
camera branch. A1 additionally exhibits camera-token scale collapse; A0 has
path-varying camera tokens, but its decoder still ignores them.

Artifact:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/first_wave/camera_interventions_seed10017_n16.json
~~~

Next action: revise the architecture/objective to force camera conditioning,
then repeat this diagnostic before running downstream probes. Positive camera
variance alone cannot solve the A0 decoder-bypass failure.

## B1 - Spatial-FiLM crossed reconstruction pilot

Date: 2026-07-24

Purpose: prevent camera-token collapse and decoder bypass without pose labels.
B1 adds per-tubelet spatial-FiLM conditioning, crossed-category weighting,
factor-ranking margins, and a camera standard-deviation hinge.

Training completed 500 steps. Final camera path separation was 0.75011 and
camera feature variance was 0.35990, so the camera representation did not
collapse. The deterministic 16-scene diagnostic produced:

~~~text
identity to original              0.411380
wrong-camera output delta         0.001580
wrong-camera normalized delta     0.002160
wrong-camera crossed gain         0.002126
wrong-camera target preference   -0.025909
zero-camera output delta          0.013255
reversed-camera output delta      0.001013
camera-token wrong-path delta     0.235601
~~~

Wrong-camera sensitivity is about 975x A1 and its crossed gain is positive,
so the decoder now uses camera tokens in the correct direction. The induced
change is still only 0.22% of the true target-camera difference, the output
remains closer to the original target, and identity reconstruction regresses
from 0.323888 to 0.411380. B1 is partial progress, not successful
disentanglement.

The original evaluation deadlocked because PyAV frame threads ran in the CUDA
main process. Evaluation now uses an ordered subset with spawned DataLoader
workers; a new regression test and the full 16-scene rerun both pass.

Artifact:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b1_spatial_film_cross_rank_seed17/camera_interventions_seed10017_n16.json
~~~

Next action: directly reconstruct the camera-induced and content-induced
feature deltas from the exact 2x2 grid, still without pose supervision.

## B2 - Symmetric factor-delta reconstruction pilot

Date: 2026-07-24

Purpose: directly align camera-induced and content-induced output differences
with exact frozen V-JEPA differences in the CamXTime 2x2 grid. Pose supervision
remains disabled. All other settings are identical to B1.

Training and the automatic 16-scene diagnostic completed. Late logged means
for steps 400-500 were pointwise reconstruction 0.44220, identity
reconstruction 0.39775, camera-delta reconstruction 0.73569, content-delta
reconstruction 0.07730, and camera feature variance 0.22470.

~~~text
identity to original              0.392633
wrong-camera output delta         0.003100
wrong-camera normalized delta     0.004036
wrong-camera crossed gain         0.005096
wrong-camera target preference   -0.039917
same-path output delta            0.000034
zero-camera output delta          0.022879
camera-token wrong-path delta     0.169527
~~~

Relative to B1, wrong-camera sensitivity is 1.96x, normalized delta is 1.87x,
crossed gain is 2.40x, and identity reconstruction improves. The response is
still far too small and remains closer to the original target. B2 supports the
delta objective, but also exposes uniform spatial-FiLM as an architectural
bottleneck.

Artifact:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b2_factor_delta_seed17/camera_interventions_seed10017_n16.json
~~~

Next action: keep the B2 loss and test a low-rank, position-dependent camera
residual in the decoder.

## B3/B4 - Spatial-basis paired pilots

Date: 2026-07-24

B3 replaces uniform spatial-FiLM with a learned rank-16 position-dependent
camera basis. B4 is identical except that the camera-delta reconstruction
weight is increased from 1.0 to 4.0. Both pose-free runs trained concurrently
on separate GPUs and completed the same deterministic 16-scene diagnostic.

~~~text
                                      B2          B3          B4
identity to original              0.392633    0.416131    0.416378
wrong-camera output delta         0.003100    0.001470    0.003801
wrong-camera normalized delta     0.004036    0.001950    0.005030
wrong-camera crossed gain         0.005096    0.002355    0.005773
wrong-camera target preference   -0.039917   -0.024222   -0.019653
same-path output delta            0.000034    0.000013    0.000019
zero-camera output delta          0.022879    0.004279    0.010532
camera-token wrong-path delta     0.169527    0.296404    0.222638
~~~

B3 disproves the hypothesis that spatially varying decoder capacity alone is
enough: intervention strength falls and identity fidelity regresses. B4 is the
strongest camera intervention so far, with a 24.6% normalized-delta and 13.3%
crossed-gain improvement over B2, while the same-path control remains small.
It still changes the reconstruction by only about 0.50% of the frozen V-JEPA
camera difference and worsens identity error by 0.023745. A 4x loss weight is
therefore not a complete solution.

Artifacts:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b3_spatial_basis_delta_seed17/camera_interventions_seed10017_n16.json
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b4_spatial_basis_camdelta4_seed17/camera_interventions_seed10017_n16.json
~~~

Next action: preserve the useful coarse spatial-FiLM path while adding
position-dependent conditioning, and replace pure loss reweighting with a
stronger architectural dependency on camera tokens.

## B5/B6 - Position-dependent FiLM versus strong uniform FiLM

Date: 2026-07-24

B5 uses a new rank-16 position-dependent FiLM interaction at camera-delta
weight 4.0. B6 is the paired control: B2's uniform spatial-FiLM with the same
weight 4.0. Both pose-free runs completed concurrently on separate GPUs.

~~~text
                                      B2          B5          B6
identity to original              0.392633    0.401040    0.383979
wrong-camera output delta         0.003100    0.001744    0.014993
wrong-camera normalized delta     0.004036    0.002316    0.019031
wrong-camera crossed gain         0.005096    0.003052    0.016996
wrong-camera target preference   -0.039917   -0.039900   -0.049589
same-path output delta            0.000034    0.000007    0.000068
zero-camera output delta          0.022879    0.003267    0.028719
camera-token wrong-path delta     0.169527    0.333882    0.520345
~~~

Position-dependent low-rank FiLM underperforms. Strong uniform FiLM is the
best result so far: B6 improves normalized delta by 4.72x and crossed gain by
3.34x over B2 while also improving identity fidelity. Its wrong-path output
response is 222x its same-path control. The output change remains only 1.90% of
the true camera difference and is not yet closer to the crossed target.

Artifacts:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/third_wave/b5_spatial_basis_film_camdelta4_seed17/camera_interventions_seed10017_n16.json
/workspace/writeable/checkpoints/camera_motion_disentangle/third_wave/b6_spatial_film_camdelta4_seed17/camera_interventions_seed10017_n16.json
~~~

Next action: hold uniform spatial-FiLM fixed and sweep camera-delta weights 8
and 16 concurrently to measure the intervention/fidelity tradeoff.

## B7/B8 - Camera-delta magnitude sweep

Date: 2026-07-24

B7 and B8 keep B6's uniform spatial-FiLM architecture and increase only the
camera-delta reconstruction weight to 8 and 16. Both pose-free runs completed
concurrently on separate GPUs.

~~~text
                                      B6          B7          B8
identity to original              0.383979    0.363430    0.381307
wrong-camera output delta         0.014993    0.039001    0.051073
wrong-camera normalized delta     0.019031    0.051257    0.066710
wrong-camera crossed gain         0.016996    0.041630    0.045620
wrong-camera target preference   -0.049589   -0.044065   -0.021173
same-path output delta            0.000068    0.000050    0.000134
~~~

B7 is the best fidelity/response tradeoff: it improves normalized response by
2.69x and crossed gain by 2.45x over B6 while improving identity error. B8's
response grows further, but crossed gain improves only 9.6% over B7 while
identity error regresses. Neither run achieves positive target preference.

Next action: keep B7's delta-MSE weight 8 and add a global cosine objective that
aligns reconstructed and frozen V-JEPA camera-induced feature differences.

## B9/B10 - Camera-delta direction sweep

Date: 2026-07-24
Status: completed

B9 and B10 use the B7 configuration and add camera-delta cosine weights 1 and
4, respectively. This remains fully self-supervised because both delta vectors
come from exact observations in the CamXTime 2x2 grid. Runs and deterministic
16-scene evaluations are assigned to separate GPUs. The B9 evaluation also
loads B7 in the same process to obtain a directly comparable delta-cosine
baseline without repeating V-JEPA feature extraction.

Both runs started at 20:44 UTC in tmux sessions `vjepa_fifth_wave_b9` and
`vjepa_fifth_wave_b10`; the full test suite passed before launch (10 passed).

Both completed 500 steps and the fixed 16-scene diagnostic:

~~~text
                                      B7          B9         B10
identity to original              0.363430    0.371721    0.398110
wrong-camera normalized delta     0.051257    0.057517    0.019775
wrong-camera delta cosine         0.245336    0.258496    0.205284
wrong-camera crossed gain         0.041630    0.045548    0.018018
wrong-camera target preference   -0.044065   -0.030920   -0.049008
same-path output delta            0.000050    0.000138    0.000047
~~~

B9 moderately improves response magnitude, direction, crossed gain, and target
preference over B7, at a small identity-fidelity cost. B10 strongly regresses,
showing that cosine weight 4 overconstrains training. Even B9 retains negative
target preference, so further cosine-weight increases are not justified.

Next action: keep B9 as the current direction-aware result and test either a
direct crossed-target preference/ranking loss or a stronger architectural
constraint against camera leakage through content tokens.

## U1/U2 - Reconstruction capacity upper bound

Date: 2026-07-24
Status: completed

Before adding another disentanglement loss, U1 and U2 measure whether the
current deterministic architecture can reconstruct frozen V-JEPA features when
trained only as an identity autoencoder. U1 retains the 384-dimensional
bottleneck; U2 changes only that width to 768. Neither run uses crossed,
invariance, ranking, delta, pose, or distribution-modeling losses.

Both runs completed 500 steps and the fixed 16-scene evaluation:

~~~text
                                      B7          U1          U2
identity MSE                       0.363430    0.238700    0.127676
identity cosine                   0.850988    0.906772    0.949828
identity retrieval top-1          1.000000    1.000000    1.000000
error / nearest-negative distance 4.297232    2.835357    1.540490
~~~

Width 768 reduces identity MSE by 46.5% relative to width 384, but the U2
reconstruction error remains above the nearest-negative target distance.

## U3/U4 - Reconstruction time and depth

Date: 2026-07-25
Status: completed

U3 keeps width 768 and depth 2 while increasing training from 500 to 2000
steps. U4 changes only depth from 2 to 4 at the same 2000 steps.

~~~text
                                      U2          U3          U4
identity MSE                       0.127676    0.050757    0.051692
identity cosine                   0.949828    0.979770    0.979367
identity retrieval top-1          1.000000    1.000000    1.000000
error / nearest-negative distance 1.540490    0.615241    0.625158
~~~

U3 reduces U2 MSE by 60.2%. U4 uses approximately twice the parameters but is
1.8% worse in identity MSE at the same step budget, so U3 is the selected
fixed-budget reconstruction baseline. This is not a convergence claim: U3's
mean training reconstruction fell from 0.064071 at steps 1410-1500 to 0.052803
at steps 1910-2000, and U4 showed the same non-plateauing trend. Longer training
and intermediate held-out evaluation are required before interpreting either
run as an asymptotic decoder ceiling.

## U5/U6 - EMA and token relation ablations

Date: 2026-07-25
Status: completed

U5 adds only EMA with decay 0.9978 to U3. U6 adds only a weight-1 spatial and
temporal token-relation objective. Both started at 11:28 UTC on separate GPUs
after the expanded test suite passed (11 passed). No crossed factorization,
pose supervision, GAN, or feature noise is active.

Because U3 had not reached a verified plateau, U5/U6 are treated as controlled
2000-step ablations only. They do not establish converged EMA or relation-loss
performance.

Both runs completed and were evaluated at steps 1000 and 2000:

~~~text
                                      U3          U5 EMA      U6 relation
identity MSE at step 1000          0.077334    0.112095    0.077686
identity MSE at step 2000          0.050757    0.059562    0.052873
identity cosine at step 2000       0.979770    0.976200    0.978917
nearest-negative ratio at step 2000
                                    0.615241    0.715987    0.639572
~~~

All validation curves were still improving substantially. Under the fixed
budget, EMA lagged the nonconverged model and token relation slightly regressed
both MSE and cosine. U3 remains the candidate for a longer convergence run.
The guarded cleanup retained both step-1000 JSON files and removed both
step-1000 checkpoints after final evaluation succeeded.

## U7 - Reconstruction convergence

Date: 2026-07-25
Status: completed

U7 resumes U3 at step 2000 with the same width-768, depth-2 architecture and
`MSE + 0.1 cosine` objective. It evaluates eight fixed factor grids every 500
steps and runs to at most step 8000. Three consecutive validations with less
than 1% relative MSE improvement and less than 0.001 cosine gain trigger a 10x
learning-rate reduction; two additional flat validations then stop the run.

Only an atomically replaced `latest.pt`, weights-only `best.pt`, and validation
JSON are retained. The preflight suite passed with 14 tests. A one-step isolated
resume smoke test recovered U3 and its optimizer successfully; the fixed
validation baseline at step 2000 was MSE 0.053449 and cosine 0.978719.

The production run launched in `tmux:vjepa_ninth_wave_u7` on physical GPU 1.
It reproduced the baseline validation and passed step 2020 without OOM or NaN.

U7 completed at step 8000. Its best validation point was step 2500; the final
16-scene MSE was 0.046807 for best versus 0.067524 for latest. U7 best improves
U3 by 7.8%, but the continued `1e-4` optimization is unstable rather than
converged. The degraded latest checkpoint was removed after JSON validation.

## U8 - Low-learning-rate convergence

Date: 2026-07-26
Status: running

U8 initializes U7-best model weights, resets AdamW at `1e-5`, and trains to at
most step 6000. The convergence monitor now compares against historical best
rather than the immediately preceding validation, so degradation followed by a
partial rebound cannot reset patience. Three non-improving validations reduce
LR to `1e-6`; two more stop training.

The revised suite passes with 15 tests. The isolated initialization smoke test
reproduced MSE 0.049426 and cosine 0.980318 at step 2500 with the reset LR.
The production run launched as `tmux:vjepa_tenth_wave_u8` on physical GPU 1 and
passed step 2520 without OOM or NaN.

U8 reached its best fixed-validation point at step 3000. Three subsequent flat
validations reduced LR to `1e-6` at step 4500; two further flat validations
stopped the run at step 5500. Final 16-scene MSE was 0.045356 with cosine
0.981901 and nearest-negative ratio 0.544672. This improves MSE by 3.1% over U7
best and 10.6% over U3.

The non-best U8 latest checkpoint was deleted after final JSON validation. A
guarded historical cleanup then removed 57 evaluated intermediate checkpoints,
reclaiming 21,634,136,773 bytes. Camera-motion checkpoints fell to about 11 GB
and writable free space rose from 35 GB to 56 GB.
