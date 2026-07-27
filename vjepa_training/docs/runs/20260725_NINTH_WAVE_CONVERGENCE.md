# Ninth-Wave Reconstruction Convergence

Date: 2026-07-25
Status: completed

## Question

Does the selected U3 deterministic autoencoder reach a held-out identity
reconstruction plateau, and what is its reconstruction ceiling before crossed
camera/content disentanglement is reintroduced?

This experiment is required because U3, U5, and U6 all improved substantially
between steps 1000 and 2000. Their previous comparisons are valid fixed-budget
pilots, but none establishes a converged decoder ceiling.

## U7 protocol

U7 resumes the complete U3 model and AdamW state from step 2000. Architecture,
data, and objective remain unchanged:

~~~text
model width=768, depth=2
loss=MSE + 0.1 cosine
camera conditioning=spatial FiLM
pose supervision=off
crossed disentanglement losses=off
EMA=off
token relation loss=off
initial LR=1e-4
maximum step=8000
~~~

Every 500 steps, U7 evaluates identity reconstruction on the same eight
deterministic CamXTime factor grids selected with seed 10017. A validation is
flat only when relative MSE improvement is below 1% and flattened-feature
cosine gain is below 0.001. Three consecutive flat validations trigger one
10x learning-rate reduction. Two further flat validations after that reduction
declare convergence and stop training early.

The eight-sample curve controls training only. The final report uses the
existing 16-scene intervention evaluator for comparability with earlier waves.

## Retention and recovery

The run atomically overwrites one resumable `latest.pt` and one weights-only
`best.pt`; it never accumulates step-numbered checkpoints. Validation history is
also written to a small JSON file. Expected stable checkpoint use is about
1.4 GB. The original U3 step-2000 checkpoint remains unchanged.

## Preflight

The expanded suite passes with 14 tests. An isolated GPU-1 smoke test resumed
U3 step 2000, restored the optimizer, ran a training step, and wrote exactly
the three intended artifacts. Its baseline eight-sample metrics were:

~~~text
identity MSE=0.053449
identity cosine=0.978719
~~~

The smoke artifacts and U3 step-1000 checkpoint were removed after validation;
their metric JSON remains. At launch preflight, `/workspace/writeable` had
approximately 52 GB free. Unrelated active downloads were left untouched.

U7 launched in tmux session `vjepa_ninth_wave_u7` on physical GPU 1. The
step-2000 validation reproduced the smoke baseline, the rolling files were
written successfully, and training passed step 2020 without OOM or NaN.

## Results

The fixed eight-sample validation reached its best point at step 2500, then
became unstable under the unchanged `1e-4` learning rate:

~~~text
step       MSE       cosine
2000    0.053449    0.978719
2500    0.049426    0.980318  best
3000    0.053999    0.978432
4500    0.051658    0.979388
5500    0.073153    0.970613
6500    0.092295    0.962768
8000    0.070645    0.971645
~~~

The final 16-scene evaluation confirms that rolling best retention worked:

~~~text
                                      U3 step 2000   U7 best 2500   U7 latest 8000
identity MSE                            0.050757        0.046807         0.067524
identity cosine                         0.979770        0.981340         0.972885
error / nearest-negative distance       0.615241        0.564130         0.801969
~~~

U7 best improves U3 MSE by 7.8%, but U7 does not establish a converged ceiling.
The original monitor compared adjacent validations; a partial rebound from a
worse point could reset its patience even while remaining below the historical
best. The monitor now compares both gains against historical best values.

The selected step-2500 weights were retained. The step-8000 `latest.pt` was
removed after its final JSON was validated because it is substantially worse
and is not a useful recovery point.
