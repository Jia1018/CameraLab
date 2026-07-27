# Tenth-Wave Low-LR Reconstruction Convergence

Date: 2026-07-26
Status: completed

## Question

Can U7's best reconstruction improve or reach a stable validation plateau when
optimization restarts at one tenth the learning rate?

## U8 protocol

U8 initializes model weights from U7 best at step 2500 and deliberately resets
AdamW. It retains the same data, architecture, and `MSE + 0.1 cosine` objective.
The initial LR is `1e-5`, the maximum step is 6000, and validation remains the
fixed eight-sample seed-10017 set every 500 steps.

Plateau decisions now compare each validation against historical best MSE and
cosine. Three validations without at least 1% relative MSE improvement or 0.001
cosine gain reduce LR to `1e-6`. Two more non-improving validations stop the
run. Rolling `latest.pt` and weights-only `best.pt` retention is unchanged.

## Preflight and launch

The revised suite passes with 15 tests, including a regression test where a
partial recovery remains worse than historical best. An isolated initialization
smoke test reproduced the U7-best eight-sample baseline:

~~~text
step=2500
identity MSE=0.049426
identity cosine=0.980318
optimizer LR=1e-5
~~~

U8 launched in `tmux:vjepa_tenth_wave_u8` on physical GPU 1. It passed step
2520 without OOM or NaN. `/workspace/writeable` had approximately 43 GB free;
unrelated active data-generation processes were not modified.

## Results

U8 improved once at step 3000, then failed to recover the historical best.
The corrected monitor reduced LR at step 4500 and stopped at step 5500:

~~~text
step       MSE       cosine       LR         action
2500    0.049426    0.980318    1e-5        continue
3000    0.047827    0.980933    1e-5        best
3500    0.048624    0.980606    1e-5        flat 1
4000    0.051384    0.979485    1e-5        flat 2
4500    0.051603    0.979386    1e-6        reduce LR
5000    0.054246    0.978305    1e-6        flat 1
5500    0.053840    0.978471    1e-6        stop
~~~

The final 16-scene comparison is:

~~~text
                                      U3 step 2000   U7 best 2500   U8 best 3000
identity MSE                            0.050757        0.046807        0.045356
identity cosine                         0.979770        0.981340        0.981901
error / nearest-negative distance       0.615241        0.564130        0.544672
~~~

U8 best improves MSE by 3.1% over U7 best and 10.6% over U3. The low-LR and
post-reduction validations establish a practical local reconstruction ceiling
for this architecture and objective. They do not imply exact invertibility.

Camera interventions remain nearly inert (`wrong_camera_output_delta=0.000006`),
which is expected because the identity-only upper-bound objective does not force
the camera branch to carry information. The next phase must add paired crossed
constraints while using U8 best as the reconstruction initialization.

After the final result JSON was validated, the non-best resumable checkpoint
was removed and the weights-only U8 best was retained.
