# Third-Wave B5/B6 Paired Pilot

Date: 2026-07-24
Status: completed (500/500 each + deterministic 16-scene evaluations)

## Question

B4 shows that a stronger camera-delta loss can improve intervention strength,
but its additive spatial basis has no content-dependent multiplicative
interaction and loses identity fidelity. This pair isolates architecture from
loss weighting at the same camera-delta weight of 4.0.

## B5 - Spatial-Basis FiLM

The camera token predicts rank-16 coefficients which combine learned,
position-dependent scale and shift fields. These fields modulate normalized
content patches at every reconstruction block:

~~~text
[scale, shift][b,t,n] = sum_r coeff[b,t,r] * basis[n,r] / sqrt(16)
patch[b,t,n] = norm(patch[b,t,n]) * (1 + tanh(scale[b,t,n])) + shift[b,t,n]
~~~

This retains direct content-camera interaction while allowing different patch
positions to respond differently.

## B6 - Uniform Spatial-FiLM Control

B6 uses B2's uniform spatial-FiLM architecture with the same camera-delta
weight of 4.0. Comparing B6 with B2 measures pure loss-weight effect; comparing
B5 with B6 measures the value of position-dependent FiLM.

Both runs use seed 17, the same 500 sampled scenes/paths, no pose supervision,
and the deterministic 16-scene intervention diagnostic with seed 10017.

## Results

Late logged means for steps 400-500:

~~~text
                                      B5          B6
pointwise reconstruction          0.447078    0.450202
identity reconstruction           0.403005    0.404104
camera-delta reconstruction       0.745804    0.724507
content-delta reconstruction      0.077231    0.077133
camera feature variance           0.169553    0.217829
~~~

Deterministic 16-scene intervention diagnostic (seed 10017):

~~~text
                                      B5          B6
identity to original              0.401040    0.383979
wrong-camera output delta         0.001744    0.014993
wrong-camera normalized delta     0.002316    0.019031
wrong-camera crossed gain         0.003052    0.016996
wrong-camera target preference   -0.039900   -0.049589
same-path output delta            0.000007    0.000068
zero-camera output delta          0.003267    0.028719
reversed-camera output delta      0.000038    0.002338
camera-token wrong-path delta     0.333882    0.520345
~~~

Artifacts:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/third_wave/b5_spatial_basis_film_camdelta4_seed17/camera_interventions_seed10017_n16.json
/workspace/writeable/checkpoints/camera_motion_disentangle/third_wave/b6_spatial_film_camdelta4_seed17/camera_interventions_seed10017_n16.json
~~~

## Interpretation

B5 rejects the position-dependent low-rank FiLM hypothesis in its current
form. Despite slightly better late-training identity reconstruction, its
wrong-camera response is weak. B6 is the clear winner: relative to B2 it
improves normalized delta by 4.72x and crossed gain by 3.34x while also reducing
identity error from 0.392633 to 0.383979. The same-path control remains 222x
smaller than the wrong-path output delta. B6 still changes the output by only
1.90% of the true camera difference and target preference remains negative, so
the next experiment should sweep stronger camera-delta weights while holding
uniform spatial-FiLM fixed.
