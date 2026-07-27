# Second-Wave B3 Spatial-Basis Pilot

Date: 2026-07-24
Status: completed (500/500 + deterministic 16-scene intervention evaluation)

Launch: `vjepa_second_wave_b3`, physical GPU 1, 2026-07-24 18:31:23 UTC.
B4 runs concurrently on physical GPU 0.

## Hypothesis

B2's symmetric factor-delta objective improves camera intervention strength,
but uniform spatial-FiLM cannot directly express different camera-induced
changes at different patch positions. A low-rank spatial camera field should
increase crossed-camera response while preserving content fidelity.

## Change From B2

At every reconstruction block, the tubelet camera token predicts rank-16
coefficients. These coefficients combine learned feature fields at all 576
spatial patch positions:

~~~text
delta[b,t,n,d] = sum_r coeff[b,t,r] * basis[n,r,d] / sqrt(16)
patch[b,t,n,d] = patch[b,t,n,d] + delta[b,t,n,d]
~~~

The coefficient projection has no bias, so a zero camera input contributes no
direct spatial-basis residual. Subsequent temporal and local attention are
unchanged. All B2 losses, including the symmetric camera/content delta terms,
remain active with identical weights. Pose supervision remains zero.

## Validation

- unit tests: 9 passed;
- Python compile and runner shell syntax: passed;
- two-step real-data GPU optimization: passed;
- step-2 checkpoint reload: passed;
- reconstructor parameters: 18,034,944;
- smoke checkpoint size: 332 MB.

## Decision Criteria

- stable optimization and non-collapsed camera tokens;
- identity reconstruction no worse than B2;
- wrong-camera normalized delta and crossed gain materially exceed B2;
- same-path output delta stays substantially below wrong-path output delta;
- target preference moves toward zero, ideally becoming positive.

Output:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b3_spatial_basis_delta_seed17
~~~

## Results

Late logged means for steps 400-500:

~~~text
pointwise reconstruction          0.450730
identity reconstruction           0.425482
camera-delta reconstruction       0.745818
content-delta reconstruction      0.078532
camera feature variance           0.279961
~~~

Deterministic 16-scene intervention diagnostic (seed 10017):

~~~text
identity to original              0.416131
wrong-camera output delta         0.001470
wrong-camera normalized delta     0.001950
wrong-camera crossed gain         0.002355
wrong-camera target preference   -0.024222
same-path output delta            0.000013
zero-camera output delta          0.004279
reversed-camera output delta      0.000137
camera-token wrong-path delta     0.296404
~~~

Artifact:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b3_spatial_basis_delta_seed17/camera_interventions_seed10017_n16.json
~~~

## Interpretation

The camera representation remains path-sensitive and the same-path control is
small, but the rank-16 spatial basis does not improve the desired output
intervention. Relative to B2, normalized delta falls from 0.004036 to 0.001950,
crossed gain falls from 0.005096 to 0.002355, and identity error worsens from
0.392633 to 0.416131. Spatially varying capacity alone is therefore not the
limiting factor under the unit camera-delta weight.
