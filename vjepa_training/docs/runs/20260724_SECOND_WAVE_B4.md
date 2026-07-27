# Second-Wave B4 Strong Camera-Delta Pilot

Date: 2026-07-24
Status: completed (500/500 + deterministic 16-scene intervention evaluation)

Launch: `vjepa_second_wave_b4`, physical GPU 0, 2026-07-24 18:31:23 UTC.
B3 runs concurrently on physical GPU 1.

B4 is the paired control for B3. It uses the same rank-16 spatial-basis
reconstructor, seed, CamXTime samples, optimizer, and self-supervised losses.
The only scientific change is:

~~~text
camera_delta_reconstruction: 1.0 -> 4.0
~~~

This tests whether the magnitude of the factor-change gradient, rather than
spatial decoder capacity, limits wrong-camera transfer. Pose supervision
remains zero. B3 and B4 run concurrently on separate GPUs and use the same
deterministic 16-scene diagnostic.

Decision: prefer B4 only if it materially increases wrong-camera normalized
delta and crossed gain without unacceptable identity reconstruction loss or
camera-token collapse.

Output:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b4_spatial_basis_camdelta4_seed17
~~~

## Results

Late logged means for steps 400-500:

~~~text
pointwise reconstruction          0.449332
identity reconstruction           0.422789
camera-delta reconstruction       0.741151
content-delta reconstruction      0.078571
camera feature variance           0.142837
~~~

Deterministic 16-scene intervention diagnostic (seed 10017):

~~~text
identity to original              0.416378
wrong-camera output delta         0.003801
wrong-camera normalized delta     0.005030
wrong-camera crossed gain         0.005773
wrong-camera target preference   -0.019653
same-path output delta            0.000019
zero-camera output delta          0.010532
reversed-camera output delta      0.000296
camera-token wrong-path delta     0.222638
~~~

Artifact:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b4_spatial_basis_camdelta4_seed17/camera_interventions_seed10017_n16.json
~~~

## Interpretation

Increasing the camera-delta weight recovers intervention sensitivity: compared
with B2, normalized delta improves by 24.6%, crossed gain by 13.3%, and target
preference moves toward zero. However, identity error worsens by 0.023745 and
the output change is still only about 0.50% of the true camera-path feature
difference. B4 is the strongest intervention result so far, but does not meet
the fidelity or effect-size criteria. The small improvement from a 4x loss
weight also shows that loss scaling alone is insufficient.
