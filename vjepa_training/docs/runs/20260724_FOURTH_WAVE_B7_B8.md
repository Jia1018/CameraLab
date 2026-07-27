# Fourth-Wave B7/B8 Camera-Delta Weight Sweep

Date: 2026-07-24
Status: completed

B6 established uniform spatial-FiLM with camera-delta weight 4.0 as the best
variant so far. B7 and B8 keep its architecture, seed, data order, optimizer,
and pose-free losses fixed while increasing only that weight:

~~~text
B7: camera_delta_reconstruction = 8.0
B8: camera_delta_reconstruction = 16.0
~~~

The paired runs measure whether intervention strength continues to scale and
where identity fidelity begins to degrade. Both use the deterministic 16-scene
intervention diagnostic with seed 10017 after step 500.

~~~text
                                      B6          B7          B8
identity to original              0.383979    0.363430    0.381307
wrong-camera output delta         0.014993    0.039001    0.051073
wrong-camera normalized delta     0.019031    0.051257    0.066710
wrong-camera crossed gain         0.016996    0.041630    0.045620
wrong-camera target preference   -0.049589   -0.044065   -0.021173
same-path output delta            0.000068    0.000050    0.000134
zero-camera output delta          0.028719    0.034170    0.064564
camera-token wrong-path delta     0.520345    0.347786    0.680577
~~~

B7 improves normalized intervention strength by 2.69x and crossed gain by
2.45x over B6, while also reducing identity error by 0.020549. B8 produces the
largest intervention: normalized delta is 30.1% above B7 and target preference
is less negative. That gain costs 0.017877 identity error, and crossed gain
improves only 9.6%, showing diminishing returns at weight 16.

Both wrong-path responses remain highly selective relative to same-path
controls, but neither reconstruction is closer to the correct crossed-camera
target than to the original. The next paired experiment therefore holds the
better-fidelity B7 setting fixed and adds an explicit camera-delta direction
objective rather than continuing the MSE weight sweep.

Artifacts:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/fourth_wave/b7_spatial_film_camdelta8_seed17/camera_interventions_seed10017_n16.json
/workspace/writeable/checkpoints/camera_motion_disentangle/fourth_wave/b8_spatial_film_camdelta16_seed17/camera_interventions_seed10017_n16.json
~~~
