# Second-Wave B2 Factor-Delta Pilot

Date: 2026-07-24
Status: completed; partial success, decision criteria not met

Launch:

- tmux session: `vjepa_second_wave_b2`;
- start: 2026-07-24 18:06:45 UTC;
- device: physical GPU 1 through `CUDA_VISIBLE_DEVICES=1`;
- the repaired 16-scene intervention diagnostic runs automatically after
  step 500.

## Hypothesis

B1 proves that spatial-FiLM makes the decoder sensitive to camera tokens, but
wrong-camera substitution explains only 0.22% of the true camera-path feature
difference. Independent pointwise reconstruction permits a weak average-target
solution. Directly aligning the factor-induced output deltas with the exact
2x2 target deltas should train both substitution direction and magnitude.

## Change From B1

For reconstructed combinations arranged as
`[target_camera, target_physics, content_source_camera,
camera_source_physics, batch, ...]`, B2 adds:

~~~text
camera delta:
  MSE((R[c=1] - R[c=0]), (Z[c=1] - Z[c=0]))

content delta:
  MSE((R[p=1] - R[p=0]), (Z[p=1] - Z[p=0]))
~~~

Both weights are 1.0. `Z` denotes frozen V-JEPA 2.1 tokens. These terms use
only the paired CamXTime grid; camera pose supervision remains zero. All other
model, data, loss, optimizer, seed, and 500-step settings are identical to B1.

## Validation

- unit tests: 8 passed;
- Python compile and runner shell syntax: passed;
- two-step real-data GPU optimization: passed;
- step-2 checkpoint serialization and reload: passed;
- initial delta losses: camera 0.74216, content 0.18824;
- temporary smoke output is isolated under `/tmp`.

## Decision Criteria

- camera and content delta losses decrease without destabilizing pointwise
  reconstruction;
- wrong-camera normalized delta materially exceeds B1's 0.002160;
- wrong-camera crossed gain remains positive and increases;
- wrong-camera target preference approaches or exceeds zero;
- identity reconstruction recovers rather than further regresses;
- camera tokens remain non-collapsed and same-path substitutions remain much
  smaller than wrong-path substitutions.

Output:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b2_factor_delta_seed17
~~~

## Results

Late-training means over logged steps 400 through 500:

~~~text
loss                           1.28517
pointwise reconstruction       0.44220
identity reconstruction        0.39775
content-swap reconstruction    0.44775
camera delta reconstruction    0.73569
content delta reconstruction   0.07730
camera feature variance        0.22470
~~~

The deterministic 16-scene diagnostic produced:

~~~text
                                      B2             B1
identity to original              0.392633       0.411380
wrong-camera output delta         0.003100       0.001580
wrong-camera normalized delta     0.004036       0.002160
wrong-camera crossed gain         0.005096       0.002126
wrong-camera target preference   -0.039917      -0.025909
same-path output delta            0.000034       0.000014
zero-camera output delta          0.022879       0.013255
camera-token wrong-path delta     0.169527       0.235601
~~~

B2 nearly doubles wrong-camera sensitivity and normalized delta, increases
crossed gain by about 2.4x, and improves identity reconstruction. The response
still accounts for only 0.40% of the target difference in squared-error units,
and remains closer to the original target. The late camera-delta loss remains
large. The delta objective is useful, but uniform spatial-FiLM cannot express
enough position-dependent camera motion.

Artifact:

~~~text
/workspace/writeable/checkpoints/camera_motion_disentangle/second_wave/b2_factor_delta_seed17/camera_interventions_seed10017_n16.json
~~~

Next action: retain B2's objective and replace uniform spatial-FiLM with a
low-rank spatial-basis camera residual that can produce a distinct feature
change at every patch position.
