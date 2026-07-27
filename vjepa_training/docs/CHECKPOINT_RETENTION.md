# Checkpoint Retention

Date: 2026-07-25

The writable volume is space constrained. Reconstruction experiments use the
following retention policy:

- During a convergence run, overwrite one resumable `latest.pt`.
- Overwrite one weights-only `best.pt` when fixed-validation MSE improves.
- Do not serialize the frozen V-JEPA backbone.
- After convergence and final evaluation, retain only the selected best model,
  its config, logs, and evaluation JSON.
- Remove intermediate checkpoints only after both the final checkpoint and a
  readable final evaluation result exist.

The eighth-wave runs predate rolling checkpoint support. Their guarded cleanup
evaluates `step_0001000.pt` on the fixed validation scenes, retains the small
JSON result, and removes that checkpoint only after `step_0002000.pt` has
completed final evaluation. Historical runs are not deleted automatically.

The ninth-wave U7 convergence run implements rolling retention directly. Its
`latest.pt` is the only full checkpoint and includes optimizer, RNG, validation
history, and convergence-monitor state. Its `best.pt` contains only the two
trainable model state dictionaries. Both files are replaced atomically, so an
interrupted write cannot destroy the previous resumable checkpoint.

U3 `step_0001000.pt` was removed after its fixed 16-scene evaluation JSON was
validated. U3 `step_0002000.pt` remains the immutable U7 starting checkpoint.

U7 completed at step 8000 but its selected best was step 2500. After both were
evaluated together and the final JSON was validated, the degraded step-8000
`latest.pt` was removed. The weights-only U7 `best.pt`, validation curve, logs,
config, and final JSON remain. U8 uses the same rolling policy.

U8 selected step 3000 and stopped at step 5500. Its final JSON was validated,
then the non-best `latest.pt` was removed. A repository-wide guarded cleanup
parsed evaluation JSON metadata and removed 57 older intermediate checkpoints
from 15 runs whose final checkpoints had valid model results. This reclaimed
21,634,136,773 bytes while preserving every final/best checkpoint and metric
file. A follow-up dry-run reported zero remaining eligible intermediates.
