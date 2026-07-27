# Eighth-Wave EMA and Token Relation Ablations

Date: 2026-07-25
Status: completed

## Baseline

U3 is the selected deterministic reconstruction baseline:

~~~text
width=768, depth=2, 2000 steps
loss=MSE + 0.1 cosine
identity MSE=0.050757
identity cosine=0.979770
nearest-negative ratio=0.615241
~~~

## Runs

~~~text
U5: U3 + decoder/factorizer EMA, decay=0.9978
U6: U3 + spatial-and-temporal token relation loss, weight=1.0
~~~

U5 evaluation uses the saved EMA weights; the checkpoint retains raw weights
for diagnosis. U6 preserves pairwise cosine geometry between spatial patch
tokens within each tubelet and temporal tokens along each spatial track.

The two additions are intentionally tested separately. They will be combined
only if both improve held-out reconstruction or complementary metrics. Neither
run uses crossed factorization losses, pose supervision, GAN loss, or feature
noise.

U3 had not reached a verified loss plateau at 2000 steps. U5/U6 are therefore
controlled fixed-budget ablations, not final asymptotic comparisons. Their
results must be followed by intermediate-checkpoint validation and longer
training of the selected objective.

Both runs started at 11:28 UTC in tmux sessions `vjepa_eighth_wave_u5`
(physical GPU 0) and `vjepa_eighth_wave_u6` (physical GPU 1). The expanded test
suite passed before launch (11 passed). Both runs passed step 50 without OOM,
NaN, or decode errors. U6's relation term fell from 0.11403 at step 1 to
0.00595 at step 50, confirming that its configured weight is numerically
meaningful without dominating the reconstruction term.

Checkpoint retention is guarded by
`scripts/cleanup_eighth_wave_intermediates.sh`. A separate cleanup session
waits for a readable final evaluation JSON and a nonempty 2000-step checkpoint
before removing the corresponding 1000-step checkpoint. Failed or incomplete
runs retain their recovery checkpoint.

## Results

Both runs completed 2000 steps. The guarded cleanup evaluated step 1000,
retained its JSON metrics, and removed both intermediate checkpoints.

~~~text
                                      U3          U5 EMA      U6 relation
identity MSE at step 1000          0.077334    0.112095    0.077686
identity MSE at step 2000          0.050757    0.059562    0.052873
identity cosine at step 2000       0.979770    0.976200    0.978917
nearest-negative ratio at step 2000
                                    0.615241    0.715987    0.639572
retrieval top-1 at step 2000       1.000000    1.000000    1.000000
~~~

None had converged: validation MSE improved by 34.4% for U3, 46.9% for U5
EMA, and 31.9% for U6 between steps 1000 and 2000. EMA lagged the rapidly
moving raw model and was 17.3% worse than U3 at step 2000. Token relation was
4.2% worse than U3 in MSE and slightly worse in cosine. U3 remains the
reconstruction candidate for a convergence run. EMA should be reconsidered
only near a validation plateau, and token relation remains a downstream
geometry ablation rather than the primary reconstruction objective.

At result collection, `/workspace/writeable` had 52G free. The drop from the
previous 75G was traced to an unrelated active `yt-dlp` process writing
`datasets/dynpose-videos`, which had reached approximately 41G. No new long
run should start until rolling checkpoint retention is implemented and free
space is rechecked.
