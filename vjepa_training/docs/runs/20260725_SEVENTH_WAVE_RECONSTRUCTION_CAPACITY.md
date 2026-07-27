# Seventh-Wave Reconstruction Capacity

Date: 2026-07-25
Status: completed

## Question

The sixth-wave U2 run showed that increasing the bottleneck width from 384 to
768 substantially improved identity reconstruction, but its error was still
1.54 times the nearest-negative target distance after 500 steps. This wave
separates insufficient optimization time from insufficient decoder depth.

## Runs

~~~text
U3: width=768, depth=2, 2000 steps, MSE + 0.1 cosine
U4: width=768, depth=4, 2000 steps, MSE + 0.1 cosine
~~~

All other model, data, optimizer, seed, and evaluation settings are held fixed.
Both runs use the same 16-scene evaluation seed (`10017`) as U1 and U2.

## Decision Rule

Identity reconstruction is considered usable for subsequent disentanglement
only if all of the following improve without a retrieval regression:

- identity MSE and cosine;
- error divided by nearest-negative target distance;
- four-way within-grid retrieval top-1.

The lower-capacity model wins when reconstruction quality is effectively tied.
EMA and token-relational reconstruction remain separate eighth-wave ablations
on the selected architecture. No camera-pose supervision is used.

## Results

Both runs completed 2000 steps and the fixed 16-scene evaluation:

~~~text
                                      U2          U3          U4
identity MSE                       0.127676    0.050757    0.051692
identity cosine                   0.949828    0.979770    0.979367
identity retrieval top-1          1.000000    1.000000    1.000000
error / nearest-negative distance 1.540490    0.615241    0.625158
wrong-camera output delta         0.000006    0.000006    0.000008
~~~

Longer training reduces U3 identity MSE by 60.2% relative to U2 and brings its
mean normalized error below the nearest-negative distance. U4 has approximately
twice as many trainable parameters as U3 (176.1M versus 88.6M), but its
identity MSE is 1.8% worse at the same 2000-step budget. U3 therefore wins the
fixed-budget selection.

This is not yet an asymptotic convergence result. Mean training reconstruction
over four 100-step windows remained:

~~~text
steps             410-500    910-1000   1410-1500  1910-2000
U3 reconstruction 0.139499    0.080163    0.064071    0.052803
U4 reconstruction 0.141932    0.082777    0.065492    0.053084
~~~

Both curves were still falling in their last 500 steps. The evidence supports
that U3 is more parameter- and step-efficient under this budget, not that U3
has a better final asymptotic ceiling. Intermediate held-out evaluations and a
longer continuation are required before declaring convergence.

As expected for identity-only training, both runs ignore camera tokens. The
eighth wave will keep U3 fixed and separately test EMA-only and
token-relation-only objectives before returning to crossed disentanglement.
