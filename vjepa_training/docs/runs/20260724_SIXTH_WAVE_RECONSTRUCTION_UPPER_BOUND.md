# Sixth-Wave Reconstruction Upper Bound

Date: 2026-07-24
Status: training and automatic evaluation running

This experiment separates decoder/bottleneck capacity from factorization loss.
It trains the same disentangler and reconstructor as a deterministic feature
autoencoder using identity reconstruction only:

~~~text
U1: model_dim=384, depth=2, identity MSE + 0.1 cosine
U2: model_dim=768, depth=2, identity MSE + 0.1 cosine
~~~

There are no crossed reconstruction, factor-invariance, ranking, delta, pose,
or distribution-modeling losses. U1 measures the best reconstruction available
to the current architecture and bottleneck without disentanglement pressure.
U2 tests whether the 384-dimensional bottleneck is the limiting factor.

Evaluation reports raw identity MSE and cosine, four-way within-grid retrieval,
and identity error normalized by the frozen-target distance to its nearest
negative. U1 is evaluated jointly with B7 to reuse V-JEPA extraction and make
the capacity comparison exact.

Both runs started concurrently at 23:45 UTC in tmux sessions
`vjepa_sixth_wave_u1` (physical GPU 1) and `vjepa_sixth_wave_u2` (physical GPU
0). The full test suite and script syntax checks passed before launch.
