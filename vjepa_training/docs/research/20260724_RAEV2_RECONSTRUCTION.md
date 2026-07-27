# RAEv2 Reconstruction Study

Date: 2026-07-24

## Scope

RAEv2 Stage 1 does not reconstruct a pretrained feature tensor into the same
feature tensor. It freezes a pretrained image encoder and learns a decoder from
its patch tokens back to RGB:

~~~text
image -> frozen representation encoder -> patch tokens -> trainable decoder -> RGB
~~~

Stage 2 then trains a diffusion/world model in that representation latent
space. Stage 2 REPA and generation losses are not direct precedents for the
current V-JEPA camera/content feature reconstruction objective.

## Stage-1 implementation

- DINOv3-L at 256x256 yields 256 patch tokens of dimension 1024. CLS and
  register tokens are discarded.
- The encoder is in eval mode, has `requires_grad=False`, and `encode()` runs
  under `torch.no_grad()`.
- The released DINOv3 Stage-1 configs use the ViT-XL decoder: 28 blocks,
  hidden dimension 1152, 16 heads, and MLP dimension 4096. The decoder has
  415,647,616 trainable parameters.
- The decoder adds a trainable CLS token and fixed 2D sinusoidal positions,
  predicts 16x16 RGB patches with a linear head, and unpatchifies them.
- Training loss is L1 RGB reconstruction + LPIPS + adaptive GAN loss. The GAN
  weight uses the ratio of reconstruction and GAN gradient norms at the final
  prediction layer. Discriminator updates start at epoch 6 and the generator
  GAN term at epoch 8.
- Decoder EMA uses decay 0.9978. AdamW uses LR 2e-4 and zero weight decay.
- During Stage-1 training, each example receives latent noise with
  `sigma ~ Uniform(0, 0.8)` and `z_noisy = z + sigma * randn_like(z)`. The
  clean RGB remains the target. Noise is disabled for sampling. The paper does
  not provide a dedicated ablation or motivation for this code-level choice.
- Encoder mean/variance statistics are computed after decoder training for
  Stage-2 latent normalization. They are not a VAE posterior and no KL loss is
  used.

## Multi-layer representation

The paper argues that earlier/middle layers retain more local spatial detail
and later layers emphasize global semantics. It preserves the original
`[N, D]` footprint rather than concatenating layers. Reconstruction improves
monotonically from K=1 to K=23 in its DINOv3-L sweep: the paper reports PSNR
18.93/rFID 0.60 for the standard last-layer RAE and PSNR 27.03/rFID 0.18 at
K=23. Linear-probe performance is reported as preserved.

There is a paper/code distinction that must be tracked in our experiments:

- The paper defines MLS as a direct sum of K normalized layer features.
- The current DINOv3 code computes their mean, then broadcasts and adds the
  spatial mean of the final selected layer to every patch.
- The DINOv2 multi-layer implementation only computes the layer mean.

Therefore a V-JEPA port should name and ablate `last`, `mean_k`, and
`mean_k_plus_final_global` explicitly. V-JEPA 2.1 Base already exposes four
normalized hierarchical layers `[2, 5, 8, 11]`, so this does not require
backbone finetuning.

## Relation to the current feature factorizer

The current model solves a stricter but different problem:

~~~text
frozen V-JEPA tokens -> content tokens + camera tokens -> frozen V-JEPA tokens
~~~

Its feature loss should stay pointwise (MSE/cosine and crossed reconstruction).
RGB LPIPS and the image discriminator do not define distances in V-JEPA feature
space. A feature discriminator could match marginal distributions while losing
per-example identity, so it cannot replace crossed reconstruction.

RAEv2 mainly raises the decoder-capacity and training-scale bar. The current
upper bounds have only two factorized blocks and 500 optimizer steps:

~~~text
U1, width 384: 22,496,649 adapter+decoder parameters
U2, width 768: 88,639,497 adapter+decoder parameters
RAEv2 ViT-XL decoder: 415,647,616 parameters
~~~

The completed fixed 16-scene evaluation is:

~~~text
                                      B7          U1          U2
identity MSE                       0.363430    0.238700    0.127676
identity cosine                   0.850988    0.906772    0.949828
identity retrieval top-1          1.000000    1.000000    1.000000
error / nearest-negative distance 4.297232    2.835357    1.540490
wrong-camera output delta         0.039001    0.000069    0.000006
~~~

U2 establishes that width is a major limitation, but even the identity-only
model has not reached a high-fidelity ceiling. As expected, identity-only
training ignores camera tokens. More disentanglement losses should not be
interpreted before decoder capacity/training duration is improved.

## Recommended transfer experiments

1. Establish a stronger identity ceiling with width 768, deeper factorized
   encoder/decoder blocks, longer training, and EMA. Do not add an identity
   skip that would trivially bypass the intended factorization.
2. Add a V-JEPA multi-layer extraction baseline. First compare final-layer
   input against four-layer mean input while retaining the final V-JEPA layer
   as the primary reconstruction target. An auxiliary head may reconstruct all
   four clean layers without changing the downstream baseline representation.
3. Initialize the disentanglement run from the identity-autoencoder weights,
   then introduce crossed and invariance objectives. This separates decoder
   optimization from factor discovery.
4. Only after a clean high-fidelity baseline, test small relative feature
   noise and EMA as isolated ablations. RAEv2's absolute tau=0.8 should not be
   copied without calibrating against V-JEPA token standard deviation and
   camera-induced feature deltas.
5. Keep pose supervision disabled in the primary method. Camera-pose targets
   remain a later ablation.

## Interpretation

Exact elementwise equality is a useful fidelity diagnostic, not a requirement
that camera and content coordinates form a unique mathematical bijection.
Joint reconstruction constrains the pair `(content, camera)` to retain the
teacher representation; crossed pairs constrain which factor may carry which
variation. The latent coordinates still have gauge freedom, and downstream
content/camera tasks and intervention tests are required to establish that the
learned split is useful.
