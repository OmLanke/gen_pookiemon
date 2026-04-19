# Training Reference

---

## Training Command

```bash
uv run python main.py --dataset pokemon --train
```

All defaults are tuned for WGAN-GP. Override as needed:

```bash
uv run python main.py --dataset pokemon --train \
  --epoch 2000 \
  --learning_rate 0.0001 \
  --beta1 0.0 \
  --n_critic 5 \
  --lambda_gp 10.0
```

---

## Training Loop

```
for each epoch:
  for each batch of 64 images:

    ① for i in range(n_critic=5):        ← critic update
        z ~ Normal(-1, 1)
        fake = G(z).detach()
        c_loss = mean(C(fake)) - mean(C(real)) + λ·GP
        optimizer_C.step()

    ② z ~ Normal(-1, 1)                  ← generator update
        g_loss = -mean(C(G(z)))
        optimizer_G.step()

    ③ log to TensorBoard
    ④ every 100 steps: save 8×8 sample grid PNG

  every 10 epochs: save checkpoint
```

Running the critic `n_critic` times before each generator step ensures the critic gives a reliable Wasserstein distance estimate before G takes a gradient step.

---

## Loss Interpretation

| Metric | What it means | Healthy sign |
|---|---|---|
| `Loss/Wasserstein` | `E[C(real)] − E[C(fake)]` — estimated Wasserstein distance | **Rises** as G improves |
| `Loss/Generator` | `−E[C(fake)]` | **Falls** as G gets better |
| `Loss/Critic` | Total critic loss including GP | Oscillates; no fixed target |
| `Loss/GradientPenalty` | GP term only | Near 0 once Lipschitz constraint holds |

Unlike DCGAN's BCE (both losses converge toward ln(2) ≈ 0.693), WGAN-GP losses have no fixed equilibrium. **The Wasserstein distance is the metric to watch** — a steady increase signals that the generator is improving.

---

## Sample Image Grid

Every 100 steps a file is saved to `samples/train_EE_NNNNNN.png`:

- `EE` — zero-padded epoch number
- `NNNNNN` — zero-padded global step counter
- 8×8 grid of 64 generated 64×64 images
- Uses a **fixed** `sample_z` across all checkpoints for consistent visual comparison

---

## Checkpoint Behaviour

Saved every 10 epochs to `checkpoint/pokemon_64_64_64/`. Last 5 retained:

```
WGAN.model-1810.pt
WGAN.model-1991.pt
WGAN.model-2172.pt
WGAN.model-2353.pt
WGAN.model-2534.pt  ← latest
```

The number is the global batch step counter (not epoch number). Training resumes from the latest checkpoint automatically.

---

## TensorBoard

```bash
uv run tensorboard --logdir logs/pokemon
# → http://localhost:6006
```

Panels:
- **Scalars**: `Loss/Critic`, `Loss/Generator`, `Loss/Wasserstein`, `Loss/GradientPenalty`
- **Images**: `Generated/samples` — live grid of generated Pokémon (updates every 100 steps)

---

## Expected Training Timeline

| Hardware | Per epoch | 2000 epochs |
|---|---|---|
| CPU only | ~5–10 min | ~7–14 days |
| Apple M-series (MPS) | ~2–5 min | ~3–7 days |
| GPU (mid-range) | ~30–90 sec | ~17–50 hours |

WGAN-GP is ~5× more compute per epoch than a standard GAN due to `n_critic=5` critic updates per batch. This is expected — the extra compute buys training stability.

---

## Inference

```bash
uv run python main.py --dataset pokemon
```

Loads the latest checkpoint, samples 64 noise vectors from `z ~ Normal(-1, 1)`, runs them through the trained generator, and saves an 8×8 grid to `samples/test_YYYY-MM-DD-HH-MM-SS.png`.

---

## Key Hyperparameters

| Parameter | Effect of increasing | Effect of decreasing |
|---|---|---|
| `learning_rate` | Faster, less stable | Slower, more stable |
| `beta1` | More momentum | Less momentum |
| `batch_size` | More stable gradients | Noisier updates |
| `n_critic` | Better W estimate, slower per epoch | Noisier estimate, faster |
| `lambda_gp` | Stricter Lipschitz enforcement | Weaker constraint, may diverge |
| `z_dim` | More generator capacity | Less capacity |
| `gf_dim` / `df_dim` | More expressive, more VRAM | Lighter, less expressive |
