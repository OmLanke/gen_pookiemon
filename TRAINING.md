# Pokemon Master — Training Reference

Everything you need to know about running and monitoring training.

---

## Training Command

```bash
python main.py \
  --dataset pokemon \
  --input_height 64 \
  --input_width 64 \
  --output_height 64 \
  --output_width 64 \
  --batch_size 64 \
  --epoch 2000 \
  --learning_rate 0.0002 \
  --beta1 0.5 \
  --generate_test_images 100 \
  --train \
  --crop False \
  --checkpoint_dir checkpoint \
  --sample_dir samples
```

Minimal version (all defaults match Pokemon config):
```bash
python main.py --dataset pokemon --train
```

---

## Training Loop Walkthrough

### Iteration Structure

```
for epoch in range(2000):
    shuffle data
    for each batch of 64 images:
        ① load batch → normalize → feed to graph
        ② sample z ~ N(-1, 1), shape [64, 100]
        ③ run d_optim  (1×)
        ④ run g_optim  (2×, always)
        ⑤ eval losses
        ⑥ if errG - (errD_fake + errD_real) > 1: run g_optim again
        ⑦ log to TensorBoard
        ⑧ every 100 steps: save sample PNG
    every 10 epochs: save checkpoint
```

### Step ③ — Discriminator Update

```python
sess.run([d_optim, d_sum], feed_dict={inputs: batch_images, z: batch_z})
```

The discriminator sees:
- Real images from the dataset (wants to output 1)
- Fake images from G(batch_z) (wants to output 0)

Its loss is the sum of both binary cross-entropy terms.

### Steps ④ — Generator Updates (×2)

```python
sess.run([g_optim, g_sum], feed_dict={z: batch_z})
sess.run([g_optim, g_sum], feed_dict={z: batch_z})
```

Notice: only `z` is fed, NOT `inputs`. The generator doesn't see real images.
Running twice ensures G gets more gradient signal than D per batch.

### Step ⑥ — Conditional Third Update

```python
errG      = g_loss.eval({z: batch_z})
errD_fake = d_loss_fake.eval({z: batch_z})
errD_real = d_loss_real.eval({inputs: batch_images})

if errG - (errD_fake + errD_real) > 1:
    sess.run([g_optim], feed_dict={z: batch_z})
```

This fires when the generator is losing by a large margin. The threshold of 1.0 was determined empirically for the Pokemon dataset.

---

## Loss Interpretation

| Scenario | d_loss | g_loss | Meaning |
|---|---|---|---|
| Early training | ~1.39 | ~0.69 | Both at random chance (ln 2 each) |
| Healthy training | 0.5–1.5 | 0.5–2.0 | Both learning, oscillating |
| D winning | ~0.0 | >3.0 | G stuck — check adaptive update |
| G winning (mode collapse) | >2.0 | ~0.1 | G found one trick, D can't catch up |
| Ideal equilibrium | ~0.69 | ~0.69 | Both at ln(2) |

---

## Sample Image Grid

Every 100 counter steps, a file is saved to `./samples/train_EE_IIII.png`:
- `EE`: zero-padded epoch number
- `IIII`: zero-padded batch index within epoch
- Contains an 8×8 grid of 64 generated 64×64 images
- Uses a **fixed** `sample_z` (same noise across all checkpoints) for consistent monitoring

---

## Checkpoint Behavior

Saved every 10 epochs to `./checkpoint/pokemon_64_64_64/`.

TF Saver keeps the **last 5 checkpoints** by default:
```
DCGAN.model-1950.*
DCGAN.model-1960.*
DCGAN.model-1970.*
DCGAN.model-1980.*
DCGAN.model-1990.*  ← latest (pointed to by `checkpoint` file)
```

The counter in the filename corresponds to the global `counter` variable (not epoch number). Counter increments once per batch.

**Resuming training:** The counter is extracted from the checkpoint filename via regex. Training resumes from where it left off.

---

## TensorBoard Monitoring

```bash
tensorboard --logdir=./logs/pokemon
# Open: http://localhost:6006
```

Available panels:
- **Scalars**: `d_loss`, `g_loss`, `d_loss_real`, `d_loss_fake` over time
- **Images**: `G` — the current generator output (updates during training)
- **Histograms**: `d`, `d_`, `z` — distribution of D scores and noise

---

## Expected Training Timeline

With a CPU (no GPU):
- ~1–2 minutes per epoch on ~800 images
- 2000 epochs ≈ 33–67 hours

With a GPU:
- ~5–15 seconds per epoch
- 2000 epochs ≈ 3–8 hours

The pre-trained checkpoint in the repo was trained to step 1990.

---

## Inference Command

```bash
python main.py --dataset pokemon
```

This:
1. Loads checkpoint from `./checkpoint/pokemon_64_64_64/`
2. Samples 64 random `z` vectors from `N(-1, 1)`
3. Runs through the trained generator (sampler)
4. Saves 8×8 grid to `./samples/test_YYYY-MM-DD-HH-MM-SS.png`

---

## Key Hyperparameter Effects

| Parameter | Effect of increasing | Effect of decreasing |
|---|---|---|
| `learning_rate` | Faster learning, less stable | Slower, more stable |
| `beta1` | More momentum, may overshoot | Less momentum, slower |
| `batch_size` | More stable gradients | Noisier, faster per iter |
| `z_dim` | More capacity, slower | Less capacity, faster |
| `gf_dim`/`df_dim` | More expressive, more VRAM | Less expressive, faster |
| `epoch` | Better convergence | May underfit |
