# Pokemon Master — Architecture Reference

Complete architectural diagrams and data flow for the DCGAN Pokemon generator.

---

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POKEMON MASTER DCGAN SYSTEM                         │
│                                                                               │
│  ┌──────────────────────┐              ┌──────────────────────────────────┐  │
│  │   DATA PREPARATION   │              │          TRAINING SYSTEM          │  │
│  │                      │              │                                  │  │
│  │  Raw PNGs (RGBA)     │              │  main.py ──► DCGAN.__init__()    │  │
│  │       │              │   ./data/    │                  │               │  │
│  │  augmentation.ipynb  │──pokemon/*.jpg─►  DCGAN.build_model()          │  │
│  │       │              │              │                  │               │  │
│  │  RGBA→RGB, resize,   │              │   Generator ◄──► Discriminator  │  │
│  │  flip, rotate 14x    │              │                  │               │  │
│  └──────────────────────┘              │  DCGAN.train()   │               │  │
│                                        │       │          │               │  │
│  ┌──────────────────────┐              │   ./samples/  ./checkpoint/      │  │
│  │      INFERENCE       │              │   ./logs/                        │  │
│  │                      │              └──────────────────────────────────┘  │
│  │  z ~ N(-1,1)^100     │                                                    │
│  │       │              │                                                    │
│  │  DCGAN.sampler()     │                                                    │
│  │       │              │                                                    │
│  │  64×64 RGB image     │                                                    │
│  └──────────────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Generator Architecture

```
INPUT: z vector [batch_size=64, z_dim=100]
Sampled from: Normal distribution N(-1, 1)

┌─────────────────────────────────────────────────────────────┐
│  z [64, 100]                                                │
│       │                                                     │
│  ┌────▼─────────────────────────────────────────────────┐  │
│  │  LINEAR (g_h0_lin)                                   │  │
│  │  [64, 100] → [64, 8192]    (512×4×4 = 8192)         │  │
│  │  init: random_normal(stddev=0.02)                    │  │
│  └────┬─────────────────────────────────────────────────┘  │
│       │                                                     │
│  ┌────▼──────────────────────┐                             │
│  │  RESHAPE                  │                             │
│  │  [64, 8192] → [64,4,4,512]│                             │
│  └────┬──────────────────────┘                             │
│       │                                                     │
│  ┌────▼──────────────────────┐                             │
│  │  BATCH NORM (g_bn0)       │                             │
│  │  epsilon=1e-5, momentum=0.9, scale=True                │
│  └────┬──────────────────────┘                             │
│       │                                                     │
│  ┌────▼──────────────────────┐                             │
│  │  LEAKY RELU (slope=0.2)   │  ← NOT standard ReLU       │
│  │  output: [64, 4, 4, 512]  │                             │
│  └────┬──────────────────────┘                             │
│       │                                                     │
│  ┌────▼──────────────────────────────────────────────────┐ │
│  │  DECONV2D (g_h1)                                      │ │
│  │  kernel: 5×5, stride: 2, padding: SAME               │ │
│  │  filters: 256  (gf_dim×4)                            │ │
│  │  init: random_normal(stddev=0.02)                     │ │
│  │  [64,4,4,512] → [64,8,8,256]                         │ │
│  └────┬──────────────────────────────────────────────────┘ │
│       │  BatchNorm(g_bn1) → LeakyReLU                      │
│       │                                                     │
│  ┌────▼──────────────────────────────────────────────────┐ │
│  │  DECONV2D (g_h2)                                      │ │
│  │  filters: 128  (gf_dim×2)                            │ │
│  │  [64,8,8,256] → [64,16,16,128]                       │ │
│  └────┬──────────────────────────────────────────────────┘ │
│       │  BatchNorm(g_bn2) → LeakyReLU                      │
│       │                                                     │
│  ┌────▼──────────────────────────────────────────────────┐ │
│  │  DECONV2D (g_h3)                                      │ │
│  │  filters: 64  (gf_dim×1)                             │ │
│  │  [64,16,16,128] → [64,32,32,64]                      │ │
│  └────┬──────────────────────────────────────────────────┘ │
│       │  BatchNorm(g_bn3) → LeakyReLU                      │
│       │                                                     │
│  ┌────▼──────────────────────────────────────────────────┐ │
│  │  DECONV2D (g_h4)                                      │ │
│  │  filters: 3  (c_dim = RGB channels)                  │ │
│  │  [64,32,32,64] → [64,64,64,3]                        │ │
│  └────┬──────────────────────────────────────────────────┘ │
│       │  NO BN on final layer                              │
│       │                                                     │
│  ┌────▼──────────────────────┐                             │
│  │  TANH activation          │  output range: [-1, 1]     │
│  │  [64, 64, 64, 3]          │                             │
│  └───────────────────────────┘                             │
└─────────────────────────────────────────────────────────────┘

OUTPUT: Fake image [64, 64, 64, 3] in range [-1, 1]
```

---

## 3. Discriminator Architecture

```
INPUT: Image [batch_size=64, 64, 64, 3] (real or fake)
Range: [-1, 1] (normalized)

┌─────────────────────────────────────────────────────────────┐
│  image [64, 64, 64, 3]                                      │
│       │                                                     │
│  ┌────▼──────────────────────────────────────────────────┐ │
│  │  CONV2D (d_h0_conv)                                   │ │
│  │  kernel: 5×5, stride: 2, padding: SAME               │ │
│  │  filters: 64  (df_dim)                               │ │
│  │  init: truncated_normal(stddev=0.02)                  │ │
│  │  [64,64,64,3] → [64,32,32,64]                        │ │
│  └────┬──────────────────────────────────────────────────┘ │
│       │  NO BatchNorm (DCGAN rule: no BN on first D layer) │
│       │                                                     │
│  ┌────▼──────────────────────┐                             │
│  │  LEAKY RELU (slope=0.2)   │                             │
│  │  output: [64, 32, 32, 64] │                             │
│  └────┬──────────────────────┘                             │
│       │                                                     │
│  ┌────▼──────────────────────────────────────────────────┐ │
│  │  CONV2D (d_h1_conv)                                   │ │
│  │  filters: 128  (df_dim×2)                            │ │
│  │  [64,32,32,64] → [64,16,16,128]                      │ │
│  └────┬──────────────────────────────────────────────────┘ │
│       │  BatchNorm(d_bn1) → LeakyReLU                      │
│       │                                                     │
│  ┌────▼──────────────────────────────────────────────────┐ │
│  │  CONV2D (d_h2_conv)                                   │ │
│  │  filters: 256  (df_dim×4)                            │ │
│  │  [64,16,16,128] → [64,8,8,256]                       │ │
│  └────┬──────────────────────────────────────────────────┘ │
│       │  BatchNorm(d_bn2) → LeakyReLU                      │
│       │                                                     │
│  ┌────▼──────────────────────────────────────────────────┐ │
│  │  CONV2D (d_h3_conv)                                   │ │
│  │  filters: 512  (df_dim×8)                            │ │
│  │  [64,8,8,256] → [64,4,4,512]                         │ │
│  └────┬──────────────────────────────────────────────────┘ │
│       │  BatchNorm(d_bn3) → LeakyReLU                      │
│       │                                                     │
│  ┌────▼──────────────────────┐                             │
│  │  RESHAPE                  │                             │
│  │  [64,4,4,512] → [64,8192] │                             │
│  └────┬──────────────────────┘                             │
│       │                                                     │
│  ┌────▼──────────────────────┐                             │
│  │  LINEAR (d_h4_lin)        │                             │
│  │  [64, 8192] → [64, 1]     │                             │
│  └────┬──────────────────────┘                             │
│       │                                                     │
│       ├──► D_logits [64, 1]   (raw, for loss computation)  │
│       │                                                     │
│  ┌────▼──────────────────────┐                             │
│  │  SIGMOID                  │                             │
│  │  output: [64, 1]          │  range: [0, 1]              │
│  └────┬──────────────────────┘                             │
│       │                                                     │
│       └──► D [64, 1]         (probability: 1=real, 0=fake) │
└─────────────────────────────────────────────────────────────┘

OUTPUT: (D: probability, D_logits: raw scores)
```

---

## 4. GAN Training Flow

```
                    TRAINING ITERATION
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  BATCH PREPARATION                                               │
│  ┌─────────────────────────────────────────────────┐            │
│  │  batch_images: load 64 real Pokemon images      │            │
│  │  normalize: pixel/127.5 - 1 → [-1, 1]          │            │
│  │  shape: [64, 64, 64, 3]                         │            │
│  └─────────────────────────────────────────────────┘            │
│  ┌─────────────────────────────────────────────────┐            │
│  │  batch_z: sample 64 noise vectors               │            │
│  │  distribution: Normal(-1, 1)                    │            │
│  │  shape: [64, 100]                               │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  STEP 1: UPDATE DISCRIMINATOR (once)                             │
│  ┌─────────────────────────────────────────────────┐            │
│  │  run: d_optim                                   │            │
│  │  feed: {inputs: batch_images, z: batch_z}       │            │
│  │                                                 │            │
│  │  D(real) → D_logits  → d_loss_real              │            │
│  │  G(z) → fake → D(fake) → D_logits_ → d_loss_fake│           │
│  │  d_loss = d_loss_real + d_loss_fake             │            │
│  │                                                 │            │
│  │  Adam(lr=0.0002, β1=0.5) updates d_vars only   │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  STEP 2: UPDATE GENERATOR (twice always)                         │
│  ┌─────────────────────────────────────────────────┐            │
│  │  run: g_optim  [1st time]                       │            │
│  │  feed: {z: batch_z}                             │            │
│  │                                                 │            │
│  │  G(z) → fake → D(fake) → D_logits_             │            │
│  │  g_loss = cross_entropy(D_logits_, ones)        │            │
│  │  (G wants D to think fake is real)              │            │
│  │                                                 │            │
│  │  run: g_optim  [2nd time — always]              │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  STEP 3: COMPUTE LOSSES                                          │
│  ┌─────────────────────────────────────────────────┐            │
│  │  errD_fake = d_loss_fake.eval({z: batch_z})     │            │
│  │  errD_real = d_loss_real.eval({inputs: imgs})   │            │
│  │  errG      = g_loss.eval({z: batch_z})          │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  STEP 4: CONDITIONAL THIRD G UPDATE                              │
│  ┌─────────────────────────────────────────────────┐            │
│  │  if errG - (errD_fake + errD_real) > 1:         │            │
│  │      run: g_optim  [3rd time]                   │            │
│  │                                                 │            │
│  │  Purpose: if G is losing badly, give it extra   │            │
│  │  gradient updates to catch up with D            │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                  │
│  STEP 5: PERIODIC ACTIONS                                        │
│  ┌─────────────────────────────────────────────────┐            │
│  │  every 100 counter steps:                       │            │
│  │    → run sampler → save 8×8 grid PNG            │            │
│  │    → filename: samples/train_EE_IIII.png        │            │
│  │                                                 │            │
│  │  every 10 epochs:                               │            │
│  │    → save checkpoint                            │            │
│  │    → filename: checkpoint/pokemon_64_64_64/     │            │
│  │               DCGAN.model-{counter}             │            │
│  └─────────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Loss Functions

```
DISCRIMINATOR LOSS (minimize: D correct on real AND fake)

  d_loss_real = mean( BCE(D(real),  target=1) )
                                              ↑ D should output 1 for real
  d_loss_fake = mean( BCE(D(G(z)),  target=0) )
                                              ↑ D should output 0 for fake
  d_loss = d_loss_real + d_loss_fake

  BCE(logits, labels) = sigmoid_cross_entropy_with_logits(logits, labels)


GENERATOR LOSS (minimize: D is fooled by fake images)

  g_loss = mean( BCE(D(G(z)),  target=1) )
                                        ↑ G wants D to output 1 for fake

EQUILIBRIUM (ideal training convergence):
  Both losses → ln(2) ≈ 0.693


ADAPTIVE TRAINING CONDITION:
  if errG - (errD_fake + errD_real) > 1:
      run extra G update step
  
  This fires when G is significantly losing to D.
  Effectively: if G_loss - D_loss > 1, give G extra help.
```

---

## 6. Variable Scoping and Weight Sharing

```
TensorFlow Variable Scopes:

  "generator/"                    "discriminator/"
  ├── g_h0_lin/Matrix             ├── d_h0_conv/w
  ├── g_h0_lin/bias               ├── d_h0_conv/biases
  ├── g_h1/w                      ├── d_h1_conv/w
  ├── g_h1/biases                 ├── d_h1_conv/biases
  ├── g_h2/w                      ├── d_h2_conv/w
  ├── g_h2/biases                 ├── d_h2_conv/biases
  ├── g_h3/w                      ├── d_h3_conv/w
  ├── g_h3/biases                 ├── d_h3_conv/biases
  ├── g_h4/w                      ├── d_h4_lin/Matrix
  ├── g_h4/biases                 ├── d_h4_lin/bias
  ├── g_bn0/...                   ├── d_bn1/...
  ├── g_bn1/...                   ├── d_bn2/...
  ├── g_bn2/...                   └── d_bn3/...
  └── g_bn3/...

VARIABLE SELECTION:
  d_vars = [v for v in tf.trainable_variables() if 'd_' in v.name]
  g_vars = [v for v in tf.trainable_variables() if 'g_' in v.name]

REUSE IN SAMPLER:
  with tf.variable_scope("generator") as scope:
      scope.reuse_variables()   ← sampler() reuses ALL generator weights
      # train=False passed to all batch_norm calls
```

---

## 7. Image Normalization Pipeline

```
RAW IMAGE (JPEG, uint8, 0-255)
         │
         ▼  scipy.misc.imread()
float64 array, shape [64, 64, 3], range [0, 255]
         │
         ▼  scipy.misc.imresize() or center_crop()
float64 array, shape [64, 64, 3], range [0, 255]
         │
         ▼  / 127.5 - 1.0
float64 array, shape [64, 64, 3], range [-1, 1]   ← TRAINING INPUT
         │
         ▼  [through Generator or loaded from checkpoint]
float32 tensor, shape [batch, 64, 64, 3], range [-1, 1]  ← GENERATOR OUTPUT
         │
         ▼  inverse_transform(): (x + 1) / 2
float32 array, shape [batch, 64, 64, 3], range [0, 1]
         │
         ▼  merge() → numpy array → scipy.misc.imsave()
uint8 PNG/JPG saved to disk, 8×8 grid of 64×64 images
```

---

## 8. Checkpoint File Structure

```
checkpoint/
└── pokemon_64_64_64/          ← named: {dataset}_{batch}_{h}_{w}
    ├── checkpoint             ← text file, pointer to latest checkpoint
    │                             content: model_checkpoint_path: "DCGAN.model-1990"
    ├── DCGAN.model-1950.meta   ← graph definition (TF MetaGraphDef protobuf)
    ├── DCGAN.model-1950.index  ← variable name→tensor mapping index
    ├── DCGAN.model-1950.data-00000-of-00001  ← actual weight values
    ├── DCGAN.model-1960.*
    ├── DCGAN.model-1970.*
    ├── DCGAN.model-1980.*
    └── DCGAN.model-1990.*     ← most recent, loaded by default

Loading:
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  self.saver.restore(sess, ckpt.model_checkpoint_path)
  counter extracted from filename via regex: (\d+)(?!.*\d)
```

---

## 9. Module Dependency Graph

```
main.py
  ├── imports: model.py (DCGAN class)
  ├── imports: utils.py (show_all_variables, visualize)
  └── uses: tensorflow (tf.app.flags, tf.Session, tf.ConfigProto)

model.py
  ├── imports: ops.py (batch_norm, conv2d, deconv2d, lrelu, linear)
  ├── imports: ops.py (image_summary, scalar_summary, histogram_summary, ...)
  ├── imports: utils.py (get_image, save_images, image_manifold_size)
  └── uses: tensorflow, numpy, scipy (via utils), glob, os, time, math, re

ops.py
  └── uses: tensorflow only (no project imports)

utils.py
  ├── imports: ops.py (image_summary — for visualize() only)
  └── uses: scipy.misc, numpy, math, os, datetime

augmentation.ipynb
  └── uses: PIL (Pillow), os (standalone, no project imports)
```

---

## 10. Filter Size Progression

```
GENERATOR (upsampling):
  z_dim → hidden →  4×4  →  8×8  → 16×16 → 32×32 → 64×64
   100     8192    512ch    256ch    128ch    64ch     3ch

DISCRIMINATOR (downsampling):
  64×64  → 32×32  → 16×16 →  8×8  →  4×4  → scalar
   3ch     64ch     128ch    256ch    512ch      1

The filter counts double/halve at each stage.
The spatial dimensions halve/double at each stride-2 step.
gf_dim = df_dim = 64 (base filter count).
```
