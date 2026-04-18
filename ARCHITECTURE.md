# Architecture Reference

WGAN-GP architecture for 64×64 Pokémon sprite generation.

---

## 1. System Overview

```
                       POOKIEMON WGAN-GP
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  DATA PREP                  TRAINING                    INFERENCE        │
│  ─────────────              ────────────────────        ──────────────   │
│  Raw PNGs (RGBA)            main.py                     main.py          │
│       ↓                         ↓                           ↓            │
│  augmentation.py            WGAN.train()               WGAN.load()       │
│       ↓                     /        \                      ↓            │
│  data/pokemon/           Generator  Critic             Generator         │
│  64×64 JPEGs                 ↓          ↓                   ↓            │
│                           samples/  checkpoint/        test_*.png        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Generator

```
INPUT: z [B, 100]  sampled from Normal(-1, 1)

  z [B, 100]
      │
  Linear       →  [B, 8192]          (gf*8 × 4 × 4 = 512 × 16)
      │
  Reshape      →  [B, 512, 4, 4]
      │
  BN + LReLU(0.2)
      │
  Deconv 5×5/s2  →  [B, 256, 8, 8]    BN + LReLU
  Deconv 5×5/s2  →  [B, 128, 16, 16]  BN + LReLU
  Deconv 5×5/s2  →  [B,  64, 32, 32]  BN + LReLU
  Deconv 5×5/s2  →  [B,   3, 64, 64]  (no BN on final layer)
      │
  Tanh  →  output range [-1, 1]

OUTPUT: fake image [B, 3, 64, 64]
```

BatchNorm is applied to all Generator hidden layers. The final deconv layer has no BN (standard practice — avoids colour shift on output).

---

## 3. Critic

```
INPUT: image [B, 3, 64, 64]  range [-1, 1]

  image [B, 3, 64, 64]
      │
  Conv 5×5/s2  →  [B,  64, 32, 32]  LReLU  (no BN)
  Conv 5×5/s2  →  [B, 128, 16, 16]  LReLU  (no BN)
  Conv 5×5/s2  →  [B, 256,  8,  8]  LReLU  (no BN)
  Conv 5×5/s2  →  [B, 512,  4,  4]  LReLU  (no BN)
      │
  Flatten      →  [B, 8192]
  Linear       →  [B, 1]

OUTPUT: unbounded score [B, 1]  (higher = more real; no sigmoid)
```

**No BatchNorm in any Critic layer.** BN creates correlation between samples in the same batch — this corrupts the gradient penalty's Lipschitz enforcement, which requires treating each sample independently.

---

## 4. WGAN-GP Training Flow

```
for each batch:

  ┌── Critic update × n_critic (default 5) ─────────────────────────────┐
  │                                                                      │
  │  z ~ Normal(-1, 1)                                                   │
  │  fake = G(z).detach()        ← stop gradients from reaching G        │
  │                                                                      │
  │  c_real = C(real)            ← critic score on real images           │
  │  c_fake = C(fake)            ← critic score on fake images           │
  │                                                                      │
  │  ε ~ Uniform(0,1)                                                    │
  │  x̂ = ε·real + (1−ε)·fake    ← random interpolation                 │
  │  GP = E[(||∇_x̂ C(x̂)||₂ − 1)²]   ← gradient penalty               │
  │                                                                      │
  │  c_loss = mean(c_fake) − mean(c_real) + λ·GP                        │
  │  optimizer_C.step()                                                  │
  └──────────────────────────────────────────────────────────────────────┘

  ┌── Generator update × 1 ────────────────────────────────────────────┐
  │                                                                     │
  │  z ~ Normal(-1, 1)                                                  │
  │  fake = G(z)                                                        │
  │  g_loss = −mean(C(fake))     ← G wants critic scores to be high    │
  │  optimizer_G.step()                                                 │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Loss Functions

```
CRITIC LOSS  (minimise: accurate Wasserstein scoring + Lipschitz constraint)

  c_loss = E[C(G(z))] − E[C(x)] + λ · GP

  where:
    GP  = E[(||∇_x̂ C(x̂)||₂ − 1)²]     (gradient penalty, λ=10)
    x̂   = ε·x + (1−ε)·G(z)             (random interpolation, ε~U(0,1))

  Wasserstein distance estimate:
    W ≈ E[C(x)] − E[C(G(z))]
    This increases as the generator improves — use it as the quality metric.


GENERATOR LOSS  (minimise: fool the critic)

  g_loss = −E[C(G(z))]

  G wants critic scores on fake images to be as high (real-looking) as possible.


WHY WASSERSTEIN DISTANCE:

  Unlike BCE, the Wasserstein distance is defined even when the real and
  generated distributions don't overlap. This means gradients don't vanish
  when the critic is highly accurate — G always gets a useful training signal.
```

---

## 6. Image Normalisation Pipeline

```
Raw JPEG [0, 255]
    ↓  / 127.5 − 1.0
float32 [-1, 1]   ← training input and generator output range (tanh)
    ↓  (x + 1) / 2
float32 [0, 1]    ← grid assembly
    ↓  × 255, clip, cast
uint8 PNG          ← saved to disk
```

---

## 7. Checkpoint Format

```
checkpoint/
└── pokemon_64_64_64/
    ├── WGAN.model-1810.pt
    ├── WGAN.model-1991.pt
    ├── WGAN.model-2172.pt
    ├── WGAN.model-2353.pt
    └── WGAN.model-2534.pt   ← latest (loaded on resume)

Each .pt file contains:
  {
    "step":      int,           ← global batch step counter
    "generator": state_dict,
    "critic":    state_dict,
  }

Last 5 checkpoints are retained. Loading always selects the highest step number.
```

---

## 8. Filter Progression

```
GENERATOR (upsampling):
  z      →   4×4  →   8×8  →  16×16 →  32×32 →  64×64
  100ch    512ch    256ch     128ch     64ch      3ch

CRITIC (downsampling):
  64×64  →  32×32 →  16×16 →   8×8  →   4×4  →  score
   3ch      64ch     128ch    256ch    512ch       1

Filter counts double (critic) / halve (generator) at each stride-2 step.
gf_dim = df_dim = 64 is the base filter count.
```

---

## 9. Module Dependencies

```
main.py
  ├── model.py  (WGAN, Generator, Critic)
  └── utils.py  (show_all_variables, visualize)

model.py
  ├── ops.py    (conv2d, deconv2d, linear, batch_norm, lrelu)
  └── utils.py  (get_image, save_images, image_manifold_size)

ops.py          (no project imports)
utils.py        (no project imports)
augmentation.py (standalone — no project imports)
```
