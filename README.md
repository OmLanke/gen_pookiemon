# Pokimon

A Deep Convolutional Generative Adversarial Network (DCGAN) that generates novel 64×64 Pokémon sprites from random noise.

```
z ~ N(-1, 1)^100  →  Generator  →  64×64 RGB image
```

---

## Examples

> Train the model and sample images appear in `samples/` every 100 steps.

```
samples/
├── train_00_000001.png   ← 8×8 grid after step 1
├── train_10_002600.png   ← 8×8 grid at epoch 10
└── test_2024-06-01-14-32-00.png  ← inference output
```

---

## Architecture

```
GENERATOR                              DISCRIMINATOR

z [64, 100]                            image [64, 64×64×3]
    │                                      │
 Linear                                 Conv 5×5/2 → LeakyReLU      (no BN)
    │                                      │
 Reshape [64, 4×4×512]                 Conv 5×5/2 → BN → LeakyReLU
    │                                      │
 BN → LeakyReLU                        Conv 5×5/2 → BN → LeakyReLU
    │                                      │
 Deconv → [8×8×256]   BN + LReLU      Conv 5×5/2 → BN → LeakyReLU
 Deconv → [16×16×128] BN + LReLU          │
 Deconv → [32×32×64]  BN + LReLU       Linear → 1
 Deconv → [64×64×3]                        │
    │                                   Sigmoid
  Tanh                                      │
    │                               (prob, logits)
 fake image [-1, 1]
```

| Hyperparameter | Default | Notes |
|---|---|---|
| Noise vector `z_dim` | 100 | — |
| Noise distribution | `Normal(-1, 1)` | — |
| Generator filters `gf_dim` | **32** | Use `--full` or `--gf_dim 64` for full quality |
| Discriminator filters `df_dim` | **32** | Use `--full` or `--df_dim 64` for full quality |
| Hidden activations | LeakyReLU(0.2) — both G and D | — |
| Output activation | Tanh (G), Sigmoid (D) | — |
| Batch size | 64 | — |
| Epochs | **500** | Use `--full` or `--epoch 2000` for full convergence |
| Learning rate | 0.0002 | — |
| Adam β1 | 0.5 | — |
| G updates per batch | 2 always + 1 conditional | — |

---

## Requirements

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) — used for all dependency and environment management

No manual `pip install` needed. `uv sync` handles everything.

---

## Setup

```bash
git clone https://github.com/yourname/pokimon
cd pokimon
uv sync --all-groups
```

This creates `.venv/` and installs PyTorch, Pillow, tqdm, torchvision, and TensorBoard automatically.

---

## Dataset

The model trains on Pokémon sprite images (Gen 1–6, ~830 raw PNGs). You need RGBA sprites with transparent backgrounds — the augmentation pipeline handles the rest.

**Recommended source:** [veekun sprite downloads](https://veekun.com/dex/downloads)

Place raw PNGs in a folder (e.g. `pokemon_sugimori_ori/`), then run the pipeline:

```bash
uv run python augmentation.py --input_dir pokemon_sugimori_ori
```

This produces ~11,600 training images in `data/pokemon/` via a 14× expansion:

| Transform | Count per image |
|---|---|
| Original (RGBA→RGB, 64×64) | 1 |
| Horizontal flip | 1 |
| Rotate ±3°, ±5°, ±7° (original) | 6 |
| Rotate ±3°, ±5°, ±7° (flip) | 6 |
| **Total** | **14** |

Augmentation runs in parallel across all CPU cores. For 830 images it completes in seconds.

---

## Training

```bash
uv run python main.py --dataset pokemon --train
```

All defaults are tuned for fast iteration on an M1 Mac (~9h for 500 epochs at gf/df=32). Common overrides:

```bash
# Full-quality run: gf_dim=64, df_dim=64, 2000 epochs (~80h on M1)
uv run python main.py --dataset pokemon --train --full

# Custom epoch count
uv run python main.py --dataset pokemon --train --epoch 1000

# Enable torch.compile for extra throughput (PyTorch ≥ 2.0)
uv run python main.py --dataset pokemon --train --compile
```

**What happens during training:**

- Sample grids saved to `samples/train_EE_NNNNNN.png` every 100 steps
- Checkpoints saved to `checkpoint/pokemon_64_64_64/` every 10 epochs (last 5 kept)
- TensorBoard logs written to `logs/pokemon/`

**Expected output:**

```
[*] Using device: mps
[*] Dataset: 11620 images
Epoch 1/500: 100%|████| 181/181 [d_loss: 0.8431, g_loss: 1.2047, ETA: 8h55m]
[*] Checkpoint saved at epoch 10, step 1811
...
```

Early training loss should hover near `ln(2) ≈ 0.693` for both G and D — that's the equilibrium of a random discriminator.

---

## Monitor with TensorBoard

```bash
uv run tensorboard --logdir logs/pokemon
# → http://localhost:6006
```

Available panels: `Loss/D`, `Loss/G`, `Loss/D_real`, `Loss/D_fake`, and a live image grid of generated samples.

---

## Inference

Generate a new 8×8 grid from a trained checkpoint:

```bash
uv run python main.py --dataset pokemon
```

Output saved to `samples/test_YYYY-MM-DD-HH-MM-SS.png`.

---

## Project Structure

```
pokimon/
├── main.py            ← CLI entry point (argparse)
├── model.py           ← Generator, Discriminator, DCGAN trainer
├── ops.py             ← Layer primitives (conv2d, deconv2d, linear, lrelu, batch_norm)
├── utils.py           ← Image I/O (Pillow), grid assembly, visualize
├── augmentation.py    ← Dataset preprocessing pipeline
├── pyproject.toml     ← uv/setuptools project config
│
├── data/
│   └── pokemon/       ← 64×64 JPEG training images (generated by augmentation.py)
├── checkpoint/
│   └── pokemon_64_64_64/  ← Saved model weights (.pt files)
├── samples/           ← Generated image grids
└── logs/
    └── pokemon/       ← TensorBoard event files
```

---

## All CLI Options

```
uv run python main.py [options]

Training:
  --train                  Enable training mode
  --epoch INT              Training epochs            (default: 500)
  --learning_rate FLOAT    Adam learning rate         (default: 0.0002)
  --beta1 FLOAT            Adam β1 momentum           (default: 0.5)
  --batch_size INT         Batch size                 (default: 64)
  --compile                Enable torch.compile()     (default: off)
  --full                   Full quality: gf/df=64, 2000 epochs

Architecture:
  --z_dim INT              Noise vector dimension     (default: 100)
  --gf_dim INT             Generator base filters     (default: 32)
  --df_dim INT             Discriminator base filters (default: 32)
  --c_dim INT              Image channels, 3=RGB      (default: 3)

Image size:
  --input_height INT       Input height               (default: 64)
  --output_height INT      Output height              (default: 64)

Dataset:
  --dataset STR            Folder under ./data/       (default: pokemon)
  --input_fname_pattern    Image glob pattern         (default: *.jpg)
  --crop                   Centre-crop instead of resize

Paths:
  --checkpoint_dir STR     Checkpoint directory       (default: checkpoint)
  --sample_dir STR         Sample output directory    (default: samples)
```

---

## Design Notes

A few intentional deviations from the original DCGAN paper, tuned for Pokémon sprites:

| Decision | Paper | This project | Why |
|---|---|---|---|
| Noise distribution | Uniform(-1, 1) | **Normal(-1, 1)** | Empirically better for this dataset |
| Generator activations | ReLU | **LeakyReLU(0.2)** | Better gradient flow through all G layers |
| G updates per batch | 1× | **2× always + 1× conditional** | Prevents discriminator from winning too fast |
| BN on first D layer | Varies | **None** | DCGAN paper recommendation |

The conditional third G update fires when `errG − (errD_fake + errD_real) > 1` — i.e. when the generator is falling significantly behind the discriminator.

---

## Performance

The implementation is optimised for Apple Silicon (M1/M2) as well as CUDA:

| Optimisation | Detail |
|---|---|
| Device | Auto-selects CUDA → Apple MPS → CPU |
| Data loading | Entire dataset pre-loaded into a single CPU tensor at startup — no per-epoch disk I/O (45× faster than PIL-per-batch) |
| Memory layout | `channels_last` (NHWC) memory format — conv/deconv are faster on Apple Silicon |
| G backward passes | Fused 2× z-batch in one forward+backward instead of two separate passes |
| Mixed precision | `torch.amp` AMP on CUDA |
| Augmentation | Multiprocess via `ProcessPoolExecutor` |
| Graph compilation | `torch.compile()` opt-in via `--compile` |

**Measured step times on M1 MacBook Air (MPS):**

| Config | ms/step | 500 epochs | 2000 epochs |
|---|---|---|---|
| gf=df=64 (full) | ~785–870 ms | ~20 h | ~80 h |
| gf=df=32 (default) | ~90–130 ms | ~4–5 h | ~18 h |

---

## License

MIT
