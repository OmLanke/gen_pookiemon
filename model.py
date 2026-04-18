"""
model.py — DCGAN Generator, Discriminator, and training harness (PyTorch)

Architecture is a faithful port of the blueprint:
  Generator  : z[100] → linear → 4×4×512 → 4× deconv → 64×64×3,  tanh out
  Discriminator: 64×64×3 → 4× conv → flatten → linear → 1,  sigmoid out

Key design decisions preserved (from BLUEPRINT.md §12):
  - Noise dist: Normal(-1, 1)  (NOT uniform)
  - Generator hidden activations: LeakyReLU(0.2) on ALL hidden layers
  - G updated 2× always + optional 3rd update if errG - errD > 1
  - BN: synchronous update (PyTorch default), scale=True (affine=True)
  - No BN on Discriminator's first conv layer
  - No BN on Generator's final deconv layer

Performance improvements over the TF1 original:
  - torch.utils.data.DataLoader with num_workers for parallel I/O
  - torch.amp (Automatic Mixed Precision) on CUDA / MPS
  - Apple Silicon MPS backend supported
  - Gradient scaler for AMP stability
  - torch.compile() opt-in (Python ≥ 3.12, PyTorch ≥ 2.0)
"""

from __future__ import annotations

import os
import re
import time
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
from tqdm import tqdm

from ops import conv2d, deconv2d, linear, batch_norm, lrelu, conv_out_size_same
from utils import get_image, save_images, image_manifold_size


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class PokemonDataset(Dataset):
    """
    Loads 64×64 JPEG images from data/{dataset_name}/.
    Returns float32 tensors in [-1, 1] with shape [C, H, W].
    """

    def __init__(
        self,
        data_dir: str,
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        crop: bool,
        grayscale: bool,
        fname_pattern: str = "*.jpg",
    ):
        self.files = sorted(glob(os.path.join(data_dir, fname_pattern)))
        if not self.files:
            raise FileNotFoundError(
                f"No images matching '{fname_pattern}' found in {data_dir!r}.\n"
                "Run augmentation.py first to build the dataset."
            )
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.crop = crop
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = get_image(
            self.files[idx],
            input_height=self.input_height,
            input_width=self.input_width,
            resize_height=self.output_height,
            resize_width=self.output_width,
            crop=self.crop,
            grayscale=self.grayscale,
        )
        # [H, W, C] or [H, W] → [C, H, W]
        tensor = torch.from_numpy(img)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # grayscale → [1, H, W]
        else:
            tensor = tensor.permute(2, 0, 1)  # [H, W, C] → [C, H, W]
        return tensor.float()


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────


class Generator(nn.Module):
    """
    Noise → Image upsampling network.

    z [B, 100]
    → linear  → [B, gf*8 * 4 * 4]
    → reshape  → [B, gf*8, 4, 4]
    → BN + LReLU
    → deconv   → [B, gf*4, 8, 8]   BN + LReLU
    → deconv   → [B, gf*2, 16, 16] BN + LReLU
    → deconv   → [B, gf*1, 32, 32] BN + LReLU
    → deconv   → [B, c_dim, 64, 64] tanh
    """

    def __init__(
        self,
        z_dim: int = 100,
        gf_dim: int = 64,
        c_dim: int = 3,
        output_height: int = 64,
        output_width: int = 64,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.c_dim = c_dim
        self.output_height = output_height
        self.output_width = output_width

        s_h16 = conv_out_size_same(
            conv_out_size_same(conv_out_size_same(conv_out_size_same(output_height)))
        )  # 4
        s_w16 = conv_out_size_same(
            conv_out_size_same(conv_out_size_same(conv_out_size_same(output_width)))
        )  # 4
        self._s_h16 = s_h16
        self._s_w16 = s_w16

        self.h0_lin = linear(z_dim, gf_dim * 8 * s_h16 * s_w16)

        # hidden: BN + LReLU
        self.bn0 = batch_norm(gf_dim * 8)
        self.h1 = deconv2d(gf_dim * 8, gf_dim * 4)
        self.bn1 = batch_norm(gf_dim * 4)
        self.h2 = deconv2d(gf_dim * 4, gf_dim * 2)
        self.bn2 = batch_norm(gf_dim * 2)
        self.h3 = deconv2d(gf_dim * 2, gf_dim * 1)
        self.bn3 = batch_norm(gf_dim * 1)
        # final: no BN
        self.h4 = deconv2d(gf_dim * 1, c_dim)

        self.act = lrelu(0.2)

    def forward(self, z: torch.Tensor, train: bool = True) -> torch.Tensor:
        # Set BN to train/eval mode explicitly (used by sampler)
        for mod in [self.bn0, self.bn1, self.bn2, self.bn3]:
            mod.training = train

        x = self.h0_lin(z)
        x = x.view(-1, self.gf_dim * 8, self._s_h16, self._s_w16)
        x = self.act(self.bn0(x))

        x = self.act(self.bn1(self.h1(x)))
        x = self.act(self.bn2(self.h2(x)))
        x = self.act(self.bn3(self.h3(x)))
        x = torch.tanh(self.h4(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Discriminator
# ─────────────────────────────────────────────────────────────────────────────


class Discriminator(nn.Module):
    """
    Image → real/fake probability.

    [B, c_dim, 64, 64]
    → conv   → [B, df,   32, 32]  LReLU (NO BN — DCGAN rule)
    → conv   → [B, df*2, 16, 16]  BN + LReLU
    → conv   → [B, df*4,  8,  8]  BN + LReLU
    → conv   → [B, df*8,  4,  4]  BN + LReLU
    → flatten → linear → [B, 1] → sigmoid
    Returns: (sigmoid_output, logits)
    """

    def __init__(
        self,
        df_dim: int = 64,
        c_dim: int = 3,
        input_height: int = 64,
        input_width: int = 64,
    ):
        super().__init__()
        self.df_dim = df_dim

        s4 = conv_out_size_same(
            conv_out_size_same(conv_out_size_same(conv_out_size_same(input_height)))
        )  # 4
        s4w = conv_out_size_same(
            conv_out_size_same(conv_out_size_same(conv_out_size_same(input_width)))
        )  # 4
        flat_dim = df_dim * 8 * s4 * s4w  # 512*4*4 = 8192

        # Layer 0: NO BN
        self.h0 = conv2d(c_dim, df_dim)
        # Layers 1-3: BN
        self.h1 = conv2d(df_dim, df_dim * 2)
        self.h2 = conv2d(df_dim * 2, df_dim * 4)
        self.h3 = conv2d(df_dim * 4, df_dim * 8)
        self.bn1 = batch_norm(df_dim * 2)
        self.bn2 = batch_norm(df_dim * 4)
        self.bn3 = batch_norm(df_dim * 8)

        self.h4 = linear(flat_dim, 1)
        self.act = lrelu(0.2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.h0(x))  # no BN
        x = self.act(self.bn1(self.h1(x)))
        x = self.act(self.bn2(self.h2(x)))
        x = self.act(self.bn3(self.h3(x)))
        x = x.view(x.size(0), -1)
        logits = self.h4(x)
        return torch.sigmoid(logits), logits


# ─────────────────────────────────────────────────────────────────────────────
# DCGAN training harness
# ─────────────────────────────────────────────────────────────────────────────


class DCGAN:
    """
    High-level wrapper that owns Generator, Discriminator, optimisers, and
    the training loop.

    Maps to the original DCGAN class in the blueprint; adapted for PyTorch.
    """

    def __init__(
        self,
        input_height: int = 64,
        input_width: int = 64,
        output_height: int = 64,
        output_width: int = 64,
        batch_size: int = 64,
        sample_num: int = 64,
        z_dim: int = 100,
        gf_dim: int = 64,
        df_dim: int = 64,
        c_dim: int = 3,
        dataset_name: str = "pokemon",
        input_fname_pattern: str = "*.jpg",
        crop: bool = False,
        checkpoint_dir: str = "checkpoint",
        sample_dir: str = "samples",
        compile_model: bool = False,
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim
        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.crop = crop
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sample_dir = Path(sample_dir)
        self.grayscale = c_dim == 1

        # ── Device selection: CUDA > MPS (Apple Silicon) > CPU ────────────────
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[*] Using device: {self.device}")

        # ── Models ────────────────────────────────────────────────────────────
        self.netG = Generator(
            z_dim=z_dim,
            gf_dim=gf_dim,
            c_dim=c_dim,
            output_height=output_height,
            output_width=output_width,
        ).to(self.device)

        self.netD = Discriminator(
            df_dim=df_dim,
            c_dim=c_dim,
            input_height=output_height,
            input_width=output_width,
        ).to(self.device)

        # Optional torch.compile for extra throughput (PyTorch ≥ 2.0)
        if compile_model and hasattr(torch, "compile"):
            print("[*] torch.compile() enabled")
            self.netG = torch.compile(self.netG)  # type: ignore[assignment]
            self.netD = torch.compile(self.netD)  # type: ignore[assignment]

        # ── AMP scaler (CUDA only; MPS uses float32 natively) ─────────────────
        self._use_amp = self.device.type == "cuda"
        self.scaler_g = torch.amp.GradScaler(enabled=self._use_amp)
        self.scaler_d = torch.amp.GradScaler(enabled=self._use_amp)

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def model_dir(self) -> str:
        return f"{self.dataset_name}_{self.batch_size}_{self.output_height}_{self.output_width}"

    def _sample_z(self, n: Optional[int] = None) -> torch.Tensor:
        """Sample noise z ~ N(-1, 1) on the correct device."""
        size = n or self.batch_size
        # mean=-1, std=1 reproduced as randn()*1 + (-1)
        return torch.randn(size, self.z_dim, device=self.device) - 1.0

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, config: object) -> None:
        """
        Main training loop faithful to blueprint §7.7 / TRAINING.md:
          - D updated once per batch
          - G updated twice always
          - G updated a third time if errG - (errD_fake + errD_real) > 1
          - Sample grid saved every 100 counter steps
          - Checkpoint saved every 10 epochs
        """
        data_dir = Path("./data") / self.dataset_name
        dataset = PokemonDataset(
            data_dir=str(data_dir),
            input_height=self.input_height,
            input_width=self.input_width,
            output_height=self.output_height,
            output_width=self.output_width,
            crop=self.crop,
            grayscale=self.grayscale,
            fname_pattern=self.input_fname_pattern,
        )
        print(f"[*] Dataset: {len(dataset)} images")

        # num_workers: use up to 4 workers; pin_memory on CUDA
        num_workers = min(4, os.cpu_count() or 1)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,  # keep batch sizes constant
            persistent_workers=(num_workers > 0),
        )

        # ── Optimisers (Adam, lr=0.0002, β1=0.5) ─────────────────────────────
        lr = getattr(config, "learning_rate", 0.0002)
        beta1 = getattr(config, "beta1", 0.5)
        d_optim = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        g_optim = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))

        # ── TensorBoard writer ────────────────────────────────────────────────
        log_dir = Path("./logs") / self.dataset_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

        # ── Fixed sample z for consistent monitoring ──────────────────────────
        sample_z = self._sample_z(self.sample_num)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # ── Resume from checkpoint if available ───────────────────────────────
        counter = 1
        could_load, checkpoint_counter = self.load(str(self.checkpoint_dir))
        if could_load:
            counter = checkpoint_counter
            print(f" [*] Resumed from step {counter}")
        else:
            print(" [!] No checkpoint found — training from scratch")

        bce = nn.BCEWithLogitsLoss()

        epochs = getattr(config, "epoch", 2000)
        start_time = time.time()

        for epoch in range(epochs):
            self.netG.train()
            self.netD.train()

            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

            for batch_images in pbar:
                batch_images = batch_images.to(self.device, non_blocking=True)
                batch_z = self._sample_z()

                real_labels = torch.ones(self.batch_size, 1, device=self.device)
                fake_labels = torch.zeros(self.batch_size, 1, device=self.device)

                # ── Update D once ─────────────────────────────────────────────
                self.netD.zero_grad(set_to_none=True)
                amp_ctx = torch.amp.autocast(
                    device_type=self.device.type, enabled=self._use_amp
                )

                with amp_ctx:
                    _, d_real_logits = self.netD(batch_images)
                    fake_imgs = self.netG(batch_z)
                    _, d_fake_logits = self.netD(fake_imgs.detach())
                    d_loss_real = bce(d_real_logits, real_labels)
                    d_loss_fake = bce(d_fake_logits, fake_labels)
                    d_loss = d_loss_real + d_loss_fake

                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.step(d_optim)
                self.scaler_d.update()

                # ── Update G twice (always) ───────────────────────────────────
                for _ in range(2):
                    self.netG.zero_grad(set_to_none=True)
                    batch_z_g = self._sample_z()
                    with amp_ctx:
                        fake_imgs_g = self.netG(batch_z_g)
                        _, d_fake_logits_g = self.netD(fake_imgs_g)
                        g_loss = bce(d_fake_logits_g, real_labels)
                    self.scaler_g.scale(g_loss).backward()
                    self.scaler_g.step(g_optim)
                    self.scaler_g.update()

                # ── Compute losses for logging / conditional update ────────────
                errD_fake = d_loss_fake.item()
                errD_real = d_loss_real.item()
                errG = g_loss.item()

                # ── Conditional third G update ────────────────────────────────
                if errG - (errD_fake + errD_real) > 1.0:
                    self.netG.zero_grad(set_to_none=True)
                    batch_z_extra = self._sample_z()
                    with amp_ctx:
                        fake_extra = self.netG(batch_z_extra)
                        _, d_fake_extra = self.netD(fake_extra)
                        g_loss_extra = bce(d_fake_extra, real_labels)
                    self.scaler_g.scale(g_loss_extra).backward()
                    self.scaler_g.step(g_optim)
                    self.scaler_g.update()

                # ── TensorBoard logging ───────────────────────────────────────
                writer.add_scalar("Loss/D", errD_real + errD_fake, counter)
                writer.add_scalar("Loss/G", errG, counter)
                writer.add_scalar("Loss/D_real", errD_real, counter)
                writer.add_scalar("Loss/D_fake", errD_fake, counter)

                pbar.set_postfix(
                    {
                        "d_loss": f"{errD_real + errD_fake:.4f}",
                        "g_loss": f"{errG:.4f}",
                        "t": f"{time.time() - start_time:.0f}s",
                    }
                )

                # ── Save sample grid every 100 steps ──────────────────────────
                if counter % 100 == 1:
                    self._save_sample_grid(sample_z, epoch, counter, writer)

                counter += 1

            # ── Checkpoint every 10 epochs ────────────────────────────────────
            if (epoch + 1) % 10 == 0:
                self.save(str(self.checkpoint_dir), counter)
                print(f"\n[*] Checkpoint saved at epoch {epoch + 1}, step {counter}")

        writer.close()
        print(f"\n[*] Training complete — {counter - 1} total steps")

    @torch.no_grad()
    def _save_sample_grid(
        self,
        sample_z: torch.Tensor,
        epoch: int,
        counter: int,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        self.netG.eval()
        samples = self.netG(sample_z, train=False)
        self.netG.train()

        # [B, C, H, W] → [B, H, W, C]
        samples_np = samples.cpu().permute(0, 2, 3, 1).numpy()
        rows, cols = image_manifold_size(samples_np.shape[0])

        out_path = self.sample_dir / f"train_{epoch:02d}_{counter:06d}.png"
        save_images(samples_np, [rows, cols], out_path)

        # Push to TensorBoard as image grid
        if writer is not None:
            # torchvision expects [B, C, H, W] in [0, 1]
            grid_tensor = samples.clamp(-1, 1) * 0.5 + 0.5
            try:
                from torchvision.utils import make_grid

                grid = make_grid(grid_tensor, nrow=cols)
                writer.add_image("Generated/samples", grid, counter)
            except ImportError:
                pass

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, checkpoint_dir: str, step: int) -> None:
        save_dir = Path(checkpoint_dir) / self.model_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"DCGAN.model-{step}.pt"
        torch.save(
            {
                "step": step,
                "generator": self.netG.state_dict(),
                "discriminator": self.netD.state_dict(),
            },
            path,
        )
        # Keep only last 5 checkpoints
        ckpts = sorted(
            save_dir.glob("DCGAN.model-*.pt"),
            key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
        )
        for old in ckpts[:-5]:
            old.unlink()

    def load(self, checkpoint_dir: str) -> tuple[bool, int]:
        ckpt_dir = Path(checkpoint_dir) / self.model_dir
        print(f" [*] Reading checkpoints from {ckpt_dir}")

        if not ckpt_dir.exists():
            return False, 0

        ckpts = sorted(
            ckpt_dir.glob("DCGAN.model-*.pt"),
            key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
        )
        if not ckpts:
            return False, 0

        latest = ckpts[-1]
        state = torch.load(latest, map_location=self.device, weights_only=True)
        self.netG.load_state_dict(state["generator"])
        self.netD.load_state_dict(state["discriminator"])
        step = state.get("step", 0)
        print(f" [*] Loaded checkpoint: {latest.name}  (step {step})")
        return True, step
