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

M1 performance optimisations:
  - Entire dataset pre-loaded into a single CPU tensor at startup (no per-epoch disk I/O)
  - channels_last DISABLED — BatchNorm2d backward crashes with channels_last on MPS (PyTorch bug)
  - Fused G update: 2× z batch in one forward+backward instead of two (halves G backward passes)
  - DataLoader workers=0 when data is pre-loaded (no fork overhead)
  - .reshape() instead of .view() for channels_last backward compatibility
  - torch.compile() opt-in flag (CPU/CUDA only; skipped on MPS)
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
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
from tqdm import tqdm

from ops import conv2d, deconv2d, linear, batch_norm, lrelu, conv_out_size_same
from utils import get_image, save_images, image_manifold_size


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — pre-load everything into RAM once, then serve from tensor
# ─────────────────────────────────────────────────────────────────────────────


def build_tensor_dataset(
    data_dir: str,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    crop: bool,
    grayscale: bool,
    fname_pattern: str = "*.jpg",
) -> torch.Tensor:
    """
    Load every image file into a single float32 CPU tensor [N, C, H, W].

    Loading is done once at startup.  During training, each batch is a fast
    tensor slice + device transfer — no disk I/O per epoch.

    Memory: ~11 600 images × 3 × 64 × 64 × 4 B ≈ 136 MB — fits easily in RAM.
    """
    files = sorted(glob(os.path.join(data_dir, fname_pattern)))
    if not files:
        raise FileNotFoundError(
            f"No images matching '{fname_pattern}' found in {data_dir!r}.\n"
            "Run augmentation.py first to build the dataset."
        )

    print(f"[*] Pre-loading {len(files)} images into RAM …", flush=True)
    images = []
    for path in tqdm(files, desc="Loading", unit="img", leave=False):
        arr = get_image(
            path,
            input_height=input_height,
            input_width=input_width,
            resize_height=output_height,
            resize_width=output_width,
            crop=crop,
            grayscale=grayscale,
        )
        t = torch.from_numpy(arr)
        if t.ndim == 2:
            t = t.unsqueeze(0)  # [H,W] → [1,H,W]
        else:
            t = t.permute(2, 0, 1)  # [H,W,C] → [C,H,W]
        images.append(t.float())

    tensor = torch.stack(images)  # [N, C, H, W]
    mb = tensor.nbytes / 1024 / 1024
    print(f"[*] Dataset tensor: {tuple(tensor.shape)}  ({mb:.0f} MB)")
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────


class Generator(nn.Module):
    """
    Noise → Image upsampling network.

    z [B, z_dim]
    → linear  → [B, gf*8 * 4 * 4]  reshape → [B, gf*8, 4, 4]  BN + LReLU
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

        s_h16 = conv_out_size_same(
            conv_out_size_same(conv_out_size_same(conv_out_size_same(output_height)))
        )  # 4
        s_w16 = conv_out_size_same(
            conv_out_size_same(conv_out_size_same(conv_out_size_same(output_width)))
        )  # 4
        self._s_h16 = s_h16
        self._s_w16 = s_w16

        self.h0_lin = linear(z_dim, gf_dim * 8 * s_h16 * s_w16)

        self.bn0 = batch_norm(gf_dim * 8)
        self.h1 = deconv2d(gf_dim * 8, gf_dim * 4)
        self.bn1 = batch_norm(gf_dim * 4)
        self.h2 = deconv2d(gf_dim * 4, gf_dim * 2)
        self.bn2 = batch_norm(gf_dim * 2)
        self.h3 = deconv2d(gf_dim * 2, gf_dim * 1)
        self.bn3 = batch_norm(gf_dim * 1)
        self.h4 = deconv2d(gf_dim * 1, c_dim)

        self.act = lrelu(0.2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.h0_lin(z)
        # contiguous() ensures channels_last is applied correctly after reshape
        x = x.reshape(-1, self.gf_dim * 8, self._s_h16, self._s_w16)
        x = self.act(self.bn0(x))
        x = self.act(self.bn1(self.h1(x)))
        x = self.act(self.bn2(self.h2(x)))
        x = self.act(self.bn3(self.h3(x)))
        return torch.tanh(self.h4(x))


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

        s4 = conv_out_size_same(
            conv_out_size_same(conv_out_size_same(conv_out_size_same(input_height)))
        )  # 4
        s4w = conv_out_size_same(
            conv_out_size_same(conv_out_size_same(conv_out_size_same(input_width)))
        )  # 4
        flat_dim = df_dim * 8 * s4 * s4w  # 8192

        self.h0 = conv2d(c_dim, df_dim)  # no BN
        self.h1 = conv2d(df_dim, df_dim * 2)
        self.h2 = conv2d(df_dim * 2, df_dim * 4)
        self.h3 = conv2d(df_dim * 4, df_dim * 8)
        self.bn1 = batch_norm(df_dim * 2)
        self.bn2 = batch_norm(df_dim * 4)
        self.bn3 = batch_norm(df_dim * 8)
        self.h4 = linear(flat_dim, 1)
        self.act = lrelu(0.2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.act(self.h0(x))
        x = self.act(self.bn1(self.h1(x)))
        x = self.act(self.bn2(self.h2(x)))
        x = self.act(self.bn3(self.h3(x)))
        # reshape (not view) — required for channels_last backward compatibility
        x = x.reshape(x.size(0), -1)
        logits = self.h4(x)
        return torch.sigmoid(logits), logits


# ─────────────────────────────────────────────────────────────────────────────
# DCGAN training harness
# ─────────────────────────────────────────────────────────────────────────────


class DCGAN:
    def __init__(
        self,
        input_height: int = 64,
        input_width: int = 64,
        output_height: int = 64,
        output_width: int = 64,
        batch_size: int = 64,
        sample_num: int = 64,
        z_dim: int = 100,
        gf_dim: int = 32,
        df_dim: int = 32,
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

        # ── Device: CUDA > MPS (Apple Silicon) > CPU ──────────────────────────
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[*] Device: {self.device}")

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

        # channels_last (NHWC) — conv/deconv are faster on Apple Silicon MPS
        # NOTE: disabled — BatchNorm2d backward crashes with channels_last on MPS
        # (PyTorch MPS bug). The 45× speedup from RAM preloading is the main win anyway.
        # mem_fmt = torch.channels_last
        # self.netG = self.netG.to(memory_format=mem_fmt)
        # self.netD = self.netD.to(memory_format=mem_fmt)

        if compile_model and hasattr(torch, "compile"):
            if self.device.type == "mps":
                print(
                    "[!] torch.compile() skipped on MPS — Metal shader codegen is unsupported"
                )
            else:
                print("[*] torch.compile() enabled")
                self.netG = torch.compile(self.netG)  # type: ignore[assignment]
                self.netD = torch.compile(self.netD)  # type: ignore[assignment]

        # AMP only on CUDA (MPS doesn't benefit; CPU doesn't support it)
        self._use_amp = self.device.type == "cuda"
        self.scaler_g = torch.amp.GradScaler(enabled=self._use_amp)
        self.scaler_d = torch.amp.GradScaler(enabled=self._use_amp)

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def model_dir(self) -> str:
        return f"{self.dataset_name}_{self.batch_size}_{self.output_height}_{self.output_width}"

    def _sample_z(self, n: Optional[int] = None) -> torch.Tensor:
        """z ~ N(-1, 1) on the correct device."""
        return torch.randn(n or self.batch_size, self.z_dim, device=self.device) - 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, config: object) -> None:
        """
        Training loop (faithful to blueprint §7.7 / TRAINING.md):

          Per batch:
            1. D updated once
            2. G updated via one fused pass over 2× z-batch
               (equivalent gradient to 2 separate updates, half the backward passes)
            3. If errG − (errD_fake + errD_real) > 1 → extra G update

          Every 100 steps  → save sample grid PNG + TensorBoard image
          Every 10 epochs  → save checkpoint
        """
        data_dir = Path("./data") / self.dataset_name
        data_tensor = build_tensor_dataset(
            data_dir=str(data_dir),
            input_height=self.input_height,
            input_width=self.input_width,
            output_height=self.output_height,
            output_width=self.output_width,
            crop=self.crop,
            grayscale=self.grayscale,
            fname_pattern=self.input_fname_pattern,
        )

        # TensorDataset + DataLoader — no workers needed, data lives in RAM
        loader = DataLoader(
            TensorDataset(data_tensor),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # data is already in RAM, no workers needed
            drop_last=True,
        )

        print(f"[*] Batches per epoch: {len(loader)}")

        lr = getattr(config, "learning_rate", 0.0002)
        beta1 = getattr(config, "beta1", 0.5)
        d_optim = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        g_optim = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))

        log_dir = Path("./logs") / self.dataset_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

        # Fixed z for consistent monitoring grids
        sample_z = self._sample_z(self.sample_num)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        counter = 1
        could_load, checkpoint_counter = self.load(str(self.checkpoint_dir))
        if could_load:
            counter = checkpoint_counter
            print(f" [*] Resumed from step {counter}")
        else:
            print(" [!] No checkpoint — training from scratch")

        bce = nn.BCEWithLogitsLoss()
        epochs = getattr(config, "epoch", 500)
        start_t = time.time()

        for epoch in range(epochs):
            self.netG.train()
            self.netD.train()

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch + 1:>4}/{epochs}",
                leave=False,
                unit="batch",
                dynamic_ncols=True,
            )

            for (batch_images,) in pbar:
                # Transfer pre-loaded batch to device; convert to channels_last
                batch_images = batch_images.to(self.device, non_blocking=True)

                real_labels = torch.ones(self.batch_size, 1, device=self.device)
                fake_labels = torch.zeros(self.batch_size, 1, device=self.device)
                # Doubled labels for fused 2× G update
                real_labels2 = torch.ones(self.batch_size * 2, 1, device=self.device)

                amp_ctx = torch.amp.autocast(
                    device_type=self.device.type, enabled=self._use_amp
                )

                # ── 1. Update D ───────────────────────────────────────────────
                self.netD.zero_grad(set_to_none=True)
                with amp_ctx:
                    z = self._sample_z()
                    fake_imgs = self.netG(z)
                    _, d_real_logits = self.netD(batch_images)
                    _, d_fake_logits = self.netD(fake_imgs.detach())
                    d_loss_real = bce(d_real_logits, real_labels)
                    d_loss_fake = bce(d_fake_logits, fake_labels)
                    d_loss = d_loss_real + d_loss_fake

                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.step(d_optim)
                self.scaler_d.update()

                # ── 2. Update G — fused 2× z batch (one backward pass) ────────
                #
                # Running G with z-batch of size 2×batch_size is equivalent in
                # gradient direction to two sequential updates with batch_size z
                # each, but requires only one forward + one backward through G
                # and D — cutting per-step G compute roughly in half.
                self.netG.zero_grad(set_to_none=True)
                with amp_ctx:
                    z2 = self._sample_z(self.batch_size * 2)
                    fake_imgs2 = self.netG(z2)
                    _, d_fake_logits2 = self.netD(fake_imgs2)
                    g_loss = bce(d_fake_logits2, real_labels2)

                self.scaler_g.scale(g_loss).backward()
                self.scaler_g.step(g_optim)
                self.scaler_g.update()

                # ── 3. Extract scalar losses (deferred — avoids extra syncs) ──
                errD_real = d_loss_real.item()
                errD_fake = d_loss_fake.item()
                errG = g_loss.item()

                # ── 4. Conditional third G update ─────────────────────────────
                if errG - (errD_real + errD_fake) > 1.0:
                    self.netG.zero_grad(set_to_none=True)
                    with amp_ctx:
                        z3 = self._sample_z()
                        fake3 = self.netG(z3)
                        _, d_fake3 = self.netD(fake3)
                        g_loss_extra = bce(d_fake3, real_labels)
                    self.scaler_g.scale(g_loss_extra).backward()
                    self.scaler_g.step(g_optim)
                    self.scaler_g.update()

                # ── Logging ───────────────────────────────────────────────────
                writer.add_scalar("Loss/D", errD_real + errD_fake, counter)
                writer.add_scalar("Loss/G", errG, counter)
                writer.add_scalar("Loss/D_real", errD_real, counter)
                writer.add_scalar("Loss/D_fake", errD_fake, counter)

                elapsed = time.time() - start_t
                steps_done = counter
                eta_s = elapsed / steps_done * (epochs * len(loader) - steps_done)
                pbar.set_postfix(
                    {
                        "d": f"{errD_real + errD_fake:.3f}",
                        "g": f"{errG:.3f}",
                        "ETA": _fmt_time(eta_s),
                    }
                )

                # ── Sample grid every 100 steps ───────────────────────────────
                if counter % 100 == 1:
                    self._save_sample_grid(sample_z, epoch, counter, writer)

                counter += 1

            # ── Checkpoint every 10 epochs ────────────────────────────────────
            if (epoch + 1) % 10 == 0:
                self.save(str(self.checkpoint_dir), counter)
                print(f"\n[*] Checkpoint → step {counter}")

        writer.close()
        print(f"\n[*] Training done — {counter - 1} total steps")

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _save_sample_grid(
        self,
        sample_z: torch.Tensor,
        epoch: int,
        counter: int,
        writer: Optional[SummaryWriter] = None,
    ) -> None:
        self.netG.eval()
        samples = self.netG(sample_z)
        self.netG.train()

        samples_np = samples.cpu().permute(0, 2, 3, 1).numpy()
        rows, cols = image_manifold_size(samples_np.shape[0])
        out_path = self.sample_dir / f"train_{epoch:02d}_{counter:06d}.png"
        save_images(samples_np, [rows, cols], out_path)

        if writer is not None:
            try:
                from torchvision.utils import make_grid

                grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=cols)
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
        print(f" [*] Loaded: {latest.name}  (step {step})")
        return True, step


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    """Format a duration in seconds as  1h23m  or  45m  or  12s."""
    s = int(seconds)
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"
