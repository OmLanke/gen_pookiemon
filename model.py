"""
model.py — WGAN-GP Generator, Critic, and training harness (PyTorch)

Architecture:
  Generator : z[100] → linear → 4×4×512 → 4× deconv → 64×64×3, tanh out
  Critic    : 64×64×3 → 4× conv → flatten → linear → 1, unbounded score

WGAN-GP (Wasserstein GAN with Gradient Penalty, Gulrajani et al. 2017):
  - Critic loss: E[C(fake)] − E[C(real)] + λ·GP
  - Generator loss: −E[C(G(z))]
  - Gradient penalty enforces the 1-Lipschitz constraint on the critic
  - Critic updated n_critic=5 times per generator step

Key design decisions:
  - No BatchNorm in Critic: BN creates inter-sample correlation that corrupts
    the gradient penalty's Lipschitz enforcement
  - Generator keeps BatchNorm on all hidden layers
  - Noise distribution: Normal(-1, 1) — empirically better for Pokémon sprites
  - Hidden activations: LeakyReLU(0.2) in both Generator and Critic
  - Adam(lr=1e-4, β1=0, β2=0.9) — WGAN-GP recommended optimizer settings

Performance:
  - DataLoader with num_workers for parallel I/O
  - Apple Silicon MPS backend supported
  - torch.compile() opt-in via --compile flag
  - AMP (automatic mixed precision) opt-in via --amp flag
    Uses FP16 Tensor Cores on T4/A100; GP computed in FP32 for correctness
"""

from __future__ import annotations

import os
import re
import time
from glob import glob
from pathlib import Path
from typing import Optional

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
    → linear   → [B, gf*8 * 4 * 4]
    → reshape   → [B, gf*8, 4, 4]
    → BN + LReLU
    → deconv    → [B, gf*4, 8, 8]   BN + LReLU
    → deconv    → [B, gf*2, 16, 16] BN + LReLU
    → deconv    → [B, gf*1, 32, 32] BN + LReLU
    → deconv    → [B, c_dim, 64, 64] tanh  (no BN on final layer)
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

        self.bn0 = batch_norm(gf_dim * 8)
        self.h1 = deconv2d(gf_dim * 8, gf_dim * 4)
        self.bn1 = batch_norm(gf_dim * 4)
        self.h2 = deconv2d(gf_dim * 4, gf_dim * 2)
        self.bn2 = batch_norm(gf_dim * 2)
        self.h3 = deconv2d(gf_dim * 2, gf_dim * 1)
        self.bn3 = batch_norm(gf_dim * 1)
        self.h4 = deconv2d(gf_dim * 1, c_dim)  # no BN

        self.act = lrelu(0.2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.h0_lin(z)
        x = x.view(-1, self.gf_dim * 8, self._s_h16, self._s_w16)
        x = self.act(self.bn0(x))

        x = self.act(self.bn1(self.h1(x)))
        x = self.act(self.bn2(self.h2(x)))
        x = self.act(self.bn3(self.h3(x)))
        x = torch.tanh(self.h4(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Critic  (no sigmoid, no BatchNorm)
# ─────────────────────────────────────────────────────────────────────────────


class Critic(nn.Module):
    """
    Image → unbounded Wasserstein score.

    No sigmoid: WGAN scores are unbounded reals (higher = more real).
    No BatchNorm: BN introduces inter-sample correlation that corrupts
    the gradient penalty's Lipschitz enforcement.

    [B, c_dim, 64, 64]
    → conv 5×5/s2 → [B, df,   32, 32]  LReLU
    → conv 5×5/s2 → [B, df*2, 16, 16]  LReLU
    → conv 5×5/s2 → [B, df*4,  8,  8]  LReLU
    → conv 5×5/s2 → [B, df*8,  4,  4]  LReLU
    → flatten → linear → [B, 1]        (unbounded score)
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
        flat_dim = df_dim * 8 * s4 * s4w  # 8192

        # No BatchNorm anywhere in the critic
        self.h0 = conv2d(c_dim, df_dim)
        self.h1 = conv2d(df_dim, df_dim * 2)
        self.h2 = conv2d(df_dim * 2, df_dim * 4)
        self.h3 = conv2d(df_dim * 4, df_dim * 8)
        self.h4 = linear(flat_dim, 1)
        self.act = lrelu(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.h0(x))
        x = self.act(self.h1(x))
        x = self.act(self.h2(x))
        x = self.act(self.h3(x))
        x = x.view(x.size(0), -1)
        return self.h4(x)  # [B, 1] unbounded score


# ─────────────────────────────────────────────────────────────────────────────
# WGAN-GP training harness
# ─────────────────────────────────────────────────────────────────────────────


class WGAN:
    """
    High-level wrapper that owns Generator, Critic, optimizers, and
    the WGAN-GP training loop (Gulrajani et al. 2017).
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
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        use_amp: bool = False,
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
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        # ── Device selection: CUDA > MPS (Apple Silicon) > CPU ────────────────
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"[*] Using device: {self.device}")

        # AMP only makes sense on CUDA; silently disable elsewhere
        self.use_amp = use_amp and (self.device.type == "cuda")
        if use_amp and not self.use_amp:
            print("[!] AMP requested but device is not CUDA — AMP disabled")
        # GradScaler is used only for the generator step (critic has GP which
        # must stay in FP32, so we skip scaling there to avoid complications)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # ── Models ────────────────────────────────────────────────────────────
        self.netG = Generator(
            z_dim=z_dim,
            gf_dim=gf_dim,
            c_dim=c_dim,
            output_height=output_height,
            output_width=output_width,
        ).to(self.device)

        self.netC = Critic(
            df_dim=df_dim,
            c_dim=c_dim,
            input_height=output_height,
            input_width=output_width,
        ).to(self.device)

        # Optional torch.compile for extra throughput (PyTorch ≥ 2.0)
        if compile_model and hasattr(torch, "compile"):
            print("[*] torch.compile() enabled")
            self.netG = torch.compile(self.netG)  # type: ignore[assignment]
            self.netC = torch.compile(self.netC)  # type: ignore[assignment]

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def model_dir(self) -> str:
        return f"{self.dataset_name}_{self.batch_size}_{self.output_height}_{self.output_width}"

    def _sample_z(self, n: Optional[int] = None) -> torch.Tensor:
        """Sample noise z ~ N(-1, 1) on the correct device."""
        size = n or self.batch_size
        return torch.randn(size, self.z_dim, device=self.device) - 1.0

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """
        WGAN-GP gradient penalty (Gulrajani et al. 2017).

        Samples random interpolations between real and fake images, runs them
        through the critic, then penalises gradients whose L2 norm deviates from 1.

          GP = E[(||∇_x̂ C(x̂)||₂ − 1)²]
          x̂  = ε·real + (1−ε)·fake,   ε ~ Uniform(0, 1)
        """
        B = real.size(0)
        alpha = torch.rand(B, 1, 1, 1, device=self.device)
        interpolated = (
            alpha * real.detach() + (1 - alpha) * fake.detach()
        ).requires_grad_(True)

        c_interp = self.netC(interpolated)

        gradients = torch.autograd.grad(
            outputs=c_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(c_interp),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(B, -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, config: object) -> None:
        """
        WGAN-GP training loop:
          - Critic updated n_critic times per generator step
          - Gradient penalty enforces the 1-Lipschitz constraint
          - Sample grid saved every 100 steps
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

        num_workers = min(4, os.cpu_count() or 1)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        # ── Optimizers: Adam(lr=1e-4, β1=0, β2=0.9) — WGAN-GP paper settings ──
        lr = getattr(config, "learning_rate", 1e-4)
        beta1 = getattr(config, "beta1", 0.0)
        c_optim = torch.optim.Adam(self.netC.parameters(), lr=lr, betas=(beta1, 0.9))
        g_optim = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.9))

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

        epochs = getattr(config, "epoch", 2000)
        n_critic = getattr(config, "n_critic", self.n_critic)
        lambda_gp = getattr(config, "lambda_gp", self.lambda_gp)
        start_time = time.time()

        # Warm up the CUDA context so cuBLAS is initialised before the first
        # backward pass (avoids the "no current CUDA context" UserWarning).
        if self.device.type == "cuda":
            _ = torch.zeros(1, device=self.device)
            torch.cuda.synchronize()

        for epoch in range(epochs):
            self.netG.train()
            self.netC.train()

            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

            for batch_images in pbar:
                batch_images = batch_images.to(self.device, non_blocking=True)

                # ── Update Critic n_critic times ──────────────────────────────
                c_loss_val = gp_val = w_dist_val = 0.0
                for _ in range(n_critic):
                    self.netC.zero_grad(set_to_none=True)
                    z = self._sample_z()

                    # Forward passes in FP16 when AMP is on
                    with torch.amp.autocast("cuda", enabled=self.use_amp):
                        fake_imgs = self.netG(z).detach()  # stop G gradients
                        c_real = self.netC(batch_images)
                        c_fake = self.netC(fake_imgs)

                    # GP must be in FP32: disable autocast and cast inputs
                    with torch.amp.autocast("cuda", enabled=False):
                        gp = self._gradient_penalty(
                            batch_images.float(), fake_imgs.float()
                        )

                    c_loss = c_fake.mean() - c_real.mean() + lambda_gp * gp
                    c_loss.backward()
                    c_optim.step()

                    c_loss_val = c_loss.item()
                    gp_val = gp.item()
                    w_dist_val = (c_real.mean() - c_fake.mean()).item()

                # ── Update Generator once ─────────────────────────────────────
                self.netG.zero_grad(set_to_none=True)
                z = self._sample_z()
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    g_loss = -self.netC(self.netG(z)).mean()
                self.scaler.scale(g_loss).backward()
                self.scaler.step(g_optim)
                self.scaler.update()

                g_loss_val = g_loss.item()

                # ── TensorBoard logging (every 10 steps to reduce I/O overhead)
                if counter % 10 == 0:
                    writer.add_scalar("Loss/Critic", c_loss_val, counter)
                    writer.add_scalar("Loss/Generator", g_loss_val, counter)
                    writer.add_scalar("Loss/Wasserstein", w_dist_val, counter)
                    writer.add_scalar("Loss/GradientPenalty", gp_val, counter)

                pbar.set_postfix(
                    {
                        "c_loss": f"{c_loss_val:.4f}",
                        "g_loss": f"{g_loss_val:.4f}",
                        "W_dist": f"{w_dist_val:.4f}",
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
        samples = self.netG(sample_z)
        self.netG.train()

        samples_np = samples.cpu().permute(0, 2, 3, 1).numpy()
        rows, cols = image_manifold_size(samples_np.shape[0])

        out_path = self.sample_dir / f"train_{epoch:02d}_{counter:06d}.png"
        save_images(samples_np, [rows, cols], out_path)

        if writer is not None:
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
        path = save_dir / f"WGAN.model-{step}.pt"
        torch.save(
            {
                "step": step,
                "generator": self.netG.state_dict(),
                "critic": self.netC.state_dict(),
            },
            path,
        )
        # Keep only last 5 checkpoints
        ckpts = sorted(
            save_dir.glob("WGAN.model-*.pt"),
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
            ckpt_dir.glob("WGAN.model-*.pt"),
            key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
        )
        if not ckpts:
            return False, 0

        latest = ckpts[-1]
        state = torch.load(latest, map_location=self.device, weights_only=True)
        self.netG.load_state_dict(state["generator"])
        self.netC.load_state_dict(state["critic"])
        step = state.get("step", 0)
        print(f" [*] Loaded checkpoint: {latest.name}  (step {step})")
        return True, step
