"""
main.py — CLI entry point for Pookiemon WGAN-GP

Train:
    uv run python main.py --dataset pokemon --train

Inference (requires a checkpoint):
    uv run python main.py --dataset pokemon

Data augmentation:
    uv run python augmentation.py --input_dir pokemon_sugimori_ori

TensorBoard:
    uv run tensorboard --logdir logs/pokemon
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model import WGAN
from utils import show_all_variables, visualize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WGAN-GP Pokémon image generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training hyperparameters
    p.add_argument("--epoch", type=int, default=2000, help="Training epochs")
    p.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Adam learning rate"
    )
    p.add_argument("--beta1", type=float, default=0.0, help="Adam β1 momentum")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size")
    p.add_argument(
        "--n_critic",
        type=int,
        default=5,
        help="Critic updates per generator step",
    )
    p.add_argument(
        "--lambda_gp",
        type=float,
        default=10.0,
        help="Gradient penalty weight λ",
    )

    # Architecture
    p.add_argument("--z_dim", type=int, default=100, help="Noise vector dimension")
    p.add_argument("--gf_dim", type=int, default=64, help="Generator base filter count")
    p.add_argument(
        "--df_dim", type=int, default=64, help="Critic base filter count"
    )
    p.add_argument(
        "--c_dim", type=int, default=3, help="Image channels (3=RGB, 1=grayscale)"
    )

    # Image dimensions
    p.add_argument("--input_height", type=int, default=64, help="Input image height")
    p.add_argument(
        "--input_width",
        type=int,
        default=None,
        help="Input width (None → same as height)",
    )
    p.add_argument("--output_height", type=int, default=64, help="Output image height")
    p.add_argument(
        "--output_width",
        type=int,
        default=None,
        help="Output width (None → same as height)",
    )

    # Dataset
    p.add_argument("--dataset", default="pokemon", help="Dataset folder under ./data/")
    p.add_argument(
        "--input_fname_pattern",
        default="*.jpg",
        help="Glob pattern for training images",
    )
    p.add_argument(
        "--crop", action="store_true", help="Use centre-crop instead of resize"
    )

    # Paths
    p.add_argument(
        "--checkpoint_dir", default="checkpoint", help="Checkpoint directory"
    )
    p.add_argument(
        "--sample_dir", default="samples", help="Generated sample output directory"
    )

    # Mode
    p.add_argument(
        "--train", action="store_true", help="Train mode (omit for inference)"
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Run visualisation after inference load",
    )

    # Performance
    p.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile() for faster training (PyTorch ≥ 2.0)",
    )

    return p.parse_args()


def main() -> None:
    config = parse_args()

    # Square defaults
    if config.input_width is None:
        config.input_width = config.input_height
    if config.output_width is None:
        config.output_width = config.output_height

    # Ensure output directories exist
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.sample_dir).mkdir(parents=True, exist_ok=True)

    wgan = WGAN(
        input_height=config.input_height,
        input_width=config.input_width,
        output_height=config.output_height,
        output_width=config.output_width,
        batch_size=config.batch_size,
        sample_num=config.batch_size,
        z_dim=config.z_dim,
        gf_dim=config.gf_dim,
        df_dim=config.df_dim,
        c_dim=config.c_dim,
        dataset_name=config.dataset,
        input_fname_pattern=config.input_fname_pattern,
        crop=config.crop,
        checkpoint_dir=config.checkpoint_dir,
        sample_dir=config.sample_dir,
        compile_model=config.compile,
        n_critic=config.n_critic,
        lambda_gp=config.lambda_gp,
    )

    print("\n── Generator ──────────────────────────────────────────────────")
    show_all_variables(wgan.netG)
    print("── Critic ─────────────────────────────────────────────────────")
    show_all_variables(wgan.netC)

    if config.train:
        wgan.train(config)
    else:
        loaded, _ = wgan.load(config.checkpoint_dir)
        if not loaded:
            print("[!] No checkpoint found. Train a model first:")
            print(f"    uv run python main.py --dataset {config.dataset} --train")
            sys.exit(1)
        visualize(wgan, config, option=0)


if __name__ == "__main__":
    main()
