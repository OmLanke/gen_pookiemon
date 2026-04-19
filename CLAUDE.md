# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Setup & Dependencies
- Install dependencies: `uv sync --all-groups`

### Dataset Preparation
- Run augmentation pipeline: `uv run python augmentation.py --input_dir <input_directory>`

### Training & Inference
- Start training: `uv run python main.py --dataset pokemon --train`
- Full-quality training: `uv run python main.py --dataset pokemon --train --full`
- Run inference (generate samples): `uv run python main.py --dataset pokemon`
- Monitor with TensorBoard: `uv run tensorboard --logdir logs/pokemon`

## Architecture & Structure

### High-Level Architecture
The project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate 64x64 Pokémon sprites. It uses a standard GAN architecture with a Generator (noise $\to$ image) and a Discriminator (image $\to$ probability).

Key architectural deviations from the original DCGAN paper:
- **Noise Distribution**: Uses `Normal(-1, 1)` instead of Uniform.
- **Activations**: Generator uses `LeakyReLU(0.2)` instead of `ReLU`.
- **Training Dynamics**: Generator is updated 2x always, with an optional 3rd update if it falls significantly behind the discriminator.

### Project Structure
- `main.py`: CLI entry point and training/inference loop.
- `model.py`: DCGAN Trainer and network definitions for Generator and Discriminator.
- `ops.py`: Low-level layer primitives (convolutions, batch norm, etc.).
- `utils.py`: Image processing, grid assembly, and visualization.
- `augmentation.py`: Preprocessing pipeline that expands a small set of raw sprites into a larger training set via flipping and rotation.
- `data/`: Stores processed training images.
- `checkpoint/`: Model weights.
- `samples/`: Output image grids.
- `logs/`: TensorBoard event files.

### Performance Optimizations
- **Apple Silicon**: Uses `channels_last` (NHWC) memory format for faster convolutions on MPS.
- **Data Loading**: Loads the entire dataset into a single CPU tensor at startup to eliminate per-epoch disk I/O.
- **Fused Pass**: Fuses multiple Generator backward passes into a single operation.
