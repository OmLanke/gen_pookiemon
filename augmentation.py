"""
augmentation.py — Data preprocessing pipeline

Converts raw Pokemon PNG sprites (RGBA, transparent background) into
the 64×64 JPEG training dataset used by the DCGAN.

Pipeline per raw image (14× expansion):
  1. RGBA → RGB  (white background composite)
  2. Resize to 64×64  (LANCZOS)
  3. Save as JPEG
  4. Horizontal flip  → {name}-f.jpg
  5. Rotate ±3°, ±5°, ±7° of original  → 6 files
  6. Rotate ±3°, ±5°, ±7° of flip      → 6 files

Usage:
    uv run python augmentation.py --input_dir pokemon_sugimori_ori --output_dir data/pokemon
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

WHITE = (255, 255, 255)
TARGET_SIZE = (64, 64)
ANGLES = [3, 5, 7]


# ─────────────────────────────────────────────────────────────────────────────
# Per-image processing (runs in worker processes)
# ─────────────────────────────────────────────────────────────────────────────


def process_image(args: tuple[Path, Path]) -> int:
    """
    Process a single PNG file: convert, resize, flip, rotate.
    Returns the number of files written.
    """
    input_path, output_dir = args
    name = input_path.stem  # e.g. "1", "3-mega", "6-mega-x"
    written = 0

    try:
        img = Image.open(input_path)

        # ── Step 1+2: RGBA → RGB + resize ────────────────────────────────────
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, WHITE)
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")

        img = img.resize(TARGET_SIZE, Image.LANCZOS)

        # ── Step 3: Save base image ───────────────────────────────────────────
        base_path = output_dir / f"{name}.jpg"
        img.save(base_path, "JPEG", quality=95)
        written += 1

        # ── Step 4: Horizontal flip ───────────────────────────────────────────
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flip_path = output_dir / f"{name}-f.jpg"
        flipped.save(flip_path, "JPEG", quality=95)
        written += 1

        # ── Steps 5+6: Rotate original and flip ──────────────────────────────
        for angle in ANGLES:
            # Clockwise (negative angle in PIL = clockwise)
            rot_cw = img.rotate(-angle, resample=Image.BICUBIC, fillcolor=WHITE)
            rot_cw.save(output_dir / f"{name}_cwr{angle}.jpg", "JPEG", quality=95)

            rot_ccw = img.rotate(angle, resample=Image.BICUBIC, fillcolor=WHITE)
            rot_ccw.save(output_dir / f"{name}_ccwr{angle}.jpg", "JPEG", quality=95)

            flip_cw = flipped.rotate(-angle, resample=Image.BICUBIC, fillcolor=WHITE)
            flip_cw.save(output_dir / f"{name}-f_cwr{angle}.jpg", "JPEG", quality=95)

            flip_ccw = flipped.rotate(angle, resample=Image.BICUBIC, fillcolor=WHITE)
            flip_ccw.save(output_dir / f"{name}-f_ccwr{angle}.jpg", "JPEG", quality=95)

            written += 4

    except Exception as exc:
        print(f"\n[!] Failed to process {input_path.name}: {exc}")

    return written


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon sprite augmentation pipeline")
    parser.add_argument(
        "--input_dir",
        default="pokemon_sugimori_ori",
        help="Directory of raw PNG sprites (default: pokemon_sugimori_ori)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/pokemon",
        help="Output directory for processed JPEGs (default: data/pokemon)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: cpu_count)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"[!] Input directory not found: {input_dir}")
        print("    Download Pokemon sprites and place them in that directory.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.glob("*.png"))
    if not png_files:
        print(f"[!] No PNG files found in {input_dir}")
        return

    print(f"[*] Found {len(png_files)} PNG files → augmenting 14× each")
    print(f"[*] Output → {output_dir}")

    tasks = [(p, output_dir) for p in png_files]
    workers = args.workers or os.cpu_count() or 1

    total_written = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_image, t): t[0] for t in tasks}
        with tqdm(total=len(futures), unit="img", desc="Processing") as pbar:
            for fut in as_completed(futures):
                total_written += fut.result()
                pbar.update(1)

    all_files = list(output_dir.glob("*.jpg"))
    print(
        f"\n[*] Done — {total_written} files written ({len(all_files)} total in {output_dir})"
    )
    print(f"[*] Expected ~{len(png_files) * 14} files (14× per raw image)")

    # Verify a sample
    sample = all_files[:3] if all_files else []
    for f in sample:
        img = Image.open(f)
        print(f"    {f.name}: {img.size} {img.mode}")


if __name__ == "__main__":
    main()
