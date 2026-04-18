# Pokemon Master — Data Pipeline

Complete guide for dataset preparation and the augmentation notebook.

---

## Overview

The dataset pipeline transforms raw Pokemon PNG sprites (transparent background) into a large collection of normalized 64×64 JPEG training images.

```
Raw PNGs (RGBA, transparent bg)
         │  augmentation.ipynb
         ▼
Step 1: RGBA → RGB (white background composite)
         │
Step 2: Resize to 64×64
         │
Step 3: Save as JPEG to data/pokemon/
         │
Step 4: Horizontal flip → save with -f suffix
         │
Step 5: Rotate ±3°, ±5°, ±7° (each original)
         │
Step 6: Rotate ±3°, ±5°, ±7° (each flip)
         │
         ▼
~12,600+ training images (14× expansion per raw image)
```

---

## Dataset Statistics

| Type | Count |
|---|---|
| Base species (Gen 1–6) | 721 |
| Mega evolutions | ~48 |
| Primal forms | 2 |
| Gender variants | 4 |
| Seasonal variants | 8 |
| Forme variants (Rotom, Deoxys, etc.) | ~20 |
| **Total raw images** | **~830** |
| **After 14× augmentation** | **~11,600+** |

---

## Raw Image Requirements

- Format: PNG with alpha channel (RGBA)
- Content: Pokemon official art / sprites on transparent background
- Naming: `{id}.png`, `{id}-mega.png`, `{id}-mega-x.png`, `{id}f.png`, etc.
- Size: any — will be resized to 64×64

---

## Augmentation Notebook Code

The complete logic from `augmentation.ipynb`, restructured for clarity:

### Step 1: RGBA to RGB Conversion

```python
from PIL import Image
import os

def rgba2rgb(rgba_path, output_path):
    """
    Composite RGBA image onto white background.
    Handles transparent Pokemon sprites.
    """
    img = Image.open(rgba_path)
    
    if img.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Use alpha channel as mask — transparent pixels become white
        background.paste(img, mask=img.split()[3])
        background.save(output_path, 'JPEG')
    else:
        # Already RGB (some sprites may not have transparency)
        img.convert('RGB').save(output_path, 'JPEG')
```

### Step 2: Resize to 64×64

```python
def resize_image(path, output_path, size=(64, 64)):
    """
    Resize image to target size.
    Uses LANCZOS resampling for quality.
    """
    img = Image.open(path)
    img = img.resize(size, Image.LANCZOS)
    img.save(output_path, 'JPEG')
```

### Step 3: Horizontal Flip

```python
def flip_image(path, output_path):
    """
    Create horizontal mirror image.
    Saved with '-f' suffix before extension.
    e.g., 1.jpg → 1-f.jpg
    """
    img = Image.open(path)
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped.save(output_path, 'JPEG')
```

### Step 4: Rotation

```python
def rotate_image(path, output_path, angle, fill_color=(255, 255, 255)):
    """
    Rotate image by angle degrees.
    Fills empty corners with white (to avoid black artifacts).
    
    Naming convention:
    - Clockwise:       {name}_cwr{angle}.jpg     (e.g., 1_cwr3.jpg)
    - Counter-clockwise: {name}_ccwr{angle}.jpg  (e.g., 1_ccwr3.jpg)
    """
    img = Image.open(path)
    # expand=False keeps original canvas size
    rotated = img.rotate(angle, fillcolor=fill_color)
    rotated.save(output_path, 'JPEG')
```

### Full Pipeline

```python
import os
from PIL import Image

INPUT_DIR  = 'pokemon_sugimori_ori'   # Raw RGBA PNGs
OUTPUT_DIR = 'data/pokemon'           # Processed JPEGs

ROTATION_ANGLES = [3, 5, 7]  # degrees, applied ± (CW and CCW)
WHITE = (255, 255, 255)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith('.png'):
        continue
    
    name = os.path.splitext(filename)[0]  # e.g., "1" or "3-mega"
    input_path = os.path.join(INPUT_DIR, filename)
    
    # ── Step 1+2: Convert RGBA→RGB and resize to 64×64 ──────────────────────
    base_path = os.path.join(OUTPUT_DIR, name + '.jpg')
    img = Image.open(input_path)
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, WHITE)
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert('RGB')
    img = img.resize((64, 64), Image.LANCZOS)
    img.save(base_path, 'JPEG')
    
    # ── Step 3: Horizontal flip ──────────────────────────────────────────────
    flip_path = os.path.join(OUTPUT_DIR, name + '-f.jpg')
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped.save(flip_path, 'JPEG')
    
    # ── Step 4+5: Rotate original and flip at ±3°, ±5°, ±7° ─────────────────
    for angle in ROTATION_ANGLES:
        # Clockwise rotation of original
        rot_cw = img.rotate(-angle, fillcolor=WHITE)  # negative = clockwise
        rot_cw.save(os.path.join(OUTPUT_DIR, f'{name}_cwr{angle}.jpg'), 'JPEG')
        
        # Counter-clockwise rotation of original
        rot_ccw = img.rotate(angle, fillcolor=WHITE)
        rot_ccw.save(os.path.join(OUTPUT_DIR, f'{name}_ccwr{angle}.jpg'), 'JPEG')
        
        # Clockwise rotation of flip
        rot_flip_cw = flipped.rotate(-angle, fillcolor=WHITE)
        rot_flip_cw.save(os.path.join(OUTPUT_DIR, f'{name}-f_cwr{angle}.jpg'), 'JPEG')
        
        # Counter-clockwise rotation of flip
        rot_flip_ccw = flipped.rotate(angle, fillcolor=WHITE)
        rot_flip_ccw.save(os.path.join(OUTPUT_DIR, f'{name}-f_ccwr{angle}.jpg'), 'JPEG')

print(f"Total images generated: {len(os.listdir(OUTPUT_DIR))}")
```

---

## Output File Naming Convention

For each raw image `{name}.png`, the following files are created:

```
{name}.jpg            ← original (resized 64×64, white bg)
{name}-f.jpg          ← horizontal flip
{name}_cwr3.jpg       ← rotated  3° clockwise
{name}_ccwr3.jpg      ← rotated  3° counter-clockwise
{name}_cwr5.jpg       ← rotated  5° clockwise
{name}_ccwr5.jpg      ← rotated  5° counter-clockwise
{name}_cwr7.jpg       ← rotated  7° clockwise
{name}_ccwr7.jpg      ← rotated  7° counter-clockwise
{name}-f_cwr3.jpg     ← flip + rotated  3° clockwise
{name}-f_ccwr3.jpg    ← flip + rotated  3° counter-clockwise
{name}-f_cwr5.jpg     ← flip + rotated  5° clockwise
{name}-f_ccwr5.jpg    ← flip + rotated  5° counter-clockwise
{name}-f_cwr7.jpg     ← flip + rotated  7° clockwise
{name}-f_ccwr7.jpg    ← flip + rotated  7° counter-clockwise
                        ─────────────────────────────────────
                        14 files per raw image
```

---

## Augmentation Math

```
Raw images:  ~830
× 14 (augmentations per image)
─────────────────────────────────
Total:       ~11,620 training images

Per epoch:
  batch_size = 64
  batches per epoch = floor(11,620 / 64) = 181
  
Over 2000 epochs:
  total iterations = 2000 × 181 = 362,000
```

---

## Why These Augmentations?

| Augmentation | Rationale |
|---|---|
| Horizontal flip | Pokemon sprites face left or right — doubles effective variety |
| Small rotations (≤7°) | Adds slight viewpoint variation without distorting the subject |
| White background fill | Avoids introducing black artifacts from rotation empty corners |
| JPEG format | Smaller file size, faster disk I/O during training |
| 64×64 target size | Matches the GAN output size; all images must be same dimensions |

Rotations are kept small (3°–7°) intentionally. Large rotations would make Pokemon look unnatural and could confuse the discriminator with degenerate examples.

---

## Verification

After running the pipeline, verify:

```python
import glob
files = glob.glob('data/pokemon/*.jpg')
print(f"Total images: {len(files)}")
# Expected: ~11,000–12,600

from PIL import Image
for f in files[:5]:
    img = Image.open(f)
    print(f"{f}: {img.size} {img.mode}")
    # Expected: (64, 64) RGB
```
