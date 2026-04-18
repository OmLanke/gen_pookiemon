# Pokemon Master — Complete Rebuild Blueprint

> A step-by-step, implementation-proof guide to rebuilding this DCGAN Pokemon image generation project from scratch.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Final Project Structure](#2-final-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Phase 1 — Dataset Preparation](#4-phase-1--dataset-preparation)
5. [Phase 2 — Core Layer Primitives (ops.py)](#5-phase-2--core-layer-primitives-opspy)
6. [Phase 3 — Image Utilities (utils.py)](#6-phase-3--image-utilities-utilspy)
7. [Phase 4 — DCGAN Model (model.py)](#7-phase-4--dcgan-model-modelpy)
8. [Phase 5 — Entry Point (main.py)](#8-phase-5--entry-point-mainpy)
9. [Phase 6 — Data Augmentation Notebook](#9-phase-6--data-augmentation-notebook)
10. [Phase 7 — Training Run](#10-phase-7--training-run)
11. [Phase 8 — Inference / Generation](#11-phase-8--inference--generation)
12. [Critical Design Decisions](#12-critical-design-decisions)
13. [Common Failure Modes](#13-common-failure-modes)

---

## 1. Project Overview

**What it is:** A Deep Convolutional Generative Adversarial Network (DCGAN) trained on Pokemon sprite images. Given random Gaussian noise, it generates new 64×64 RGB images that look like novel Pokemon.

**What it is NOT:** A web app, an API, a mobile app. It is a pure Python CLI program.

**Key numbers at a glance:**

| Property | Value |
|---|---|
| Generated image size | 64 × 64 pixels, RGB |
| Noise vector dimension (z_dim) | 100 |
| Generator depth | 5 layers (linear + 4 deconvolutions) |
| Discriminator depth | 5 layers (4 convolutions + linear) |
| Batch size | 64 |
| Training epochs | 2000 |
| Learning rate | 0.0002 |
| Adam β1 | 0.5 |
| Dataset size | ~900+ images (after augmentation: ~12,600+) |

---

## 2. Final Project Structure

```
Pokemon Master/
├── docs/                          ← (these blueprint files)
├── data/
│   └── pokemon/                   ← 64×64 JPEG training images
│       └── *.jpg
├── checkpoint/
│   └── pokemon_64_64_64/          ← saved TF checkpoints
│       ├── checkpoint
│       └── DCGAN.model-{step}.*
├── samples/                       ← generated image grids (auto-created)
│   ├── train_EE_IIII.png
│   └── test_YYYY-MM-DD-HH-MM-SS.png
├── logs/
│   └── pokemon/                   ← TensorBoard event files (auto-created)
├── main.py                        ← CLI entry point
├── model.py                       ← DCGAN class
├── ops.py                         ← TF layer primitives
├── utils.py                       ← image I/O and visualization
└── augmentation.ipynb             ← data preprocessing notebook
```

---

## 3. Environment Setup

### 3.1 Python Version

Use **Python 3.6.x** exactly. The project uses `scipy.misc.imread` and `scipy.misc.imsave`, which were deprecated in SciPy 1.3 and **removed in SciPy 1.6**. Python 3.6 ships naturally with a compatible pip ecosystem.

```bash
# Using pyenv (recommended)
pyenv install 3.6.15
pyenv local 3.6.15
```

### 3.2 Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
```

### 3.3 Install Dependencies

There is no `requirements.txt`. Install these exact versions:

```bash
pip install tensorflow==1.15.5       # Last stable TF 1.x release
pip install numpy==1.16.6            # Compatible with TF 1.15
pip install scipy==1.2.3             # Last version with scipy.misc.imread
pip install Pillow==6.2.2            # For augmentation notebook
pip install six==1.16.0              # Python 2/3 compat (for xrange)
pip install jupyter==1.0.0           # For augmentation.ipynb
```

> **Critical:** TensorFlow 1.15 is the last 1.x release and includes `tf.contrib`. If you use TF 2.x, everything breaks — the entire codebase uses TF 1.x API (`tf.Session`, `tf.placeholder`, `tf.app.flags`, `tf.contrib.*`).

### 3.4 Verify Installation

```python
import tensorflow as tf
print(tf.__version__)   # Must be 1.15.x
import scipy.misc
scipy.misc.imread       # Must not raise AttributeError
```

---

## 4. Phase 1 — Dataset Preparation

### 4.1 Obtain Raw Images

Download Pokemon sprite images (Generation 1-6, 721 base species + variants):
- Source: https://veekun.com/dex/downloads (official game sprites)
- Alternative: https://www.kaggle.com/dollarakshay/pokemon-images

You need PNG files with **transparent backgrounds** (RGBA format). Place them in a temporary folder, e.g., `pokemon_sugimori_ori/`.

Expected files: numbered `1.png` through `721.png`, plus variant forms:
- Mega evolutions: `3-mega.png`, `6-mega-x.png`, `6-mega-y.png`, ...
- Primal forms: `382-primal.png`, `383-primal.png`
- Gender variants: `521f.png`, `592f.png`, `668f.png`, `678f.png`
- Seasonal forms: `585-spring.png`, `585-summer.png`, `585-autumn.png`, `585-winter.png`, `586-spring.png`, ...
- Rotom formes: `479-fan.png`, `479-frost.png`, `479-heat.png`, `479-mow.png`, `479-wash.png`
- Deoxys formes: `386-attack.png`, `386-defense.png`, `386-speed.png`
- Giratina: `487-origin.png`
- Pikachu costumes: `25-belle.png`, `25-libre.png`, `25-phd.png`, `25-pop-star.png`, `25-rock-star.png`

### 4.2 Run Augmentation Pipeline

See `augmentation.ipynb` and [DATA_PIPELINE.md](./DATA_PIPELINE.md) for the full pipeline.

**Summary of what the notebook does:**
1. RGBA → RGB (white background composite)
2. Resize all to 64×64
3. Save to `data/pokemon/` as JPEG
4. For each image: create a horizontal flip (saved with `-f` suffix)
5. For each original AND flip: rotate at ±3°, ±5°, ±7° (6 rotated variants each)
6. Net result: **14× dataset expansion**

---

## 5. Phase 2 — Core Layer Primitives (`ops.py`)

Build `ops.py` first — it has no dependencies on other project files.

### 5.1 TensorBoard Compatibility Shim

```python
import tensorflow as tf

try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter
```

### 5.2 `batch_norm` Class

```python
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(
            x,
            decay=self.momentum,
            updates_collections=None,     # IMPORTANT: synchronous update
            epsilon=self.epsilon,
            scale=True,
            is_training=train
        )
```

> **Critical:** `updates_collections=None` forces BN statistics to update synchronously during the forward pass. Without this, BN mean/variance don't update during training.

### 5.3 `conv2d` Function

```python
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv
```

Key parameters: **5×5 kernel, stride 2, SAME padding, stddev=0.02 truncated normal init**.

### 5.4 `deconv2d` Function

```python
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        try:
            deconv = tf.nn.conv2d_transpose(
                input_, w,
                output_shape=output_shape,
                strides=[1, d_h, d_w, 1]
            )
        except AttributeError:  # TF < 0.7
            deconv = tf.nn.deconv2d(
                input_, w,
                output_shape=output_shape,
                strides=[1, d_h, d_w, 1]
            )
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        return deconv
```

> **Critical difference from conv2d:** weight tensor shape is `[k_h, k_w, output_channels, input_channels]` — the last two dims are **reversed** compared to conv2d.

### 5.5 `lrelu` Function

```python
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)
```

Leaky ReLU with slope 0.2 in the negative region.

### 5.6 `linear` Function

```python
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [shape[1], output_size], tf.float32,
            tf.random_normal_initializer(stddev=stddev)
        )
        bias = tf.get_variable(
            "bias", [output_size],
            initializer=tf.constant_initializer(bias_start)
        )
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        return tf.matmul(input_, matrix) + bias
```

---

## 6. Phase 3 — Image Utilities (`utils.py`)

### 6.1 Image Reading

```python
import scipy.misc
import numpy as np

def imread(path, grayscale=False):
    if grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    return scipy.misc.imread(path).astype(np.float)
```

### 6.2 Image Transformation (Normalize to [-1, 1])

```python
def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.  # → [-1, 1]
```

**This normalization is critical** — tanh output is in `[-1, 1]`, so inputs must match.

### 6.3 Center Crop

```python
def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w]
    )
```

### 6.4 Get Image (Load + Transform)

```python
def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)
```

### 6.5 Save Image Grid

```python
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images + 1.) / 2.  # [-1,1] → [0,1]

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j*h:j*h+h, i*w:i*w+w, :] = image
        return img
    # grayscale case omitted for brevity
```

### 6.6 Visualize (Inference Mode)

```python
import datetime

def visualize(sess, dcgan, config, option):
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    if option == 0:
        z_sample = np.random.normal(-1, 1, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [image_frame_dim, image_frame_dim],
                    './samples/test_%s.png' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
```

### 6.7 Grid Size Helper

```python
def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w
```

### 6.8 Show All Variables

```python
import tensorflow.contrib.slim as slim

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
```

---

## 7. Phase 4 — DCGAN Model (`model.py`)

This is the core of the project. Build in this order: `__init__` → `build_model` → `generator` → `discriminator` → `sampler` → `train` → `save`/`load`.

### 7.1 Imports and Class Skeleton

```python
from __future__ import division
import os, time, math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import batch_norm, conv2d, deconv2d, lrelu, linear
from ops import image_summary, scalar_summary, histogram_summary, merge_summary, SummaryWriter
from utils import *

class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, ...):
        ...
    def build_model(self):
        ...
    def train(self, config):
        ...
    def generator(self, z, y=None):
        ...
    def sampler(self, z, y=None):
        ...
    def discriminator(self, image, y=None, reuse=False):
        ...
    def save(self, checkpoint_dir, step):
        ...
    def load(self, checkpoint_dir):
        ...
```

### 7.2 `__init__`

```python
def __init__(self, sess,
             input_height=108, input_width=108,
             output_height=64, output_width=64,
             batch_size=64, sample_num=64,
             y_dim=None, z_dim=100,
             gf_dim=64, df_dim=64,
             gfc_dim=1024, dfc_dim=1024,
             c_dim=3, dataset_name='default',
             input_fname_pattern='*.jpg',
             crop=False,
             checkpoint_dir=None, sample_dir=None):

    self.sess = sess
    self.crop = crop
    self.batch_size = batch_size
    self.sample_num = sample_num
    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width
    self.y_dim = y_dim
    self.z_dim = z_dim
    self.gf_dim = gf_dim   # 64 — generator base filter count
    self.df_dim = df_dim   # 64 — discriminator base filter count
    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim
    self.c_dim = c_dim     # 3 for RGB

    # Batch normalization layers
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')
    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    # Load file paths
    self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))

    self.grayscale = (self.c_dim == 1)

    self.build_model()
```

### 7.3 `build_model`

```python
def build_model(self):
    if self.y_dim:
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
        self.y = None

    if self.crop:
        image_dims = [self.input_height, self.input_width, self.c_dim]
    else:
        image_dims = [self.output_height, self.output_width, self.c_dim]

    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

    self.G = self.generator(self.z, self.y)
    self.D, self.D_logits = self.discriminator(self.inputs, self.y, reuse=False)
    self.sampler = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

    # Summaries
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    # Losses
    def sigmoid_cross_entropy_with_logits(x, y):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

    self.d_loss_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    self.d_loss = self.d_loss_real + self.d_loss_fake

    # Loss summaries
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()
```

### 7.4 Generator

```python
def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = self.output_height, self.output_width        # 64, 64
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)  # 32, 32
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2) # 16, 16
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2) # 8, 8
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2) # 4, 4

        # Project and reshape
        self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
        self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = lrelu(self.g_bn0(self.h0))    # ← NOTE: lrelu, NOT relu

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = lrelu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = lrelu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = lrelu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
        return tf.nn.tanh(h4)
```

Helper:
```python
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))
```

### 7.5 Sampler (Inference)

```python
def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()   # ← CRITICAL: reuse generator weights
        # ... identical structure to generator() but with train=False in BN calls ...
        h0 = lrelu(self.g_bn0(self.h0, train=False))
        # etc.
```

### 7.6 Discriminator

```python
def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        # Layer 0: NO batch norm on first layer (DCGAN rule)
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        # → [batch, 32, 32, 64]

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        # → [batch, 16, 16, 128]

        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        # → [batch, 8, 8, 256]

        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        # → [batch, 4, 4, 512]

        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
        # → [batch, 1]

        return tf.nn.sigmoid(h4), h4
```

### 7.7 Training Loop

```python
def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)

    tf.global_variables_initializer().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

    self.writer = SummaryWriter("./logs/" + self.model_dir, self.sess.graph)

    sample_z = np.random.normal(-1, 1, size=(self.sample_num, self.z_dim))
    sample_files = self.data[0:self.sample_num]
    sample = [get_image(...) for f in sample_files]
    sample_inputs = np.array(sample).astype(np.float32)

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
        counter = checkpoint_counter

    for epoch in xrange(config.epoch):
        self.data = glob(os.path.join("./data", config.dataset, config.input_fname_pattern))
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

        for idx in xrange(0, batch_idxs):
            batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
            batch = [get_image(bf, ...) for bf in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            batch_z = np.random.normal(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

            # Update D
            _, summary_str = self.sess.run([d_optim, self.d_sum],
                feed_dict={self.inputs: batch_images, self.z: batch_z})
            self.writer.add_summary(summary_str, counter)

            # Update G (twice)
            _, summary_str = self.sess.run([g_optim, self.g_sum],
                feed_dict={self.z: batch_z})
            self.writer.add_summary(summary_str, counter)
            _, summary_str = self.sess.run([g_optim, self.g_sum],
                feed_dict={self.z: batch_z})
            self.writer.add_summary(summary_str, counter)

            # Compute losses
            errD_fake = self.d_loss_fake.eval({self.z: batch_z})
            errD_real = self.d_loss_real.eval({self.inputs: batch_images})
            errG = self.g_loss.eval({self.z: batch_z})

            # Conditional third G update
            if errG - (errD_fake + errD_real) > 1:
                self.sess.run([g_optim], feed_dict={self.z: batch_z})

            counter += 1
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (
                epoch, config.epoch, idx, batch_idxs,
                time.time() - start_time, errD_real+errD_fake, errG))

            # Save samples every 100 steps
            if np.mod(counter, 100) == 1:
                samples, d_loss, g_loss = self.sess.run(
                    [self.sampler, self.d_loss, self.g_loss],
                    feed_dict={self.z: sample_z, self.inputs: sample_inputs})
                manifold_h, manifold_w = image_manifold_size(samples.shape[0])
                save_images(samples, [manifold_h, manifold_w],
                            './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))

        # Save checkpoint every 10 epochs
        if np.mod(epoch, 10) == 0:
            self.save(config.checkpoint_dir, counter)
```

### 7.8 Save / Load

```python
@property
def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

def load(self, checkpoint_dir):
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        return True, counter
    return False, 0
```

---

## 8. Phase 5 — Entry Point (`main.py`)

```python
import os
import scipy.misc
import numpy as np
import tensorflow as tf

from model import DCGAN
from utils import show_all_variables, visualize

# --- Flag definitions ---
flags = tf.app.flags
flags.DEFINE_integer("epoch", 2000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 64, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")
FLAGS = flags.FLAGS

def main(_):
    # Square output
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    # Create output directories
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
            sess,
            input_height=FLAGS.input_height,
            input_width=FLAGS.input_width,
            output_height=FLAGS.output_height,
            output_width=FLAGS.output_width,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            y_dim=None,
            z_dim=FLAGS.generate_test_images,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir
        )

        show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            visualize(sess, dcgan, FLAGS, 0)

if __name__ == '__main__':
    tf.app.run()
```

---

## 9. Phase 6 — Data Augmentation Notebook

See [DATA_PIPELINE.md](./DATA_PIPELINE.md) for the full augmentation pipeline with code.

---

## 10. Phase 7 — Training Run

```bash
# Activate environment
source venv/bin/activate

# Start training (Pokemon dataset, 64x64 output)
python main.py \
  --dataset pokemon \
  --input_height 64 \
  --input_width 64 \
  --output_height 64 \
  --output_width 64 \
  --batch_size 64 \
  --epoch 2000 \
  --train \
  --crop False

# Monitor with TensorBoard (separate terminal)
tensorboard --logdir=./logs/pokemon
```

**Expected output:**
```
Epoch: [ 0/ 2000] [   0/ 14] time: 1.2345, d_loss: 1.38629436, g_loss: 0.69314718
Epoch: [ 0/ 2000] [   1/ 14] time: 2.3456, d_loss: 1.30000000, g_loss: 0.75000000
...
```

Sample images saved to `./samples/train_00_0001.png` (8×8 grid) every 100 counter steps.
Checkpoints saved to `./checkpoint/pokemon_64_64_64/` every 10 epochs.

---

## 11. Phase 8 — Inference / Generation

```bash
python main.py \
  --dataset pokemon \
  --input_height 64 \
  --output_height 64 \
  --batch_size 64

# Output: ./samples/test_2024-01-01-12-00-00.png
```

---

## 12. Critical Design Decisions

These are the project's intentional deviations from the standard DCGAN paper that **must not be changed**:

| Decision | Standard DCGAN | This Project | Why |
|---|---|---|---|
| Noise distribution | `Uniform(-1, 1)` | `Normal(-1, 1)` | Empirically found to work better for this dataset |
| Generator hidden activations | ReLU | **Leaky ReLU (0.2)** | Applied to all 4 hidden G layers; helps gradient flow |
| G optimizer steps per batch | 1 | **2 always + 1 conditional** | Prevents D from winning too fast |
| BN `updates_collections` | Default | `None` (synchronous) | Required for correct BN behavior in TF 1.x |
| First D layer BN | Yes (some implementations) | **No BN on D's first layer** | DCGAN paper recommendation |

---

## 13. Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: module 'scipy.misc' has no attribute 'imread'` | SciPy >= 1.3 | Downgrade: `pip install scipy==1.2.3` |
| `AttributeError: module 'tensorflow' has no attribute 'app'` | TF 2.x installed | Install: `pip install tensorflow==1.15.5` |
| `ValueError: Variable generator/... already exists` | `reuse_variables()` not called in sampler | Ensure `scope.reuse_variables()` is called in `sampler()` |
| `InvalidArgumentError: logits and labels must have the same shape` | TF version mismatch in cross-entropy API | Use `tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)` with **keyword args** |
| D loss drops to 0 immediately | G optimizer not running enough | Check the conditional third G update logic |
| Generated images are all the same (mode collapse) | Usually caused by D winning too fast | Verify 2× G update and conditional 3rd update |
| Checkpoint not found on load | Directory naming mismatch | Checkpoint dir = `{dataset}_{batch}_{h}_{w}` = `pokemon_64_64_64` |
| `xrange is not defined` | Python 3 without `six` | `pip install six` and use `from six.moves import xrange` |
| BN not updating during training | Missing `updates_collections=None` | Set it in `batch_norm.__call__` |
