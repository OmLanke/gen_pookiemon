# Pokemon Master — Complete Implementation Guide

Exact, copy-paste-ready code for every file in the project, with annotations explaining every decision.

---

## File: `ops.py`

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard API compatibility shim (TF <1.0 vs ≥1.0)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Batch Normalization wrapper
# ─────────────────────────────────────────────────────────────────────────────
class batch_norm(object):
    """
    Wraps tf.contrib.layers.batch_norm.
    
    Critical settings:
    - updates_collections=None  → forces SYNCHRONOUS BN update (required!)
    - scale=True                → learnable gamma scaling
    - epsilon=1e-5, decay=0.9   → standard stable settings
    """
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(
            x,
            decay=self.momentum,
            updates_collections=None,   # CRITICAL: sync update
            epsilon=self.epsilon,
            scale=True,
            is_training=train
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convolution
# ─────────────────────────────────────────────────────────────────────────────
def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    """
    Standard 2D convolution used in Discriminator.
    
    - kernel: 5×5 (DCGAN standard)
    - stride: 2×2 (halves spatial dims)
    - padding: SAME (output H/W = ceil(input H/W / stride))
    - init: truncated_normal(0, 0.02) — small, stable initialization
    """
    with tf.variable_scope(name):
        w = tf.get_variable(
            'w',
            [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=tf.truncated_normal_initializer(stddev=stddev)
        )
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable(
            'biases', [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


# ─────────────────────────────────────────────────────────────────────────────
# Transposed Convolution (Deconvolution)
# ─────────────────────────────────────────────────────────────────────────────
def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    """
    Transposed convolution used in Generator for upsampling.
    
    KEY DIFFERENCE from conv2d:
    - weight shape: [k_h, k_w, output_channels, input_channels]  ← REVERSED last 2
    - output_shape must be explicitly provided (TF 1.x requirement)
    - init: random_normal (not truncated_normal — generator uses this)
    
    TF API compatibility: tries conv2d_transpose, falls back to deconv2d
    """
    with tf.variable_scope(name):
        # NOTE: shape is [k_h, k_w, out_channels, in_channels] — reversed!
        w = tf.get_variable(
            'w',
            [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        try:
            deconv = tf.nn.conv2d_transpose(
                input_, w,
                output_shape=output_shape,
                strides=[1, d_h, d_w, 1]
            )
        except AttributeError:  # TF < 0.7 fallback
            deconv = tf.nn.deconv2d(
                input_, w,
                output_shape=output_shape,
                strides=[1, d_h, d_w, 1]
            )

        biases = tf.get_variable(
            'biases', [output_shape[-1]],
            initializer=tf.constant_initializer(0.0)
        )
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        return deconv


# ─────────────────────────────────────────────────────────────────────────────
# Leaky ReLU
# ─────────────────────────────────────────────────────────────────────────────
def lrelu(x, leak=0.2, name="lrelu"):
    """
    Leaky ReLU: f(x) = max(x, 0.2x)
    
    Used EVERYWHERE in this project (both G and D hidden layers).
    The original DCGAN paper used ReLU in G and LeakyReLU in D.
    This project uses LeakyReLU in BOTH — a deliberate modification.
    
    slope=0.2 is the DCGAN paper's recommended value for discriminator.
    """
    return tf.maximum(x, leak * x)


# ─────────────────────────────────────────────────────────────────────────────
# Fully Connected Layer
# ─────────────────────────────────────────────────────────────────────────────
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """
    Fully connected (dense) layer.
    
    Used in:
    - Generator: first layer (z → 8192)
    - Discriminator: last layer (8192 → 1)
    
    init: random_normal(0, 0.02) for weights, 0.0 for biases
    """
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        try:
            matrix = tf.get_variable(
                "Matrix", [shape[1], output_size], tf.float32,
                tf.random_normal_initializer(stddev=stddev)
            )
        except ValueError:
            raise

        bias = tf.get_variable(
            "bias", [output_size],
            initializer=tf.constant_initializer(bias_start)
        )

        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        return tf.matmul(input_, matrix) + bias
```

---

## File: `utils.py`

```python
from __future__ import division

import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import os
import gzip

from PIL import Image
import datetime

from ops import image_summary

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def show_all_variables():
    """Print all trainable variable names and shapes."""
    import tensorflow.contrib.slim as slim
    import tensorflow as tf
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def imread(path, grayscale=False):
    """
    Read an image from disk as a float array.
    Returns float64 array, NOT uint8.
    """
    if grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    """
    Arrange images into a grid.
    
    images: numpy array [N, H, W, C]
    size: [rows, cols] — must satisfy rows*cols >= N
    
    Returns: numpy array [rows*H, cols*W, C]
    """
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter must have dimensions: HxW, HxWx3 or HxWx4')


def imsave(images, size, path):
    """Save merged image grid to path."""
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    """
    Crop the center of image x to [crop_h, crop_w], then resize to [resize_h, resize_w].
    Used when config.crop=True.
    """
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    """
    Crop/resize image and normalize to [-1, 1].
    
    CRITICAL: The / 127.5 - 1 normalization MUST match tanh output range.
    """
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    """
    Undo normalization: [-1, 1] → [0, 1]
    Applied before saving images to disk.
    """
    return (images + 1.) / 2.


def save_images(images, size, image_path):
    """Save a batch of images as a grid PNG."""
    return imsave(inverse_transform(images), size, image_path)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    """
    Full image loading pipeline: read → transform → normalize.
    Returns float array in range [-1, 1].
    """
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def image_manifold_size(num_images):
    """
    Compute a square-ish grid size for num_images images.
    Returns (rows, cols) such that rows*cols == num_images.
    
    For batch_size=64: returns (8, 8)
    """
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def visualize(sess, dcgan, config, option):
    """
    Run the trained sampler and save output image grid.
    Called in inference mode (no --train flag).
    
    option=0: sample random z from N(-1,1), generate, save timestamped PNG
    """
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    if option == 0:
        z_sample = np.random.normal(-1, 1, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(
            samples,
            [image_frame_dim, image_frame_dim],
            './samples/test_%s.png' % datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        )
```

---

## File: `model.py`

```python
from __future__ import division

import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import re

from ops import (
    batch_norm, conv2d, deconv2d, lrelu, linear,
    image_summary, scalar_summary, histogram_summary,
    merge_summary, SummaryWriter
)
from utils import (
    imread, get_image, save_images,
    image_manifold_size, show_all_variables
)


def conv_out_size_same(size, stride):
    """
    Calculate output size of SAME-padded convolution.
    For DCGAN: always stride=2, so output = ceil(size/2).
    Used to pre-compute spatial dims for deconv output_shape.
    """
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(
        self,
        sess,
        input_height=108,
        input_width=108,
        output_height=64,
        output_width=64,
        batch_size=64,
        sample_num=64,
        y_dim=None,         # Set to None — unconditional GAN
        z_dim=100,          # Noise vector dimension
        gf_dim=64,          # Generator base filter count
        df_dim=64,          # Discriminator base filter count
        gfc_dim=1024,       # Generator fully connected dim (not used in this architecture)
        dfc_dim=1024,       # Discriminator fully connected dim (not used)
        c_dim=3,            # RGB = 3 channels
        dataset_name='default',
        input_fname_pattern='*.jpg',
        crop=False,
        checkpoint_dir=None,
        sample_dir=None
    ):
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

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        # ── Batch normalization instances ─────────────────────────────────────
        # Each gets a unique name → unique TF variable scope
        # Discriminator: 3 BN layers (NOT on first conv layer — DCGAN rule)
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        # Generator: 4 BN layers (NOT on final tanh layer)
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        # Load file paths at init time
        self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        """
        Construct the entire TensorFlow computation graph.
        Defines: placeholders, G, D, losses, summaries, saver.
        """
        # ── Conditional GAN support (unused — y=None throughout) ─────────────
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        # ── Input dimensions ─────────────────────────────────────────────────
        if self.crop:
            image_dims = [self.input_height, self.input_width, self.c_dim]
        else:
            image_dims = [self.output_height, self.output_width, self.c_dim]

        # ── Placeholders ─────────────────────────────────────────────────────
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images'
        )
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        # ── Build networks ───────────────────────────────────────────────────
        self.G              = self.generator(self.z, self.y)
        self.D, self.D_logits   = self.discriminator(self.inputs, self.y, reuse=False)
        self.sampler        = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        # ── Summaries ────────────────────────────────────────────────────────
        self.d_sum  = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum  = image_summary("G", self.G)

        # ── Loss functions ───────────────────────────────────────────────────
        def sigmoid_xent(logits, labels):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

        self.d_loss_real = tf.reduce_mean(sigmoid_xent(self.D_logits,  tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_xent(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss      = tf.reduce_mean(sigmoid_xent(self.D_logits_, tf.ones_like(self.D_)))
        self.d_loss      = self.d_loss_real + self.d_loss_fake

        # ── Loss summaries ───────────────────────────────────────────────────
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum      = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum      = scalar_summary("d_loss", self.d_loss)

        # ── Separate variable lists for separate optimizers ──────────────────
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """
        Main training loop.
        
        Key behaviors:
        - D updated once per batch
        - G updated TWICE per batch (always)
        - G updated a THIRD time if errG - (errD_fake + errD_real) > 1
        - Sample grid saved every 100 counter steps
        - Checkpoint saved every 10 epochs
        """
        # ── Optimizers (separate for G and D) ────────────────────────────────
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                    .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        # ── Merged summaries ──────────────────────────────────────────────────
        self.g_sum = merge_summary([
            self.z_sum, self.d__sum, self.G_sum,
            self.d_loss_fake_sum, self.g_loss_sum
        ])
        self.d_sum = merge_summary([
            self.z_sum, self.d_sum,
            self.d_loss_real_sum, self.d_loss_sum
        ])
        self.writer = SummaryWriter("./logs/" + self.model_dir, self.sess.graph)

        # ── Fixed sample for consistent monitoring ────────────────────────────
        sample_z = np.random.normal(-1, 1, size=(self.sample_num, self.z_dim))
        sample_files = self.data[0:self.sample_num]
        sample = [
            get_image(
                sample_file,
                input_height=self.input_height,
                input_width=self.input_width,
                resize_height=self.output_height,
                resize_width=self.output_width,
                crop=self.crop,
                grayscale=self.grayscale
            )
            for sample_file in sample_files
        ]
        if self.grayscale:
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        # ── Resume from checkpoint if available ───────────────────────────────
        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # ── Training loop ─────────────────────────────────────────────────────
        for epoch in xrange(config.epoch):
            self.data = glob(os.path.join(
                "./data", config.dataset, self.input_fname_pattern))
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, int(batch_idxs)):
                batch_files = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [
                    get_image(
                        batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale
                    )
                    for batch_file in batch_files
                ]
                if self.grayscale:
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.normal(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # ── Update D once ─────────────────────────────────────────────
                _, summary_str = self.sess.run(
                    [d_optim, self.d_sum],
                    feed_dict={self.inputs: batch_images, self.z: batch_z}
                )
                self.writer.add_summary(summary_str, counter)

                # ── Update G twice ────────────────────────────────────────────
                _, summary_str = self.sess.run(
                    [g_optim, self.g_sum],
                    feed_dict={self.z: batch_z}
                )
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run(
                    [g_optim, self.g_sum],
                    feed_dict={self.z: batch_z}
                )
                self.writer.add_summary(summary_str, counter)

                # ── Compute losses for logging and conditional update ──────────
                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG      = self.g_loss.eval({self.z: batch_z})

                # ── Conditional third G update ────────────────────────────────
                if errG - (errD_fake + errD_real) > 1:
                    self.sess.run([g_optim], feed_dict={self.z: batch_z})

                counter += 1
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" % (
                    epoch, config.epoch, idx, batch_idxs,
                    time.time() - start_time, errD_real + errD_fake, errG
                ))

                # ── Save sample images every 100 steps ───────────────────────
                if np.mod(counter, 100) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                            },
                        )
                        manifold_h, manifold_w = image_manifold_size(samples.shape[0])
                        save_images(
                            samples,
                            [manifold_h, manifold_w],
                            './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx)
                        )
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except Exception as e:
                        print(e)

            # ── Save checkpoint every 10 epochs ──────────────────────────────
            if np.mod(epoch, 10) == 0:
                self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, y=None, reuse=False):
        """
        5-layer CNN discriminator.
        
        Architecture:
          input [B,64,64,3]
          → conv 5×5/2 → [B,32,32,64]   lrelu (NO BN — first layer rule)
          → conv 5×5/2 → [B,16,16,128]  BN + lrelu
          → conv 5×5/2 → [B,8,8,256]    BN + lrelu
          → conv 5×5/2 → [B,4,4,512]    BN + lrelu
          → linear [B,1] → sigmoid
          
        Returns: (sigmoid_output, logits)
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # Layer 0: NO batch norm (DCGAN convention)
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # [B, 32, 32, 64]

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            # [B, 16, 16, 128]

            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            # [B, 8, 8, 256]

            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            # [B, 4, 4, 512]

            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')
            # [B, 1]

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, y=None):
        """
        5-layer deconvolution generator.
        
        Architecture:
          z [B,100]
          → linear → [B,8192] reshape → [B,4,4,512]   BN + lrelu
          → deconv 5×5/2 → [B,8,8,256]                BN + lrelu
          → deconv 5×5/2 → [B,16,16,128]              BN + lrelu
          → deconv 5×5/2 → [B,32,32,64]               BN + lrelu
          → deconv 5×5/2 → [B,64,64,3]                tanh
          
        NOTE: ALL hidden layers use LeakyReLU (NOT standard ReLU).
        NOTE: NO BN on final layer.
        
        Returns: image tensor [B, 64, 64, 3] in range [-1, 1]
        """
        with tf.variable_scope("generator") as scope:
            # Spatial dimensions at each scale (computed for generality)
            s_h,   s_w   = self.output_height, self.output_width          # 64, 64
            s_h2,  s_w2  = conv_out_size_same(s_h, 2),  conv_out_size_same(s_w, 2)   # 32
            s_h4,  s_w4  = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)  # 16
            s_h8,  s_w8  = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)  # 8
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)  # 4

            # Project noise to spatial volume
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True
            )
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = lrelu(self.g_bn0(self.h0))  # [B, 4, 4, 512]

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4],
                name='g_h1', with_w=True
            )
            h1 = lrelu(self.g_bn1(self.h1))  # [B, 8, 8, 256]

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                name='g_h2', with_w=True
            )
            h2 = lrelu(self.g_bn2(h2))  # [B, 16, 16, 128]

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1],
                name='g_h3', with_w=True
            )
            h3 = lrelu(self.g_bn3(h3))  # [B, 32, 32, 64]

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim],
                name='g_h4', with_w=True
            )
            # NO BN, NO lrelu — directly tanh
            return tf.nn.tanh(h4)  # [B, 64, 64, 3]

    def sampler(self, z, y=None):
        """
        IDENTICAL to generator() but:
        1. Reuses generator's weights (scope.reuse_variables())
        2. Passes train=False to all batch_norm calls
        
        This is used for inference/evaluation — not training.
        """
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()  # CRITICAL

            s_h,   s_w   = self.output_height, self.output_width
            s_h2,  s_w2  = conv_out_size_same(s_h, 2),  conv_out_size_same(s_w, 2)
            s_h4,  s_w4  = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8,  s_w8  = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True
            )
            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = lrelu(self.g_bn0(self.h0, train=False))  # ← train=False

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4],
                name='g_h1', with_w=True
            )
            h1 = lrelu(self.g_bn1(self.h1, train=False))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2],
                name='g_h2', with_w=True
            )
            h2 = lrelu(self.g_bn2(h2, train=False))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1],
                name='g_h3', with_w=True
            )
            h3 = lrelu(self.g_bn3(h3, train=False))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim],
                name='g_h4', with_w=True
            )
            return tf.nn.tanh(h4)

    @property
    def model_dir(self):
        """
        Auto-generates checkpoint directory name.
        Format: {dataset}_{batch_size}_{output_h}_{output_w}
        Example: pokemon_64_64_64
        """
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width
        )

    def save(self, checkpoint_dir, step):
        """Save checkpoint. Creates directory if needed."""
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        """
        Load latest checkpoint if it exists.
        Returns (success: bool, step_count: int)
        """
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
```

---

## File: `main.py`

```python
import os
import scipy.misc
import numpy as np
import tensorflow as tf

from model import DCGAN
from utils import show_all_variables, visualize

# ─────────────────────────────────────────────────────────────────────────────
# CLI Flag Definitions
# ─────────────────────────────────────────────────────────────────────────────
flags = tf.app.flags

flags.DEFINE_integer("epoch", 2000, "Epoch to train [2000]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 64, "Input image height [64]")
flags.DEFINE_integer("input_width", None, "Input image width. If None, uses input_height [None]")
flags.DEFINE_integer("output_height", 64, "Output image height [64]")
flags.DEFINE_integer("output_width", None, "Output image width. If None, uses output_height [None]")
flags.DEFINE_string("dataset", "celebA", "Dataset name under ./data/ [celebA]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern for images [*.jpg]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Sample output directory [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for inference [False]")
flags.DEFINE_boolean("crop", False, "Enable center cropping [False]")
flags.DEFINE_boolean("visualize", False, "Enable visualization mode [False]")
flags.DEFINE_integer("generate_test_images", 100, "Doubles as z_dim; images to generate in test [100]")

FLAGS = flags.FLAGS


def main(_):
    # ── Square defaults ───────────────────────────────────────────────────────
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    # ── Create output directories ─────────────────────────────────────────────
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # ── Session config: allow GPU memory growth ───────────────────────────────
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
            sess,
            input_height=int(FLAGS.input_height),
            input_width=int(FLAGS.input_width),
            output_height=int(FLAGS.output_height),
            output_width=int(FLAGS.output_width),
            batch_size=int(FLAGS.batch_size),
            sample_num=int(FLAGS.batch_size),
            y_dim=None,
            z_dim=FLAGS.generate_test_images,  # z_dim = 100
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir,
        )

        show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run inference")
            visualize(sess, dcgan, FLAGS, 0)


if __name__ == '__main__':
    tf.app.run()
```

---

## Parameter Quick Reference

| Parameter | Value | Location | Notes |
|---|---|---|---|
| `z_dim` | 100 | `main.py --generate_test_images` | Also z input size |
| `batch_size` | 64 | `main.py --batch_size` | Must match sample grid (8×8=64) |
| `output_height/width` | 64 | `main.py --output_height` | Generated image size |
| `gf_dim` | 64 | `model.py DCGAN.__init__` | G base filter count |
| `df_dim` | 64 | `model.py DCGAN.__init__` | D base filter count |
| `epoch` | 2000 | `main.py --epoch` | Total training epochs |
| `learning_rate` | 0.0002 | `main.py --learning_rate` | Adam lr |
| `beta1` | 0.5 | `main.py --beta1` | Adam momentum |
| `lrelu leak` | 0.2 | `ops.py lrelu()` | Negative slope |
| `stddev conv` | 0.02 | `ops.py conv2d()` | Truncated normal init |
| `stddev deconv` | 0.02 | `ops.py deconv2d()` | Random normal init |
| `BN epsilon` | 1e-5 | `ops.py batch_norm` | BN numerical stability |
| `BN momentum` | 0.9 | `ops.py batch_norm` | BN moving average decay |
| `noise dist` | `Normal(-1,1)` | `model.py train()` | NOT Uniform |
| `crop` | False | `main.py --crop` | Use resize, not crop |
| `c_dim` | 3 | `model.py DCGAN.__init__` | RGB |
