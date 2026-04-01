""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Data loading and preprocessing functions
"""

import jax.numpy as jnp
import jax
from torch.utils.data import TensorDataset, DataLoader, default_collate, Subset
import numpy as np
from jax.tree import map
from torch import randperm, stack, tensor
from torchvision.transforms import v2
from jax import random, jit, vmap, lax
import jax.image as jimg
import dm_pix as pix

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


# Custom DataLoader
class NumpyLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(
            dataset,
            **kwargs
        )

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = np.stack(images)
        labels = np.array(labels)
        return images, labels


def to_dataloader(data, batch_size, num_classes, fits_in_memory=True, augmentation=False):
    loader = []
    if fits_in_memory:
        for dataset in data:
            images, labels = prepare_data(
                dataset[:][0], dataset[:][1], batch_size, num_classes)
            loader.append((images, labels))
    else:
        for dataset in data:
            dataloader = NumpyLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0)
            loader.append(dataloader)
    return loader


def reshape_perm(dataset, perm):
    data, labels = dataset
    return data.reshape(data.shape[0], data.shape[1], -1)[:, :, perm].reshape(data.shape), labels


def prepare_data(data, targets, batch_size, num_classes):
    num_samples = len(data) - len(data) % batch_size
    data = jnp.array(data[:num_samples], dtype=jnp.float32)
    targets = jax.nn.one_hot(
        jnp.array(targets[:num_samples], dtype=jnp.int32), num_classes=num_classes)
    return data.reshape(-1, batch_size, *data.shape[1:]), targets.reshape(-1, batch_size, num_classes)


def prepare_data_val_split(tensor_dataset_list, rng, val_split=0.1):
    """ Prepare data for training and validation split.

    Args:
        tensor_dataset_list (list[TensorDataset]): The list of TensorDataset containing the training datasets.
        rng (jax.random.key): The random number generator.
        val_split (float): The validation split.

    Returns:
        tuple: The training and validation dataset as TensorDataset.
    """
    train_dataset_list = []
    val_dataset_list = []
    for tensor_dataset in tensor_dataset_list:
        data, labels = tensor_dataset[:]
        num_samples = data.shape[0]
        perm = randperm(num_samples)
        data = data[perm]
        labels = labels[perm]
        split = int(num_samples * (1 - val_split))
        train_dataset_list.append(TensorDataset(data[:split], labels[:split]))
        val_dataset_list.append(TensorDataset(data[split:], labels[split:]))
    return train_dataset_list, val_dataset_list


def shuffle_dataset(dataloader, key):
    images, labels = dataloader
    permutation = jax.random.permutation(
        key, images.shape[0])
    images = images[permutation]
    labels = labels[permutation]
    return images, labels


def split_dataset(dataloader, n_splits, key=None, fits_in_memory=True, augmentation=False):
    if not fits_in_memory:
        # unchanged branch
        dataset = dataloader.dataset
        total_size = len(dataset)
        split_size = total_size // n_splits
        transform = build_cifar_augmentation() if augmentation else None
        splits = []
        for i in range(n_splits):
            start = i * split_size
            end = start + split_size if i < n_splits - 1 else total_size
            imgs = []
            lbls = []
            for idx in range(start, end):
                x, y = dataset[idx]
                imgs.append(x)
                lbls.append(y)
            imgs = stack(imgs)
            lbls = stack(lbls)
            if transform is not None:
                imgs = transform(imgs)
            subset = TensorDataset(imgs, lbls)
            splits.append(
                NumpyLoader(
                    subset,
                    batch_size=dataloader.batch_size,
                    shuffle=True,
                    drop_last=True
                )
            )
        return splits

    images, labels = dataloader
    augment_fn = cifar_augment_batch

    @jax.jit(static_argnames=("n_splits", "use_aug"))
    def _split(images, labels, key, n_splits, use_aug):
        N = images.shape[0]  # 390
        split_size = N // n_splits
        end_idx = split_size * n_splits

        images = images[:end_idx]
        labels = labels[:end_idx]

        def body_fn(carry, inputs):
            key = carry
            x, y = inputs 
            key, subkey = random.split(key)
            if use_aug:
                x = augment_fn(x, subkey)

            return key, (x, y)

        # scan over the 390 axis
        new_key, key = random.split(key)
        _, (xs, ys) = lax.scan(body_fn, new_key, (images, labels))
        # xs: (390, 128, 3, 32, 32)

        # reshape into splits
        xs = xs.reshape(n_splits, split_size, *xs.shape[1:])
        ys = ys.reshape(n_splits, split_size, *ys.shape[1:])
        return xs, ys
    xs, ys = _split(images, labels, key, n_splits, augmentation)
    return list(zip(xs, ys))

# CIFAR strong augmentation (example)

def build_cifar_augmentation():
    return v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(15),
        v2.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )])



# Constants
_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
_STD = jnp.array([0.2023, 0.1994, 0.2010])


def _pad_crop_batch(images, keys):
    B, C, H, W = images.shape
    padded = jnp.pad(images, ((0, 0), (0, 0), (4, 4), (4, 4)), mode="constant")
    offsets = random.randint(keys, (B, 2), 0, 9)

    def crop(img, offset):
        h, w = offset
        return lax.dynamic_slice(img, (0, h, w), (C, 32, 32))

    return vmap(crop)(padded, offsets)


def _flip_batch(images, keys):
    flips = random.bernoulli(keys, 0.5, (images.shape[0],))
    return jnp.where(
        flips[:, None, None, None],
        jnp.flip(images, axis=3),  # width axis
        images
    )


def _rotate_batch(images, keys):
    angles = random.uniform(keys, (images.shape[0],), minval=-15.0, maxval=15.0)
    angles = angles * jnp.pi / 180.0
    # dm_pix expects NHWC
    images_nhwc = jnp.transpose(images, (0, 2, 3, 1))
    rotated = vmap(pix.rotate, in_axes=(0, 0))(images_nhwc, angles)
    return jnp.transpose(rotated, (0, 3, 1, 2))

@jax.jit
def _color_jitter_batch(images, key, brightness=0.2, contrast=0.2, saturation=0.2):
    """
    images: (B, C, H, W) in [0,1]
    key: single PRNGKey for the batch
    """
    B = images.shape[0]
    k_brightness, k_contrast, k_saturation = random.split(key, 3)
    # Brightness
    bright_offsets = random.uniform(k_brightness, (B,1,1,1), minval=-brightness, maxval=brightness)
    images = images + bright_offsets
    # Contrast
    mean = jnp.mean(images, axis=(1,2,3), keepdims=True)
    contrast_factors = random.uniform(k_contrast, (B,1,1,1), minval=1-contrast, maxval=1+contrast)
    images = (images - mean) * contrast_factors + mean
    # Saturation
    gray = jnp.mean(images, axis=1, keepdims=True)
    sat_factors = random.uniform(k_saturation, (B,1,1,1), minval=1-saturation, maxval=1+saturation)
    images = gray + (images - gray) * sat_factors
    return images

@jit
def cifar_augment_batch(images, key):
    """
    images: (B, C, H, W)
    key: single PRNGKey for the batch
    """
    k_pad, k_flip, k_rotate, k_color = random.split(key, 4)
    images = _flip_batch(images, k_flip)
    images = lax.cond(random.bernoulli(k_pad, 0.5), lambda _: _pad_crop_batch(images, k_pad), lambda _: images, operand=None)
    images = lax.cond(random.bernoulli(k_rotate, 0.5), lambda _: _rotate_batch(images, k_rotate), lambda _: images, operand=None)
    images = lax.cond(random.bernoulli(k_color, 0.5), lambda _: _color_jitter_batch(images, k_color), lambda _: images, operand=None)
    images = (images - _MEAN[None, :, None, None]) / _STD[None, :, None, None]
    return images
