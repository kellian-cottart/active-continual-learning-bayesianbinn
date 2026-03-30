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


def split_dataset(dataloader, n_splits, fits_in_memory=True, augmentation=False):
    if fits_in_memory:
        images, labels = dataloader
        split_size = images.shape[0] // n_splits
        end_idx = split_size * n_splits

        images = images[:end_idx]
        labels = labels[:end_idx]

        splits = [
            (images[i * split_size:(i + 1) * split_size],
             labels[i * split_size:(i + 1) * split_size])
            for i in range(n_splits)
        ]
        return splits

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
                shuffle=False,
                drop_last=True
            )
        )

    return splits

# CIFAR strong augmentation (example)


def build_cifar_augmentation():
    return v2.Compose([
        v2.ToImage(),
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(15),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2),
        v2.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )])
