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
from torch import randperm


def numpy_collate(batch):
    return map(np.asarray, default_collate(batch))


class NumpyLoader(DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)

    # define subscript method to access the dataset
    def __getitem__(self, index):
        return self.dataset[index]


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


def to_dataloader(data, batch_size, num_classes, fits_in_memory=True):
    loader = []
    if fits_in_memory:
        for dataset in data:
            images, labels = prepare_data(
                dataset[:][0], dataset[:][1], batch_size, num_classes)
            loader.append((images, labels))
    else:
        for dataset in data:
            dataloader = NumpyLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True)
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


def split_dataset(dataloader, n_splits, fits_in_memory=True):
    if fits_in_memory:
        images, labels = dataloader
        split_size = images.shape[0] // n_splits
        end_idx = split_size * n_splits
        images = images[:end_idx]
        labels = labels[:end_idx]
        splits = []
        for i in range(n_splits):
            start = i * split_size
            end = start + split_size
            splits.append((images[start:end], labels[start:end]))
    else:
        total_size = len(dataloader.dataset)
        split_size = total_size // n_splits
        splits = []
        for i in range(n_splits):
            start = i * split_size
            end = start + split_size if i < n_splits - 1 else total_size
            subset = Subset(
                dataloader.dataset, list(range(start, end)))
            splits.append(NumpyLoader(
                subset, batch_size=dataloader.batch_size, shuffle=False, drop_last=True))
    return splits
