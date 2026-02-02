""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Miscellaneous utility functions
"""


import jax
import jax.numpy as jnp
import equinox as eqx
from copy import deepcopy
from torch.utils.data import TensorDataset
from torch import randperm, prod, tensor
from jax.tree import leaves, map
from utils.dataFunctions import to_dataloader


def compute_memory_occupation(configuration, model, opt_state, si_parameters=None, ewc_streaming_parameters=None, ewc_online_parameters=None, ewc_parameters=None, norm_params=None):
    """Compute memory occupation for model, optimizer, and regularization parameters."""
    model_memory = jnp.sum(
        jnp.array(leaves(map(lambda x: x.nbytes, eqx.filter(model, eqx.is_array)))))
    opt_memory = jnp.sum(jnp.array(
        leaves(map(lambda x: x.nbytes, eqx.filter(opt_state, eqx.is_array)))))
    extra_memory = 0

    if "si" in configuration and si_parameters is not None:
        extra_memory = jnp.sum(jnp.array(leaves(
            map(lambda x: x.nbytes, eqx.filter(si_parameters["omega"], eqx.is_array)))))
        extra_memory += jnp.sum(jnp.array(leaves(
            map(lambda x: x.nbytes, eqx.filter(si_parameters["w_k"], eqx.is_array)))))
    elif "stream-ewc" in configuration and ewc_streaming_parameters is not None:
        extra_memory = jnp.sum(jnp.array(leaves(map(lambda x: x.nbytes, eqx.filter(
            ewc_streaming_parameters["old_param"], eqx.is_array)))))
        extra_memory += jnp.sum(jnp.array(leaves(map(lambda x: x.nbytes, eqx.filter(
            ewc_streaming_parameters["fisher"], eqx.is_array)))))
    elif "online-ewc" in configuration and ewc_online_parameters is not None:
        extra_memory = jnp.sum(jnp.array(leaves(map(lambda x: x.nbytes, eqx.filter(
            ewc_online_parameters["old_param"], eqx.is_array)))))
        extra_memory += jnp.sum(jnp.array(leaves(map(lambda x: x.nbytes,
                                eqx.filter(ewc_online_parameters["fisher"], eqx.is_array)))))
    elif "ewc" in configuration and ewc_parameters is not None:
        extra_memory = jnp.sum(jnp.array(leaves(
            map(lambda x: x.nbytes, eqx.filter(ewc_parameters["old_param"], eqx.is_array)))))
        extra_memory += jnp.sum(jnp.array(leaves(
            map(lambda x: x.nbytes, eqx.filter(ewc_parameters["fisher"], eqx.is_array)))))
    elif "synapticmetaplasticity" in configuration["optimizer"] and norm_params is not None:
        extra_memory = jnp.sum(jnp.array(
            leaves(map(lambda x: x.nbytes, eqx.filter(norm_params, eqx.is_array)))))

    return model_memory, opt_memory, extra_memory


def load_ood_dataset(OOD, loader, task_params, configuration, num_classes, rng, FITS_IN_MEMORY):
    """Load and prepare OOD dataset based on the specified OOD type.

    Args:
        OOD: Type of OOD dataset to load
        loader: Dataset loader instance
        task_params: Task configuration parameters
        configuration: Full configuration dictionary
        num_classes: Number of classes
        rng: JAX random key
        FITS_IN_MEMORY: Whether dataset fits in memory

    Returns:
        tuple: (ood_dataloader, ood_core_keys, rng)
    """
    if "fashion" in OOD:
        _, test_ood, shape_ood, _ = loader.task_selection(
            "fashion")
    elif "pmnist" in OOD:
        _, test_ood, shape_ood, _ = loader.task_selection(
            "mnist")
        # permute the test_ood dataset with a random permutation
        ood_permutation = randperm(prod(tensor(shape_ood)))
        images, labels = test_ood[0][:]
        images = images.reshape(
            images.shape[0], -1)[:, ood_permutation].reshape(images.shape)
        test_ood[0] = TensorDataset(images, labels)
    elif "openloris" in OOD:
        new_tasks_params = deepcopy(task_params)
        new_tasks_params["classes"] = new_tasks_params.get(
            "ood_classes", ["pencil"])
        test_ood, _, shape_ood, _ = loader.task_selection(
            "openloris", **new_tasks_params)

    ood_dataloader = to_dataloader(
        test_ood, configuration["test_batch_size"], num_classes, fits_in_memory=FITS_IN_MEMORY)
    ookey, rng = jax.random.split(rng)
    ood_core_keys = jax.random.split(
        ookey, (configuration["n_tasks"], configuration["epochs"]))

    return ood_dataloader, ood_core_keys, rng
