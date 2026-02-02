""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Active learning wrappers for uncertainty functions and training
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial
from utils.uncertaintyFunctions import variation_ratio, element_uncertainty, variation_ratio_with_true_label


@eqx.filter_jit
def variation_ratio_wrapper(images, init_state, model, samples, rng):
    predictions, _ = jax.vmap(model, in_axes=(0, None, None, None), out_axes=(
        0, None))(images, init_state, samples, rng) if samples > 0 else jax.vmap(
        model, in_axes=(0, None, None), out_axes=(0, None))(images, init_state)
    return jax.vmap(variation_ratio)(predictions)


@eqx.filter_jit
def epistemic_wrapper(images, init_state, model, samples, rng):
    predictions, _ = jax.vmap(model, in_axes=(0, None, None, None), out_axes=(
        0, None))(images, init_state, samples, rng) if samples > 0 else jax.vmap(
        model, in_axes=(0, None), out_axes=(0, None))(images, init_state)
    _, epistemic_u = element_uncertainty(predictions)
    return epistemic_u


@eqx.filter_jit
def aleatoric_wrapper(images, init_state, model, samples, rng):
    predictions, _ = jax.vmap(model, in_axes=(0, None, None, None), out_axes=(
        0, None))(images, init_state, samples, rng) if samples > 0 else jax.vmap(
        model, in_axes=(0, None), out_axes=(0, None))(images, init_state)
    aleatoric_u, _ = element_uncertainty(predictions)
    return aleatoric_u


@eqx.filter_jit
def predictive_wrapper(images, init_state, model, samples, rng):
    predictions, _ = jax.vmap(model, in_axes=(0, None, None, None), out_axes=(
        0, None))(images, init_state, samples, rng) if samples > 0 else jax.vmap(
        model, in_axes=(0, None, None), out_axes=(0, None))(images, init_state)
    aleatoric_u, epistemic_u = element_uncertainty(predictions)
    return aleatoric_u + epistemic_u


@eqx.filter_jit
def variation_ratio_with_label_wrapper(images, init_state, model, samples, rng, labels):
    predictions, _ = jax.vmap(model, in_axes=(0, None, None, None), out_axes=(
        0, None))(images, init_state, samples, rng) if samples > 0 else jax.vmap(
        model, in_axes=(0, None), out_axes=(0, None))(images, init_state)
    vr = jax.vmap(variation_ratio_with_true_label)(predictions, labels)
    return vr


def apply_active_learning_strategy(model, images, labels, samples, rng, init_state, predictions):
    mode_map = {
        0: epistemic_wrapper,
        1: aleatoric_wrapper,
        2: predictive_wrapper,
        3: variation_ratio_wrapper,
        5: variation_ratio_with_label_wrapper
    }
    mode = model.active_learning.get("mode", 0)
    wrapper_fn = mode_map.get(mode, epistemic_wrapper)
    if mode == 4:
        return labels, predictions, jnp.ones(labels.shape, dtype=bool)
    if mode == 5:
        wrapper_fn = partial(
            wrapper_fn, labels=labels)
    uncertainty = wrapper_fn(images, init_state, model,
                             model.active_learning.get("samples", samples), rng)
    if model.active_learning.get("threshold", -jnp.inf) > 0.0:
        mask = jnp.repeat(jnp.expand_dims(uncertainty >= model.active_learning.get(
            "threshold", 0.0), axis=-1), labels.shape[1], axis=-1)
    else:
        fn = jnp.argmin if model.active_learning.get(
            "reverse", False) else jnp.argmax
        idx = fn(uncertainty, keepdims=True)
        predictions = predictions[idx, :, :]
        labels = labels[idx, :]
        mask = jnp.ones(labels.shape, dtype=bool)
    return labels, predictions, mask
