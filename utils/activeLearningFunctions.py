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


def apply_active_learning_strategy(model, images, labels, samples, rng, init_state, predictions, opt_state=None):
    mode_map = {
        0: epistemic_wrapper,
        1: aleatoric_wrapper,
        2: predictive_wrapper,
        3: variation_ratio_wrapper,
        5: variation_ratio_with_label_wrapper,
        6: variation_ratio_wrapper,
    }
    mode = model.active_learning.get("mode", 0)
    base_threshold = model.active_learning.get("threshold", -jnp.inf)
    wrapper_fn = mode_map.get(mode, epistemic_wrapper)
    if mode == 4:
        return labels, predictions, jnp.ones(labels.shape, dtype=bool)
    if mode == 5:
        wrapper_fn = partial(wrapper_fn, labels=labels)
    uncertainty = wrapper_fn(
        images,
        init_state,
        model,
        model.active_learning.get("samples", samples),
        rng,
    )

    def compute_threshold(_):
        K = model.active_learning.get("samples", samples)
        current_ratio = opt_state["step"] / (opt_state["seen"] + 1.0)
        target_budget = model.active_learning.get("budget", 0.0)
        scaler = model.active_learning.get("scaler", 1.0)
        eps = 1e-8  # avoid division by zero
        error = current_ratio - target_budget
        th_cont = target_budget + \
            jnp.sign(error) * jnp.power(jnp.abs(error), scaler)
        idx = jnp.round(K * th_cont)
        idx = jnp.clip(idx, 0, K)
        threshold = idx / K
        return threshold

    threshold = jax.lax.cond(
        mode == 6,
        compute_threshold,
        lambda _: base_threshold,
        operand=None,
    )

    def threshold_branch(th):
        mask = uncertainty >= th
        mask = jnp.repeat(mask[:, None], labels.shape[1], axis=1)
        return labels, predictions, mask

    def argmax_branch(_):
        fn = jnp.argmin if model.active_learning.get(
            "reverse", False) else jnp.argmax
        idx = fn(uncertainty, keepdims=True)
        return (
            labels[idx, :],
            predictions[idx, :, :],
            jnp.ones(labels[idx, :].shape, dtype=bool),
        )

    labels, predictions, mask = jax.lax.cond(
        threshold > 0.0,
        threshold_branch,
        argmax_branch,
        threshold,
    )

    return labels, predictions, mask
