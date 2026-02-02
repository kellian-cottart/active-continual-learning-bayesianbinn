""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Uncertainty functions for active learning strategies
"""

import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def variation_ratio(predictions):
    # if no samples, add a dimension
    if len(predictions.shape) != 3:
        predictions = jnp.expand_dims(predictions, axis=1)
    # compute softmax (n_elements, n_samples, n_classes)
    softmax_predictions = jax.nn.softmax(predictions, axis=-1)
    probs = jax.nn.softmax(softmax_predictions, axis=-1)
    avg_probs = jnp.mean(probs, axis=0)
    avg_choice = jnp.argmax(avg_probs, axis=-1)
    choice = jnp.argmax(probs, axis=-1)
    var_ratio = 1 - jnp.mean(choice == avg_choice)
    return var_ratio


@eqx.filter_jit
def variation_ratio_with_true_label(predictions, labels):
    # if no samples, add a dimension
    if len(predictions.shape) != 3:
        predictions = jnp.expand_dims(predictions, axis=1)
    # compute softmax (n_elements, n_samples, n_classes)
    softmax_predictions = jax.nn.softmax(predictions, axis=-1)
    probs = jax.nn.softmax(softmax_predictions, axis=-1)
    choice = jnp.argmax(probs, axis=-1)
    true_choice = jnp.argmax(labels, axis=-1)
    var_ratio = 1 - jnp.mean(choice == true_choice)
    return var_ratio


@eqx.filter_jit
def element_uncertainty(predictions):
    # if no samples, add a dimension
    if len(predictions.shape) != 3:
        predictions = jnp.expand_dims(predictions, axis=1)
    # compute softmax (n_elements, n_samples, n_classes)
    softmax_predictions = jax.nn.softmax(predictions, axis=-1)
    mean_prediction = jnp.mean(softmax_predictions, axis=1)
    predictive = -jnp.sum(mean_prediction *
                          jnp.log2(mean_prediction), axis=-1)
    aleatoric = -jnp.mean(jnp.sum(softmax_predictions *
                          jnp.log2(softmax_predictions), axis=-1), axis=1)
    # replace nan by 0
    aleatoric = jnp.nan_to_num(aleatoric)
    predictive = jnp.nan_to_num(predictive)
    epistemic = predictive - aleatoric
    return aleatoric, epistemic


@eqx.filter_jit
def compute_uncertainty(predictions):
    # Compute aleatoric and epistemic uncertainty OUT: (n_elements, n_classes)
    aleatoric_uncertainty, epistemic_uncertainty = element_uncertainty(
        predictions)
    variation_ratios = jax.vmap(variation_ratio)(predictions)
    return aleatoric_uncertainty, epistemic_uncertainty, variation_ratios


@eqx.filter_jit
def compute_roc_auc(uncertainty, uncertainty_ood):
    # Compute ROC AUC using vectorized operations
    max_val = jnp.maximum(uncertainty.max(), uncertainty_ood.max())
    thresholds = jnp.linspace(0, max_val, 1000)

    # Vectorized comparison
    tpr = jnp.mean(uncertainty[None, :] < thresholds[:, None], axis=1)
    fpr = jnp.mean(uncertainty_ood[None, :] < thresholds[:, None], axis=1)

    # Compute AUC using the trapezoidal rule
    auc = jnp.trapezoid(y=tpr, x=fpr)
    return auc
