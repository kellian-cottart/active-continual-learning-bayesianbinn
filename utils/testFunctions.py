""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Testing functions for evaluating models
"""


import jax
from jax.numpy import expand_dims, ndarray, array
import equinox as eqx
from functools import partial
from jax.tree import map
from utils.uncertaintyFunctions import compute_uncertainty
from typing import NamedTuple


class UncertaintyMetrics(NamedTuple):
    """Structure to hold uncertainty metrics in a JAX-compatible format."""
    aleatoric: ndarray
    epistemic: ndarray
    variation_ratio: ndarray


def main_test_fn(test_dataset, num_classes, test_samples, test_ck, model, model_state, norm_params, is_permuted=False, max_permutations=None, permutations=None, fits_in_memory=False):
    if is_permuted == True:
        images, labels = test_dataset[0]
        accuracies, uncertainties = test_fn_permuted_mnist(
            model=model,
            images=images,
            labels=labels,
            rng=test_ck,
            state=model_state,
            max_parallel_permutation=max_permutations,
            permutations=permutations,
            test_samples=test_samples,
            norm_params=norm_params,
        )
    elif fits_in_memory == True:
        accuracies = []
        uncertainties = []
        for task_id, task_dataset in enumerate(test_dataset):
            if norm_params is not None:
                model = model.load_tree_norm(
                    map(lambda x: x[task_id], norm_params))
            images = task_dataset[0]
            labels = task_dataset[1]
            acc, pred = test_fn_memory(model=model,
                                       images=images,
                                       labels=labels,
                                       rng=test_ck,
                                       state=model_state,
                                       test_samples=test_samples)
            accuracies.append(acc)
            uncertainties.append(compute_uncertainty(pred))
        try:
            accuracies = array(accuracies)
            uncertainties = UncertaintyMetrics(
                aleatoric=array([u[0] for u in uncertainties]),
                epistemic=array([u[1] for u in uncertainties]),
                variation_ratio=array([u[2] for u in uncertainties])
            )
        except:
            accuracies = list(accuracies)
            uncertainties = UncertaintyMetrics(
                aleatoric=list(u[0] for u in uncertainties),
                epistemic=list(u[1] for u in uncertainties),
                variation_ratio=list(u[2] for u in uncertainties)
            )
    else:
        accuracies = []
        predictions = []
        uncertainties = []
        for task_id, task_dataset in enumerate(test_dataset):
            if norm_params is not None:
                model = model.load_tree_norm(
                    map(lambda x: x[task_id], norm_params))
            task_accuracies, task_predictions, task_uncertainties = [], [], []
            for images, labels in task_dataset:
                images = array(images)
                labels = jax.nn.one_hot(labels, num_classes=num_classes)
                acc, pred = compute_accuracy(
                    model=model,
                    images=images,
                    labels=labels,
                    state=model_state,
                    samples=test_samples,
                    rng=test_ck)
                task_accuracies.append(acc)
                task_predictions.append(pred)
                task_uncertainties.append(compute_uncertainty(pred))
            accuracies.append(array(task_accuracies).reshape(-1,))
            predictions.append(array(task_predictions).reshape(-1,))
            uncertainties.append((array([u[0] for u in task_uncertainties]).reshape(-1,),
                                  array(
                                      [u[1] for u in task_uncertainties]).reshape(-1,),
                                  array([u[2] for u in task_uncertainties]).reshape(-1,)))

        try:
            accuracies = array(accuracies)
            predictions = array(predictions)
            uncertainties = UncertaintyMetrics(
                aleatoric=array([u[0] for u in uncertainties]),
                epistemic=array([u[1] for u in uncertainties]),
                variation_ratio=array([u[2] for u in uncertainties])
            )
        except:
            accuracies = list(accuracies)
            predictions = list(predictions)
            uncertainties = UncertaintyMetrics(
                aleatoric=list(u[0] for u in uncertainties),
                epistemic=list(u[1] for u in uncertainties),
                variation_ratio=list(u[2] for u in uncertainties)
            )
    return accuracies, uncertainties


def test_fn_bayesian(model, images, state, samples, rng):
    return jax.vmap(model, axis_name="batch",  in_axes=(0, None, None, None), out_axes=(0, None))(images, state, samples, rng)


def test_fn_deterministic(model, images, state):
    return jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))(images, state)


@eqx.filter_jit
def compute_accuracy(model, images, labels, state, samples=None, rng=None):
    if samples is not None:
        predictions, _ = test_fn_bayesian(model, images, state, samples, rng)
        output = jax.nn.log_softmax(predictions, axis=-1).mean(axis=1)
    else:
        predictions, _ = test_fn_deterministic(model, images, state)
        output = jax.nn.log_softmax(predictions, axis=-1)
    accuracy = (output.argmax(axis=-1) == labels.argmax(axis=-1))
    return accuracy, predictions


@eqx.filter_jit
def test_fn_memory(model: eqx.Module,
                   images: ndarray,
                   labels: ndarray,
                   rng,
                   state,
                   test_samples=None,):
    def compute_accuracies_predictions(images, labels, test_samples, model, state, rng):
        def scan_f(carry, data):
            image, label = data
            accuracy, predictions = compute_accuracy(
                model, image, label, state, test_samples, rng)
            return carry, (accuracy, predictions)

        _, (accuracies, predictions) = jax.lax.scan(
            f=scan_f,
            init=(),
            xs=(images, labels))
        return accuracies, predictions

    accuracies, predictions = compute_accuracies_predictions(
        images, labels, test_samples, model, state, rng)
    accuracies = expand_dims(accuracies, 0)
    predictions = predictions.reshape(
        predictions.shape[0] * predictions.shape[1], *predictions.shape[2:])
    return accuracies, predictions


@eqx.filter_jit
def test_fn_permuted_mnist(model: eqx.Module,
                           images: ndarray,
                           labels: ndarray,
                           rng,
                           state,
                           max_parallel_permutation=1,
                           permutations=None,
                           test_samples=None,
                           norm_params=None):
    if max_parallel_permutation < permutations.shape[0]:
        batched_permutations = permutations.reshape(permutations.shape[0] // max_parallel_permutation,
                                                    max_parallel_permutation, *permutations.shape[1:])
        if norm_params is not None:
            def reshape(x):
                return x.reshape(
                    x.shape[0] // max_parallel_permutation, max_parallel_permutation, *x.shape[1:])
            norm_params = map(reshape, norm_params)
    else:
        batched_permutations = expand_dims(permutations, 0)
        if norm_params is not None:
            norm_params = map(lambda x: expand_dims(x, 0), norm_params)

    def distribute_batches_permutations(carry, data, model):
        (batched_permutations, batch_norm_param) = data

        def vmap_perms(permutation, batch_norm_p, model):
            def distribute_batches(carry, data, model):
                (image, label) = data
                accuracy, predictions = compute_accuracy(
                    model, image, label, state, test_samples, rng)
                aleatoric_u, epistemic_u, variation_ratio_u = compute_uncertainty(
                    predictions)
                uncertainty = UncertaintyMetrics(
                    aleatoric=aleatoric_u,
                    epistemic=epistemic_u,
                    variation_ratio=variation_ratio_u
                )
                return carry, (accuracy, uncertainty)

            permuted_images = images.reshape(
                images.shape[0], images.shape[1], -1)[:, :, permutation].reshape(images.shape)
            if batch_norm_p is not None:
                model = model.load_tree_norm(batch_norm_p)

            _, (accuracy, uncertainty) = jax.lax.scan(
                f=partial(distribute_batches, model=model),
                init=(),
                xs=(permuted_images, labels))

            return accuracy, uncertainty

        return carry, jax.vmap(vmap_perms, in_axes=(0, 0 if batch_norm_param is not None else None, None))(batched_permutations, batch_norm_param, model)

    _, (accuracies, uncertainty) = jax.lax.scan(
        f=partial(distribute_batches_permutations, model=model),
        init=(),
        xs=(batched_permutations, norm_params)
    )

    accuracies = accuracies.reshape(
        accuracies.shape[0] * accuracies.shape[1], *accuracies.shape[2:]).mean(axis=1)

    uncertainty = UncertaintyMetrics(
        aleatoric=uncertainty.aleatoric.reshape(
            uncertainty.aleatoric.shape[0] * uncertainty.aleatoric.shape[1], -1),
        epistemic=uncertainty.epistemic.reshape(
            uncertainty.epistemic.shape[0] * uncertainty.epistemic.shape[1], -1),
        variation_ratio=uncertainty.variation_ratio.reshape(
            uncertainty.variation_ratio.shape[0] * uncertainty.variation_ratio.shape[1], -1)
    )

    return accuracies, uncertainty
