""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Synaptic metaplasticity optimizer from Bonnet et al. 2025
"""

import optax
from jax.numpy import where, clip
from jax.tree import map, leaves
from customLayers.linears import GaussianParameter
import jax.numpy as jnp


def discriminant(param):
    """ Discriminate between Bayesian parameters"""
    return hasattr(param, 'mu') and hasattr(param, 'sigma') and param.mu is not None and param.sigma is not None


def mesu(
        lr_mu: float = 1,
        lr_sigma: float = 1,
        sigma_prior: float = 0.2,
        mu_prior: float = 0,
        N: int = 1e5,
        clamp_grad: float = 0.) -> optax.GradientTransformation:
    """
    Optax gradient transformation for MESU.

    Args:
        lr_mu: Learning rate for mu parameters.
        lr_sigma: Learning rate for sigma parameters.
        mu_prior: Prior mean value.
        N: Number of batches for synaptic memory.
        clamp_grad: Gradient clamping threshold.

    Returns:
        optax.GradientTransformation: The MESU update rule.
    """

    def init(params):
        # Check if all parameters are bayesian i.e that the path name contains mu or sigma
        return {'step': 0, }

    def update(gradients, state, params=None):
        # Update the parameters using sgd
        def update_mesu(param, grad):
            """ Update the parameters based on the gradients and the prior"""
            # If clamp_grad > 0, then clamp gradients between -clamp_grad/sigma and clamp_grad/sigma
            if clamp_grad > 0:
                grad = map(lambda x: clip(x, -clamp_grad /
                           param.sigma, clamp_grad/param.sigma), grad)
            variance = param.sigma ** 2
            prior_attraction_mu = variance * \
                (mu_prior - param.mu) / (N * (sigma_prior ** 2))
            prior_attraction_sigma = param.sigma * \
                (sigma_prior ** 2 - variance) / (N * (sigma_prior ** 2))
            mu_update = lr_mu * (-variance * grad.mu + prior_attraction_mu)
            sigma_update = lr_sigma * 0.5 * (-variance *
                                             grad.sigma + prior_attraction_sigma)
            return GaussianParameter(mu_update, sigma_update)
        # check if all gradients are 0, if so, don't update
        sum_grads = jnp.sum(jnp.array([jnp.sum(jnp.abs(param))
                            for param in leaves(gradients)])) > 0.0
        updates = map(update_mesu, params, gradients, is_leaf=discriminant)
        # if sum_grads is 0, then we don't want to update the parameters, we don't want forgetting,
        # hence updates must be 0
        updates = map(lambda x: GaussianParameter(
            x.mu * sum_grads, x.sigma * sum_grads), updates, is_leaf=discriminant)
        return updates, {'step': state['step'] + sum_grads}

    return optax.GradientTransformation(init, update)
