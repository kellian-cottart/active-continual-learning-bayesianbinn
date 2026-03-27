""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: BiMU optimizer from Cottart et al. 2026
"""

import optax
from jax.lax import tanh, abs
from jax.tree import map, leaves
import jax.numpy as jnp
import jax


def bimu(
    lr: float = 1.0,
    lr_max: float = 1.0,
    likelihood_multiplier: float = 1.0,
    kl_multiplier: float = 1.0,
    N: int = 60000,
) -> optax.GradientTransformation:
    """
    Optax gradient transformation for Binary Metaplasticity for Uncertainty (BiMU).

    Args:
        lr (float): The multiplying factor scaling the update for increased convergence.
        lr_max (float): The maximum learning rate of the optimizer for a synapse.
        likelihood_multiplier (float): The likelihood_multiplier, the higher the more asymmetric the learning rate.
        N (int): The memory window factor.

    Returns:
        optax.GradientTransformation: The BHU update rule.
    """
    def init(params):
        return {'step': 0, 'seen': 0}

    def update(gradients, state, params=None):
        def update_bhu(param, grad):
            tanh_param = tanh(param)
            inv_cosh_sqr = kl_multiplier * (1-tanh_param*tanh_param)
            grad = likelihood_multiplier * grad
            second_derivative = 2.0 * abs(grad) + 1 / lr_max
            lr_asymmetry = 1 / (inv_cosh_sqr + 2.0*grad *
                                tanh_param + second_derivative)
            forgetting = (param * inv_cosh_sqr) / N
            return - lr_asymmetry*(lr*grad + forgetting)

        # check if all gradients are 0, if so, don't update
        sum_grads = jnp.sum(jnp.array([jnp.sum(jnp.abs(param))
                            for param in leaves(gradients)])) > 0.0
        updates = map(update_bhu, params, gradients)
        # if sum_grads is 0, then we don't want to update the parameters, we don't want forgetting,
        # hence updates must be 0
        updates = map(lambda x: x * sum_grads, updates)
        return updates, {'step': state['step'] + sum_grads, 'seen': state['seen'] + 1}
    return optax.GradientTransformation(init, update)
