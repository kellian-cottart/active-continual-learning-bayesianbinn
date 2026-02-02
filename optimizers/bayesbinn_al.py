""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: BayesBiNN optimizer from Meng et al. 2021 + active learning cutoff added
"""


import optax
from jax.tree import map
from jax.numpy import zeros_like
from jax.tree import leaves
import jax.numpy as jnp
import jax


def bayesbinn_al(lr: float = 0.001, prior_strength: float = 1.0):
    """
    bayesbinn_al

    Args:
        lr: Learning rate for the optimizer.
    """

    def init(params):
        return {
            'step': 0,
            'prior': map(lambda x: zeros_like(x), params)
        }

    def update(gradients_mu, state, params=None):
        # check if all gradients are 0, if so, don't update
        sum_grads = jnp.sum(
            jnp.array([jnp.sum(jnp.abs(param)) for param in leaves(gradients_mu)]))
        should_update = sum_grads > 0.0

        def update_fn(grad_mu, param, prior):
            output = -lr*grad_mu - lr*prior_strength * (param-prior)
            return output
        updates = map(update_fn, gradients_mu, params, state["prior"])
        updates = map(lambda x: x * sum_grads, updates)
        state['step'] += should_update
        return updates, state

    return optax.GradientTransformation(init, update)
