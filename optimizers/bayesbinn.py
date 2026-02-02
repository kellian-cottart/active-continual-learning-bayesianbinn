""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: BayesBiNN optimizer from Meng et al. 2021
"""

import optax
from jax.tree import map
from jax.numpy import zeros_like
from jax.tree import leaves
import jax.numpy as jnp
import jax


def bayesbinn(lr: float = 0.001, prior_strength: float = 1.0):
    """
    bayesbinn

    Args:
        lr: Learning rate for the optimizer.
    """

    def init(params):
        return {
            'step': 0,
            'prior': map(lambda x: zeros_like(x), params)
        }

    def update(gradients_mu, state, params=None):
        def update_fn(grad_mu, param, prior):
            output = -lr*grad_mu - lr*prior_strength * (param-prior)
            return output
        updates = map(update_fn, gradients_mu, params, state["prior"])
        state['step'] += 1
        return updates, state

    return optax.GradientTransformation(init, update)
