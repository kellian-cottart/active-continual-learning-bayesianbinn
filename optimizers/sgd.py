""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: SGD optimizer
"""
import optax
from jax.tree import map
import jax.numpy as jnp
from jax.tree import leaves


def sgd(lr: float = 0.001,):
    """
    SGD

    Args:
        lr: Learning rate for the optimizer.
    """

    def init(params):
        return {
            'step': 0,
        }

    def update(gradients, state, params=None):
        # check if all gradients are 0, if so, don't update
        sum_grads = jnp.sum(jnp.array([jnp.sum(jnp.abs(param))
                            for param in leaves(gradients)])) > 0.0
        return map(lambda g: -lr * g, gradients), {'step': state['step'] + sum_grads}

    return optax.GradientTransformation(init, update)
