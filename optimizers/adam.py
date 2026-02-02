""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: ADAMW optimizer with weight decay
"""


import optax
from jax.tree import map
import jax.numpy as jnp
from jax.tree import leaves


def adamw(learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
    """
    Adam optimizer with weight regularization

    Args:
        learning_rate: Learning rate for the optimizer.
        beta1: Exponential decay rate for the first moment estimates.
        beta2: Exponential decay rate for the second moment estimates.
        eps: Small constant for numerical stability.
        weight_decay: Weight decay (L2 regularization) coefficient.
    """

    def init(params):
        return {
            'step': 0,
            'm': map(lambda p: jnp.zeros_like(p), params),
            'v': map(lambda p: jnp.zeros_like(p), params),
        }

    def update(gradients, state, params=None):
        # check if all gradients are 0, if so, don't update
        sum_grads = jnp.sum(jnp.array([jnp.sum(jnp.abs(param))
                            for param in leaves(gradients)])) > 0.0
        step = state['step'] + sum_grads

        # Add weight decay to gradients
        if weight_decay > 0.0 and params is not None:
            gradients = map(lambda g, p: g + weight_decay *
                            p, gradients, params)

        # Update biased first moment estimate (only if sum_grads > 0)
        m = map(lambda m_prev, g: sum_grads * (beta1 * m_prev + (1 - beta1) * g) + (1 - sum_grads) * m_prev,
                state['m'], gradients)

        # Update biased second moment estimate (only if sum_grads > 0)
        v = map(lambda v_prev, g: sum_grads * (beta2 * v_prev + (1 - beta2) * g ** 2) + (1 - sum_grads) * v_prev,
                state['v'], gradients)

        # Bias correction
        m_hat = map(lambda m_val: m_val / (1 - beta1 ** step), m)
        v_hat = map(lambda v_val: v_val / (1 - beta2 ** step), v)

        # Compute update
        updates = map(lambda m_val, v_val: -learning_rate * m_val /
                      (jnp.sqrt(v_val) + eps), m_hat, v_hat)

        updates = map(lambda x: x * sum_grads, updates)

        return updates, {'step': step, 'm': m, 'v': v}

    return optax.GradientTransformation(init, update)
