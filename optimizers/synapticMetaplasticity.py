""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Synaptic metaplasticity optimizer from Laborieux et al. 2021
"""

from jax.numpy import zeros_like, where
import optax
from jax.tree import map
from jax.lax import tanh, abs, sqrt, sign
import jax


def synapticMetaplasticity(lr: float = 0.001, metaplasticity: float = 0, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0):
    """
    """

    def init(params):
        # initialize moments
        exp_avg = map(lambda p: zeros_like(p), params)
        exp_avg_sq = map(lambda p: zeros_like(p), params)
        return {
            'step': 0,
            'exp_avg': exp_avg,
            'exp_avg_sq': exp_avg_sq
        }

    def update(gradients, state, params=None):
        step = state['step'] + 1
        gradients = map(lambda g, p: g + weight_decay*p, gradients, params)
        # update moment 1
        exp_avg = map(lambda g, m: b1 * m + (1 - b1) *
                      g, gradients, state['exp_avg'])
        # update moment 2
        exp_avg_sq = map(lambda g, v: b2 * v + (1 - b2) *
                         g**2, gradients, state['exp_avg_sq'])
        bias_correction1 = 1 - b1**step
        bias_correction2 = 1 - b2**step

        def update_adam(param, m, v):
            corrected_step = lr * sqrt(bias_correction2) / bias_correction1
            step_size = corrected_step * m / (sqrt(v) + eps)
            lr_meta = 1
            if not param.ndim == 1:  # if the parameter is not the bias, we do the metaplastic update
                f_meta = 1 - tanh(metaplasticity*abs(param))**2
                lr_meta = where(
                    sign(param) == sign(m), f_meta, 1)
            return - lr_meta * step_size
        # Update the parameters and moments
        updates = map(update_adam, params, exp_avg, exp_avg_sq)
        return updates, {'step': step, 'exp_avg': exp_avg, 'exp_avg_sq': exp_avg_sq}

    return optax.GradientTransformation(init, update)
