""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Binary Bayesian MLP classifiers with different normalization layers
"""

from customLayers import BinaryBayesianLinear
from equinox import Module
from equinox.nn import LayerNorm
from jax.numpy import ravel
from jax.random import split
from jax import vmap
from jax.nn import relu


def forward(x, layers, key, backwards=False, temperature=1):
    for layer in layers:
        if isinstance(layer, BinaryBayesianLinear):
            layer_fn = layer if backwards else layer.sample
            l_key, key = split(key, 2)
            x = layer_fn(x, key=l_key, temperature=temperature)
        else:   # activation function
            x = layer(x)
    return x


class BaseBinaryBayesianMLP(Module):
    layers: list[BinaryBayesianLinear]
    temperature: float
    active_learning: dict

    def __init__(self, key, layers, temperature, use_bias=False, activation_fn=None, active_learning=None, **kwargs):
        super().__init__()
        keys = split(key, len(layers) - 1)
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(BinaryBayesianLinear(
                in_features=layers[i],
                out_features=layers[i + 1],
                use_bias=use_bias,
                key=keys[i]
            ))
            self.layers.append(LayerNorm(
                shape=(layers[i + 1],),
                use_weight=False,
                use_bias=False,
            ))
            if i < len(layers) - 2 and activation_fn:
                self.layers.append(activation_fn)
        self.temperature = temperature
        self.active_learning = active_learning

    def __call__(self, x, state, samples, key, *, backwards=False):
        samples_keys = split(key, samples)
        x = ravel(x)
        x = vmap(forward, in_axes=(None, None, 0, None, None))(
            x, self.layers, samples_keys, backwards, self.temperature)
        return x, state
