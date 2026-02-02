""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: BayesBiNN MLP classifiers with different normalization layers
"""
from customLayers import BayesBiNNLinear
from equinox.nn import LayerNorm
from jax.nn import relu
from jax.numpy import ravel
from jax.random import split
from equinox import Module, field
from jax import vmap


class BaseBayesBiNNMLP(Module):
    layers: list[BayesBiNNLinear]
    temperature: float = field(static=True)
    active_learning: dict

    def __init__(self, key, layers, temperature, use_bias=False, activation_fn=None, active_learning=None, **kwargs):
        super().__init__()
        keys = split(key, len(layers) - 1)
        self.layers = []
        self.temperature = temperature
        for i in range(len(layers) - 1):
            self.layers.append(BayesBiNNLinear(
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
        self.active_learning = active_learning

    def __call__(self, x, state, samples, key, *, backwards=False):
        x = ravel(x)
        if backwards:
            for layer in self.layers:
                if isinstance(layer, BayesBiNNLinear):
                    l_key, key = split(key, 2)
                    if backwards:
                        x = layer(x)
                else:   # activation function
                    x = layer(x)
        else:
            s_key = split(key, samples)

            def forward(x, layers, s_l_key):
                for layer in layers:
                    if isinstance(layer, BayesBiNNLinear):
                        l_key, s_l_key = split(s_l_key, 2)
                        x = layer.sample(x, key=l_key)
                    else:
                        x = layer(x)
                return x
            x = vmap(forward, in_axes=(None, None, 0))(x, self.layers, s_key)

        return x, state
