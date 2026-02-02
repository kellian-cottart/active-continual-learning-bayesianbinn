""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Binary MLP classifiers with different normalization layers
"""

# Define a simple model using Equinox
from customLayers import BinaryLinear
from equinox import Module, filter, is_array, partition, combine
from jax.random import split
from jax.numpy import ravel
from customLayers import *
from equinox.nn import LayerNorm, BatchNorm
from jax.nn import relu


class BaseBinaryMLP(Module):
    layers: list[BinaryLinear]
    active_learning: dict

    def __init__(self, key, layers=[784, 50, 10], use_bias=None, activation_fn=None, norm_fn=None, norm_params=None, active_learning=None, **kwargs):
        keys = split(key, len(layers) - 1)
        self.layers = []
        # Add `BayesianLinear` and `relu` alternately except after the last linear layer
        for i in range(len(layers) - 1):
            # Add `BayesianLinear` layer
            self.layers.append(BinaryLinear(
                in_features=layers[i],
                out_features=layers[i + 1],
                key=keys[i]
            ))
            if norm_fn == LayerNorm:
                norm_params["shape"] = (layers[i + 1],)
                self.layers.append(norm_fn(**norm_params))
            elif norm_fn == BatchNorm:
                norm_params["input_size"] = layers[i + 1]
                self.layers.append(norm_fn(**norm_params))
            if i < len(layers) - 2 and activation_fn:
                self.layers.append(activation_fn)
        self.active_learning = active_learning

    def __call__(self, x, state, *, backwards=False):
        x = ravel(x)
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                x, state = layer(x, state)
            else:
                x = layer(x)
        return x, state

    def return_tree_norm(self):
        """ Return a jax pytree with only the LayerNorm """
        # remove layers that are not LayerNorm
        def discriminant(leaf):
            return isinstance(leaf, LayerNorm) or isinstance(leaf, BatchNorm)

        def filter_spec(leaf):
            return hasattr(leaf, "weight") or hasattr(leaf, "bias")
        return filter(filter(
            self, filter_spec=filter_spec, is_leaf=discriminant), is_array)

    def load_tree_norm(self, tree):
        """ Load a jax pytree with only the LayerNorm """
        def discriminant(leaf):
            return isinstance(leaf, LayerNorm) or isinstance(leaf, BatchNorm)

        def filter_spec(leaf):
            return hasattr(leaf, "weight") or hasattr(leaf, "bias")

        # split self into norm and normless
        norm, normless = partition(
            self, filter_spec=filter_spec, is_leaf=discriminant)
        # combine normless tree with the tree containing the task batchnorm
        combined = combine(normless, tree)
        # recombine to retrieve string parameters of norm
        return combine(combined, norm)


class BinaryMLPBatchNorm(BaseBinaryMLP):
    def __init__(self, key, layers=[784, 50, 10], use_bias=None, activation_fn=None, ** kwargs):
        norm_fn = BatchNorm
        norm_params = {
            "axis_name": "batch",
            "channelwise_affine": True,
            "momentum": 0.1,
            "eps": 1e-5,
            "inference": False,
        }
        super().__init__(key, layers, use_bias, activation_fn, norm_fn, norm_params, **kwargs)


class BinaryMLPLayerNorm(BaseBinaryMLP):
    def __init__(self, key, layers=[784, 50, 10], use_bias=None, activation_fn=None, ** kwargs):
        norm_fn = LayerNorm
        norm_params = {
            "use_weight": True,
            "use_bias": True,
        }
        super().__init__(key, layers, use_bias, activation_fn, norm_fn, norm_params, **kwargs)
