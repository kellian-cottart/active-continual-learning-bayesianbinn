from customLayers import *
from equinox import Module
from equinox.nn import LayerNorm, AvgPool2d, Dropout, MaxPool2d, BatchNorm
from jax.numpy import ravel, repeat
from jax.random import split
from jax import vmap
from functools import partial


def forward(x, state, layers, key, backwards=False, temperature=1.0):
    """Forward pass for one sample through layers"""
    for layer in layers:
        l_key, key = split(key)
        if isinstance(layer, (BinaryBayesianLinear, BinaryBayesianConv2D)):
            layer_fn = layer if backwards else layer.sample
            x = layer_fn(x, key=l_key, temperature=temperature)
        elif isinstance(layer, BatchNorm):
            x, state = layer(x, state)
        elif isinstance(layer, Dropout):
            l_key, key = split(key, 2)
            x = layer(x, inference=not backwards, key=l_key)
        else:
            x = layer(x)
    return x, state


class BaseBinaryBayesianCNN(Module):
    layers: list
    temperature: float
    active_learning: dict
    
    def __init__(self, key, layers=None, temperature=1.0, active_learning=None, **kwargs):
        super().__init__()
        self.layers = []
        self.temperature = temperature
        self.active_learning = active_learning

    def __call__(self, x, state, samples, key, *, backwards=False):
        keys = split(key, samples)
        # vmap over samples dimension
        states = jax.tree.map(lambda x: repeat(
            x[None, ...], samples, axis=0), state)
        x, states = vmap(forward,
                         in_axes=(None, 0, None, 0, None, None)
                         )(
            x, states, self.layers, keys, backwards, self.temperature
        )
        # take first sample's state (they should all be the same)
        state = jax.tree.map(lambda x: x[0], states)
        return x, state
