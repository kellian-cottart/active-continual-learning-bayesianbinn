from customLayers import *
from equinox import Module
from equinox.nn import LayerNorm, AvgPool2d, Dropout, MaxPool2d
from jax.numpy import ravel
from jax.random import split
from jax import vmap


def forward(x, layers, key, backwards=False, temperature=1.0):
    """Forward pass for one sample through layers"""
    for layer in layers:
        l_key, key = split(key)
        if isinstance(layer, (BinaryBayesianLinear, BinaryBayesianConv2D, BinaryResidualBlock)):
            layer_fn = layer if backwards else layer.sample
            x = layer_fn(x, key=l_key, temperature=temperature)
        elif isinstance(layer, Dropout):
            l_key, key = split(key, 2)
            x = layer(x, inference=not backwards, key=l_key)
        else: 
            x = layer(x)
    return x

class BaseBinaryBayesianCNN(Module):
    layers: list
    temperature: float

    def __init__(self, key, layers=None, temperature=1.0, **kwargs):
        super().__init__()
        self.layers = []
        self.temperature = temperature

    def __call__(self, x, state, samples, key, *, backwards=False):
        keys = split(key, samples)
        # vmap over samples dimension
        x = vmap(forward, in_axes=(None, None, 0, None, None))(
            x, self.layers, keys, backwards, self.temperature
        )
        return x, state

class BinaryResidualBlock(Module):
    """Residual block for JAX Binary Bayesian CNNs"""
    conv1: BinaryBayesianConv2D
    norm1: LayerNorm
    conv2: BinaryBayesianConv2D
    norm2: LayerNorm
    downsample: BinaryBayesianConv2D | None
    activation_fn: callable

    def __init__(self, key, in_ch, out_ch, activation_fn, spatial_size, stride=1, use_bias=False):
        super().__init__()
        k1, k2, k3 = split(key, 3)

        # First conv
        self.conv1 = BinaryBayesianConv2D( key=k1, in_channels=in_ch, out_channels=out_ch,kernel_size=3, stride=stride, padding="SAME", use_bias=use_bias)
        self.norm1 = LayerNorm(shape=(out_ch, spatial_size, spatial_size), use_weight=False, use_bias=False)

        # Second conv
        self.conv2 = BinaryBayesianConv2D(
            key=k2, in_channels=out_ch, out_channels=out_ch,
            kernel_size=3, stride=1, padding="SAME", use_bias=use_bias
        )
        self.norm2 = LayerNorm(
            shape=(out_ch, spatial_size, spatial_size),  # same spatial size
            use_weight=False, use_bias=False
        )
        # Optional downsample if channels or stride change
        if stride != 1 or in_ch != out_ch:
            self.downsample = BinaryBayesianConv2D(
                key=k3, in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, stride=stride, padding="SAME", use_bias=use_bias
            )
        else:
            self.downsample = None

        self.activation_fn = activation_fn

    def __call__(self, x, key, temperature=1.0, backwards=False):
        k1, k2, k3 = split(key, 3)
        out = self.activation_fn(self.norm1(self.conv1.sample(x, key=k1, temperature=temperature)))
        out = self.norm2(self.conv2.sample(out, key=k2, temperature=temperature))
        identity = x if self.downsample is None else self.downsample.sample(x, key=k3, temperature=temperature)
        out = self.activation_fn(out + identity)
        return out
    
    def sample(self, x, key, temperature=1.0):
        return self.__call__(x, key, temperature, backwards=False)
