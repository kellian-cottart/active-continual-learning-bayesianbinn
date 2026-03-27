from customLayers import *
from equinox import Module
from equinox.nn import LayerNorm, AvgPool2d, Dropout, MaxPool2d
from jax.numpy import ravel
from jax.random import split
from jax import vmap


def forward(x, layers, key, backwards=False, temperature=1.0):
    for layer in layers:
        l_key, key = split(key, 2)
        if isinstance(layer, BinaryBayesianLinear) or isinstance(layer, BaseBinaryBayesianConv):
            layer_fn = layer if backwards else layer.sample
            x = layer_fn(x, key=l_key, temperature=temperature)
        elif isinstance(layer, Dropout):
            l_key, key = split(key, 2)
            x = layer(x, inference=not backwards, key=l_key)
        else:   # activation function
            x = layer(x)
    return x


class BaseBinaryBayesianCNN(Module):
    layers: list[BinaryBayesianConv2D]
    temperature: float

    def __init__(self, key, layers, temperature, use_bias=False, activation_fn=None, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.layers = []

    def __call__(self, x, state, samples, key, *, backwards=False):
        samples_keys = split(key, samples)
        x = vmap(forward, in_axes=(None, None, 0, None, None))(
            x, self.layers, samples_keys, backwards, self.temperature)
        return x, state


class BinaryBayesianCNNCifar100(BaseBinaryBayesianCNN):
    def __init__(self, key, layers, temperature, use_bias=False, activation_fn=None, **kwargs):
        super().__init__(key, layers, temperature, use_bias, activation_fn, **kwargs)
        conv_layers = [
            (3, 32),
            (32, 32),
            (32, 64),
            (64, 64),
        ]
        keys = split(key, len(conv_layers) * 2 + len(layers) * 2 - 2)
        for i, (in_channels, out_channels) in enumerate(conv_layers):
            self.layers.append(BinaryBayesianConv2D(
                key=keys[i * 2],
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding="SAME",
                use_bias=use_bias,
            ))
            self.layers.append(LayerNorm(
                shape=(out_channels, 32//2**i, 32//2**i),
                use_weight=False,
                use_bias=False,
            ))
            self.layers.append(AvgPool2d(kernel_size=2, stride=2, padding=0,))
            self.layers.append(Dropout(0.2))
            self.layers.append(activation_fn)
        self.layers.append(ravel)
        for i in range(len(layers) - 1):
            self.layers.append(BinaryBayesianLinear(
                in_features=layers[i],
                out_features=layers[i + 1],
                use_bias=use_bias,
                key=keys[len(conv_layers) * 2 + i * 2]
            ))
            self.layers.append(LayerNorm(
                shape=(layers[i + 1],),
                use_weight=False,
                use_bias=False,
            ))
            if i < len(layers) - 2:
                self.layers.append(activation_fn)


class BinaryBayesianCNNMNIST(BaseBinaryBayesianCNN):
    def __init__(self, key, layers, temperature, use_bias=False, activation_fn=None, **kwargs):
        super().__init__(key, layers, temperature, use_bias, activation_fn, **kwargs)
        conv_layers = [
            (1, 32),
            (32, 64),
            (64, 128),
            (128, 128),
        ]
        keys = split(key, len(conv_layers) * 2 + len(layers) * 2 - 2)
        for i, (in_channels, out_channels) in enumerate(conv_layers):
            self.layers.append(BinaryBayesianConv2D(
                key=keys[i * 2],
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding="SAME",
                use_bias=use_bias,
            ))
            self.layers.append(LayerNorm(
                shape=(out_channels, 28//2**i, 28//2**i),
                use_weight=False,
                use_bias=False,
            ))
            self.layers.append(AvgPool2d(kernel_size=2, stride=2, padding=0,))
            self.layers.append(activation_fn)
        self.layers.append(ravel)
        for i in range(len(layers) - 1):
            self.layers.append(BinaryBayesianLinear(
                in_features=layers[i],
                out_features=layers[i + 1],
                use_bias=use_bias,
                key=keys[len(conv_layers) * 2 + i * 2]
            ))
            self.layers.append(LayerNorm(
                shape=(layers[i + 1],),
                use_weight=False,
                use_bias=False,
            ))
            if i < len(layers) - 2:
                self.layers.append(activation_fn)


class BinaryBayesianCNNCore50(BaseBinaryBayesianCNN):
    def __init__(self, key, layers, temperature, use_bias=False, activation_fn=None, **kwargs):
        super().__init__(key, layers, temperature, use_bias, activation_fn, **kwargs)
        conv_layers = [
            (3, 64),
            (64, 64),
            (64, 128),
            (128, 128),
            (128, 256),
            (256, 256),
            (256, 512),
        ]
        keys = split(key, len(conv_layers) * 2 + len(layers) * 2 - 2)
        for i, (in_channels, out_channels) in enumerate(conv_layers):
            self.layers.append(BinaryBayesianConv2D(
                key=keys[i * 2],
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding="SAME",
                use_bias=use_bias,
            ))
            self.layers.append(LayerNorm(
                shape=(out_channels, 128//2**i, 128//2**i),
                use_weight=False,
                use_bias=False,
            ))
            self.layers.append(AvgPool2d(kernel_size=2, stride=2, padding=0,))
            self.layers.append(activation_fn)
        self.layers.append(ravel)
        for i in range(len(layers) - 1):
            self.layers.append(BinaryBayesianLinear(
                in_features=layers[i],
                out_features=layers[i + 1],
                use_bias=use_bias,
                key=keys[len(conv_layers) * 2 + i * 2]
            ))
            self.layers.append(LayerNorm(
                shape=(layers[i + 1],),
                use_weight=False,
                use_bias=False,
            ))
            if i < len(layers) - 2:
                self.layers.append(activation_fn)
