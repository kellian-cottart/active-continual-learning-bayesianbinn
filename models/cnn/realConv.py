from equinox import Module
from jax.numpy import ravel
from jax.random import split
from typing import Callable
from customLayers.convolutions.realConv import RealConv2D
from equinox.nn import Linear, LayerNorm, MaxPool2d, AvgPool2d, BatchNorm, Dropout


class BaseRealCNN(Module):
    layers: list

    def __init__(self, layers=None):
        super().__init__()
        self.layers = []

    def __call__(self, x, state, key, *, backwards=False):
        """
        Forward pass for batched input x.
        - is_training: controls BatchNorm behavior
        - key: unused but kept for API consistency
        """
        # generate len(layers) keys for dropout and batchnorm if needed
        if key is not None:
            new_key = split(key, len(self.layers))
        for i, layer in enumerate(self.layers):
            if isinstance(layer, BatchNorm):
                x, state = layer(x, state)
            elif isinstance(layer, Dropout):
                x = layer(
                    x, key=new_key[i] if backwards else None, inference=not backwards)
            else:
                x = layer(x)
        return x, state


class RealCNNCifar100(BaseRealCNN):
    def __init__(
        self,
        key,
        layers: list[int],  # list of fully connected layer widths
        activation_fn: Callable,
        use_bias: bool = True,
    ):
        super().__init__(layers)
        conv_blocks = [
            (3, 128),
            (128, 128),
            (128, 256),
            (256, 256),
            (256, 512),
            (512, 512),
        ]

        num_keys = len(conv_blocks) + len(layers) - 1
        keys = split(key, num_keys)
        spatial_size = 32  # CIFAR input

        # Convolutional blocks
        for i in range(0, len(conv_blocks), 2):
            for j in range(2):
                in_ch, out_ch = conv_blocks[i + j]
                self.layers.append(RealConv2D(
                    key=keys[i + j],
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding="SAME",
                    use_bias=use_bias,
                ))
                self.layers.append(activation_fn)
                self.layers.append(BatchNorm(axis_name="batch",
                                             input_size=out_ch,
                                             channelwise_affine=False,
                                             momentum=0.1,
                                             eps=1e-5,
                                             inference=False,))
            # MaxPool after the block
            self.layers.append(
                MaxPool2d(kernel_size=2, stride=2, padding=0))
            
            spatial_size //= 2
        # Flatten for FC layers
        self.layers.append(ravel)

        # Fully connected layers
        fc_in_features = conv_blocks[-1][1] * spatial_size * spatial_size
        all_fc_layers = [fc_in_features] + layers

        for i in range(len(all_fc_layers) - 1):
            self.layers.append(Linear(
                key=keys[len(conv_blocks) + i],
                in_features=all_fc_layers[i],
                out_features=all_fc_layers[i + 1],
                use_bias=use_bias,
            ))
            # BatchNorm for FC layers
            self.layers.append(BatchNorm(
                input_size=all_fc_layers[i + 1],
                axis_name="batch",
                channelwise_affine=True,
                momentum=0.1,
                eps=1e-5,
                mode="batch",
                inference=False,
            ))
            if i < len(all_fc_layers) - 2:
                self.layers.append(activation_fn)
