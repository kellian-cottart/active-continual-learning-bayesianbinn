
from .binaryBayesianBase import *

class BinaryBayesianCNNCifar100(BaseBinaryBayesianCNN):
    def __init__(self, key, layers, temperature, use_bias=False, activation_fn=None, **kwargs):
        super().__init__(key, layers, temperature)
        conv_blocks = [
            (3, 128),
            (128, 128),
            (128, 256),
            (256, 256),
            (256, 512),
            (512, 512),
        ]

        keys = split(key, len(conv_blocks) + len(layers) - 1)

        spatial_size = 32  # CIFAR input

        for i in range(0, len(conv_blocks), 2):
            # Two conv layers per block
            for j in range(2):
                in_ch, out_ch = conv_blocks[i + j]
                self.layers.append(BinaryBayesianConv2D(
                    key=keys[i + j],
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    stride=1,
                    padding="SAME",
                    use_bias=use_bias,
                ))
                self.layers.append(LayerNorm(
                    shape=(out_ch, spatial_size, spatial_size),
                    use_weight=False,
                    use_bias=False,
                ))
                self.layers.append(activation_fn)
            
            # MaxPool after the block
            self.layers.append(MaxPool2d(kernel_size=2, stride=2, padding=0))
            spatial_size = spatial_size // 2  # update spatial size after pooling

        self.layers.append(ravel)  # flatten

        # Compute flattened input for first linear layer
        fc_in_features = conv_blocks[-1][1] * spatial_size * spatial_size
        layers = [fc_in_features] + layers

        # Fully connected layers
        for i in range(len(layers) - 1):
            self.layers.append(BinaryBayesianLinear(
                in_features=layers[i],
                out_features=layers[i + 1],
                use_bias=use_bias,
                key=keys[len(conv_blocks) + i]
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
        super().__init__(key, layers, temperature)
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
        super().__init__(key, layers, temperature)
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
