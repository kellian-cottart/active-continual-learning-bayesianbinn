from .binaryBayesianBase import *
from jax.random import split

class BinaryBayesianResNetCIFAR(BaseBinaryBayesianCNN):
    """Modular residual Binary Bayesian CNN for CIFAR"""

    def __init__(self, key, layers, temperature, use_bias=False, activation_fn=None, **kwargs):
        super().__init__(key, layers, temperature)
        static_layers = layers.copy()

        # Split keys for each residual block + FC
        k1, k2, k3, k4, k_fc = split(key, 5)

        # Track spatial size for LayerNorm
        spatial_size = 32  # CIFAR input
        in_ch = 3

        # Residual blocks with progressive channels
        blocks = [(k1, 64), (k2, 128), (k3, 256), (k4, 512)]
        for k, out_ch in blocks:
            self.layers.append(BinaryResidualBlock(
                key=k,
                in_ch=in_ch,
                out_ch=out_ch,
                activation_fn=activation_fn,
                spatial_size=spatial_size,
                stride=1,
                use_bias=use_bias
            ))
            self.layers.append(AvgPool2d(kernel_size=2, stride=2))
            spatial_size //= 2  # Update spatial size after pooling
            in_ch = out_ch

        # Flatten
        self.layers.append(ravel)

        # Fully connected layers
        fc_in_features = in_ch * spatial_size * spatial_size
        fc_layers = [fc_in_features] + static_layers

        # Split keys for FC layers
        num_fc = len(fc_layers) - 1
        fc_keys = split(k_fc, num_fc)

        for i in range(num_fc):
            self.layers.append(BinaryBayesianLinear(
                in_features=fc_layers[i],
                out_features=fc_layers[i + 1],
                use_bias=use_bias,
                key=fc_keys[i]
            ))
            self.layers.append(LayerNorm(
                shape=(fc_layers[i + 1],),
                use_weight=False,
                use_bias=False,
            ))
            if i < num_fc - 1:
                self.layers.append(activation_fn)
