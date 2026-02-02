""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Binary Linear layer
"""

from jax.numpy import shape, dot, ones, sign, float32
from typing import Literal, Union
from jaxtyping import PRNGKeyArray, Array
from math import sqrt
from equinox import Module, field
from equinox import _misc
from jax.random import split, normal, uniform
from jax import custom_vjp
import jax


@ custom_vjp
def unclamped_sign_function(x):
    return (sign(x)).astype(float32)
# Custom forward and backward functions


def unclamped_sign_function_fwd(x):
    y = unclamped_sign_function(x)
    return y, x


def unclamped_sign_function_bwd(res, g):
    return (g,)


# Register the custom VJP
unclamped_sign_function.defvjp(
    unclamped_sign_function_fwd, unclamped_sign_function_bwd)


class BinaryLinear(Module, strict=True):
    """Performs a linear transformation."""

    weight: dict[str, Array]
    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """ Initialises the Binary Linear Layer

        Args:
            in_features: The input size. The input to the layer should be a vector of
                shape `(in_features,)`
            out_features: The output size. The output from the layer will be a vector
                of shape `(out_features,)`.
            use_bias: Whether to add on a bias as well.
            dtype: The dtype to use for the weight and the bias in this layer.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
                on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for GaussianParameter
                initialisation. (Keyword only argument.)
        """
        dtype = _misc.default_floating_dtype() if dtype is None else dtype
        wkey, key = split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / sqrt(in_features_)
        wshape = (out_features_, in_features_)
        self.weight = uniform(wkey, wshape, minval=-lim, maxval=lim)
        self.in_features = in_features
        self.out_features = out_features

    def __call__(self, x):
        """ Call function for Binary linear layer
        Do the forward pass by doing a linear transformation on the binary weights
        """
        return dot(unclamped_sign_function(self.weight), x)
