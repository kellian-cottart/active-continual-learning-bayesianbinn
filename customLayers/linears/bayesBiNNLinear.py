""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: BayesBiNN Linear Layer
"""


from jax.lax import tanh, logistic
from jax.numpy import ones_like
from jax.random import bernoulli
from typing import Literal, Union
from jaxtyping import PRNGKeyArray, Array
from equinox import Module, field
from equinox import _misc
from jax.random import split, uniform
from jax import custom_vjp
from jax.numpy import log, dot, float32
import jax

class BayesBiNNLinear(Module, strict=True):
    """Performs a linear transformation."""

    weight: dict[str, Array]
    bias: dict[str, Array]
    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """ Initialises the Bayesian Linear Layer

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
        wkey, bkey = split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 0
        wshape = (out_features_, in_features_)
        bshape = (out_features_,)
        self.weight = uniform(wkey, wshape, minval=-lim, maxval=lim)
        self.bias = uniform(bkey, bshape, minval=-lim,
                            maxval=lim) if use_bias else None
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: Array) -> Array:
        output = dot(self.weight, x)
        if self.use_bias:
            output += self.bias   
        return output

    def sample(self, x: Array, *, key: PRNGKeyArray) -> Array:
        """ Sample the weights according to a Bernoulli distribution """
        wkey, bkey = split(key, 2)
        p = logistic(2 * self.weight)
        weights = 2 * bernoulli(wkey, p).astype(float32) - 1
        output = dot(weights, x)
        if self.use_bias:
            p_bias = logistic(2 * self.bias)
            biases = 2 * bernoulli(bkey, p_bias).astype(float32) - 1
            output += biases
        return output
