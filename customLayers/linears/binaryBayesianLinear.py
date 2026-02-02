""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Binary Bayesian Linear layer
"""

from jax.numpy import dot, log, float32
from jax.lax import tanh, logistic
from jax.random import bernoulli
from typing import Literal, Union
from jaxtyping import PRNGKeyArray, Array
from math import sqrt
from equinox import Module, field
from equinox import _misc
from jax.random import split, uniform
from equinox.nn import Linear


class BinaryBayesianLinear(Linear):
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
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            dtype=dtype,
            key=key,
        )

    def __call__(self, x: Array, *, key: PRNGKeyArray, temperature: float = 1) -> Array:
        wkey, bkey = split(key, 2)
        # Gumbel-Softmax trick
        epsilon = uniform(wkey, self.weight.shape,
                          minval=1e-10, maxval=1 - 1e-10)
        logit_epsilon = log(epsilon) - log(1 - epsilon)
        weights = tanh((self.weight + 0.5 * logit_epsilon) / temperature)
        output = dot(weights, x)

        if self.use_bias:
            epsilon_bias = uniform(bkey, self.bias.shape,
                                   minval=1e-10, maxval=1 - 1e-10)
            logit_epsilon_bias = log(epsilon_bias) - log(1 - epsilon_bias)
            biases = tanh(
                (self.bias + 0.5 * logit_epsilon_bias) / self.temperature)
            output += biases
        return output

    def sample(self, x: Array, *, key: PRNGKeyArray, temperature: float = 1) -> Array:
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
