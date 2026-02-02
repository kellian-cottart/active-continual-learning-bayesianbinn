""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Sign activation function with custom gradient
"""


import jax.numpy as jnp
from jax import custom_vjp

# Define the sign function with a custom gradient


@custom_vjp
def sign_function(x, offset):
    return (jnp.sign(x - offset)).astype(jnp.float32)
# Custom forward and backward functions


def sign_function_fwd(x, offset):
    y = sign_function(x, offset)
    return y, (x, offset)


def sign_function_bwd(res, g):
    x, offset = res
    grad_output = g * ((x > offset-1) & (x < 1+offset)
                       ).astype(jnp.float32)
    return grad_output, None


# Register the custom VJP
sign_function.defvjp(sign_function_fwd, sign_function_bwd)

# Sign Activation Layer


class SignActivation:
    def __init__(self, offset=0):
        self.offset = offset

    def __call__(self, x):
        return sign_function(x, self.offset)

    def __repr__(self):
        return f"SignActivation(offset={self.offset})"
