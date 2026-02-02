""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Reverse Binary Gate Activation function with custom gradient
"""

import jax.numpy as jnp
from jax import custom_vjp

# Define the r_binary_gate function with a custom gradient


@custom_vjp
def r_binary_gate_function(tensor_input, width=1):
    """Forward pass of the GateElephant function: returns 0 if -width < input < width, 1 otherwise."""
    return (jnp.abs(tensor_input) > width).astype(jnp.float32)

# Define the forward and backward passes for the r_binary_gate function


def r_binary_gate_function_fwd(tensor_input, width):
    y = r_binary_gate_function(tensor_input, width)
    return y, (tensor_input, width)


def r_binary_gate_function_bwd(res, grad_output):
    """ Backward pass surrogate of the GateElephant function.

    1 when -3*width/2 < input < -width/2
    -1 when width/2 < input < 3*width/2
    0 otherwise
    """
    tensor_input, width = res
    grad_input = grad_output * (
        ((tensor_input > width / 2) & (tensor_input < 3 * width / 2)).astype(jnp.float32) -
        ((tensor_input > -3 * width / 2) &
         (tensor_input < -width / 2)).astype(jnp.float32)
    )
    return grad_input, None


# Register the custom forward and backward functions
r_binary_gate_function.defvjp(
    r_binary_gate_function_fwd, r_binary_gate_function_bwd)


class ReverseBinaryGateActivation:
    """ GateElephant Activation Layer, applies r_binary_gate function to the input tensor.
    Output: 0 or 1
    """

    def __init__(self, width=1):
        self.width = width

    def __call__(self, tensor_input):
        """Apply the GateElephant function with the specified width."""
        return r_binary_gate_function(tensor_input, self.width)

    def __repr__(self):
        return f"ReverseBinaryGateActivation(width={self.width})"
