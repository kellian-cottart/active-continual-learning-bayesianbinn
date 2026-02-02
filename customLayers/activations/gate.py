""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Gate activation function with custom gradient
"""

import jax.numpy as jnp
from jax import custom_vjp

# Define the gate function with a custom gradient


@custom_vjp
def gate_function(tensor_input, width=1):
    """Forward pass of the Gate function: returns 1 if -width < input < width, -1 otherwise."""
    return 2 * (jnp.abs(tensor_input) < width).astype(jnp.float32) - 1

# Define the forward and backward passes for the gate function


def gate_function_fwd(tensor_input, width):
    y = gate_function(tensor_input, width)
    return y, (tensor_input, width)


def gate_function_bwd(res, grad_output):
    """ Backward pass surrogate of the Gate function.

    1 when -3*width/2 < input < -width/2
    -1 when width/2 < input < 3*width/2
    0 otherwise
    """
    tensor_input, width = res
    grad_input = grad_output * (
        ((tensor_input > -3 * width / 2) & (tensor_input < -width / 2)).astype(jnp.float32) -
        ((tensor_input > width / 2) &
         (tensor_input < 3 * width / 2)).astype(jnp.float32)
    )

    return grad_input, None


# Register the custom forward and backward functions
gate_function.defvjp(gate_function_fwd, gate_function_bwd)


class GateActivation:
    """ Gate Activation Layer, applies gate function to the input tensor."""

    def __init__(self, width=1):
        self.width = width

    def __call__(self, tensor_input):
        """Apply the Gate function with the specified width."""
        return gate_function(tensor_input, self.width)

    def __repr__(self):
        return f"GateActivation(width={self.width})"
