""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Gaussian parameter class
"""

from jaxtyping import Array
from equinox import Module


class GaussianParameter(Module):
    mu: Array
    sigma: Array

    def __init__(self, mu: Array, sigma: Array):
        self.mu = mu
        self.sigma = sigma
