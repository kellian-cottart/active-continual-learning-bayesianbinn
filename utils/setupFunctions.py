""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Setup functions for networks and optimizers
"""


import jax
from models import *
from optimizers import *
import optax
import equinox as eqx
import os


def load_weights(model, path, task=None, epoch=None):
    """
    Load the weights from a given path into the model.
    """
    path = os.path.join(path, "model.eqx")
    new_model = eqx.tree_deserialise_leaves(path, model)
    return new_model


def configure_networks(configuration, rng):
    # make a dictionary of maps
    select_network = {
        "bayesianmlp": BaseBayesianMLP,
        "bayesianmlplayernorm": BayesianMLPLayerNorm,
        "mlp": BaseMLP,
        "mlpbatchnorm": MLPBatchNorm,
        "mlplayernorm": MLPLayerNorm,
        "binarybayesianmlp": BaseBinaryBayesianMLP,
        "binarymlp": BaseBinaryMLP,
        "binarymlpbatchnorm": BinaryMLPBatchNorm,
        "binarymlplayernorm": BinaryMLPLayerNorm,
        "bayesbinnmlp": BaseBayesBiNNMLP,
        "binarybayesianconvcifar100": BinaryBayesianCNNCifar100,
        "binarybayesianconvmnist": BinaryBayesianCNNMNIST,
        "binarybayesianconvcore50": BinaryBayesianCNNCore50,
        "binarybayesianresnet": BinaryBayesianResNetCIFAR,
        "realconvcifar100": RealCNNCifar100,
    }
    if not "network_params" in configuration:
        raise ValueError("Network parameters 'network_params' not found")
    if not "activation_fn" in configuration["network_params"]:
        raise ValueError("Activation function 'activation_fn' not found")

    select_activation = {
        "relu": jax.nn.relu,
        "sigmoid": jax.nn.sigmoid,
        "tanh": jax.nn.tanh,
        "gate": GateActivation,
        "sign": SignActivation,
        "gateelephant": GateElephantActivation,
        "reversegate": ReverseGateActivation,
        "reversebinarygate": ReverseBinaryGateActivation,
    }
    # Certain activation functions don't take any parameters, so we try defaulting to the function alone
    try:
        configuration["network_params"]["activation_fn"] = select_activation[configuration["network_params"]["activation_fn"]](
            **configuration["network_params"]["activation_params"]
        )
    except (TypeError, KeyError) as e:
        try:
            configuration["network_params"]["activation_fn"] = select_activation[configuration["network_params"]["activation_fn"]]()
        except TypeError as e:
            configuration["network_params"]["activation_fn"] = select_activation[configuration["network_params"]["activation_fn"]]

    # Instantiate the model and the initial state
    try:
        key, rng = jax.random.split(rng, 2)
        # make_with_state separates the model and the initial state for batch norm tracking stats
        model, model_state = eqx.nn.make_with_state(select_network[configuration["network"]])(
            key=key, **configuration["network_params"])
    except KeyError as e:
        raise KeyError("Error with provided keys: ", e)

    return model, model_state


def configure_optimizer(configuration, model):
    select_optimizer = {
        "sgd": sgd,
        "adam": optax.adamw,
        "mesu": mesu,
        "bgd": bgd,
        "bimu": bimu,
        "bayesbinn": bayesbinn,
        "bayesbinn_al": bayesbinn_al,
        "synapticmetaplasticity": synapticMetaplasticity,
        "adamw": adamw,
    }
    if not "optimizer_params" in configuration:
        raise ValueError("Optimizer parameters not found")
    try:
        optimizer = select_optimizer[configuration["optimizer"]](
            **configuration["optimizer_params"]
        )
    except KeyError as e:
        raise KeyError("Error with provided keys: ", e)
    opt_state = optimizer.init(model)
    return optimizer, opt_state
