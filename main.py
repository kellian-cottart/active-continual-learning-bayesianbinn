""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File: Main training file (Loads configurations, dataset, model, optimizer and runs the training loop, exports accuracies and uncertainties)
Arguments: 
    -c, --config: Configuration file name (without .json)
    -it, --n_iterations: Number of iterations to run the config file for
    -v, --verbose: Whether to display the pbar or not
    -ood, --ood: Which dataset to compute the ood on (fashion, pmnist, None)
    -gpu, --gpu: GPU ID to use
    -wh, --weight_histogram: Whether to save weight histograms
    -eln, --extract_layer_norm: Whether to retrieve the output of the layer norm of the model
    -fits, --fits_in_memory: Whether the dataset fits in memory or not
    -train, --train_accuracy: Whether to display the train accuracy, requires -v verbose
    -euf, --extract_uncertainties_full: Whether to extract the epistemic uncertainty histogram on the train set at each epoch
    -pca, --per_class_acc: Whether to compute per class accuracy
"""

from optimizers import *
from utils import *
from models import *
from datetime import datetime
import os
import json
from shutil import rmtree
import argparse
from copy import deepcopy
import numpy as np
from torch import manual_seed
from time import time
import jax
import jax.numpy as jnp
from jax.numpy import expand_dims
import equinox as eqx

import traceback

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
# argparse allows to load a configuration from a file
CONFIGURATION_LOADING_FOLDER = "configurations"
# first argument is name of config file
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Configuration file name",
                    type=str)
parser.add_argument(
    "-it", "--n_iterations", help="Number of iterations to run the config file for", type=int, default=1)
parser.add_argument(
    "-v", "--verbose", help="Whether to display the pbar or not", action="store_true")
parser.add_argument(
    "-ood", "--ood", help="Which dataset to compute the ood on (fashion, pmnist, None)", type=str, default=None)
parser.add_argument(
    "-gpu", "--gpu", help="GPU ID to use", type=int, default=0)
parser.add_argument(
    "-wh", "--weight_histogram", help="Whether to save weight histograms", action="store_true")
parser.add_argument(
    "-eln", "--extract_layer_norm", help="Whether to retrieve the output of the layer norm of the model", action="store_true"
)
parser.add_argument(
    "-fits", "--fits_in_memory", help="Whether the dataset fits in memory or not", action="store_true")
parser.add_argument(
    "-train", "--train_accuracy", help="Whether to display the train accuracy, requires -v verbose", action="store_true")
parser.add_argument(
    "-euf", "--extract_uncertainties_full", help="Whether to extract the epistemic uncertainty histogram on the train set at each epoch", action="store_true"
)
parser.add_argument(
    "-pca", "--per_class_acc", help="Whether to compute per class accuracy", action="store_true"
)
PER_CLASS_ACC = parser.parse_args().per_class_acc
args = parser.parse_args()
CONFIG_FILE = json.load(
    open(os.path.join(CONFIGURATION_LOADING_FOLDER, args.config+".json")))
for k, v in CONFIG_FILE.items():
    if isinstance(v, str):
        CONFIG_FILE[k] = v.lower()
N_ITERATIONS = args.n_iterations
OOD = args.ood
VERBOSE = args.verbose
WEIGHT_HIST = args.weight_histogram
EXTRACT_LAYER_NORM = args.extract_layer_norm
FITS_IN_MEMORY = args.fits_in_memory
TRAIN_ACC = args.train_accuracy
EXTRACT_UNCERTAINTIES_FULL = args.extract_uncertainties_full

# set the device
jax.config.update("jax_platform_name", "gpu")

if __name__ == "__main__":
    # Create a timestamp
    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S-")
    MAIN_FOLDER = "results"
    os.makedirs(MAIN_FOLDER, exist_ok=True)
    for k in range(N_ITERATIONS):
        configuration = deepcopy(CONFIG_FILE)
        configuration["seed"] += k
        print(f"========== CONFIGURATION {k} ==========")
        print(json.dumps(configuration, indent=4))
        print("========================================")
        FOLDER = f"{TIMESTAMP}{configuration['task']}-t={configuration['n_tasks']}-e={configuration['epochs']}-opt={configuration['optimizer']}"
        if configuration["optimizer"] == "mesu" or configuration["optimizer"] == "bimu":
            FOLDER += f"-N={int(configuration['optimizer_params']['N'])}-{int(configuration['n_test_samples'])}f-{int(configuration['n_train_samples'])}b"
        ewc_parameters = None
        ewc_streaming_parameters = None
        ewc_online_parameters = None
        si_parameters = None
        if "stream-ewc" in configuration:
            ewc_streaming_parameters = deepcopy(configuration["stream-ewc"])
            FOLDER += f"-stream-ewc={configuration['stream-ewc']['importance']}"
        elif "online-ewc" in configuration:
            ewc_online_parameters = deepcopy(configuration["online-ewc"])
            FOLDER += f"-online-ewc={configuration['online-ewc']['importance']}"
        elif "ewc" in configuration:
            ewc_parameters = deepcopy(configuration["ewc"])
            FOLDER += f"-ewc={configuration['ewc']['importance']}"
        elif "si" in configuration:
            si_parameters = deepcopy(configuration["si"])
            FOLDER += f"-si={configuration['si']['coefficient']}"
        if "use_bias" in configuration["network_params"]:
            FOLDER += "-bias"
        SAVE_PATH = os.path.join(MAIN_FOLDER, FOLDER)
        CONFIGURATION_PATH = os.path.join(SAVE_PATH, f"config{k}")
        DATA_PATH = os.path.join(CONFIGURATION_PATH, "accuracy")
        WEIGHTS_PATH = os.path.join(CONFIGURATION_PATH, "weights")
        UNCERTAINTY_PATH = os.path.join(CONFIGURATION_PATH, "uncertainty")
        TRAIN_PATH = os.path.join(CONFIGURATION_PATH, "train_accuracy")
        HISTOGRAM_PATH = os.path.join(CONFIGURATION_PATH, "histograms")
        for path in [SAVE_PATH, CONFIGURATION_PATH, DATA_PATH, WEIGHTS_PATH, UNCERTAINTY_PATH, TRAIN_PATH, HISTOGRAM_PATH]:
            os.makedirs(path, exist_ok=True)
        # save config
        with open(CONFIGURATION_PATH + "/config.json", "w") as f:
            json.dump(configuration, f, indent=4)
        try:
            # Initialize the random number generator
            manual_seed(configuration["seed"])
            np.random.seed(configuration["seed"])
            rng = jax.random.key(configuration["seed"])
            # Load the dataset
            loader = GPULoading(device="cpu")
            task_params = configuration["task_params"] if "task_params" in configuration else {
            }
            max_permutations = configuration["max_parallel_permutation"] if "max_parallel_permutation" in configuration else 1
            train_samples = configuration["n_train_samples"] if "n_train_samples" in configuration else None
            test_samples = configuration["n_test_samples"] if "n_test_samples" in configuration else None
            train, test, shape, num_classes = loader.task_selection(
                configuration["task"], **task_params)

            if "unbalanced" in task_params:
                threshold = task_params["unbalanced"].get(
                    "threshold_max", 700)
                # retrieve the index of the classes under represented for each task of the training dataset
                indexes_under_represented = []
                indexes_well_represented = []
                for task_dataset in train:
                    current_class_under_represented = []
                    current_class_well_represented = []
                    train_labels = task_dataset[:][1]
                    # convert to jnp array
                    train_labels = jnp.array(
                        train_labels)
                    class_counts = jnp.bincount(
                        train_labels, minlength=num_classes)
                    for class_idx, count in enumerate(class_counts):
                        if count < threshold:
                            current_class_under_represented.append(
                                int(class_idx))
                        else:
                            current_class_well_represented.append(
                                int(class_idx))
                    indexes_under_represented.append(
                        jnp.array(current_class_under_represented))
                    indexes_well_represented.append(
                        jnp.array(current_class_well_represented))
                # Convert JAX arrays to NumPy and store as object arrays
                indexes_under_represented_np = np.array(
                    [np.asarray(x) for x in indexes_under_represented],
                    dtype=object
                )
                indexes_well_represented_np = np.array(
                    [np.asarray(x) for x in indexes_well_represented],
                    dtype=object
                )
                with open(os.path.join(CONFIGURATION_PATH, "indexes_under_represented.npy"), "wb") as f:
                    np.save(f, indexes_under_represented_np, allow_pickle=True)

                with open(os.path.join(CONFIGURATION_PATH, "indexes_well_represented.npy"), "wb") as f:
                    np.save(f, indexes_well_represented_np, allow_pickle=True)

            if "active_learning" in configuration["network_params"] and configuration["network_params"]["active_learning"]["mode"] == 4:
                # this mode cuts the amount of data used for training by the value given
                for k in range(len(train)):
                    # reduction factor is between 0 and 1
                    n_train_examples = len(train[k])
                    number_of_remaining_samples = n_train_examples * \
                        (1-configuration["network_params"]
                         ["active_learning"]["reduction_factor"])
                    # take a random subset of the training data
                    permutation = randperm(n_train_examples)
                    selected_indices = permutation[:int(
                        number_of_remaining_samples)]
                    train_images = train[k][selected_indices][0]
                    train_labels = train[k][selected_indices][1]
                    train[k] = TensorDataset(train_images, train_labels)

            is_permuted = configuration["task"] == "permutedmnist"
            # Permutations
            permutations = None
            if is_permuted:
                perm_keys, rng = jax.random.split(rng, 2)
                perm_keys = jax.random.split(
                    perm_keys, configuration["n_tasks"])
                permutations = jnp.array(
                    [jax.random.permutation(key, jnp.array(shape).prod()) for key in perm_keys])
                # save permutations vector
                with open(os.path.join(CONFIGURATION_PATH, "permutations.npy"), "wb") as f:
                    jnp.save(f, permutations)

            model_key, rng = jax.random.split(rng)
            # Configure the model
            model, model_state = configure_networks(configuration, model_key)
            print("Training on", configuration["task"])
            # Configure the optimizer
            optimizer, opt_state = configure_optimizer(
                configuration, eqx.filter(model, eqx.is_array))

            def initialize_ewc_parameters(model):
                old_param = eqx.filter(model, eqx.is_array)
                fisher = map(lambda x: jnp.zeros_like(x), old_param)
                return old_param, fisher
            if ewc_parameters is not None:
                old_param, fisher = initialize_ewc_parameters(model)
                old_param = map(lambda x: expand_dims(x, 0).repeat(
                    configuration["n_tasks"], axis=0), old_param)
                fisher = map(lambda x: expand_dims(x, 0).repeat(
                    configuration["n_tasks"], axis=0), fisher)
                ewc_parameters["old_param"], ewc_parameters["fisher"] = old_param, fisher

            if ewc_online_parameters is not None:
                ewc_online_parameters["old_param"], ewc_online_parameters["fisher"] = initialize_ewc_parameters(
                    model)
            if ewc_streaming_parameters is not None:
                ewc_streaming_parameters["old_param"], ewc_streaming_parameters["fisher"] = initialize_ewc_parameters(
                    model)

            def initialize_si_parameters(model):
                old_param = eqx.filter(model, eqx.is_array)
                omega = map(lambda x: jnp.zeros_like(x), old_param)
                w_k = map(lambda x: jnp.zeros_like(x), old_param)
                return old_param, omega, w_k

            if si_parameters is not None:
                si_parameters["old_param"], si_parameters["omega"], si_parameters["w_k"] = initialize_si_parameters(
                    model)

            # Synaptic metaplasticity imposes to store a normalization parameter at the end of each task
            # and to evaluate the model based on this normalization parameter
            # We pre-allocate the tree with the normalization parameters
            norm_params = None
            if isinstance(model, BaseBinaryMLP):
                tree_norm = model.return_tree_norm()
                norm_params = map(lambda x: expand_dims(x, 0).repeat(
                    configuration["n_tasks"], axis=0), tree_norm)
            # GENERATING A HUGE ARRAY OF KEYS, ASSURING THAT THE KEYS ARE UNIQUE
            trkey, tekey, rng = jax.random.split(rng, 3)
            training_core_keys = jax.random.split(
                trkey, (configuration["n_tasks"], configuration["epochs"]))
            test_core_keys = jax.random.split(
                tekey, (configuration["n_tasks"], configuration["epochs"]))
            pbar = tqdm(range(configuration["n_tasks"]), desc="Tasks") if VERBOSE else range(
                configuration["n_tasks"])
            epoch_pbar = tqdm(range(configuration["epochs"]), desc="Epochs") if VERBOSE else range(
                configuration["epochs"])
            if OOD is not None:
                ood_dataloader, ood_core_keys, rng = load_ood_dataset(
                    OOD, loader, task_params, configuration, num_classes, rng, FITS_IN_MEMORY)
            n_splits_per_epoch = configuration.get("n_splits_per_epoch", 1)

            # ---------- Memory Occupation --------------
            model_memory, opt_memory, extra_memory = compute_memory_occupation(
                configuration, model, opt_state, si_parameters, ewc_streaming_parameters,
                ewc_online_parameters, ewc_parameters, norm_params)
            # save the memory occupation as the sum of the model, optimizer and extra parameters
            memory_occupation = jnp.array(
                model_memory + opt_memory + extra_memory)
            jnp.save(os.path.join(CONFIGURATION_PATH,
                     "memory_occupation.npy"), memory_occupation)

            # ---------- Preparing Dataloaders --------------
            train = to_dataloader(
                train, configuration["train_batch_size"], num_classes, fits_in_memory=FITS_IN_MEMORY)
            test_dataloader = to_dataloader(
                test, configuration["test_batch_size"], num_classes, fits_in_memory=FITS_IN_MEMORY)

            # ---------- Training Loop --------------
            for task_id, task in enumerate(pbar):

                if is_permuted:
                    task_train_dataloader = reshape_perm(
                        train[0], permutations[task_id])
                elif len(train) == configuration["n_tasks"]:
                    task_train_dataloader = train[task_id]
                else:
                    raise ValueError("Length of train and n_tasks do not match: ", len(
                        train), " != ", configuration["n_tasks"])

                for epoch in epoch_pbar:
                    rng, key = jax.random.split(rng)
                    split_train_dataloader = split_dataset(
                        task_train_dataloader, n_splits_per_epoch, fits_in_memory=FITS_IN_MEMORY)
                    split_epoch_pbar = tqdm(range(n_splits_per_epoch), desc="Splits") if VERBOSE else range(
                        n_splits_per_epoch)

                    if "layerwise" in configuration:
                        layerwise = configuration["layerwise"]
                        opt_state["layer_to_train"] = layerwise[str(epoch)] if str(
                            epoch) in layerwise else opt_state["layer_to_train"]
                    if VERBOSE:
                        pbar.set_description(
                            f"Task {task+1}/{configuration['n_tasks']} - Epoch {epoch+1}/{configuration['epochs']}")
                    train_ck = training_core_keys[task_id, epoch]
                    test_ck = test_core_keys[task_id, epoch]
                    for split_epoch in split_epoch_pbar:
                        model, opt_state, losses, ewc_streaming_parameters, model_state, si_parameters = train_fn(
                            model=model,
                            dataset=split_train_dataloader[split_epoch],
                            num_classes=num_classes,
                            opt_state=opt_state,
                            optimizer=optimizer,
                            train_ck=train_ck,
                            train_samples=train_samples,
                            ewc_online_parameters=ewc_online_parameters,
                            ewc_streaming_parameters=ewc_streaming_parameters,
                            ewc_parameters=ewc_parameters,
                            si_parameters=si_parameters,
                            init_state=model_state
                        )
                        if isinstance(model, BaseBinaryMLP):
                            # we need to save the normalization weights at the end of the epoch
                            new_params = model.return_tree_norm()
                            norm_params = map(lambda old, new: old.at[task_id].set(
                                new), norm_params, new_params)
                        accuracies, uncertainties = main_test_fn(
                            test_dataset=test_dataloader,
                            num_classes=num_classes,
                            test_samples=test_samples,
                            test_ck=test_ck,
                            model=model,
                            model_state=model_state,
                            norm_params=norm_params,
                            is_permuted=is_permuted,
                            max_permutations=max_permutations,
                            permutations=permutations,
                            fits_in_memory=FITS_IN_MEMORY
                        )
                        current_iterations = 0
                        if "step" in opt_state:
                            current_iterations = opt_state["step"].item() if not isinstance(
                                opt_state["step"], int) else opt_state["step"]
                        if TRAIN_ACC:
                            tr_accuracies, tr_uncertainties = main_test_fn(
                                test_dataset=train,
                                num_classes=num_classes,
                                test_samples=test_samples,
                                test_ck=test_ck,
                                model=model,
                                model_state=model_state,
                                norm_params=norm_params,
                                is_permuted=is_permuted,
                                max_permutations=max_permutations,
                                permutations=permutations,
                                fits_in_memory=FITS_IN_MEMORY
                            )
                            if VERBOSE:
                                tqdm.write("======== Train ========")
                            for i, acc in enumerate(tr_accuracies):
                                tr_accuracy = tr_accuracies[i].reshape(
                                    -1).mean()
                                if VERBOSE:
                                    tqdm.write(f"{tr_accuracy.item()*100:.2f}%", end="\t" if i % 10 !=
                                               9 and i != len(tr_accuracies) - 1 else "\n")
                                if PER_CLASS_ACC:
                                    if VERBOSE:
                                        tqdm.write(f"\n--- Dataset {i} ---\n")
                                    classes_acc = []
                                    tr_acc = tr_accuracies[i].reshape(-1)
                                    # classes are one hot encoded, we want the argmax of the second dimension,
                                    n_classes = train[i][1].argmax(
                                        -1).max() + 1
                                    min_classes = train[i][1].argmax(
                                        -1).min()
                                    train_labels = train[i][1].reshape(
                                        -1, num_classes)
                                    for label in range(min_classes, n_classes):
                                        label_acc = tr_acc[jnp.argmax(
                                            train_labels, axis=1) == label]
                                        if len(label_acc) > 0:
                                            label_acc = label_acc.mean()
                                            classes_acc.append(label_acc)
                                            if VERBOSE:
                                                tqdm.write(
                                                    f"C{label}: {label_acc.item()*100:.2f}%", end="\t")

                                        # save classes_acc
                                        with open(os.path.join(TRAIN_PATH, f"per-class-acc-split={split_epoch}-task={task}-epoch={epoch}-dataset={i}.npy"), "wb") as f:
                                            jnp.save(f, jnp.array(classes_acc))

                                        # save the accuracies on well-represented class and under-represented classes
                                tqdm.write("")
                        if VERBOSE:
                            tqdm.write("======== Test ========")
                        # First pass: collect all metrics
                        all_accuracies = []
                        all_per_class_acc = []
                        for i, acc in enumerate(accuracies):
                            acc = accuracies[i].reshape(-1)
                            accuracy = acc.mean()
                            all_accuracies.append(accuracy)
                            if PER_CLASS_ACC:
                                classes_acc = []
                                test_labels = test_dataloader[i][1].reshape(
                                    -1, num_classes)
                                for label in range(num_classes):
                                    label_acc = acc[jnp.argmax(
                                        test_labels, axis=1) == label]
                                    if len(label_acc) > 0:
                                        label_acc = label_acc.mean()
                                        classes_acc.append(label_acc)
                                all_per_class_acc.append(
                                    jnp.array(classes_acc))

                        # save results
                        if PER_CLASS_ACC:
                            with open(os.path.join(DATA_PATH, f"per-class-acc-split={split_epoch}-task={task}-epoch={epoch}.npy"), "wb") as f:
                                jnp.save(f, jnp.array(all_per_class_acc))
                        # Second pass: display results
                        if VERBOSE:
                            for i, accuracy in enumerate(all_accuracies):
                                tqdm.write(f"{accuracy.item()*100:.2f}%", end="\t" if i %
                                           10 != 9 and i != len(all_accuracies) - 1 else "\n")
                            if PER_CLASS_ACC:
                                for i, classes_acc in enumerate(all_per_class_acc):
                                    tqdm.write(f"\n--- Dataset {i} ---\n")
                                    for label, label_acc in enumerate(classes_acc):
                                        tqdm.write(
                                            f"C{label}: {label_acc.item()*100:.2f}%", end="\t")
                                    tqdm.write("")

                        if "unbalanced" in task_params:
                            # compute the mean accuracy on the under-represented classes for all tasks and the mean accuracy on the well-represented classes
                            well_represented_acc = []
                            under_represented_acc = []
                            for idx, test_set in enumerate(test_dataloader):
                                # get the labels
                                if not isinstance(test_set, NumpyLoader):
                                    test_labels = test_set[:][1].argmax(
                                        -1).squeeze()
                                else:
                                    test_labels = test_set[:][1]
                                test_labels = jnp.array(
                                    test_labels).reshape(-1)
                                acc = accuracies[idx].reshape(-1)
                                # retrieve the accuracy for the classes in indexes_under_represented[idx]
                                under_represented_classes = jnp.array(
                                    indexes_under_represented[idx])
                                under_represented_acc.append(
                                    acc[jnp.isin(test_labels, under_represented_classes)])
                                well_represented_acc.append(
                                    acc[~jnp.isin(test_labels, under_represented_classes)])
                            under_represented_acc = jnp.concatenate(
                                under_represented_acc)
                            well_represented_acc = jnp.concatenate(
                                well_represented_acc)
                            # save the under-represented and well-represented accuracies
                            with open(os.path.join(DATA_PATH, f"ura-split={split_epoch}-task={task}-epoch={epoch}.npy"), "wb") as f:
                                jnp.save(
                                    f, under_represented_acc.mean().item())
                            with open(os.path.join(DATA_PATH, f"wra-split={split_epoch}-task={task}-epoch={epoch}.npy"), "wb") as f:
                                jnp.save(
                                    f, well_represented_acc.mean().item())
                        if VERBOSE:
                            if "unbalanced" in task_params:
                                tqdm.write(
                                    f"URC acc: {under_represented_acc.mean().item()*100:.2f}%\t WRC acc: {well_represented_acc.mean().item()*100:.2f}%")
                            tqdm.write("")
                            # add loss to the bar
                            tqdm.write(
                                f"Loss: {jnp.mean(losses):.4f} - N_iterations: {current_iterations} - AVG Test acc: {jnp.array([acc.reshape(-1).mean() for acc in accuracies]).mean().item()*100:.2f}%")
                        # save the accuracies
                        saved_test_accuracies = accuracies.reshape(
                            len(accuracies), -1).mean(axis=-1) if not isinstance(accuracies, list) else jnp.array([acc.reshape(len(
                                acc), -1).mean() for acc in accuracies])
                        with open(os.path.join(DATA_PATH, f"split={split_epoch}-task={task}-epoch={epoch}.npy"), "wb") as f:
                            jnp.save(f, saved_test_accuracies)
                        if TRAIN_ACC:
                            saved_tr_accuracies = tr_accuracies.reshape(
                                len(tr_accuracies), -1).mean(axis=-1) if not isinstance(tr_accuracies, list) else jnp.array([acc.reshape(len(
                                    acc), -1).mean() for acc in tr_accuracies])
                            with open(os.path.join(TRAIN_PATH, f"split={split_epoch}-task={task}-epoch={epoch}.npy"), "wb") as f:
                                jnp.save(f, saved_tr_accuracies)
                        # save the number of iterations
                        np.save(os.path.join(DATA_PATH, f"iterations-split={split_epoch}-task={task}-epoch={epoch}.npy"),
                                current_iterations)
                    if OOD is not None:
                        # Compute uncertainty
                        ood_k = ood_core_keys[task_id, epoch]
                        # Extract uncertainty components
                        alea_u, epi_u, var_u = uncertainties
                        # Select ID uncertainties (train if available, else test)
                        if TRAIN_ACC:
                            tr_alea_u, tr_epi_u, tr_var_u = tr_uncertainties
                            alea_id, epi_id, var_id = tr_alea_u[task_id], tr_epi_u[task_id], tr_var_u[task_id]
                        else:
                            alea_id, epi_id, var_id = alea_u[task_id], epi_u[task_id], var_u[task_id]

                        # Select OOD uncertainties (handle single or multiple tasks)
                        def get_ood(u): return u[0] if len(
                            u) == 1 else u[task_id]
                        ood_test = [get_ood(ood_dataloader)]
                        _,  ood_uncertainties = main_test_fn(
                            test_dataset=ood_test,
                            num_classes=num_classes,
                            test_samples=test_samples,
                            test_ck=ood_k,
                            model=model,
                            model_state=model_state,
                            norm_params=norm_params,
                            is_permuted=False,
                            fits_in_memory=FITS_IN_MEMORY
                        )
                        alea_ood, epi_ood, var_ood = ood_uncertainties
                        alea_ood = alea_ood.squeeze()
                        epi_ood = epi_ood.squeeze()
                        var_ood = var_ood.squeeze()

                        # Compute ROC-AUC metrics
                        roc_metrics = {
                            "roc-auc-aleatoric": compute_roc_auc(alea_id, alea_ood),
                            "roc-auc-epistemic": compute_roc_auc(epi_id, epi_ood),
                            "roc-auc-variation-ratio": compute_roc_auc(var_id, var_ood),
                        }

                        uncertainty_metrics = {
                            "id-aleatoric": alea_id,
                            "id-epistemic": epi_id,
                            "ood-aleatoric": alea_ood,
                            "ood-epistemic": epi_ood,
                            "id-variation-ratio": var_id,
                            "ood-variation-ratio": var_ood,
                        }
                        if EXTRACT_UNCERTAINTIES_FULL:
                            for metric_name, metric_data in uncertainty_metrics.items():
                                np.save(os.path.join(
                                    UNCERTAINTY_PATH, f"{metric_name}-task={task}-epoch={epoch}.npy"), metric_data)
                        for metric_name, metric_data in roc_metrics.items():
                            np.save(os.path.join(
                                UNCERTAINTY_PATH, f"{metric_name}-task={task}-epoch={epoch}.npy"), metric_data)
                # ewc requires saving at the end of the task the current model parameters
                if ewc_parameters is not None or ewc_online_parameters is not None:
                    fisher = compute_fisher(
                        model=model,
                        dataset=task_train_dataloader,
                    )
                    if ewc_online_parameters is not None:
                        ewc_online_parameters["old_param"] = eqx.filter(
                            model, eqx.is_array)
                        ewc_online_parameters["fisher"] = map(
                            lambda old, new: ewc_online_parameters["downweighting"] * old + new, ewc_online_parameters["fisher"], fisher)
                    elif ewc_parameters is not None:
                        ewc_parameters["old_param"] = map(lambda old, new: old.at[task_id].set(new),
                                                          ewc_parameters["old_param"], eqx.filter(model, eqx.is_array))
                        ewc_parameters["fisher"] = map(lambda old, new: old.at[task_id].set(new),
                                                       ewc_parameters["fisher"], fisher)
                if si_parameters is not None:
                    epsilon = si_parameters["damping_factor"]
                    difference = map(lambda old, new: (
                        new - old)**2, si_parameters["old_param"], eqx.filter(model, eqx.is_array))
                    si_parameters["omega"] = map(lambda omega, diff, w: omega + relu(w/(diff + epsilon)),
                                                 si_parameters["omega"],
                                                 difference,
                                                 si_parameters["w_k"])
                    si_parameters["w_k"] = map(
                        lambda x: jnp.zeros_like(x), si_parameters["w_k"])
                    si_parameters["old_param"] = eqx.filter(
                        model, eqx.is_array)
                # if we are doing bayesbinn, we update the prior with the current parameters at the end of the task
                if isinstance(model, BaseBayesBiNNMLP):
                    opt_state["prior"] = jax.tree.map(
                        lambda param: param, eqx.filter(model, eqx.is_array)
                    )
            if WEIGHT_HIST:
                # save  weights
                filter_weights = eqx.filter(model, eqx.is_array)
                output_leaves = leaves(filter_weights)
                for i, leaf in enumerate(output_leaves):
                    with open(os.path.join(WEIGHTS_PATH, f"layer={i}.npy"), "wb") as f:
                        jnp.save(f, leaf)
            # save the number of iterations
            np.save(os.path.join(CONFIGURATION_PATH,
                    "iterations.npy"), current_iterations)
            # serialize the model
            eqx.tree_serialise_leaves(os.path.join(
                CONFIGURATION_PATH, "model"), model)
        except (KeyboardInterrupt, SystemExit, Exception):
            print(traceback.format_exc())
            rmtree(SAVE_PATH)
