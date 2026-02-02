""" 
SPDX-License-Identifier: CC-BY-4.0
Code for "Active Continual Learning with Metaplastic Binary Bayesian Neural Networks"
Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz
Portions of the code are adapted from the Pytorch project (BSD-3-Clause)
Author: Kellian Cottart <kellian.cottart@gmail.com>
Date: 2025-30-01 

File description: Dataset downloading and loading
"""

import numpy as np
import idx2numpy
from torchvision.transforms import v2
from torchvision import models, datasets
import os
import requests
import pickle
import sys
from tqdm import tqdm
import hashlib
from torch import tensor, load, save, cat, from_numpy, LongTensor, Tensor, no_grad, randperm, prod, ones, clip, zeros, zeros_like
from torch.nn import AvgPool2d
from torch.nn import Sequential
from collections import Counter, defaultdict
from torch.utils.data import TensorDataset
from tqdm import tqdm
from PIL import Image
import gdown
from datasets import load_dataset
import urllib.request
import zipfile


PATH_MNIST_X_TRAIN = "datasets/MNIST/raw/train-images-idx3-ubyte"
PATH_MNIST_Y_TRAIN = "datasets/MNIST/raw/train-labels-idx1-ubyte"
PATH_MNIST_X_TEST = "datasets/MNIST/raw/t10k-images-idx3-ubyte"
PATH_MNIST_Y_TEST = "datasets/MNIST/raw/t10k-labels-idx1-ubyte"

PATH_FASHION_MNIST_X_TRAIN = "datasets/FashionMNIST/raw/train-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TRAIN = "datasets/FashionMNIST/raw/train-labels-idx1-ubyte"
PATH_FASHION_MNIST_X_TEST = "datasets/FashionMNIST/raw/t10k-images-idx3-ubyte"
PATH_FASHION_MNIST_Y_TEST = "datasets/FashionMNIST/raw/t10k-labels-idx1-ubyte"

PATH_EMNIST_X_TRAIN = "datasets/EMNIST/raw/emnist-balanced-train-images-idx3-ubyte"
PATH_EMNIST_Y_TRAIN = "datasets/EMNIST/raw/emnist-balanced-train-labels-idx1-ubyte"
PATH_EMNIST_X_TEST = "datasets/EMNIST/raw/emnist-balanced-test-images-idx3-ubyte"
PATH_EMNIST_Y_TEST = "datasets/EMNIST/raw/emnist-balanced-test-labels-idx1-ubyte"

PATH_KMNIST_X_TRAIN = "datasets/KMNIST/raw/train-images-idx3-ubyte"
PATH_KMNIST_Y_TRAIN = "datasets/KMNIST/raw/train-labels-idx1-ubyte"
PATH_KMNIST_X_TEST = "datasets/KMNIST/raw/t10k-images-idx3-ubyte"
PATH_KMNIST_Y_TEST = "datasets/KMNIST/raw/t10k-labels-idx1-ubyte"

PATH_CIFAR10 = "datasets/cifar-10-batches-py"
PATH_CIFAR10_DATABATCH = [
    f"{PATH_CIFAR10}/data_batch_{i}" for i in range(1, 6)]
PATH_CIFAR10_TESTBATCH = f"{PATH_CIFAR10}/test_batch"

PATH_CIFAR100 = "datasets/cifar-100-python"
PATH_CIFAR100_DATABATCH = [f"{PATH_CIFAR100}/train"]
PATH_CIFAR100_TESTBATCH = f"{PATH_CIFAR100}/test"
PATH_CIFAR100_META = f"{PATH_CIFAR100}/meta"

REPOSITORY_CORE50_NPZ_128 = "http://bias.csr.unibo.it/maltoni/download/core50/core50_imgs.npz"
REPOSITORY_CORE50_PATHS = "https://vlomonaco.github.io/core50/data/paths.pkl"
REPOSITORY_CORE50_LABELS = "https://vlomonaco.github.io/core50/data/labels.pkl"
REPOSITORY_CORE50_LUP = "https://vlomonaco.github.io/core50/data/LUP.pkl"


class GPULoading:
    """ Load local datasets on GPU using the TensorDataset

    Args:
        device (str, optional): Device to use. Defaults to "cuda:0".
    """

    def __init__(self, device="cuda:0", root="datasets", *args, **kwargs):
        self.device = device
        self.root = root
        if "test_batch_size" in kwargs:
            self.test_batch_size = kwargs["test_batch_size"]
        if "train_batch_size" in kwargs:
            self.train_batch_size = kwargs["train_batch_size"]
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def task_selection(self, task, padding=0, *args, **kwargs):
        """ Select the task to load

        Args:
            task (str): Name of the task

        Returns:
            train (TensorDataset): Training dataset
            test (TensorDataset): Testing dataset
            shape (tuple): Shape of the data
            target_size (int): Number of classes
        """
        self.padding = padding

        if "emnist" in task.lower():
            train, test = self.emnist(*args, **kwargs)
        elif "kmnist" in task.lower():
            train, test = self.kmnist(*args, **kwargs)
        elif "fullpmnist" in task.lower():
            train, test = self.permuted_mnist_full(
                *args, **kwargs)
        elif "mnist" in task.lower() and "fashion" in task.lower():
            train = []
            test = []
            mnist_train, mnist_test = self.mnist(*args, **kwargs)
            fashion_train, fashion_test = self.fashion_mnist(*args, **kwargs)
            train.append(mnist_train)
            train.append(fashion_train)
            test.append(mnist_test)
            test.append(fashion_test)
        elif "mnist" in task.lower():
            train, test = self.mnist(*args, **kwargs)
        elif "fashion" in task.lower():
            train, test = self.fashion_mnist(*args, **kwargs)
        elif "dilcifar100" in task.lower():
            train, test = self.domain_incremental_cifar100(*args, **kwargs)
        elif "cifar100" in task.lower():
            train, test = self.cifar100(*args, **kwargs)
        elif "cifar10" in task.lower():
            train, test = self.cifar10(*args, **kwargs)
        elif "core50" in task.lower():
            scenario = task.split("-")[1]
            train, test = self.core50(
                scenario=scenario, *args, **kwargs)
        elif "openloris" in task.lower():
            train, test = self.openloris(*args, **kwargs)
        elif "camelyon17" in task.lower():
            train, test = self.camelyon17wilds(*args, **kwargs)
        elif "animals" in task.lower():
            train, test = self.animals(*args, **kwargs)
        if not isinstance(train, list):
            train = [train]
        if not isinstance(test, list):
            test = [test]
        shape = train[0][0][0].shape
        # max index of the targets + 1 in all datasets
        target_size = max([int(test_dataset[:][1].max().item())
                          for test_dataset in test]) + 1
        return train, test, shape, target_size

    def camelyon17wilds(self, *args, **kwargs):
        # save the dataset
        if not os.path.exists("datasets/camelyon17wilds/train.pt") or not os.path.exists("datasets/camelyon17wilds/test.pt") or not os.path.exists("datasets/camelyon17wilds/validation_OOD.pt") or not os.path.exists("datasets/camelyon17wilds/validation_ID.pt"):
            # DenseNet-121
            dense = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1).to("cuda:0")
            transforms = models.DenseNet121_Weights.IMAGENET1K_V1.transforms().to("cuda:0")
            # Remove classification layer and add avgpooling
            model = Sequential(*list(dense.children())
                               [:-1] + [AvgPool2d(kernel_size=7, stride=1, padding=0)])
            model.eval()
            ds = load_dataset("wltjr1007/Camelyon17-WILDS",
                              cache_dir="datasets/")
            ds = ds.with_format("torch")
            if not os.path.exists("datasets/camelyon17wilds/validation_OOD.pt"):
                validation_center_OOD = ds["validation"]["center"]
                validation_x = ds["validation"]["image"]
                validation_y = ds["validation"]["label"]
                validation_x_OOD, validation_y_OOD = [], []
                validation_x_ID, validation_y_ID = [], []
                pbar = tqdm(total=len(validation_center_OOD),
                            desc="Extracting features from validation set")
                with pbar:
                    for image, label, center in zip(validation_x, validation_y, validation_center_OOD):
                        image = image.float().to("cuda:0")[
                            :-1, :, :].unsqueeze(0)
                        label = label.unsqueeze(0).cpu()
                        with no_grad():
                            image = model(transforms(image)).cpu()
                            if center != 1:
                                validation_x_ID.append(image)
                                validation_y_ID.append(label)
                            else:
                                validation_x_OOD.append(image)
                                validation_y_OOD.append(label)
                        pbar.update(1)
                validation_OOD = TensorDataset(
                    cat(validation_x_OOD).cpu(), cat(validation_y_OOD).cpu())
                validation_ID = TensorDataset(
                    cat(validation_x_ID).cpu(), cat(validation_y_ID).cpu())
                # Save the validation sets
                os.makedirs("datasets/camelyon17wilds", exist_ok=True)
                save(validation_OOD, "datasets/camelyon17wilds/validation_OOD.pt")
                save(validation_ID, "datasets/camelyon17wilds/validation_ID.pt")
            train_x = ds["train"]["image"]
            train_y = ds["train"]["label"]
            test_x = ds["test"]["image"]
            test_y = ds["test"]["label"]
            datasets = [train_x, test_x]
            labels = [train_y, test_y]
            if not os.path.exists("datasets/camelyon17wilds/train.pt") or not os.path.exists("datasets/camelyon17wilds/test.pt"):
                descs = ["Extracting features from train set",
                         "Extracting features from test set"]
                save_names = ["train", "test"]
                for dataset, label, save_name, desc in zip(datasets, labels, save_names, descs):
                    new_dataset = []
                    new_labels = []
                    pbar = tqdm(total=len(dataset), desc=desc)
                    with pbar:
                        for image, label in zip(dataset, label):
                            image = image.float().to("cuda:0")[
                                :-1, :, :].unsqueeze(0)
                            label = label.unsqueeze(0).cpu()
                            with no_grad():
                                new_dataset.append(
                                    model(transforms(image)).cpu())
                                new_labels.append(label)
                            pbar.update(1)
                    current = TensorDataset(
                        cat(new_dataset).cpu(), cat(new_labels).cpu())
                    os.makedirs("datasets/camelyon17wilds", exist_ok=True)
                    save(current, f"datasets/camelyon17wilds/{save_name}.pt")

        train = load("datasets/camelyon17wilds/train.pt", weights_only=False)
        test = load("datasets/camelyon17wilds/test.pt", weights_only=False)
        validation_OOD = load(
            "datasets/camelyon17wilds/validation_OOD.pt", weights_only=False)
        validation_ID = load(
            "datasets/camelyon17wilds/validation_ID.pt", weights_only=False)

        dataset_normalisation = kwargs.get("dataset_normalisation", "none")
        train_x = normalisation(
            train[:][0].cpu().numpy(), dataset_normalisation=dataset_normalisation)
        test_x = normalisation(
            test[:][0].cpu().numpy(), dataset_normalisation=dataset_normalisation)
        train = TensorDataset(
            from_numpy(train_x).to(self.device), train[:][1].to(self.device))
        test = TensorDataset(
            from_numpy(test_x).to(self.device), test[:][1].to(self.device))
        validation_ID_x = normalisation(
            validation_ID[:][0].cpu().numpy(), dataset_normalisation=dataset_normalisation)
        validation_OOD_x = normalisation(
            validation_OOD[:][0].cpu().numpy(), dataset_normalisation=dataset_normalisation)
        validation_ID = TensorDataset(
            from_numpy(validation_ID_x).to(self.device), validation_ID[:][1].to(self.device))
        validation_OOD = TensorDataset(
            from_numpy(validation_OOD_x).to(self.device), validation_OOD[:][1].to(self.device))

        test = [validation_ID, validation_OOD, test]
        # Shuffle the training data
        rand_perm = randperm(len(train)).to(self.device)
        train = train[rand_perm]
        return train, test

    def fashion_mnist(self, *args, **kwargs):
        if not os.path.exists(PATH_FASHION_MNIST_X_TRAIN):
            datasets.FashionMNIST("datasets", download=True)
        return self.mnist_like(PATH_FASHION_MNIST_X_TRAIN, PATH_FASHION_MNIST_Y_TRAIN,
                               PATH_FASHION_MNIST_X_TEST, PATH_FASHION_MNIST_Y_TEST, *args, **kwargs)

    def mnist(self, *args, **kwargs):
        if not os.path.exists(PATH_MNIST_X_TRAIN):
            datasets.MNIST("datasets", download=True)
        return self.mnist_like(PATH_MNIST_X_TRAIN, PATH_MNIST_Y_TRAIN,
                               PATH_MNIST_X_TEST, PATH_MNIST_Y_TEST, *args, **kwargs)

    def emnist(self, *args, **kwargs):
        if not os.path.exists(PATH_EMNIST_X_TRAIN):
            datasets.EMNIST("datasets", download=True, split="balanced")
        return self.mnist_like(PATH_EMNIST_X_TRAIN, PATH_EMNIST_Y_TRAIN,
                               PATH_EMNIST_X_TEST, PATH_EMNIST_Y_TEST, *args, **kwargs)

    def kmnist(self, *args, **kwargs):
        if not os.path.exists(PATH_KMNIST_X_TRAIN):
            datasets.KMNIST("datasets", download=True)
        return self.mnist_like(PATH_KMNIST_X_TRAIN, PATH_KMNIST_Y_TRAIN,
                               PATH_KMNIST_X_TEST, PATH_KMNIST_Y_TEST, *args, **kwargs)

    def mnist_like(self, path_train_x, path_train_y, path_test_x, path_test_y, *args, **kwargs):
        """ Load a local dataset on GPU corresponding either to MNIST or FashionMNIST

        Args:
            batch_size (int): Batch size
            path_train_x (str): Path to the training data
            path_train_y (str): Path to the training labels
            path_test_x (str): Path to the testing data
            path_test_y (str): Path to the testing labels
        """
        # load ubyte dataset
        train_x = idx2numpy.convert_from_file(
            path_train_x).astype(np.float32)
        train_y = idx2numpy.convert_from_file(
            path_train_y).astype(np.float32)
        test_x = idx2numpy.convert_from_file(
            path_test_x).astype(np.float32)
        test_y = idx2numpy.convert_from_file(
            path_test_y).astype(np.float32)
        # Normalize and pad the data
        dataset_normalisation = kwargs.get("dataset_normalisation", "none")
        train_x = normalisation(
            train_x, dataset_normalisation=dataset_normalisation)
        train_x = padding(train_x, self.padding)
        test_x = normalisation(
            test_x, dataset_normalisation=dataset_normalisation)
        test_x = padding(test_x, self.padding)
        return TensorDataset(train_x, Tensor(train_y).type(LongTensor)), TensorDataset(test_x, Tensor(test_y).type(LongTensor))

    def permuted_mnist_full(self, n_tasks=10, *args, **kwargs):
        if not os.path.exists(PATH_MNIST_X_TRAIN):
            datasets.MNIST("datasets", download=True)
        train_dataset, test_dataset = self.mnist_like(PATH_MNIST_X_TRAIN, PATH_MNIST_Y_TRAIN,
                                                      PATH_MNIST_X_TEST, PATH_MNIST_Y_TEST, *args, **kwargs)
        permutations = [randperm(784).cpu() for _ in range(n_tasks)]
        # create a dataset with n tasks all blended together
        train_x, train_y = train_dataset.data, train_dataset.targets
        test_x, test_y = test_dataset.data, test_dataset.targets
        test_data, test_labels, train_data, train_labels = [], [], [], []
        for i in range(n_tasks):
            perm = permutations[i]
            train_x_new = train_x.view(-1,
                                       784)[:, perm].view(-1, 1, 28, 28).clone()
            test_x_new = test_x.view(-1,
                                     784)[:, perm].view(-1, 1, 28, 28).clone()
            train_data.append(train_x_new)
            test_data.append(test_x_new)
            train_labels.append(train_y)
            test_labels.append(test_y)
        train_data = cat(train_data)
        test_data = cat(test_data)
        train_labels = cat(train_labels)
        test_labels = cat(test_labels)
        return train_dataset, test_dataset

    def cifar10(self, iterations=10, *args, **kwargs):
        """ Load a local dataset on GPU corresponding to CIFAR10 """
        # Deal with the training data
        if not os.path.exists("datasets/CIFAR10/raw"):
            datasets.CIFAR10("datasets", download=True)
        path_databatch = PATH_CIFAR10_DATABATCH
        path_testbatch = PATH_CIFAR10_TESTBATCH
        if "feature_extraction" in kwargs and kwargs["feature_extraction"] == True:
            folder = "datasets/cifar10_resnet18"
            os.makedirs(folder, exist_ok=True)
            if not os.listdir(folder) or not os.path.exists(f"{folder}/cifar10_{iterations}_features_train.pt"):
                train_x = []
                train_y = []
                for path in path_databatch:
                    with open(path, 'rb') as f:
                        dict = pickle.load(f, encoding='bytes')
                    train_x.append(dict[b'data'])
                    train_y.append(dict[b'labels'])
                train_x = np.concatenate(train_x)
                train_y = np.concatenate(train_y)
                # Deal with the test data
                with open(path_testbatch, 'rb') as f:
                    dict = pickle.load(f, encoding='bytes')
                test_x = dict[b'data']
                test_y = dict[b'labels']
                # Deflatten the data
                train_x = train_x.reshape(-1, 3, 32, 32)
                test_x = test_x.reshape(-1, 3, 32, 32)
                self.feature_extraction(
                    folder, train_x, train_y, test_x, test_y, task="cifar10", iterations=iterations)
            train_x = load(
                f"{folder}/cifar10_{iterations}_features_train.pt", weights_only=False)
            train_y = load(
                f"{folder}/cifar10_{iterations}_target_train.pt", weights_only=False)
            test_x = load(
                f"{folder}/cifar10_{iterations}_features_test.pt", weights_only=False)
            test_y = load(
                f"{folder}/cifar10_{iterations}_target_test.pt", weights_only=False)
        else:
            train_x = []
            train_y = []
            for path in path_databatch:
                with open(path, 'rb') as f:
                    dict = pickle.load(f, encoding='bytes')
                train_x.append(dict[b'data'])
                train_y.append(dict[b'labels'])
            train_x = np.concatenate(train_x)
            train_y = np.concatenate(train_y)
            # Deal with the test data
            with open(path_testbatch, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
            test_x = dict[b'data']
            test_y = dict[b'labels']
            # Deflatten the data
            train_x = train_x.reshape(-1, 3, 32, 32)
            test_x = test_x.reshape(-1, 3, 32, 32)
            # Normalize and pad the data
            train_x = normalisation(train_x, padding=self.padding)
            test_x = normalisation(test_x, padding=self.padding)
        return self.to_dataset(train_x, train_y, test_x, test_y)

    def cifar100(self, iterations=10, *args, **kwargs):
        """ Load a local dataset on GPU corresponding to CIFAR100 """
        if not os.path.exists("datasets/CIFAR100/raw"):
            datasets.CIFAR100("datasets", download=True)
        if "feature_extraction" in kwargs and kwargs["feature_extraction"] == True:
            folder = "datasets/cifar100_resnet18"
            os.makedirs(folder, exist_ok=True)
            if not os.listdir(folder) or not os.path.exists(f"{folder}/cifar100_{iterations}_features_train.pt"):
                path_databatch = PATH_CIFAR100_DATABATCH
                path_testbatch = PATH_CIFAR100_TESTBATCH
                with open(path_databatch[0], "rb") as f:
                    data = pickle.load(f, encoding="bytes")
                    train_x = data[b"data"]
                    train_y = data[b"fine_labels"]
                with open(path_testbatch, "rb") as f:
                    data = pickle.load(f, encoding="bytes")
                    test_x = data[b"data"]
                    test_y = data[b"fine_labels"]
                train_x = train_x.reshape(-1, 3, 32, 32)
                test_x = test_x.reshape(-1, 3, 32, 32)
                self.feature_extraction(
                    folder, train_x, train_y, test_x, test_y, task="cifar100", iterations=iterations)
            train_x = load(
                f"{folder}/cifar100_{iterations}_features_train.pt", weights_only=False)
            train_y = load(
                f"{folder}/cifar100_{iterations}_target_train.pt", weights_only=False)
            test_x = load(
                f"{folder}/cifar100_{iterations}_features_test.pt", weights_only=False)
            test_y = load(
                f"{folder}/cifar100_{iterations}_target_test.pt", weights_only=False)
        else:
            train_x, fine_labels, test_x, test_fine_labels, coarse_labels, test_coarse_labels = self.read_cifar100()
            train_x = train_x.reshape(-1, 3, 32, 32)
            test_x = test_x.reshape(-1, 3, 32, 32)
            train_y = fine_labels
            test_y = test_fine_labels
            # Normalize and pad the data
            train_x = normalisation(train_x, padding=self.padding)
            test_x = normalisation(test_x, padding=self.padding)
        return self.to_dataset(train_x, train_y, test_x, test_y)

    def feature_extraction(self, folder, train_x, train_y, test_x, test_y, task="cifar100", iterations=10):
        """ Extract features using a resnet18 model

        Args:
            folder (str): Folder to save the features
            train_x (tensor): Training data
            train_y (tensor): Training labels
            test_x (tensor): Testing data
            test_y (tensor): Testing labels
            task (str, optional): Name of the task. Defaults to "cifar100".
            iterations (int, optional): Number of passes to make. Defaults to 10.
        """
        print(f"Extracting features from {task}...")
        resnet18 = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        # Remove the classification layer
        resnet18 = Sequential(
            *list(resnet18.children())[:-1])
        # Freeze the weights of the feature extractor
        for param in resnet18.parameters():
            param.requires_grad = False
        # Transforms to apply to augment the data
        transform_train = v2.Compose([
            v2.feature_extraction(220, antialias=True),
            v2.RandomHorizontalFlip(),
        ])
        transform_test = v2.Compose([
            v2.feature_extraction(220, antialias=True),
        ])
        # Extract the features
        features_train = []
        target_train = []
        features_test = []
        target_test = []
        # Normalize
        train_x = from_numpy(train_x).float() / 255
        test_x = from_numpy(test_x).float() / 255
        if len(train_x.size()) == 3:
            train_x = train_x.unsqueeze(1)
            test_x = test_x.unsqueeze(1)
        # Converting the data to a GPU TensorDataset (allows to load everything in the GPU memory at once)
        train_dataset = TensorDataset(
            train_x, Tensor(train_y).type(
                LongTensor), device=self.device)
        test_dataset = TensorDataset(test_x, Tensor(test_y).type(
            LongTensor), device=self.device)
        train_dataset = DataLoader(
            train_dataset, batch_size=1024, shuffle=True, drop_last=False, transform=transform_train, device=self.device)
        test_dataset = DataLoader(
            test_dataset, batch_size=1024, shuffle=True, device=self.device, transform=transform_test)
        # Make n passes to extract the features
        for _ in range(iterations):
            for data, target in train_dataset:
                features_train.append(resnet18(data))
                target_train.append(target)
        for data, target in test_dataset:
            features_test.append(resnet18(data))
            target_test.append(target)

        # Concatenate the features
        features_train = cat(features_train)
        target_train = cat(target_train)
        features_test = cat(features_test)
        target_test = cat(target_test)
        # Save the features
        save(features_train,
             f"{folder}/{task}_{iterations}_features_train.pt")
        save(
            target_train, f"{folder}/{task}_{iterations}_target_train.pt")
        save(features_test,
             f"{folder}/{task}_{iterations}_features_test.pt")
        save(
            target_test, f"{folder}/{task}_{iterations}_target_test.pt")

    def to_dataset(self, train_x, train_y, test_x, test_y):
        """ Create a DataLoader to load the data in batches

        Args:
            train_x (tensor): Training data
            train_y (tensor): Training labels
            test_x (tensor): Testing data
            test_y (tensor): Testing labels
            batch_size (int): Batch size

        Returns:
            DataLoader, DataLoader: Training and testing DataLoader

        """
        train_dataset = TensorDataset(
            train_x, Tensor(train_y).type(
                LongTensor))
        test_dataset = TensorDataset(test_x, Tensor(test_y).type(
            LongTensor))
        return train_dataset, test_dataset

    def read_cifar100(self):
        with open(PATH_CIFAR100_DATABATCH[0], "rb") as f:
            data = pickle.load(f, encoding="bytes")
            training_data = data[b"data"]
            fine_labels = data[b"fine_labels"]
            coarse_labels = data[b"coarse_labels"]
        with open(PATH_CIFAR100_TESTBATCH, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            test_data = data[b"data"]
            test_fine_labels = data[b"fine_labels"]
            test_coarse_labels = data[b"coarse_labels"]
        return training_data, fine_labels, test_data, test_fine_labels, coarse_labels, test_coarse_labels

    def domain_incremental_cifar100(
            self, feature_extraction=False, full=False, *args, **kwargs):
        if not os.path.exists("datasets/CIFAR100/raw"):
            datasets.CIFAR100("datasets", download=True)
        train_datasets, test_datasets = self.cifar100_cil_dataset_generation(
            full=full)
        if feature_extraction:
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )
            features = Sequential(
                *list(resnet.children())[:-1])
            transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
            for i in range(len(train_datasets)):
                train_dataset = train_datasets[i]
                test_dataset = test_datasets[i]
                train_datasets[i] = self.set_to_feature_set(
                    train_dataset, features, transform)
                test_datasets[i] = self.set_to_feature_set(
                    test_dataset, features, transform)
        return train_datasets, test_datasets

    def cifar100_cil_dataset_generation(self, full=False):
        # Normalize and pad the data
        training_data, fine_labels, test_data, test_fine_labels, coarse_labels, test_coarse_labels = self.read_cifar100()
        training_data = training_data.reshape(-1, 3, 32, 32)
        test_data = test_data.reshape(-1, 3, 32, 32)
        training_data = normalisation(training_data)
        training_data = padding(training_data, self.padding)
        test_data = normalisation(test_data)
        test_data = padding(test_data, self.padding)
        rescale = v2.Resize(224)
        training_data = rescale(training_data)
        test_data = rescale(test_data)
        # scale data to imagenet size
        # I want to retrieve the class number for each fine label, and sort them by coarse label
        fine_to_coarse = {}
        for (fine, coarse) in zip(fine_labels, coarse_labels):
            if fine not in fine_to_coarse:
                fine_to_coarse[fine] = coarse
        fine_to_coarse = dict(sorted(fine_to_coarse.items()))
        # Organize fine labels by coarse labels (superclasses)
        coarse_to_fine = defaultdict(list)
        for fine, coarse in fine_to_coarse.items():
            coarse_to_fine[coarse].append(fine)
        coarse_to_fine = dict(sorted(coarse_to_fine.items()))
        selected_classes = set()
        datasets_class_mapping = []
        for i in range(5):
            dataset_fine_classes = []
            for coarse, fine_list in coarse_to_fine.items():
                available_fine_classes = [
                    fine for fine in fine_list if fine not in selected_classes]
                if available_fine_classes:
                    chosen_fine = np.random.choice(available_fine_classes)
                    dataset_fine_classes.append(chosen_fine)
                    selected_classes.add(chosen_fine)
            datasets_class_mapping.append(dataset_fine_classes)
        train_datasets = []
        test_datasets = []
        for i, dataset_fine_classes in enumerate(datasets_class_mapping):
            train_x, train_y = [], []
            test_x, test_y = [], []
            # training data
            for j, fine in enumerate(fine_labels):
                if fine in dataset_fine_classes:
                    train_x.append(training_data[j])
                    train_y.append(coarse_labels[j])
            train_x = from_numpy(np.array(train_x).reshape(-1, 3, 224, 224))
            train_y = from_numpy(np.array(train_y))
            # testing data
            for j, fine in enumerate(test_fine_labels):
                if fine in dataset_fine_classes:
                    test_x.append(test_data[j])
                    test_y.append(test_coarse_labels[j])
            test_x = from_numpy(np.array(test_x).reshape(-1, 3, 224, 224))
            test_y = from_numpy(np.array(test_y))
            # normalize and pad the data
            train_dataset, test_dataset = self.to_dataset(
                train_x, train_y, test_x, test_y)
            # extract the features from each dataset
            train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
        if full:
            # blend all the datasets
            train_x, train_y = [], []
            test_x, test_y = [], []
            for i in range(len(train_datasets)):
                train_x.append(train_datasets[i].data)
                train_y.append(train_datasets[i].targets)
                test_x.append(test_datasets[i].data)
                test_y.append(test_datasets[i].targets)
            train_x = cat(train_x)
            train_y = cat(train_y)
            test_x = cat(test_x)
            test_y = cat(test_y)
            train_datasets = [TensorDataset(
                train_x, train_y)]
            test_datasets = [TensorDataset(
                test_x, test_y)]
        return train_datasets, test_datasets

    def set_to_feature_set(self, dataset, features, transform):
        batch_size = 64
        number_of_batches = len(dataset) // batch_size if len(
            dataset) % batch_size == 0 else len(dataset) // batch_size + 1
        storage = []
        pbar = tqdm(total=number_of_batches, desc="Extracting features")
        for i in range(number_of_batches):
            with no_grad():
                batch, targets = dataset[i * batch_size:(i + 1) * batch_size]
                batch = transform(batch)
                batch = features(batch)
            storage.append((batch.to("cpu"), targets.to("cpu")))
            pbar.update(1)
        pbar.close()
        # turn storage into a dataset
        features, targets = zip(*storage)
        features, targets = cat(features), cat(targets)
        return TensorDataset(features, targets)

    def core50(self, scenario="ni", run=0, download=True, *args, **kwargs):

        core_class = CORe50(scenario=scenario,
                            run=run,
                            download=download,
                            device=self.device)
        train_datasets, test_dataset = core_class.get_dataset()
        if "feature_extraction" in kwargs and kwargs["feature_extraction"] == True:
            train_datasets, test_dataset = core_class.extract_features(
                train_datasets, test_dataset)
        return train_datasets, test_dataset

    def openloris(self, *args, **kwargs):
        if not os.path.exists("datasets"):
            os.makedirs("datasets", exist_ok=True)
        dataset = OpenLORIS(root="datasets")
        dataset.download_dataset(*args, **kwargs)
        return dataset.get_dataset(*args, **kwargs)

    def animals(self, *args, **kwargs):
        n_test_examples_per_class = 50
        data_dir = "datasets/animals"
        train_data_dir = os.path.join(data_dir, "train")
        test_data_dir = os.path.join(data_dir, "test")
        classes_to_pick = {
            "Butterfly": 0,
            "Lizard": 1,
            "Fish": 2,
            "Monkey": 3,
            "Spider": 4,
            "Eagle": 5,
            "Frog": 6,
            "Jellyfish": 7,
            "Penguin": 8,
            "Whale": 9,
            "Zebra": 10,
            "Crocodile": 11,
            "Leopard": 12,
            "Sheep": 13,
            "Raccoon": 14,
            "Raven": 15,
            "Panda": 16,
            "Lynx": 17,
            "Bull": 18,
            "Scorpion": 19,
        }
        # print("Classes to pick:", classes_to_pick)
        n_classes = len(classes_to_pick)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            PATH = "https://www.kaggle.com/api/v1/datasets/download/antoreepjana/animals-detection-images-dataset"
            # Download animals dataset with progress bar
            pbar = tqdm(unit='B', unit_scale=True,
                        desc=f"Downloading Animals dataset to {data_dir}...")
            zip_path = os.path.join(data_dir, "animals.zip")
            with urllib.request.urlopen(PATH) as response, open(zip_path, 'wb') as out_file:
                file_size = int(response.getheader('Content-Length', 0))
                pbar.total = file_size
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(zip_path)

        def get_class_counts(data_dir):
            classes = os.listdir(data_dir)
            class_counts = {
                cls: len(os.listdir(os.path.join(data_dir, cls)))
                for cls in classes if os.path.isdir(os.path.join(data_dir, cls))
            }
            return dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

        train_class_counts = get_class_counts(train_data_dir)
        test_class_counts = get_class_counts(test_data_dir)
        # new dict out of the sum of both dicts
        combined_class_counts = {cls: train_class_counts.get(cls, 0) + test_class_counts.get(cls, 0)
                                 for cls in set(train_class_counts) | set(test_class_counts)}
        combined_class_counts = dict(sorted(
            combined_class_counts.items(), key=lambda item: item[1], reverse=True))
        # print("Examples counts (combined):")
        # for cls, count in combined_class_counts.items():
        #     print(f"{cls}: {count}")
        feature_extractor = kwargs.get("feature_extractor", "resnet18")
        dataset_path = f"{data_dir}/dataset_{feature_extractor}_{n_classes}.pt"

        if not os.path.exists(dataset_path):
            print("Preparing the Animals dataset...")
            # extract the images from the folders
            dataset = self.extract_animals_folders(
                classes_to_pick, data_dir, **kwargs)
            # save the datasets
            save(dataset, dataset_path)

        dataset = load(dataset_path, weights_only=False)

        # From each class in the dataset, take n_test_examples_per_class for the test set, and the rest for the train set
        images, labels = dataset[:][0], dataset[:][1]
        train_indices = []
        test_indices = []
        for class_idx in range(len(labels.unique())):
            class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
            # Shuffle the indices
            class_indices = class_indices[randperm(len(class_indices))]
            test_indices.extend(
                class_indices[:n_test_examples_per_class].tolist())
            train_indices.extend(
                class_indices[n_test_examples_per_class:].tolist())
        train_dataset = TensorDataset(
            images[train_indices], labels[train_indices])
        test_dataset = TensorDataset(
            images[test_indices], labels[test_indices])
        # shuffle the training dataset
        train_dataset = train_dataset[randperm(len(train_dataset))]

        # apply normalisation fct to the datasets
        dataset_normalisation = kwargs.get("dataset_normalisation", "none")
        # normalize each dataset:
        train_x = normalisation(
            train_dataset[:][0].numpy(), dataset_normalisation=dataset_normalisation)
        train_dataset = TensorDataset(train_x, train_dataset[:][1])

        test_x = normalisation(
            test_dataset[:][0].numpy(), dataset_normalisation=dataset_normalisation)
        test_dataset = TensorDataset(test_x, test_dataset[:][1])

        n_split = kwargs.get("interleaved", 0)
        if n_split > 0:
            # Retrieve images and labels from each class individually
            images, labels = train_dataset[:][0], train_dataset[:][1]
            classes_images = []
            classes_labels = []
            for class_idx in range(len(labels.unique())):
                class_indices = (labels == class_idx).nonzero(as_tuple=True)[0]
                class_images = images[class_indices]
                class_labels = labels[class_indices]
                # We subdivide each class dataset into n_split parts
                n_data = len(class_images) // n_split
                split_images = []
                split_labels = []
                for i in range(n_split):
                    split_images.append(
                        class_images[i * n_data:(i + 1) * n_data])
                    split_labels.append(
                        class_labels[i * n_data:(i + 1) * n_data])
                classes_images.append(split_images)
                classes_labels.append(split_labels)
            # Now we interleave randomly the parts of each class to create the final train dataset
            interleaved_images = []
            interleaved_labels = []
            for i in range(n_split):
                for class_idx in range(len(labels.unique())):
                    interleaved_images.append(classes_images[class_idx][i])
                    interleaved_labels.append(classes_labels[class_idx][i])

            # randomly shuffle the interleaved parts
            perm = randperm(len(interleaved_images))
            interleaved_images = cat([interleaved_images[i] for i in perm])
            interleaved_labels = cat([interleaved_labels[i] for i in perm])
            train_dataset = TensorDataset(
                interleaved_images, interleaved_labels)

            if kwargs.get("CL", False):
                # We want to create a CL scenario where each task contains 5 classes
                n_classes_per_task = 5
                n_tasks = len(labels.unique()) // n_classes_per_task
                # randomly sort classes indexes
                class_indices = list(range(len(labels.unique())))
                class_indices = np.array(class_indices)
                np.random.shuffle(class_indices)
                # For each task, get the corresponding class indices
                task_class_indices = [class_indices[i:i + n_classes_per_task]
                                      for i in range(0, len(class_indices), n_classes_per_task)]
                # create a dictionnary mapping old class index to new class index for each task
                task_class_dict = {
                    task_idx: {old_idx: new_idx for new_idx,
                               old_idx in enumerate(task_class_indices[task_idx])}
                    for task_idx in range(n_tasks)
                }
                # For each task, we want a train dataset containing only the classes of that task, with remapped labels
                train_datasets = []
                for task_idx in range(n_tasks):
                    class_dict = task_class_dict[task_idx]
                    task_images = []
                    task_labels = []
                    for image, label in zip(train_dataset[:][0], train_dataset[:][1]):
                        if label.item() in class_dict:
                            task_images.append(image.unsqueeze(0))
                            task_labels.append(
                                from_numpy(np.array(class_dict[label.item()])).unsqueeze(0))
                    task_images = cat(task_images)
                    task_labels = cat(task_labels)
                    train_datasets.append(TensorDataset(
                        task_images, task_labels))
                train_dataset = train_datasets
                # then remap the labels of the test dataset using the flattened task_class_dict
                flat_class_dict = {item: new_idx for task_dict in task_class_dict.values(
                ) for item, new_idx in task_dict.items()}
                test_images = test_dataset[:][0]
                test_labels = test_dataset[:][1]
                remapped_test_images = []
                remapped_test_labels = []
                for image, label in zip(test_images, test_labels):
                    if label.item() in flat_class_dict:
                        remapped_test_images.append(image.unsqueeze(0))
                        remapped_test_labels.append(
                            from_numpy(np.array(flat_class_dict[label.item()])).unsqueeze(0))
                remapped_test_images = cat(remapped_test_images)
                remapped_test_labels = cat(remapped_test_labels)
                test_dataset = TensorDataset(
                    remapped_test_images, remapped_test_labels)

        subfeatures = kwargs.get("subfeatures", None)
        if subfeatures is not None:
            # Reduce the number of features by keeping only the first subfeatures features, nb of
            shape = prod(tensor(train_x[:][0].shape[1:]))
            random_indexes = np.random.choice(
                shape, subfeatures, replace=False)
            train_x = train_dataset[:][0].reshape(len(train_x), -1, 1, 1)
            train_x = train_x[:, random_indexes]
            train_y = train_dataset[:][1]
            train_dataset = TensorDataset(train_x, train_y)
            test_x = test_dataset[:][0].reshape(len(test_x), -1, 1, 1)
            test_x = test_x[:, random_indexes]
            test_y = test_dataset[:][1]
            test_dataset = TensorDataset(test_x, test_y)

        return train_dataset, test_dataset

    def extract_animals_folders(self, classes_to_pick, data_dir, **kwargs):
        images = []
        labels = []
        total_images = 0
        # Count total images for progress bar (train + test)
        for split in ["train", "test"]:
            split_dir = os.path.join(data_dir, split)
            for cls_name in classes_to_pick.keys():
                cls_dir = os.path.join(split_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                total_images += len([
                    name for name in os.listdir(cls_dir)
                    if os.path.isfile(os.path.join(cls_dir, name))
                ])
        pbar = tqdm(total=total_images, desc=f"Loading images from {data_dir}")
        features, transform = get_feature_extractor(
            kwargs["feature_extractor"], device=self.device)

        for split in ["train", "test"]:
            split_dir = os.path.join(data_dir, split)
            for cls_name in classes_to_pick.keys():
                cls_dir = os.path.join(split_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    if os.path.isfile(img_path):
                        current_image = Image.open(img_path).convert("RGB")
                        with no_grad():
                            current_image = transform(
                                current_image).unsqueeze(0)
                            current_image = features(
                                current_image).squeeze().to("cpu")
                        images.append(current_image.numpy())
                        labels.append(classes_to_pick[cls_name])
                    pbar.update(1)
        pbar.close()
        images = np.array(images)
        labels = np.array(labels)
        return TensorDataset(from_numpy(images), from_numpy(labels).type(LongTensor))


class CORe50:
    """ Load the CORe50 dataset
    INSPIRED BY Vincenzo Lomonaco

    Args:
        root (str, optional): Root folder for the dataset. Defaults to "datasets".
        scenario (str, optional): Scenario to load. Defaults to "ni".
        run (int, optional): Run to load. Defaults to 0.
        start_batch (int, optional): Starting batch. Defaults to 0.
        download (bool, optional): Download the dataset. Defaults to True.
        device (str, optional): Device to use. Defaults to "cuda:0".
    """

    def __init__(self, root="datasets", scenario="ni", run=0, download=True, device="cpu"):
        self.root = os.path.join(root, "core50")
        self.scenario = scenario
        self.run = run
        self.device = device
        self.batch_scenario = {
            "ni": 8,
            'nc': 9,
            'nic': 79,
            'nicv2_79': 79,
            'nicv2_196': 196,
            'nicv2_391': 391
        }
        self.md5 = {
            "core50_imgs.npz": "3689d65d0a1c760b87821b114c8c4c6c",
            "labels.pkl": "281c95774306a2196f4505f22fd60ab1",
            "paths.pkl": "b568f86998849184df3ec3465290f1b0",
            "LUP.pkl": "33afc26faa460aca98739137fdfa606e"
        }
        if not os.path.exists(self.root) or not os.listdir(self.root):
            os.makedirs(self.root, exist_ok=True)
            self.download_dataset()

        bin_path = os.path.join(self.root, "core50_imgs.bin")
        if not os.path.exists(bin_path):
            data = np.load(os.path.join(self.root, "core50_imgs.npz"))['x']
            data.tofile(bin_path)

        self.data = np.fromfile(bin_path, dtype=np.uint8).reshape(
            164866, 128, 128, 3)
        self.labels = pickle.load(
            open(os.path.join(self.root, "labels.pkl"), "rb"))
        self.paths = pickle.load(
            open(os.path.join(self.root, "paths.pkl"), "rb"))
        self.lup = pickle.load(open(os.path.join(self.root, "LUP.pkl"), "rb"))

    def download_dataset(self):
        """ Download the dataset """
        files_to_download = [
            ("core50_imgs.npz", REPOSITORY_CORE50_NPZ_128),
            ("paths.pkl", REPOSITORY_CORE50_PATHS),
            ("labels.pkl", REPOSITORY_CORE50_LABELS),
            ("LUP.pkl", REPOSITORY_CORE50_LUP)
        ]
        for file_name, url in files_to_download:
            file_path = os.path.join(self.root, file_name)
            if not os.path.exists(file_path):
                print(f"Downloading {file_name}...")
                self.download_file(url, file_path)

    def checksum(self, file_path):
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(4096):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    def download_file(self, url, file_path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)
        with open(file_path, 'wb') as file:
            for data in response.iter_content(1024):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if not self.checksum(file_path) == self.md5[os.path.basename(file_path)]:
            print("Checksum failed. Deleting file.")
            os.remove(file_path)
            sys.exit(1)
        else:
            print("Checksum validated for " + file_path)

    def get_dataset(self):
        # There is only one full dataset for testing
        test_indexes = self.lup[self.scenario][self.run][-1]
        test_x = normalisation(np.moveaxis(
            self.data[test_indexes], 3, 1), dataset_normalisation="none")
        test_y = tensor(
            self.labels[self.scenario][self.run][-1]).to("cpu")
        test_dataset = TensorDataset(test_x, test_y)
        # ...but multiple training datasets divided in "batches"
        train_datasets = []
        pbar = tqdm(total=self.batch_scenario[self.scenario],
                    desc="Loading training datasets...")
        with pbar:
            for i in range(self.batch_scenario[self.scenario]):
                train_indexes = self.lup[self.scenario][self.run][i]
                train_x = normalisation(np.moveaxis(
                    self.data[train_indexes], 3, 1), dataset_normalisation="none")
                train_y = tensor(
                    self.labels[self.scenario][self.run][i], device="cpu")
                train_dataset = TensorDataset(train_x, train_y)
                train_datasets.append(train_dataset)
                pbar.update(1)
        return train_datasets, test_dataset

    def extract_features(self, train_datasets, test_dataset):
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT).to("cuda:0")
        model = Sequential(model.features, model.avgpool).to("cuda:0")
        transforms = models.EfficientNet_B0_Weights.DEFAULT.transforms()
        new_train_datasets = []
        batch_size = 128

        def extract_features_from_loader(loader, model, device):
            features = []
            targets = []
            pbar = tqdm(total=len(loader),
                        desc="Extracting features from loader...")
            with pbar:
                for data, target in loader:
                    with no_grad():
                        data = model(transforms(data.to(device)))
                    features.append(data.to("cpu"))
                    targets.append(target.to("cpu"))
                    pbar.update(1)
            # cat all datasets
            return cat(features), cat(targets)
        # progress bar
        for train_dataset in train_datasets:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False)
            features, targets = extract_features_from_loader(
                train_loader, model, device="cuda:0")
            new_train_datasets.append(TensorDataset(features, targets))
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        test_features, test_targets = extract_features_from_loader(
            test_loader, model, device="cuda:0")
        new_test_dataset = TensorDataset(test_features, test_targets)
        return new_train_datasets, new_test_dataset


class OpenLORIS:
    """ Load the OpenLORIS dataset"""

    def __init__(self, root="datasets", device="cuda:0"):
        self.root = os.path.join(root, "OpenLORIS")
        self.device = device

    # What we are going to do is make 9 datasets out of the segments of the dataset
    # with classes corresponding to each classes fond in the segment
    # Then domain incremental learning manifests itself as

    def download_and_extract(self, url, zip_path, md5_checksum):
        # wait for download using gdown
        gdown.cached_download(url=url, path=zip_path, quiet=False,
                              hash="md5:"+md5_checksum, postprocess=gdown.extractall)

    def download_dataset(self, validation=False, *args, **kwargs):
        folder = "OpenLORIS"
        id_train = "https://drive.google.com/uc?id=1eNYCT4WN5Gq_Fpsk2HSjuN56mSkhahkZ"
        id_test = "https://drive.google.com/uc?id=1MO1EtyZZio2kekTChLptqm0UywftQkkN"
        train_md5_checksum = "4c1369a0da893ef2fce28b78a2443a55"
        test_md5_checksum = "05bd74fb07968586f646a6987ac3a6c6"

        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)

        if not os.path.exists(f"{self.root}/train"):
            print(f"Downloading train datasets...")
            self.download_and_extract(
                id_train, f"{self.root}/train.zip", train_md5_checksum)

        if not os.path.exists(f"{self.root}/test"):
            print(f"Downloading test datasets...")
            self.download_and_extract(
                id_test, f"{self.root}/test.zip", test_md5_checksum)

        if validation:
            id_validation = "https://drive.google.com/uc?id=14yoZMh7eDFtj1jnHzdzCYcRz_VQ2ED0e"
            validation_md5_checksum = "ad4bc64a4d1911ce022f1bbf213b7ba6"
            if not os.path.exists(f"{self.root}/validation"):
                print(f"Downloading validation datasets...")
                self.download_and_extract(
                    id_validation, f"{self.root}/validation.zip", validation_md5_checksum)

    def get_dataset(self,
                    classes=["bottle", "bowl", "corkscrew", "cottonswab", "cup", "cushion", "glasses", "knife", "ladle",
                             "mask", "paper_cutter", "pencil", "plasticbag", "plug", "pot", "scissors", "stapler", "thermometer", "toy"],
                    validation=False,
                    *args,
                    **kwargs):
        # add the first letter of the categories
        categories = ["illumination", "occlusion",  "pixel", "clutter"]
        all_classes = ["bottle", "bowl", "corkscrew", "cottonswab", "cup", "cushion", "glasses", "knife", "ladle",
                       "mask", "paper_cutter", "pencil", "plasticbag", "plug", "pot", "scissors", "stapler", "thermometer", "toy"]
        test_or_validation = "validation" if validation else "test"
        train_path = os.path.join(self.root, "train")
        test_path = os.path.join(self.root, test_or_validation)
        n_segments = 9
        feature_extractor = kwargs.get("feature_extractor", "efficientnet")
        model, transforms = get_feature_extractor(
            feature_extractor, self.device)

        def benchmark2_data_load(test_train_path):
            test_train_datasets = []
            paths_to_categories = [os.path.join(
                test_train_path, f"{categories[j]}") for j in range(4)]
            paths_to_difficulties = []
            for path in paths_to_categories:
                for segment in range(0, n_segments, 3):
                    difficulties = [
                        os.path.join(path, f"segment{segment+1}"),
                        os.path.join(path, f"segment{segment+2}"),
                        os.path.join(path, f"segment{segment+3}")
                    ]
                    paths_to_difficulties.append(difficulties)

            pbar = tqdm(total=len(paths_to_difficulties),
                        desc="Loading difficulty levels for Benchmark 2: Sequential factors analysis")
            with pbar:
                for path in paths_to_difficulties:
                    pbar_difficulty = tqdm(total=len(path),
                                           desc="Loading difficulty levels...")
                    x, y = [], []
                    with pbar_difficulty:
                        for difficulty in path:
                            for folders in os.listdir(difficulty):
                                for k, class_name in enumerate(all_classes):
                                    if class_name in folders:
                                        for img in os.listdir(os.path.join(difficulty, folders)):
                                            img_path = os.path.join(
                                                difficulty, folders, img)
                                            img = Image.open(img_path)
                                            img = np.array(img)
                                            img = np.moveaxis(img, 2, 0)
                                            img = np.expand_dims(img, axis=0)
                                            img = from_numpy(
                                                img).float().to(self.device)
                                            with no_grad():
                                                img = transforms(img)
                                                img = model(img)
                                            x.append(img.to("cpu"))
                                            y.append(tensor([k]).to("cpu"))
                            pbar_difficulty.update(1)
                    pbar.update(1)
                    x = cat(x)
                    y = cat(y)
                    test_train_datasets.append(TensorDataset(x, y))
            return test_train_datasets

        feature_extractor = kwargs.get("feature_extractor", "efficientnet")

        if not os.path.exists(f"{self.root}/train-{feature_extractor}.pt"):
            train_datasets = benchmark2_data_load(train_path)
            save(train_datasets, f"{self.root}/train-{feature_extractor}.pt")
        else:
            train_datasets = load(
                f"{self.root}/train-{feature_extractor}.pt", weights_only=False)
        if not os.path.exists(f"{self.root}/{test_or_validation}-{feature_extractor}.pt"):
            test_datasets = benchmark2_data_load(test_path)
            save(test_datasets,
                 f"{self.root}/{test_or_validation}-{feature_extractor}.pt")
        else:
            test_datasets = load(
                f"{self.root}/{test_or_validation}-{feature_extractor}.pt", weights_only=False)
        # y = train_datasets[0][:][1]
        # class_counts = Counter(y.tolist())
        # # map name to class idx
        # print("Class counts")
        # for class_idx, count in enumerate(class_counts.items()):
        #     print(
        #         f"Class {all_classes[class_idx]} ({class_idx}): {count} examples")

        if kwargs.get("unbalanced", False):
            # Change the train dataset to contain less examples of certain classes
            n_rand = kwargs.get("n_rand", 1)
            # count the number of examples per class in each dataset
            new_train_datasets = []
            for _ in range(n_rand):
                for i in range(len(train_datasets)):
                    x, y = train_datasets[i][:][0], train_datasets[i][:][1]
                    class_counts = Counter(y.tolist())
                    # for all classes below 700 examples, remove from 0.9 to 0.98 of the examples randomly
                    for class_idx, count in class_counts.items():
                        if count < kwargs["unbalanced"].get("threshold_max", 700):
                            remove_ratio = np.random.uniform(kwargs["unbalanced"].get("remove_ratio_min", 0.9),
                                                             kwargs["unbalanced"].get("remove_ratio_max", 0.98))
                            class_indices = (y == class_idx).nonzero(
                                as_tuple=True)[0]
                            n_min = kwargs["unbalanced"].get("n_min", 30)
                            n_remove = clip(
                                tensor([int(count * remove_ratio)]), tensor([0]), tensor([len(class_indices)-n_min])).item()
                            remove_indices = np.random.choice(
                                class_indices.numpy(), n_remove, replace=False)
                            mask = ones(len(y), dtype=bool)
                            mask[remove_indices] = False
                            x = x[mask]
                            y = y[mask]
                    # shuffle the dataset
                    perm = randperm(len(y))
                    new_train_datasets.append(TensorDataset(x[perm], y[perm]))
            train_datasets = new_train_datasets
            x, y = train_datasets[i][:][0], train_datasets[i][:][1]

            # Make the test set by evening the number of examples per class
            for i in range(len(test_datasets)):
                x, y = test_datasets[i][:][0], test_datasets[i][:][1]
                class_counts = Counter(y.tolist())
                min_count = min(class_counts.values())
                balanced_x = []
                balanced_y = []
                for class_idx in class_counts.keys():
                    class_indices = (y == class_idx).nonzero(as_tuple=True)[0]
                    selected_indices = class_indices[:min_count]
                    balanced_x.append(x[selected_indices])
                    balanced_y.append(y[selected_indices])
                balanced_x = cat(balanced_x)
                balanced_y = cat(balanced_y)
                test_datasets[i] = TensorDataset(balanced_x, balanced_y)
            # randomly shuffle the new train & test datasets
            # keep the first 12 datasets
            if n_rand > 1:
                if len(train_datasets) > 12:
                    if kwargs.get("shuffle", True):
                        # for datasets from 11 to end, concat into a big dataset, and then split into datasets of the same size as the first 11
                        extra_train_data = []
                        for i in range(12, len(train_datasets)):
                            extra_train_data.append(train_datasets[i][:][0])
                        extra_train_labels = []
                        for i in range(12, len(train_datasets)):
                            extra_train_labels.append(train_datasets[i][:][1])
                        extra_train_data = cat(extra_train_data)
                        extra_train_labels = cat(extra_train_labels)
                        # permute the extra data
                        perm = randperm(len(extra_train_labels))
                        extra_train_data = extra_train_data[perm]
                        extra_train_labels = extra_train_labels[perm]
                        new_extra_datasets = []
                        start_idx = 0
                        for i in range(12, len(train_datasets)):
                            end_idx = start_idx + len(train_datasets[i][:][1])
                            new_extra_datasets.append(
                                TensorDataset(extra_train_data[start_idx:end_idx],
                                              extra_train_labels[start_idx:end_idx]))
                            start_idx = end_idx
                    else:
                        # here dataset are just repeated. We have 12 tasks, and we want tasks 12 to be next to 24, 36, etc and same for test datasets
                        # order should be
                        new_extra_datasets = []
                        for i in range(12):
                            for j in range(12, len(train_datasets), 12):
                                new_extra_datasets.append(
                                    train_datasets[j + i])
                train_datasets = train_datasets[:12] + new_extra_datasets

        dataset_normalisation = kwargs.get("dataset_normalisation", "none")
        if dataset_normalisation != "none":
            # Concatenate all training data to compute normalization stats
            all_train_x = cat([train_datasets[i][:][0]
                               for i in range(len(train_datasets))])
            normalisation_options = {
                "zero_mean": v2.Normalize(mean=(0.0,), std=(1.0,)),
                "min_max": v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
                "standardised": v2.Normalize(mean=all_train_x.mean(dim=(0, 2, 3)), std=all_train_x.std(dim=(0, 2, 3))),
                "binarized-gate": v2.Compose([v2.Normalize(mean=all_train_x.mean(dim=(0, 2, 3)), std=all_train_x.std(dim=(0, 2, 3))), v2.Lambda(lambda x: (abs(x) > 1).float())]),
                "binarized-sign": v2.Compose([v2.Normalize(mean=all_train_x.mean(dim=(0, 2, 3)), std=all_train_x.std(dim=(0, 2, 3))), v2.Lambda(lambda x: (x >= 0).float() * 2 - 1)]),
            }
            norm_fn = normalisation_options.get(
                dataset_normalisation, normalisation_options["standardised"])
            # Apply the same normalization to all training datasets
            for i in range(len(train_datasets)):
                train_x = train_datasets[i][:][0]
                train_x = norm_fn(train_x)
                train_y = train_datasets[i][:][1]
                train_datasets[i] = TensorDataset(train_x, train_y)

            # Apply the same normalization to all test datasets
            for i in range(len(test_datasets)):
                test_x = test_datasets[i][:][0]
                test_x = norm_fn(test_x)
                test_y = test_datasets[i][:][1]
                test_datasets[i] = TensorDataset(test_x, test_y)

        subfeatures = kwargs.get("subfeatures", None)
        if subfeatures is not None:
            # Reduce the number of features by keeping only the first subfeatures features, nb of
            shape = prod(tensor(train_datasets[0][:][0].shape[1:]))
            random_indexes = np.random.choice(
                shape, subfeatures, replace=False)
            for i in range(len(train_datasets)):
                train_x = train_datasets[i][:][0].reshape(
                    len(train_datasets[i][:][0]), -1, 1, 1)
                train_x = train_x[:, random_indexes]
                train_y = train_datasets[i][:][1]
                train_datasets[i] = TensorDataset(train_x, train_y)
            for i in range(len(test_datasets)):
                test_x = test_datasets[i][:][0].reshape(
                    len(test_datasets[i][:][0]), -1, 1, 1)
                test_x = test_x[:, random_indexes]
                test_y = test_datasets[i][:][1]
                test_datasets[i] = TensorDataset(test_x, test_y)

        # from the train and test datasets, only keep the classes in the classes argument

        def filter_classes(datasets, classes, all_classes):
            filtered_datasets = []
            class_indices = [all_classes.index(
                class_name) for class_name in classes]
            for dataset in datasets:
                x, y = dataset[:][0], dataset[:][1]
                mask = zeros(len(y), dtype=bool)
                for class_idx in class_indices:
                    mask |= (y == class_idx)
                filtered_x = x[mask]
                filtered_y = y[mask]
                # remap the labels to be from 0 to len(classes)-1
                remapped_y = zeros_like(filtered_y)
                for new_idx, class_idx in enumerate(class_indices):
                    remapped_y[filtered_y == class_idx] = new_idx
                filtered_datasets.append(
                    TensorDataset(filtered_x, remapped_y))
            return filtered_datasets
        train_datasets = filter_classes(train_datasets, classes, all_classes)
        test_datasets = filter_classes(test_datasets, classes, all_classes)
        return train_datasets, test_datasets


def normalisation(data, dataset_normalisation="standardised"):
    """ Normalize the pixels in train_x and test_x using transform

    Args:
        train_x (np.array): Training data
        test_x (np.array): Testing data

    Returns:
        tensor, tensor: Normalized training and testing data
    """
    # Completely convert train_x and test_x to float torch tensors
    data = from_numpy(data).float() / 255
    if len(data.size()) == 3:
        data = data.unsqueeze(1)
    normalisation_options = {
        "standardised": v2.Normalize(mean=data.mean(dim=(0, 2, 3)),
                                     std=data.std(dim=(0, 2, 3))),
        "zero_mean": v2.Normalize(mean=(0.0,), std=(1.0,)),
        "min_max": v2.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        "none": v2.Lambda(lambda x: x),
        "binarized-gate": v2.Compose([v2.Normalize(mean=data.mean(dim=(0, 2, 3)), std=data.std(dim=(0, 2, 3))), v2.Lambda(lambda x: (abs(x) > 1).float())]),
        "binarized-sign": v2.Compose([v2.Normalize(mean=data.mean(dim=(0, 2, 3)), std=data.std(dim=(0, 2, 3))), v2.Lambda(lambda x: (x >= 0).float() * 2 - 1)]),
    }
    normalisation_fn = normalisation_options.get(
        dataset_normalisation, normalisation_options["standardised"])
    return normalisation_fn(data)


def padding(data, padding_size=0):
    """ Pad the images with zeros on each side

    Args:
        data (np.array): Data to pad
        padding_size (int, optional): Size of the padding. Defaults to 4.

    Returns:
        tensor: Padded data
    """
    if len(data.size()) == 3:
        data = data.unsqueeze(1)
    pad_fn = v2.Pad(padding_size)
    return pad_fn(data)


def get_feature_extractor(feature_extractor_name, device):
    """Get feature extractor model and transforms based on name.

    Args:
    feature_extractor_name (str): Name of the feature extractor
    device (str): Device to load the model on

    Returns:
    tuple: (model, transforms) for the specified feature extractor
    """
    extractors = {
        "efficientnetb0": lambda: (
            Sequential(
                models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.DEFAULT).features,
                models.efficientnet_b0(
                    weights=models.EfficientNet_B0_Weights.DEFAULT).avgpool
            ).to(device),
            models.EfficientNet_B0_Weights.DEFAULT.transforms()
        ),
        "efficientnetb1": lambda: (
            Sequential(
                models.efficientnet_b1(
                    weights=models.EfficientNet_B1_Weights.DEFAULT).features,
                models.efficientnet_b1(
                    weights=models.EfficientNet_B1_Weights.DEFAULT).avgpool
            ).to(device),
            models.EfficientNet_B1_Weights.DEFAULT.transforms()
        ),
        "efficientnetb2": lambda: (
            Sequential(
                models.efficientnet_b2(
                    weights=models.EfficientNet_B2_Weights.DEFAULT).features,
                models.efficientnet_b2(
                    weights=models.EfficientNet_B2_Weights.DEFAULT).avgpool
            ).to(device),
            models.EfficientNet_B2_Weights.DEFAULT.transforms()
        ),
        "vgg19": lambda: (
            Sequential(
                *list(models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).children())[:-1]).to(device),
            models.VGG19_Weights.IMAGENET1K_V1.transforms()
        ),
        "resnet18": lambda: (
            Sequential(
                *list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1]).to(device),
            models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        )
    }

    if feature_extractor_name not in extractors:
        raise ValueError(
            f"Unknown feature extractor: {feature_extractor_name}")

    return extractors[feature_extractor_name]()
