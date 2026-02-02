# Active Continual Learning with Binarized Bayesian Neural Networks

This repository contains the code used for the experiments in **Active Continual Learning with Binarized Bayesian Neural Networks**. It provides a unified framework to train, evaluate, and analyze Bayesian binarized models under continual learning settings, with a strong focus on uncertainty estimation and out-of-distribution (OOD) detection.

Main contributors:

- [Kellian COTTART](https://scholar.google.com/citations?hl=en&user=Akg-AH4AAAAJ)
- [Théo Ballet](https://scholar.google.com/citations?user=WWfZoQcAAAAJ&hl=en)
- [Djohan BONNET](https://scholar.google.com/citations?user=1cSwOPIAAAAJ&hl=en)
  
Research director:
- [Damien QUERLIOZ](https://scholar.google.com/citations?user=2-EKdW4AAAAJ&hl=en)

- [Active Continual Learning with Binarized Bayesian Neural Networks](#active-continual-learning-with-binarized-bayesian-neural-networks)
  - [Environment Setup](#environment-setup)
  - [Project Structure](#project-structure)
  - [Main Training File (`main.py`)](#main-training-file-mainpy)
    - [Command-line Arguments](#command-line-arguments)
    - [Example Usage](#example-usage)
  - [Reproducing Figures and Tables](#reproducing-figures-and-tables)
    - [Main Paper Results](#main-paper-results)
    - [Appendix Experiments](#appendix-experiments)
  - [Notes](#notes)
  - [Citation](#citation)
  - [License](#license)

## Environment Setup

The repository uses a Conda environment (environment.yml). Key dependencies:

- Conda channels: nvidia, defaults
- Conda packages:
    - python=3.12
    - matplotlib
    - pip

- Pip packages (installed via pip: section in environment.yml):
    - --extra-index-url https://download.pytorch.org/whl/cu128
    - torch==2.9.1+cu128
    - torchvision
    - idx2numpy
    - tqdm
    - jax[cuda12]
    - jaxlib
    - equinox
    - optax
    - seaborn
    - gdown
    - datasets

You can recreate the environment with:
```bash
conda env create -f environment.yml
conda activate binarized
```

## Project Structure

The project is organized around a single **main file**, a set of **configurations**, and a collection of **scripts** used to reproduce all figures and appendix results.

```text
active-continual-learning-bayesianbinn/
│
├── configurations/          # JSON configuration files (models, datasets, optimizers)
├── customLayers/            # Custom neural network layers, activations
├── datasets/                # Dataset loaders and utilities
├── figures-*/               # Output figures (main paper & appendix)
├── models/                  # Model definitions
├── optimizers/              # Optimizers and regularization methods (EWC, SI, etc.)
├── results-*/               # Serialized experiment results
├── scripts/                 # Bash scripts to reproduce figures and tables
├── utils/                   # Utility functions
│
├── main.py                  # Main training & evaluation entry point
├── environment.yml          # Conda environment specification
└── README.md                # This file
```

## Main Training File (`main.py`)

The core of the project is the `main.py` file.

* Loading configurations
* Initializing datasets, models, and optimizers
* Running the continual learning training loop
* Exporting accuracies and uncertainty metrics

### Command-line Arguments

```text
-c, --config
    Configuration file name (without .json)

-it, --n_iterations
    Number of times to run the configuration

-v, --verbose
    Display progress bar and intermediate metrics

-ood, --ood
    Dataset for OOD detection: {fashion, pmnist, None}

-gpu, --gpu
    GPU ID to use

-wh, --weight_histogram
    Save weight histograms during training

-eln, --extract_layer_norm
    Extract layer normalization outputs

-fits, --fits_in_memory
    Whether the dataset fits in memory

-train, --train_accuracy
    Display training accuracy (requires --verbose)

-euf, --extract_uncertainties_full
    Extract epistemic uncertainty histograms per epoch

-pca, --per_class_acc
    Compute per-class accuracy
```

### Example Usage

```bash
python main.py \
    --main-pmnist-1000tasks-100neurons \
    --n_iterations 5 \
    --ood fashion \
    --gpu 0 \
    --verbose
```

---

## Reproducing Figures and Tables

All figures and tables from the paper and appendix can be reproduced using the scripts in the `scripts/` folder.

### Main Paper Results

* **PMNIST results**

  ```bash
  bash scripts/main-pmnist-table.sh
  ```

* **Animals dataset**

  ```bash
  bash scripts/main-animals-al.sh
  ```

* **OpenLORIS dataset**

  ```bash
  bash scripts/main-openloris-al.sh
  bash scripts/main-openloris-table.sh
  ```

### Appendix Experiments

* **PMNIST – Memory window N**

  ```bash
  bash scripts/appendix-pmnist-table-N.sh
  ```

* **PMNIST – Activation functions**

  ```bash
  bash scripts/appendix-pmnist-table-activation.sh
  ```

* **PMNIST – Model size**

  ```bash
  bash scripts/appendix-pmnist-table-size.sh
  ```

* **OpenLORIS – Standardized evaluation**

  ```bash
  bash scripts/appendix-openloris-standardized-table.sh
  ```

* **OpenLORIS – Variation ratio samples**

  ```bash
  bash scripts/appendix-openloris-al-variation-ratio.sh
  ```

Each script launches a sequence of `main.py` runs with the appropriate configuration files and automatically stores the results.

## Notes

* Experiments can be computationally expensive; GPU usage is recommended.
* Interrupting training safely cleans up partial results.


## Citation

Please reference this work as

```bibtex
```

## License

This project is licensed under the CC-BY 4.0 License - see the [LICENSE](LICENSE) file for details.
