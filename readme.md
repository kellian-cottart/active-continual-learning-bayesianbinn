# Active Continual Learning with Metaplastic Binary Bayesian Neural Networks

**ICML 2026** (poster, Seoul) · [OpenReview](https://openreview.net/forum?id=SPZd0HVyiS)

Kellian Cottart, Théo Ballet, Djohan Bonnet, Damien Querlioz — Université Paris-Saclay, CNRS, C2N.

This repository is the code behind the paper. It trains and evaluates **BiMU** (Binary Metaplasticity from
Synaptic Uncertainty), a Bayesian continual-learning rule for networks whose weights are single bits, and the
**active continual learning** setting in which the network decides from its own predictive uncertainty which
samples are worth labelling.

## The idea in one paragraph

A binary weight is a Bernoulli variable. Training a Bayesian binary network means updating the probability that
each weight is +1, and that probability doubles as a measure of how sure the network is about that weight. BiMU
turns this into a learning rule: the further a weight is from certain, the more it is allowed to move, so the
network keeps learning where it is uncertain and protects what it already knows without task boundaries or
replay. It is the binary counterpart of MESU ([Bonnet, Cottart et al., Nature Communications 2025](https://github.com/kellian-cottart/mesu-pmnist)),
derived here for discrete weights so that latent parameters never saturate. The same uncertainty, read at the
output through the variation ratio, drives the active-learning trigger: only samples the network is unsure about
are queried, under a labelling budget (see [activelearning.md](activelearning.md) for the dynamic threshold).

## What is in the repository

| Component | Where | Notes |
| --- | --- | --- |
| Training and evaluation | `main.py` | One entry point, driven by JSON configurations |
| Optimizers | `optimizers/` | `bimu.py` (ours), `bayesbinn.py`, `bayesbinn_al.py`, `mesu.py`, `bgd.py`, `synapticMetaplasticity.py`, `adam.py` (STE), `sgd.py`; EWC and SI are regularisers in the training loop, selected by configuration |
| Models | `models/` | Binary Bayesian MLP and CNN |
| Layers and activations | `customLayers/` | Bayesian binary linear and convolution layers, binary activations (`reversebinarygate`, ...) |
| Datasets | `utils/gpuLoading.py` | Permuted MNIST, Animals, OpenLORIS; downloaded on first use |
| Active learning | `utils/activeLearningFunctions.py`, `utils/uncertaintyFunctions.py` | Variation ratio, budget-controlled threshold |
| Hyperparameter search | `hyperparameters.py`, `hpo-configurations/` | Optuna |
| Reproduction scripts | `scripts/` | One script per table or figure |
| Notebooks | `main-*.ipynb`, `appendix-*.ipynb` | Turn the exported results into the paper's figures and tables |
| Microcontroller-style inference | `appendix-cpp-code-benchmark/` | Trained Bayesian binary MLP re-implemented in plain C++, validated against JAX golden vectors, timed with `make run` |

The code is JAX / Equinox. PyTorch is used for data loading and seeding only.

## Setup

```bash
conda env create -f environment.yml
conda activate binarized
```

The environment pins Python 3.12, `torch==2.9.1+cu128`, `jax[cuda12]`, `equinox`, `optax`. A GPU is strongly
recommended: the Permuted MNIST table alone trains 1000 sequential tasks per run.

## Running an experiment

```bash
python main.py --config main-pmnist-1000tasks-100neurons/bimu --n_iterations 5 --ood fashion --gpu 0 --verbose
```

`--config` names a file under `configurations/` without the `.json` extension. Each configuration fixes the
network, the optimizer and its hyperparameters, the task, the number of tasks and epochs, and the number of
Monte-Carlo samples used at train and test time. Results are written under `results/`.

| Argument | Meaning |
| --- | --- |
| `-c, --config` | configuration file, relative to `configurations/`, without `.json` |
| `-it, --n_iterations` | number of independent runs (seeds) |
| `-ood, --ood` | dataset for out-of-distribution detection: `fashion`, `pmnist`, or none |
| `-gpu, --gpu` | GPU id |
| `-v, --verbose` | progress bar and intermediate metrics |
| `-train, --train_accuracy` | also report training accuracy (needs `-v`) |
| `-fits, --fits_in_memory` | keep the dataset on the GPU |
| `-wh, --weight_histogram` | save weight histograms during training |
| `-euf, --extract_uncertainties_full` | epistemic-uncertainty histograms on the train set at every epoch |
| `-eln, --extract_layer_norm` | export layer-norm outputs |
| `-pca, --per_class_acc` | per-class accuracy |

## Reproducing the paper

Each script below launches the `main.py` runs for one table or figure and stores the results. The matching
notebook (`main-*` for the paper, `appendix-*` for the appendix) then produces the plot or table.

### Main results

| Result | Script | Notebook |
| --- | --- | --- |
| Permuted MNIST, 1000 tasks | `bash scripts/main-pmnist-table.sh` | `main-permuted-1000tasks-100neurons.ipynb` |
| Animals, active learning | `bash scripts/main-animals-al.sh` | `main-animals-al-comparison.ipynb` |
| OpenLORIS, active learning | `bash scripts/main-openloris-al.sh` | `main-openloris-al-comparison.ipynb` |
| OpenLORIS, table | `bash scripts/main-openloris-table.sh` | `main-openloris.ipynb` |

### Appendix

| Result | Script | Notebook |
| --- | --- | --- |
| Permuted MNIST, memory window N | `bash scripts/appendix-pmnist-table-N.sh` | `appendix-permuted-N.ipynb` |
| Permuted MNIST, activation functions | `bash scripts/appendix-pmnist-table-activation.sh` | `appendix-permuted-activation.ipynb`, `appendix-permuted-activation-1task.ipynb` |
| Permuted MNIST, model size | `bash scripts/appendix-pmnist-table-size.sh` | `appendix-permuted-1000tasks-2000neurons.ipynb` |
| OpenLORIS, standardized evaluation | `bash scripts/appendix-openloris-standardized-table.sh` | `appendix-openloris-standardized.ipynb` |
| OpenLORIS, variation-ratio samples | `bash scripts/appendix-openloris-al-variation-ratio.sh` | `appendix-openloris-al-variation-ratio.ipynb` |
| OpenLORIS, dynamic threshold and budget | `bash scripts/appendix-openloris-al-variation-threshold.sh` | `appendix-openloris-al-variation-budget.ipynb`, `appendix-openloris-al-switch-time.ipynb`, `appendix-openloris-al-backwards.ipynb` |
| Wall-clock time | (from the main runs) | `appendix-wall-clock-time-bimu.ipynb` |
| C++ inference benchmark | `cd appendix-cpp-code-benchmark && make run` | — |

Interrupting a run cleans up its partial results.

## Citation

```bibtex
@inproceedings{cottart2026active,
  title     = {Active Continual Learning with Metaplastic Binary Bayesian Neural Networks},
  author    = {Cottart, Kellian and Ballet, Th{\'e}o and Bonnet, Djohan and Querlioz, Damien},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=SPZd0HVyiS}
}
```

The continuous-weight rule this work specialises is
[Bayesian continual learning and forgetting in neural networks](https://doi.org/10.1038/s41467-025-64601-w),
Nature Communications 16, 9614 (2025).

## License

CC-BY 4.0, see [LICENSE](LICENSE). Portions of the data-loading code are adapted from PyTorch (BSD-3-Clause).
