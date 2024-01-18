# When to Switch: Planning and Learning For Partially Observable Multi-Agent Pathfinding

This repository provides the implementation of the "When to Switch" paper, offering various policies and algorithms
designed to address the challenging problem of finding non-conflicting paths for a set of agents in an environment that
is only partially observable to each agent (PO-MAPF).
The repository includes two main policies: one is based on search-based re-planning (**RePlan**), and the other is based
on reinforcement learning (**EPOM**).
Additionally, the repository features three implementations of mixed policies, which switch between **RePlan** and **EPOM**.

## Installation

Install all dependencies using:

```bash
pip install -r docker/requirements.txt
```

## Inference Example


To download pretrained weights, use this [link 137MB](https://github.com/AIRI-Institute/when-to-switch/releases/download/v0/weights.zip)

Execute **EPOM**, **RePlan**, **ASwitcher**, **LSwitcher**, and **HSwitcher** to generate animations using pre-trained
weights with the following command:

```bash
python example.py
```


The animations will be stored in the ```renders``` folder.

## Training EPOM

To train **EPOM**, execute ```train_epom.py``` with the ```learning/train.yaml``` config file:

```bash
python train_epom.py --config_path="learning/train.yaml"
```

## Training LSwitcher

To train **LSwitcher** estimator for the **RePlan** or **EPOM** algorithm, use the commands below:

```bash
python train_lswitcher.py --algo="RePlan"
```

```bash
python train_lswitcher.py --algo="EPOM"
```

## Citation

If you use this repository in your research or wish to reference it, please cite our TNNLS paper:

```bibtex
@article{skrynnik2023switch,
    title = {When to Switch: Planning and Learning for Partially Observable Multi-Agent Pathfinding},
    author = {Skrynnik, Alexey and Andreychuk, Anton and Yakovlev, Konstantin and Panov, Aleksandr I},
    journal = {IEEE Transactions on Neural Networks and Learning Systems},
    year = {2023},
    publisher = {IEEE}
}
```
