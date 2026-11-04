"""Utilities intentionally provided to students.

The project focuses on the PyTorch model/training pipeline. Dataset engineering,
metric boilerplate, and reproducibility helpers are provided so students can
concentrate on Dataset/DataLoader usage, nn.Module, forward passes, losses,
backpropagation, optimization, validation, and model selection.
"""
from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
import torchmetrics
import yaml
import collections
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

@dataclass
class Result:
    model_type: type[nn.Module]
    model_args: dict=field(default_factory=dict)
    history: list[dict]=field(default_factory=list)
    best_epoch: int=-1
    best_model_state: collections.OrderedDict=None

@dataclass
class Stats:
    mean: dict[np.ndarray]=field(default_factory=dict)
    std: dict[np.ndarray]=field(default_factory=dict)

def TODO(dataset,
         batch_size: int=1,
         shuffle: bool=False,
         num_workers: int=0,
         collate_fn: callable=None,
         pin_memory: bool=False):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        items = ", ".join(f"{k}={v!r}" for k, v in sorted(self.__dict__.items()))
        return f"Struct({items})"


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return Struct(**(yaml.safe_load(f) or {}))


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def infer_feature_dim(dataset) -> int:
    if len(dataset) == 0:
        raise ValueError("Cannot infer feature dimension from an empty dataset.")
    if hasattr(dataset, "feature_dim"):
        return int(dataset.feature_dim)
    return int(dataset[0]["input"].shape[-1])


def classification_metrics(probabilities: torch.Tensor,
                           labels: torch.Tensor,
                           n_class: int):
    """Return macro AUROC, accuracy (%), and macro F1.

    Metric implementation is provided because writing AUROC/F1 from scratch is
    not a learning objective of this project.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, C]")
    labels = labels.long()
    device = probabilities.device
    auroc_metric = torchmetrics.AUROC(
        num_classes=int(n_class), task="multiclass", average="macro"
    ).to(device)
    f1_metric = torchmetrics.F1Score(
        num_classes=int(n_class), task="multiclass", average="macro"
    ).to(device)
    auc = float(auroc_metric(probabilities, labels).item())
    f1 = float(f1_metric(probabilities, labels).item())
    acc = float((probabilities.argmax(dim=1) == labels).float().mean().item() * 100.0)
    return {"auc": auc, "acc": acc, "f1": f1}

def compute_stats(results: dict[Result]):
    # extract metrics (assume epochs increment by 1 and are the same across runs)
    metrics = {'train_loss': [], 'val_loss': []}
    for _, result in results.items():
        for key in metrics.keys():
            metrics[key].append([row[key] for row in result.history])
    # compute stats
    stats = Stats()
    for key in metrics.keys():
        stats.mean[key] = np.array(metrics[key]).mean(axis=0)
        stats.std[key] = np.array(metrics[key]).std(axis=0)
    return stats

def visualise_stats(results: dict[Result]):
    stats = compute_stats(results)
    num_epochs = next(iter(stats.mean.values())).size
    epochs = np.arange(1,num_epochs+1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    for key, mean in stats.mean.items():
        ax.plot(epochs, mean, label=metric_label(key))
    for key, std in stats.std.items():
        mean = stats.mean[key]
        ax.fill_between(
            epochs,
            mean - std,
            mean + std,
            alpha=0.3,
            label=None,
        )
    ax.legend()
    ax.grid(alpha=0.2)
    plt.show()

def metric_label(key: str) -> str:
    match key:
        case 'train_loss':
            return 'train loss'
        case 'val_loss':
            return 'validation loss'
        case _:
            return ''