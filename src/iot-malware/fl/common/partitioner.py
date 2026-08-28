"""
Non-IID Data Partitioner for Federated Learning.

This module provides data partitioning algorithms (specifically Dirichlet distribution-based)
to simulate non-IID client dataset splits.

NOTE: This partitioner works directly on class labels and sample indices.
It is domain-agnostic and model-agnostic. The exact same functions here will later be
applied to Member 1's tabular dataset and Member 2's assembly dataset once they are
ready, without requiring any modifications to this code.
"""

from typing import Dict, List, Union
import numpy as np


def generate_dummy_labels(
    num_samples: int = 1000,
    num_classes: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic array of class labels (e.g. 0 for benign, 1 for malware)
    to simulate dataset labels for federated partitioning prior to real data integration.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=num_samples)
    return labels


def partition_dirichlet(
    labels: Union[List[int], np.ndarray],
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Partition dataset sample indices across clients using a Dirichlet distribution.

    Args:
        labels: 1D array/list of class labels for the dataset.
        num_clients: Number of federated clients (e.g. 10).
        alpha: Dirichlet distribution concentration parameter.
               Lower alpha (e.g. 0.05-0.5) creates high non-IID class skew across clients.
               Higher alpha (e.g. 10-100) creates nearly uniform / IID partitions.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping client_id (e.g. 'client_0') to list of sample indices assigned to that client.
    """
    labels_arr = np.array(labels)
    num_samples = len(labels_arr)
    num_classes = len(np.unique(labels_arr))
    rng = np.random.default_rng(seed)

    client_indices: Dict[str, List[int]] = {f"client_{i}": [] for i in range(num_clients)}

    for c in range(num_classes):
        idx_c = np.where(labels_arr == c)[0]
        rng.shuffle(idx_c)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        counts = (proportions * len(idx_c)).astype(int)
        remainder = len(idx_c) - counts.sum()

        if remainder > 0:
            fractional_parts = (proportions * len(idx_c)) - counts
            largest_indices = np.argsort(fractional_parts)[::-1]
            for r in range(remainder):
                counts[largest_indices[r]] += 1

        start = 0
        for i in range(num_clients):
            end = start + counts[i]
            client_indices[f"client_{i}"].extend(idx_c[start:end].tolist())
            start = end

    for i in range(num_clients):
        client_key = f"client_{i}"
        client_indices[client_key] = rng.permutation(client_indices[client_key]).tolist()

    return client_indices
