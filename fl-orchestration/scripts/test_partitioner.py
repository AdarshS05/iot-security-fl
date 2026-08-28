import numpy as np
import pytest

from common.partitioner import generate_dummy_labels, partition_dirichlet


def test_all_samples_assigned_exactly_once():
    """Confirm every sample index is assigned to exactly one client (no duplicates, none missing)."""
    num_samples = 500
    num_clients = 5
    labels = generate_dummy_labels(num_samples=num_samples, num_classes=2, seed=42)

    partitions = partition_dirichlet(
        labels=labels,
        num_clients=num_clients,
        alpha=0.5,
        seed=42,
    )

    all_assigned_indices = []
    for client_id, indices in partitions.items():
        all_assigned_indices.extend(indices)

    assert len(all_assigned_indices) == num_samples
    assert set(all_assigned_indices) == set(range(num_samples))


def test_alpha_skewness_comparison():
    """
    Confirm low dirichlet_alpha (e.g. 0.05) produces higher variance in per-client
    class ratios (more skewed) than high dirichlet_alpha (e.g. 100).
    """
    num_samples = 2000
    labels = generate_dummy_labels(num_samples=num_samples, num_classes=2, seed=123)

    partition_skewed = partition_dirichlet(labels=labels, num_clients=10, alpha=0.05, seed=123)
    partition_uniform = partition_dirichlet(labels=labels, num_clients=10, alpha=100.0, seed=123)

    def compute_class_ratio_variance(partitions):
        ratios = []
        for indices in partitions.values():
            if len(indices) > 0:
                client_labels = labels[indices]
                ratio_class_1 = np.mean(client_labels == 1)
                ratios.append(ratio_class_1)
        return np.var(ratios)

    var_skewed = compute_class_ratio_variance(partition_skewed)
    var_uniform = compute_class_ratio_variance(partition_uniform)

    assert var_skewed > var_uniform


def test_random_seed_reproducibility():
    """Confirm the same random_seed always produces identical partitions."""
    num_samples = 400
    labels = generate_dummy_labels(num_samples=num_samples, num_classes=2, seed=99)

    part1 = partition_dirichlet(labels=labels, num_clients=4, alpha=0.5, seed=99)
    part2 = partition_dirichlet(labels=labels, num_clients=4, alpha=0.5, seed=99)

    assert part1 == part2
