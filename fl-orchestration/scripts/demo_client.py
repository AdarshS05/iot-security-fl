import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import flwr as fl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.partitioner import generate_dummy_labels, partition_dirichlet
from common.metrics import compute_metrics

logging.getLogger("flwr").setLevel(logging.WARNING)


def forward_pass(X, weights):
    """Simple 2-layer feed-forward forward pass (ReLU hidden activation, softmax output)."""
    w1, w2 = weights[0], weights[1]
    h = np.maximum(0, np.dot(X, w1))
    logits = np.dot(h, w2)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs


class DemoClient(fl.client.NumPyClient):
    """Flower NumPyClient connecting over real network socket to demo_server.py."""

    def __init__(
        self,
        client_id: int,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        X: np.ndarray,
        labels: np.ndarray,
    ):
        self.client_id = client_id
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.X = X
        self.labels = labels
        self.num_samples = len(train_indices) + len(val_indices)

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        server_round = config.get("server_round", 1)
        timestamp_rx = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"[{timestamp_rx}] [CLIENT {self.client_id}] Received updated global model from server, round {server_round}",
            flush=True,
        )

        X_tr = self.X[self.train_indices]
        y_tr = self.labels[self.train_indices]

        w1, w2 = parameters[0].copy(), parameters[1].copy()
        lr = 0.02
        epochs = 5

        for _ in range(epochs):
            h = np.maximum(0, np.dot(X_tr, w1))
            exp_logits = np.exp(np.dot(h, w2) - np.max(np.dot(h, w2), axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            y_onehot = np.zeros((len(y_tr), 2))
            y_onehot[np.arange(len(y_tr)), y_tr] = 1.0

            grad_logits = (probs - y_onehot) / len(y_tr)
            grad_w2 = np.dot(h.T, grad_logits)
            grad_h = np.dot(grad_logits, w2.T) * (h > 0)
            grad_w1 = np.dot(X_tr.T, grad_h)

            w1 -= lr * grad_w1
            w2 -= lr * grad_w2

        updated_weights = [w1, w2]
        payload_bytes = sum(w.nbytes for w in updated_weights)

        X_val = self.X[self.val_indices]
        y_val = self.labels[self.val_indices]

        val_probs = forward_pass(X_val, updated_weights)
        val_proba = val_probs[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)

        metrics = compute_metrics(y_val, val_pred, val_proba)

        timestamp_tx = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp_tx}] [CLIENT {self.client_id}] Round {server_round} Validation Metrics -> "
            f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, Loss: {metrics['loss']:.4f}",
            flush=True,
        )
        print(
            f"[{timestamp_tx}] [CLIENT {self.client_id}] Sending update to server: {payload_bytes} bytes, round {server_round}",
            flush=True,
        )

        return updated_weights, self.num_samples, metrics

    def evaluate(self, parameters, config):
        X_val = self.X[self.val_indices]
        y_val = self.labels[self.val_indices]
        val_probs = forward_pass(X_val, parameters)
        val_proba = val_probs[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)
        metrics = compute_metrics(y_val, val_pred, val_proba)
        return float(metrics["loss"]), len(self.val_indices), metrics


def main():
    parser = argparse.ArgumentParser(description="Flower Network Demo Client")
    parser.add_argument("--client_id", type=int, default=1, help="Client identifier (1, 2, 3...)")
    args = parser.parse_args()

    labels = generate_dummy_labels(num_samples=1000, num_classes=2, seed=42)
    partitions = partition_dirichlet(labels=labels, num_clients=10, alpha=0.5, seed=42)

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1.0, (1000, 10))
    for i in range(1000):
        if labels[i] == 1:
            X[i, :5] += 0.8
        else:
            X[i, :5] -= 0.8

    client_key = f"client_{args.client_id - 1}"
    indices = np.array(partitions.get(client_key, partitions["client_0"]))

    split_idx = int(len(indices) * 0.8)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] [CLIENT {args.client_id}] Connecting to server at 127.0.0.1:8080 "
        f"(Train samples: {len(train_indices)}, Val samples: {len(val_indices)})",
        flush=True,
    )

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=DemoClient(
            client_id=args.client_id,
            train_indices=train_indices,
            val_indices=val_indices,
            X=X,
            labels=labels,
        ).to_client(),
    )


if __name__ == "__main__":
    main()
