import sys
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import flwr as fl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.metrics_logger import log_global_metrics

logging.getLogger("flwr").setLevel(logging.WARNING)


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy that logs weight transfer details, global aggregation stats, and metrics."""

    def aggregate_fit(self, server_round, results, failures):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n==================== [SERVER] Round {server_round} Started ====================", flush=True)
        print(f"[{timestamp}] [SERVER] Received updates from {len(results)} clients:", flush=True)

        for client_proxy, fit_res in results:
            payload_bytes = sum(len(b) for b in fit_res.parameters.tensors)
            cid = client_proxy.cid
            print(
                f"  --> [{timestamp}] [SERVER] Received update from Client '{cid}': {payload_bytes} bytes payload",
                flush=True,
            )

        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            flattened = np.concatenate([arr.ravel() for arr in ndarrays])
            mean_val = float(np.mean(flattened))
            std_val = float(np.std(flattened))

            total_samples = sum(fit_res.num_examples for _, fit_res in results)
            if total_samples > 0:
                global_acc = sum(fit_res.metrics.get("accuracy", 0.0) * fit_res.num_examples for _, fit_res in results) / total_samples
                global_prec = sum(fit_res.metrics.get("precision", 0.0) * fit_res.num_examples for _, fit_res in results) / total_samples
                global_rec = sum(fit_res.metrics.get("recall", 0.0) * fit_res.num_examples for _, fit_res in results) / total_samples
                global_f1 = sum(fit_res.metrics.get("f1_score", 0.0) * fit_res.num_examples for _, fit_res in results) / total_samples
                global_loss = sum(fit_res.metrics.get("loss", 0.0) * fit_res.num_examples for _, fit_res in results) / total_samples
                global_auc = sum(fit_res.metrics.get("auc_roc", 0.0) * fit_res.num_examples for _, fit_res in results) / total_samples
            else:
                global_acc = global_prec = global_rec = global_f1 = global_loss = global_auc = 0.0

            num_clients = len(results)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(
                f"[{timestamp}] [SERVER] Round {server_round} Aggregated Global Weights -> Mean: {mean_val:.6f}, Std: {std_val:.6f}",
                flush=True,
            )
            print(
                f"[{timestamp}] [SERVER] Round {server_round} Global Metrics -> "
                f"Accuracy: {global_acc:.4f}, Precision: {global_prec:.4f}, Recall: {global_rec:.4f}, "
                f"F1: {global_f1:.4f}, Loss: {global_loss:.4f}, AUC: {global_auc:.4f}",
                flush=True,
            )
            print(f"==================== [SERVER] Round {server_round} Completed ====================\n", flush=True)

            log_global_metrics(
                {
                    "round_number": server_round,
                    "accuracy": global_acc,
                    "precision": global_prec,
                    "recall": global_rec,
                    "f1_score": global_f1,
                    "loss": global_loss,
                    "auc_roc": global_auc,
                    "num_clients": num_clients,
                    "mean_weight": mean_val,
                    "std_weight": std_val,
                },
                csv_path="docs/metrics_log.csv",
            )

        return aggregated_parameters, metrics


def main():
    print("=== Starting Real Flower Network Server on localhost:8080 ===", flush=True)

    rng = np.random.default_rng(42)
    initial_weights = [
        rng.normal(0, 0.5, (10, 5)),
        rng.normal(0, 0.5, (5, 2)),
    ]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)

    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=initial_parameters,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )

    config = fl.server.ServerConfig(num_rounds=5)

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=config,
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
