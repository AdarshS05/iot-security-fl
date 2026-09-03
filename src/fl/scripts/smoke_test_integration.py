import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.partitioner import generate_dummy_labels, partition_dirichlet
from common.telemetry import serialize_update, deserialize_update, SUPPORTED_SCHEMA_VERSION



def main():
    print("=== Starting Telemetry & Partitioner Smoke Integration Test ===")

    num_samples = 1000
    num_clients = 10
    num_classes = 2
    alpha = 0.5
    seed = 42

    labels = generate_dummy_labels(num_samples=num_samples, num_classes=num_classes, seed=seed)
    client_partitions = partition_dirichlet(labels=labels, num_clients=num_clients, alpha=alpha, seed=seed)

    print(f"Generated Dirichlet partition for {num_clients} clients across {num_samples} total samples.")

    target_client = "client_0"
    client_indices = client_partitions[target_client]
    client_num_samples = len(client_indices)

    mock_weights = [0.125, -0.45, [0.01, 0.02, 0.03], 1.2]
    model_type = "tabular"
    round_number = 1

    print(f"Selected '{target_client}' with {client_num_samples} local samples.")
    print("Serializing client weight update message...")

    serialized_payload = serialize_update(
        client_id=target_client,
        round_number=round_number,
        model_type=model_type,
        weights=mock_weights,
        num_samples=client_num_samples,
    )

    print(f"Serialized Payload (JSON preview): {serialized_payload[:120]}...")

    print("Deserializing payload and validating against telemetry schema...")
    deserialized_data = deserialize_update(serialized_payload)

    assert deserialized_data["client_id"] == target_client
    assert deserialized_data["round_number"] == round_number
    assert deserialized_data["model_type"] == model_type
    assert deserialized_data["weights"] == mock_weights
    assert deserialized_data["num_samples"] == client_num_samples
    assert deserialized_data["schema_version"] == SUPPORTED_SCHEMA_VERSION

    print("\nSUCCESS: All data fields matched perfectly after round-trip serialization!")
    print(f"- Client ID: {deserialized_data['client_id']}")
    print(f"- Round Number: {deserialized_data['round_number']}")
    print(f"- Model Type: {deserialized_data['model_type']}")
    print(f"- Sample Count: {deserialized_data['num_samples']}")
    print(f"- Timestamp: {deserialized_data['timestamp']}")
    print(f"- Schema Version: {deserialized_data['schema_version']}")


if __name__ == "__main__":
    main()
