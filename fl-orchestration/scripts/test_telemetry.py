import json
import pytest
import jsonschema

from common.telemetry import (
    serialize_update,
    deserialize_update,
    SchemaVersionError,
    SUPPORTED_SCHEMA_VERSION,
)


def test_round_trip_serialization():
    """Test serializing and deserializing a sample weight update for round-trip consistency."""
    client_id = "client_node_01"
    round_number = 3
    model_type = "tabular"
    weights = [0.25, -0.75, 1.5, [0.1, 0.2, 0.3]]
    num_samples = 120

    serialized_str = serialize_update(
        client_id=client_id,
        round_number=round_number,
        model_type=model_type,
        weights=weights,
        num_samples=num_samples,
    )

    assert isinstance(serialized_str, str)

    deserialized_data = deserialize_update(serialized_str)

    assert deserialized_data["client_id"] == client_id
    assert deserialized_data["round_number"] == round_number
    assert deserialized_data["model_type"] == model_type
    assert deserialized_data["weights"] == weights
    assert deserialized_data["num_samples"] == num_samples
    assert deserialized_data["schema_version"] == SUPPORTED_SCHEMA_VERSION
    assert "timestamp" in deserialized_data


def test_missing_required_field_fails_validation():
    """Test that a payload missing a required field (e.g. client_id) fails schema validation."""
    invalid_payload = {
        "round_number": 1,
        "schema_version": "1.0",
        "model_type": "assembly",
        "weights": [0.5, 0.5],
        "num_samples": 50,
        "timestamp": "2026-08-01T00:00:00+00:00",
    }

    json_str = json.dumps(invalid_payload)

    with pytest.raises(jsonschema.ValidationError):
        deserialize_update(json_str)


def test_unsupported_schema_version_flagged():
    """Test that an unsupported schema_version is detected and raises SchemaVersionError."""
    payload_future_version = {
        "client_id": "client_node_02",
        "round_number": 5,
        "schema_version": "99.0",
        "model_type": "tabular",
        "weights": [1.0, 2.0],
        "num_samples": 200,
        "timestamp": "2026-08-01T00:00:00+00:00",
    }

    json_str = json.dumps(payload_future_version)

    with pytest.raises(SchemaVersionError) as exc_info:
        deserialize_update(json_str)

    assert "Unsupported schema_version '99.0'" in str(exc_info.value)
