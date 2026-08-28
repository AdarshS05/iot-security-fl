"""
Telemetry and serialization layer for the federated learning system.

NOTE: This implementation currently uses JSON format for weight updates.
Protobuf (or FlatBuffers / msgpack) can be swapped in here later if needed
for performance or reduced network overhead without breaking caller code.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union
import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "config" / "telemetry_schema.json"

SUPPORTED_SCHEMA_VERSION = "1.0"


def _load_schema() -> Dict[str, Any]:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


_SCHEMA_CACHE = None


def get_schema() -> Dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = _load_schema()
    return _SCHEMA_CACHE


class SchemaVersionError(ValueError):
    """Raised when an unsupported schema version is encountered."""
    pass


def serialize_update(
    client_id: str,
    round_number: int,
    model_type: str,
    weights: Union[List[Any], Any],
    num_samples: int,
    schema_version: str = SUPPORTED_SCHEMA_VERSION,
) -> str:
    """
    Serialize client weight update into a JSON string matching telemetry_schema.json.

    Auto-fills schema_version and timestamp if not explicitly provided.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    elif isinstance(weights, list):
        def _convert(item):
            if hasattr(item, "tolist"):
                return item.tolist()
            if isinstance(item, list):
                return [_convert(x) for x in item]
            return item
        weights = [_convert(x) for x in weights]

    update_dict = {
        "client_id": client_id,
        "round_number": round_number,
        "schema_version": schema_version,
        "model_type": model_type,
        "weights": weights,
        "num_samples": num_samples,
        "timestamp": timestamp,
    }

    schema = get_schema()
    jsonschema.validate(instance=update_dict, schema=schema)

    return json.dumps(update_dict)


def deserialize_update(json_string: str) -> Dict[str, Any]:
    """
    Parse a JSON string representing a weight update and validate it against telemetry_schema.json.
    Also validates that the schema_version is supported.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}") from e

    schema = get_schema()
    jsonschema.validate(instance=data, schema=schema)

    version = data.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Unsupported schema_version '{version}'. Supported version is '{SUPPORTED_SCHEMA_VERSION}'."
        )

    return data
