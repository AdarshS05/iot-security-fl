#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python3"

echo "=== Starting Real Flower Network Communication Demo ==="

"${PYTHON_BIN}" "${SCRIPT_DIR}/demo_server.py" &
SERVER_PID=$!

sleep 2

"${PYTHON_BIN}" "${SCRIPT_DIR}/demo_client.py" --client_id 1 &
CLIENT1_PID=$!

"${PYTHON_BIN}" "${SCRIPT_DIR}/demo_client.py" --client_id 2 &
CLIENT2_PID=$!

"${PYTHON_BIN}" "${SCRIPT_DIR}/demo_client.py" --client_id 3 &
CLIENT3_PID=$!

wait ${SERVER_PID} ${CLIENT1_PID} ${CLIENT2_PID} ${CLIENT3_PID}

echo ""
echo "Demo complete — see logs above for message transfer between processes"
