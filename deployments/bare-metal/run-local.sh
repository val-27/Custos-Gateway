#!/usr/bin/env bash
set -euo pipefail

# Bare-metal local launcher for Custos (Topology B / Topology C).
# Usage:
#   ./deployments/bare-metal/run-local.sh [custos|baseline]

MODE="${1:-custos}"
SOCKET_DIR="${CUSTOS_SOCKET_DIR:-/var/run/custos}"
SOCKET_PATH="${SOCKET_DIR}/ext_proc.sock"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${SOCKET_DIR}"

cleanup() {
  if [[ -n "${CUSTOS_PID:-}" ]] && kill -0 "${CUSTOS_PID}" 2>/dev/null; then
    echo "Stopping custos-proc (PID ${CUSTOS_PID})..."
    kill "${CUSTOS_PID}" || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${MODE}" == "custos" ]]; then
  echo "Building custos-proc in release mode..."
  cargo build --release --manifest-path "${REPO_ROOT}/Cargo.toml" -p custos-proc

  echo "Starting custos-proc on ${SOCKET_PATH}..."
  "${REPO_ROOT}/target/release/custos-proc" \
    --socket-path "${SOCKET_PATH}" \
    --max-nesting-depth 16 \
    --max-tensor-elements 16777216 \
    --max-message-size-mb 32 &
  CUSTOS_PID=$!

  while [[ ! -S "${SOCKET_PATH}" ]]; do
    sleep 0.05
  done
  echo "Launching Envoy with Custos ext_proc filter (${SCRIPT_DIR}/envoy-custos.yaml)..."
  exec envoy -c "${SCRIPT_DIR}/envoy-custos.yaml"
elif [[ "${MODE}" == "baseline" ]]; then
  echo "Launching Envoy baseline without Custos (${SCRIPT_DIR}/envoy-baseline.yaml)..."
  exec envoy -c "${SCRIPT_DIR}/envoy-baseline.yaml"
else
  echo "Unknown mode: ${MODE}. Expected 'custos' or 'baseline'." >&2
  exit 1
fi
