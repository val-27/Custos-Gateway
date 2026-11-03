# Bare-Metal Scientific Evaluation Topologies

## Purpose
These configurations support the **scientific evaluation methodology** for benchmarking Custos on a dedicated bare-metal Linux testbed (paired with the [`Custos-Bench`](../../Custos-Bench) load harness).

To isolate the exact latency, throughput, CPU/RSS overhead, and survivability impact of zero-copy L7 wire inspection, we evaluate three topologies over identical hardware and network paths:

| Evaluation Topology | Data Path | Configuration | Purpose |
| :--- | :--- | :--- | :--- |
| **Topology A: Unprotected Server** | `Client -> Triton (:8001 or :9001)` | Direct to `tritonserver` | Establishes raw inference latency and demonstrates runtime vulnerability / crashes under exploit bursts. |
| **Topology B: Standard Proxy Baseline** | `Client -> Envoy (:8001) -> Triton (:9001)` | [`envoy-baseline.yaml`](./envoy-baseline.yaml) | Isolates standard L7 HTTP/2 proxy forwarding cost without `ext_proc` payload inspection. |
| **Topology C: Custos Protected Ingress** | `Client -> Envoy (:8001) ==[UDS]==> custos-proc -> Triton (:9001)` | [`envoy-custos.yaml`](./envoy-custos.yaml) | Evaluates zero-copy TLV wire validation efficacy, microsecond latency delta over Topology B, and server survival under attack. |

## Quick Start

### 1. Start Upstream Triton Inference Server
Ensure Triton is listening on gRPC port `9001` with the `resnet50` ONNX model loaded:
```bash
tritonserver --model-repository=/path/to/models --grpc-port=9001 --http-port=8000 --metrics-port=8002
```

### 2. Launch Topology B (Standard Envoy Baseline)
```bash
./deployments/bare-metal/run-local.sh baseline
```

### 3. Launch Topology C (Envoy + Custos `ext_proc` over UDS)
```bash
./deployments/bare-metal/run-local.sh custos
```
By default, `run-local.sh` builds `custos-proc` in `--release` mode, starts it on `/var/run/custos/ext_proc.sock` with per-request admitted stdout logging disabled (`CUSTOS_LOG_ADMITTED=false`) to avoid stdout I/O contention during high-RPS benchmarks, and launches Envoy with [`envoy-custos.yaml`](./envoy-custos.yaml).
