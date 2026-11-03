# Custos-Gateway

**High-Performance, Zero-Copy Layer 7 Admission Control for AI Inference Traffic Using Rust and Envoy `ext_proc`**

---

## 1. Problem Statement & Architectural Rationale

Modern AI inference runtimes—such as **NVIDIA Triton Inference Server**—rely on high-throughput gRPC streams transporting binary Protocol Buffers (Protobuf) tensor payloads (`KServe v2` / `inference.GRPCInferenceService/ModelInfer`). Because model runtimes execute tight compute-bound operations on specialized GPU and CPU hardware, they assume well-formed wire inputs and are acutely vulnerable to resource-exhausting or structurally malformed payloads:

* **Protobuf Nesting Bombs:** Deeply nested `Wire Type 2` length-delimited submessages that exhaust C++ recursive parser call stacks.
* **Tensor Shape Overflows:** Anomalous dimension arrays (negative dimensions, 64-bit integer wraparounds, or multi-gigabyte dimension products) that trigger unhandled out-of-memory (OOM) heap allocations before tensor buffer bounds are checked.
* **Malformed Varints:** Unterminated Base-128 varint sequences (`11+` continuation bytes with `MSB = 1` or 10th-byte 64-bit overflows) that stall wire parsers.

### Why Layer 7 `ext_proc` Instead of Layer 2/3 `AF_XDP` / `eBPF`?
Custos pivoted from a kernel-bypass `AF_XDP` packet filter to an **unprivileged, zero-copy Layer 7 admission control daemon** written in Rust due to three physical systems boundaries verified during early prototyping:
1. **Cryptographic Blindness (TLS 1.3 / mTLS):** Ephemeral per-connection keys render NIC-level packet rings opaque ciphertext; the Linux eBPF verifier prohibits the unbounded loops and dynamic allocations required to terminate TLS in-kernel.
2. **Multi-Frame TCP Segmentation & HTTP/2 Multiplexing:** Inference tensors (`600 KB` to `9.6+ MB`) span hundreds of MTU frames that can arrive out of order and are multiplexed across stateful HTTP/2 streams using HPACK header compression. Inspecting application payloads at the packet ring requires re-implementing a full user-space TCP reassembly and HTTP/2 state machine.
3. **Privilege & Blast-Radius Constraints:** `AF_XDP` requires `CAP_NET_ADMIN`, `CAP_BPF`, and host networking on GPU nodes, introducing security exposure and colliding with cluster CNIs (such as Cilium).

```
                     ┌──────────────────────────────────────────────────────────┐
                     │                Unprivileged Ingress Plane                │
                     │                                                          │
Downstream Client ───┼──► Envoy Proxy (:8001) ═══[ UDS: ext_proc.sock ]═══► custos-proc
 (TLS 1.3 / HTTP/2)  │     • Terminates TLS 1.3          (0660)             (custos-core)
                     │     • Reassembles TCP                                • Zero-alloc TLV scan
                     │     • Demultiplexes HTTP/2                           • ≤10B varint bound
                     └───────────────┬──────────────────────────────────────────┬┘
                                     │ ALLOW (Continue)                         │ DENY (ImmediateResponse)
                                     ▼                                          ▼
                          NVIDIA Triton Server (:9001)               HTTP 400 / gRPC INVALID_ARGUMENT (3)
```

---

## 2. Repository Structure

```text
Custos-Gateway/
├── Cargo.toml                         # Rust 2024 workspace (LTO="fat", panic="abort")
├── Dockerfile                         # Multi-stage Debian Bookworm image (UID/GID 10001)
├── proto/envoy/service/ext_proc/v3/   # Envoy ExternalProcessor v3 protobuf definitions
├── crates/
│   ├── custos-core/                   # #![no_std], #![forbid(unsafe_code)] zero-copy TLV wire scanner
│   │   └── src/
│   │       ├── lib.rs                 # Public API & comprehensive unit test suite
│   │       ├── scanner.rs             # Single-pass TLV state machine & KServe v2 walker
│   │       ├── varint.rs              # Bounded 10-byte Base-128 varint decoder
│   │       └── error.rs               # Exhaustive ParseError taxonomy
│   └── custos-proc/                   # Async Tokio/Tonic Envoy ext_proc daemon over UDS
│       └── src/
│           ├── main.rs                # CLI, UDS lifecycle (0660), signal handling, e2e UDS test
│           └── service.rs             # ExternalProcessor stream handler & atomic ServiceMetrics
└── deployments/
    ├── bare-metal/                    # Scientific evaluation configs (Topology B & Topology C)
    └── k8s-topology1-gateway-sidecar/ # Practical production Kubernetes deployment
```

---

## 3. Core Technical Invariants (`custos-core`)

`custos-core` is compiled with `#![no_std]` and `#![forbid(unsafe_code)]`. It validates contiguous byte slices (`&[u8]`) in a **single pass with zero heap allocations**:

1. **gRPC Frame Header Verification:**
   Validates that the leading byte has compression disabled (`0x00`) and that bytes `1..5` (big-endian `u32` declared length) match `slice.len() - 5`.
2. **Bounded 10-Byte Varint Decoding (`varint.rs`):**
   Unrolls Base-128 varint decoding up to a strict 10-byte ceiling. If byte index `9` (the 10th byte) has any bit above `0x01` set (which would overflow `u64::MAX`) or still carries a continuation bit (`0x80`), `ParseError::MalformedVarint` is raised immediately.
3. **Iterative Stack-Allocated Nesting Tracker (`scanner.rs`):**
   Tracks `Wire Type 2` submessage scopes using a fixed stack array `[ScopeFrame; MAX_DEPTH]` (`MAX_DEPTH = 16`). Any submessage exceeding depth `16` is rejected with `ParseError::NestingLimitExceeded(17)` without recursive function calls.
4. **KServe v2 Context-Aware Schema Walker & $O(1)$ Bulk Skipping:**
   Follows the KServe v2 `inference.ModelInferRequest` wire specification:
   - **Root Fields `1` (`model_name`), `2` (`model_version`), `3` (`id`), and `7` (`raw_input_contents`):** Skipped in **$O(1)$ time** by advancing the slice cursor past their length-delimited boundaries. Multi-megabyte float32 tensor buffers in Field 7 are never parsed as Protobuf tags, preventing both latency linear scans and false positives.
   - **Root Field `5` (`inputs` -> `InferInputTensor`):** Pushes an `MsgContext::InputTensor` frame onto the scope stack.
   - **`InferInputTensor` Field `3` (`shape`):** Supports both **packed repeated `int64`** (`Wire Type 2`, tag `0x1A`) and **unpacked repeated `int64`** (`Wire Type 0`, tag `0x18`). Each dimension is validated (`dim > 0`, rejecting negative dimensions with `ParseError::NegativeDimension`) and multiplied into a running product via `i64::checked_mul` (catching 64-bit integer wraparounds with `ParseError::ShapeOverflow`). If the product exceeds `DEFAULT_MAX_TENSOR_ELEMENTS` (`16,777,216` elements = `64 MiB` for `FP32`), `ParseError::BufferCeilingExceeded` is returned immediately.
5. **Shape-to-Payload Buffer Parity Verification (`CVE-2025-23320`):**
   Maps `InferInputTensor` Field `2` (`datatype`: `FP32`, `FP64`, `INT64`, `FP16`, `BF16`, `INT32`, `INT8`, `UINT8`, `BOOL`, etc.) to its exact element byte width and verifies at the end of the message that $\sum (\prod \text{dim}_i \times \text{sizeof}(\text{datatype})) == \text{len}(\text{raw\_input\_contents})$, rejecting buffer under-reads and over-reads with `ParseError::PayloadSizeMismatch`.
6. **Shallow AST / Unknown-Field Flood Ceiling (`CVE-2021-22569` / `CVE-2022-1941`):**
   Bounds the total number of Protobuf TLV tags scanned per message to `DEFAULT_MAX_TAGS_SCANNED` (`1,024`), preventing shallow unknown-field floods (`D = 0..1`) from triggering $10\times$–$50\times$ `UnknownFieldSet` heap amplification in downstream C++/Java/Python parsers.
7. **KServe v2 RPC Method Allowlisting (`CVE-2024-0088` / `CVE-2025-23329` / `CVE-2024-53880`):**
   At the `RequestHeaders` stage in `custos-proc`, only inference and read-only health RPCs (`ModelInfer`, `ModelStreamInfer`, `ServerLive`, `ServerReady`, `ModelReady`, `ModelMetadata`, `ModelConfig`) are admitted. Administrative and shared-memory RPCs (`SystemSharedMemoryRegister`, `CudaSharedMemoryRegister`, `RepositoryModelLoad`, `RepositoryModelUnload`) are rejected immediately with `HTTP 403 Forbidden` and `gRPC PERMISSION_DENIED (7)` (`custos_denied:unauthorized_rpc_method`).

---

## 4. Building & Testing

```bash
# Run all unit and end-to-end UDS integration tests
cargo test --workspace

# Build optimized release binary (fat LTO, single codegen unit, stripped)
cargo build --release -p custos-proc
```

---

## 5. Deployment & Evaluation Architectures

Located in [`deployments/k8s-topology1-gateway-sidecar/`](./deployments/k8s-topology1-gateway-sidecar/README.md) (Kubernetes) and [`deployments/bare-metal/`](./deployments/bare-metal/README.md) (Bare-Metal). Paired with [`Custos-Bench`](../Custos-Bench) and [`Custos-Testbed`](../Custos-Testbed) to evaluate three architectures:
* **Unprotected Control (Direct Triton):** Client directly targets NVIDIA Triton Inference Server (`:8001`).
* **Proxy Baseline (Envoy Only):** Client targets Envoy Proxy ([`envoy-baseline.yaml`](./deployments/bare-metal/envoy-baseline.yaml), `:8002`) forwarding to Triton without `ext_proc`.
* **Custos Production (Gateway + Sidecar):** Client targets Envoy Proxy ([`envoy-custos.yaml`](./deployments/bare-metal/envoy-custos.yaml) / [`k8s-topology1-gateway-sidecar`](./deployments/k8s-topology1-gateway-sidecar/README.md), `:8003` / `:8001`) co-located with the `custos-proc` Native Sidecar enforcing fail-closed `ext_proc` admission over `/var/run/custos/ext_proc.sock`.

```bash
# Start Topology C (Custos + Envoy) locally
./deployments/bare-metal/run-local.sh custos

# Start Topology B (Envoy baseline) locally
./deployments/bare-metal/run-local.sh baseline
```

### B. Practical Production Deployment (Kubernetes Gateway Sidecar)
Located in [`deployments/k8s-topology1-gateway-sidecar/`](./deployments/k8s-topology1-gateway-sidecar/README.md). Deploys Envoy Proxy and `custos-proc` (as a Kubernetes Native Sidecar sharing a RAM-backed `emptyDir` Unix socket volume) inside a dedicated, unprivileged Ingress Gateway Pod in front of the GPU-backed Triton Inference Server Service.

---

## 6. `custos-proc` CLI Reference

| Flag | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `--socket-path` | `CUSTOS_SOCKET_PATH` | `/var/run/custos/ext_proc.sock` | Unix Domain Socket path bound with `0660` permissions |
| `--max-nesting-depth` | `CUSTOS_MAX_NESTING_DEPTH` | `16` | Maximum allowed Protobuf submessage depth (`1..=16`) |
| `--max-tensor-elements` | `CUSTOS_MAX_TENSOR_ELEMENTS` | `16777216` | Maximum shape element product per `InferInputTensor` |
| `--max-tags-scanned` | `CUSTOS_MAX_TAGS_SCANNED` | `1024` | Maximum Protobuf TLV tags scanned per message (`CVE-2021-22569`) |
| `--enforce-payload-parity` | `CUSTOS_ENFORCE_PAYLOAD_PARITY` | `true` | Enforce byte parity between declared tensor shape and `raw_input_contents` (`CVE-2025-23320`) |
| `--max-message-size-mb` | `CUSTOS_MAX_MESSAGE_SIZE_MB` | `32` | Maximum gRPC message size accepted from Envoy in MiB |
| `--log-format` | `CUSTOS_LOG_FORMAT` | `json` | Structured log output format (`json`, `compact`, `pretty`) |
| `--log-level` | `CUSTOS_LOG_LEVEL` | `info` | Tracing filter directive when `RUST_LOG` is unset |
| `--log-admitted` | `CUSTOS_LOG_ADMITTED` | `true` | Emit per-request `INFO` logs on `ALLOW` (set `false` during high-RPS benchmarks) |
| `--stats-interval-secs` | `CUSTOS_STATS_INTERVAL_SECS` | `60` | Periodic cumulative telemetry heartbeat interval in seconds (`0` to disable) |
