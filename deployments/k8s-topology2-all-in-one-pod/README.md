# Topology 2: All-in-One 3-Container Inference Pod

## Architecture
In **Topology 2**, Envoy Proxy, `custos-proc`, and NVIDIA Triton Inference Server all run inside the **same Pod**, sharing the Pod network namespace (`127.0.0.1`) and a RAM-backed `emptyDir` volume (`/var/run/custos/ext_proc.sock`):

1. **`custos-proc` (Native Sidecar `initContainers` with `restartPolicy: Always`):**
   - Starts first, creates `/var/run/custos/ext_proc.sock` (`0660`) inside the in-memory `emptyDir` (`medium: Memory`, `sizeLimit: 16Mi`), and runs with `runAsNonRoot: true`, `readOnlyRootFilesystem: true`, and `capabilities: drop: ["ALL"]`.
2. **`envoy` (Ingress Sidecar Container):**
   - Listens on `0.0.0.0:8001` for external KServe v2 gRPC inference traffic.
   - Buffers up to `16 MiB` of request body and invokes `custos-proc` over `/var/run/custos/ext_proc.sock` (`failure_mode_allow: false`, `request_body_mode: BUFFERED`).
   - Forwards validated requests over Pod loopback (`127.0.0.1:9001`).
3. **`triton` (Inference Runtime Container):**
   - Binds its gRPC server strictly to Pod loopback (`--grpc-address=127.0.0.1`, `--grpc-port=9001`) so external callers cannot bypass Envoy and `custos-proc`.

## Deployment
```bash
kubectl apply -f deployments/k8s-topology2-all-in-one-pod/configmap-envoy.yaml
kubectl apply -f deployments/k8s-topology2-all-in-one-pod/deployment-inference-pod.yaml
```
