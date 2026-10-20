# Topology 1: Dedicated Gateway Pod Sidecar

## Architecture
In **Topology 1**, Envoy Proxy and `custos-proc` run together inside a dedicated, CPU-bound **Ingress Gateway Pod**, decoupled from the GPU-backed **NVIDIA Triton Inference Server Pod**:

1. **Gateway Pod (`custos-gateway-deployment`):**
   - `custos-proc` runs as a Kubernetes Native Sidecar (`initContainers` with `restartPolicy: Always`) and binds `/var/run/custos/ext_proc.sock` (`0660`) inside a RAM-backed `emptyDir` volume (`medium: Memory`, `sizeLimit: 16Mi`).
   - A startup probe guarantees `/var/run/custos/ext_proc.sock` exists before the main `envoy` container starts.
   - Envoy listens on port `8001`, buffers downstream KServe v2 `ModelInferRequest` frames up to `16 MiB`, and inspects them synchronously via `envoy.filters.http.ext_proc` (`failure_mode_allow: false`, `request_body_mode: BUFFERED`).
   - Hardened security context: `runAsNonRoot: true`, `runAsUser: 10001`, `fsGroup: 10001`, `readOnlyRootFilesystem: true`, and `capabilities: drop: ["ALL"]`.
2. **Inference Pod (`triton-inference-deployment` & `triton-inference-svc`):**
   - Runs NVIDIA Triton Inference Server on port `9001` behind a ClusterIP Service (`triton-inference-svc:9001`).
   - Only requests admitted by `custos-proc` traverse the cluster network to reach Triton.

## Deployment
```bash
kubectl apply -f deployments/k8s-topology1-gateway-sidecar/configmap-envoy.yaml
kubectl apply -f deployments/k8s-topology1-gateway-sidecar/deployment-triton-pod.yaml
kubectl apply -f deployments/k8s-topology1-gateway-sidecar/deployment-gateway-pod.yaml
```
