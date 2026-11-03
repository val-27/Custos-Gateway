# Kubernetes Cluster Deployment & Evaluation Guide

This directory contains the Kubernetes manifests to deploy and evaluate all three unified architectures on your **2-node Kubernetes cluster**:

1. **Unprotected Control (`:8001`):** Direct to `triton-inference-svc` ([`deployment-triton-pod.yaml`](./deployment-triton-pod.yaml)).
2. **Proxy Baseline (`:8002`):** Standard Envoy HTTP/2 ingress Pod forwarding to Triton without `custos-proc` ([`deployment-envoy-baseline-pod.yaml`](./deployment-envoy-baseline-pod.yaml)).
3. **Custos Production (`:8003` / `:8001`):** Dedicated Ingress Gateway Pod running Envoy Proxy + the `custos-proc` Kubernetes Native Sidecar sharing a RAM-backed `emptyDir` Unix Domain Socket (`/var/run/custos/ext_proc.sock`) ([`configmap-envoy.yaml`](./configmap-envoy.yaml) & [`deployment-gateway-pod.yaml`](./deployment-gateway-pod.yaml)).

---

## 1. Deploy All Three Architectures to Your 2-Node Cluster

```bash
# 1. Build the custos-proc container image and make it available to your cluster nodes
docker build -t custos-proc:latest ../..

# 2. Apply the Triton Backend (Unprotected Control), Proxy Baseline, and Custos Production manifests
kubectl apply -f deployment-triton-pod.yaml
kubectl apply -f deployment-envoy-baseline-pod.yaml
kubectl apply -f configmap-envoy.yaml
kubectl apply -f deployment-gateway-pod.yaml

# 3. Verify all Pods and Services are Ready
kubectl get pods,svc -l app.kubernetes.io/part-of=custos
```

---

## 2. Expose Ports `:8001` (Control), `:8002` (Baseline), and `:8003` (Production) for `Custos-Bench`

If running `custos-bench` from outside the cluster (or from your benchmark node via port-forwarding or NodePort/ClusterIP), bind the three ports expected by `Custos-Bench/scripts/run_scientific_suite.sh`:

```bash
kubectl port-forward svc/triton-inference-svc 8001:8001 &
kubectl port-forward svc/envoy-baseline-svc   8002:8002 &
kubectl port-forward svc/custos-gateway-svc   8003:8003 &
```

Then execute the automated 5-trial scientific benchmark suite:

```bash
../../../Custos-Bench/scripts/run_scientific_suite.sh \
  --mode lan \
  --target-host <CLUSTER_NODE_OR_LOCALHOST_IP> \
  --trials 5
```
