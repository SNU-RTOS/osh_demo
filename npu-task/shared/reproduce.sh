#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
namespace="${NAMESPACE:-default}"
result_file="${RESULT_FILE:-${repo_dir}/npu-task/results/multi-npu-share.json}"

"${repo_dir}/npu-task/shared/build-images.sh"
docker build -t npu-task-controller:share "${repo_dir}/npu-task"
docker build -t hailo-monitor-exporter:share -f "${repo_dir}/npu-task/Dockerfile.monitor-exporter" "${repo_dir}/npu-task"
docker save npu-task-controller:share hailo-monitor-exporter:share npu-share-dispatcher:local npu-share-client:local | k3s ctr -n k8s.io images import -

kubectl apply -f "${repo_dir}/npu-task/deploy/crd.yaml"
kubectl apply -f "${repo_dir}/npu-task/deploy/controller.yaml"
kubectl -n npu-task-system set image deployment/npu-task-controller controller=npu-task-controller:share
kubectl -n npu-task-system rollout status deployment/npu-task-controller --timeout=120s
kubectl apply -f "${repo_dir}/npu-task/deploy/monitor-exporter.yaml"
kubectl -n npu-task-system set image daemonset/hailo-monitor-exporter exporter=hailo-monitor-exporter:share
kubectl -n npu-task-system rollout restart daemonset/hailo-monitor-exporter
kubectl -n npu-task-system rollout status daemonset/hailo-monitor-exporter --timeout=120s

kubectl -n "${namespace}" delete nputask exclusive-a exclusive-b exclusive-c exclusive-d --ignore-not-found --wait=true
kubectl -n npu-task-system delete deployment npu-share-dispatcher-0 npu-share-dispatcher-1 --ignore-not-found --wait=true
kubectl -n "${namespace}" apply -f "${repo_dir}/npu-task/shared/exclusive-workloads.yaml"
for wave in "exclusive-a exclusive-b" "exclusive-c exclusive-d"; do
    for task in ${wave}; do
        kubectl -n "${namespace}" patch "nputask/${task}" --type=merge -p '{"spec":{"suspend":false}}'
    done
    for task in ${wave}; do
        kubectl -n "${namespace}" wait "nputask/${task}" --for=jsonpath='{.status.phase}'=Succeeded --timeout=300s
        pod="$(kubectl -n "${namespace}" get "nputask/${task}" -o jsonpath='{.status.podName}')"
        kubectl -n "${namespace}" logs "${pod}" | grep -q 'NPU_E2E_OK mode=exclusive.*runs=500'
    done
done

kubectl apply -f "${repo_dir}/npu-task/shared/dispatchers.yaml"
kubectl -n npu-task-system rollout status deployment/npu-share-dispatcher-0 --timeout=180s
kubectl -n npu-task-system rollout status deployment/npu-share-dispatcher-1 --timeout=180s

kubectl -n "${namespace}" delete nputask share-a share-b share-c share-d --ignore-not-found --wait=true
kubectl -n "${namespace}" delete nputask share-overflow --ignore-not-found --wait=true
kubectl -n "${namespace}" apply -f "${repo_dir}/npu-task/shared/workloads.yaml"

for task in share-a share-b share-c share-d; do
    deadline=$((SECONDS + 180))
    while (( SECONDS < deadline )); do
        phase="$(kubectl -n "${namespace}" get "nputask/${task}" -o jsonpath='{.status.phase}')"
        [[ "${phase}" == Running ]] && break
        [[ "${phase}" == Succeeded || "${phase}" == Failed ]] && break
        sleep 1
    done
    [[ "${phase}" == Running ]]
done

kubectl -n "${namespace}" apply -f "${repo_dir}/npu-task/shared/overflow.yaml"
kubectl -n "${namespace}" wait nputask/share-overflow --for=jsonpath='{.status.phase}'=Pending --timeout=60s
test "$(kubectl -n "${namespace}" get nputask/share-overflow -o jsonpath='{.status.conditions[0].reason}')" = InsufficientNPUShare

monitor_ip="$(kubectl -n npu-task-system get service hailo-monitor-exporter -o jsonpath='{.spec.clusterIP}')"
monitor_metrics=""
for _ in $(seq 1 30); do
    monitor_metrics="$(curl -fsS "http://${monitor_ip}:9788/metrics" || true)"
    grep -q '^hailo_device_utilization_percent' <<<"${monitor_metrics}" && break
    sleep 1
done
printf '%s\n' "${monitor_metrics}"
grep -q '^hailo_device_utilization_percent' <<<"${monitor_metrics}"

for task in share-a share-b share-c share-d; do
    kubectl -n "${namespace}" wait "nputask/${task}" --for=jsonpath='{.status.phase}'=Succeeded --timeout=300s
    pod="$(kubectl -n "${namespace}" get "nputask/${task}" -o jsonpath='{.status.podName}')"
    kubectl -n "${namespace}" logs "${pod}" | grep -q 'NPU_E2E_OK mode=shared.*runs=2000'
done

kubectl -n "${namespace}" wait nputask/share-overflow --for=jsonpath='{.status.phase}'=Succeeded --timeout=300s
overflow_pod="$(kubectl -n "${namespace}" get nputask/share-overflow -o jsonpath='{.status.podName}')"
kubectl -n "${namespace}" logs "${overflow_pod}" | grep -q 'NPU_E2E_OK mode=shared.*runs=10'

for dispatcher in npu-share-dispatcher-0 npu-share-dispatcher-1; do
    metrics="$(kubectl -n npu-task-system exec deployment/${dispatcher} -c share-scheduler -- curl -fsS http://127.0.0.1:9790/metrics)"
    printf '%s\n' "${metrics}"
    test "$(grep -c '^npu_share_grants_total' <<<"${metrics}")" -ge 2
done

test "$(kubectl -n "${namespace}" get nputask share-a -o jsonpath='{.status.npuID}')" = npu-0
test "$(kubectl -n "${namespace}" get nputask share-b -o jsonpath='{.status.npuID}')" = npu-1
test "$(kubectl -n "${namespace}" get nputask share-c -o jsonpath='{.status.npuID}')" = npu-0
test "$(kubectl -n "${namespace}" get nputask share-d -o jsonpath='{.status.npuID}')" = npu-1

kubectl -n "${namespace}" get nputask share-a share-b share-c share-d \
    -o custom-columns=NAME:.metadata.name,NPU:.status.npuID,SHARE:.status.allocatedShare,PHASE:.status.phase
python3 "${repo_dir}/npu-task/shared/evaluate.py" --namespace "${namespace}" --output "${result_file}"
echo "Multi-NPU fractional-sharing E2E passed"
