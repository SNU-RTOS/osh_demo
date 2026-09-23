// Package sharing reconstructs logical NPU allocations from Kubernetes Pods.
// Dispatcher Pods are the only Pods that own physical hailo.ai/npu resources.
package sharing

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/SNU-RTOS/osh_demo/npu-task/resource"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	ComponentLabel         = "npu.snu-rtos.io/component"
	DispatcherComponent    = "dispatcher"
	DispatcherIDAnn        = "npu.snu-rtos.io/dispatcher-id"
	DispatcherEndpointAnn  = "npu.snu-rtos.io/dispatcher-endpoint"
	DispatcherSchedulerAnn = "npu.snu-rtos.io/scheduler-endpoint"
	DispatcherCapacityAnn  = "npu.snu-rtos.io/capacity"
	AssignedNPUAnn         = "npu.snu-rtos.io/assigned-npu"
	AllocatedShareAnn      = "npu.snu-rtos.io/allocated-share"
	WorkloadIDAnn          = "npu.snu-rtos.io/workload-id"
	ModelKeyAnn            = "npu.snu-rtos.io/model-key"
)

type Registry interface {
	Snapshot(context.Context) ([]resource.NPUState, error)
}

type PodRegistry struct{ Client client.Client }

var brokerHTTPClient = &http.Client{Timeout: 500 * time.Millisecond}

func brokerEpoch(pod *corev1.Pod) string {
	if pod.Status.PodIP == "" {
		return ""
	}
	response, err := brokerHTTPClient.Get("http://" + pod.Status.PodIP + ":9790/metrics")
	if err != nil {
		return ""
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return ""
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return ""
	}
	return parseBrokerEpoch(string(body))
}

func parseBrokerEpoch(metrics string) string {
	const marker = `npu_share_broker_info{epoch="`
	for _, line := range strings.Split(metrics, "\n") {
		if strings.HasPrefix(line, marker) {
			if end := strings.Index(line[len(marker):], `"`); end >= 0 {
				return line[len(marker) : len(marker)+end]
			}
		}
	}
	return ""
}

func (r PodRegistry) Snapshot(ctx context.Context) ([]resource.NPUState, error) {
	var pods corev1.PodList
	if err := r.Client.List(ctx, &pods); err != nil {
		return nil, fmt.Errorf("list pods: %w", err)
	}
	states := make([]resource.NPUState, 0)
	byID := make(map[string]int)
	for i := range pods.Items {
		pod := &pods.Items[i]
		if pod.Labels[ComponentLabel] != DispatcherComponent || pod.Spec.NodeName == "" || pod.DeletionTimestamp != nil || !podReady(pod) {
			continue
		}
		id, endpoint := pod.Annotations[DispatcherIDAnn], pod.Annotations[DispatcherEndpointAnn]
		schedulerEndpoint := pod.Annotations[DispatcherSchedulerAnn]
		if id == "" || endpoint == "" || schedulerEndpoint == "" {
			continue
		}
		if _, duplicate := byID[id]; duplicate {
			return nil, fmt.Errorf("duplicate ready dispatcher ID %q", id)
		}
		capacity := resource.DefaultCapacity
		if raw := pod.Annotations[DispatcherCapacityAnn]; raw != "" {
			parsed, err := strconv.Atoi(raw)
			if err != nil || parsed <= 0 {
				return nil, fmt.Errorf("dispatcher %q has invalid capacity %q", id, raw)
			}
			capacity = parsed
		}
		byID[id] = len(states)
		states = append(states, resource.NPUState{ID: id, NodeName: pod.Spec.NodeName, Endpoint: endpoint, SchedulerEndpoint: schedulerEndpoint, BrokerEpoch: brokerEpoch(pod), Health: resource.HealthHealthy, Capacity: capacity})
	}
	for i := range pods.Items {
		pod := &pods.Items[i]
		if pod.DeletionTimestamp != nil || pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
			continue
		}
		id := pod.Annotations[AssignedNPUAnn]
		index, exists := byID[id]
		if !exists {
			continue
		}
		share, err := strconv.Atoi(pod.Annotations[AllocatedShareAnn])
		if err != nil || share <= 0 {
			return nil, fmt.Errorf("assigned pod %s/%s has invalid share", pod.Namespace, pod.Name)
		}
		state := &states[index]
		if pod.Spec.NodeName != "" && pod.Spec.NodeName != state.NodeName {
			return nil, fmt.Errorf("assigned pod %s/%s is on node %q, dispatcher is on %q", pod.Namespace, pod.Name, pod.Spec.NodeName, state.NodeName)
		}
		state.Workloads = append(state.Workloads, resource.WorkloadAllocation{WorkloadID: pod.Annotations[WorkloadIDAnn], ModelKey: pod.Annotations[ModelKeyAnn], Share: share})
		state.Allocated += share
		if state.Allocated > state.Capacity {
			return nil, fmt.Errorf("dispatcher %q is over-allocated: %d/%d", id, state.Allocated, state.Capacity)
		}
	}
	sort.Slice(states, func(i, j int) bool { return states[i].ID < states[j].ID })
	return states, nil
}

func podReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}
