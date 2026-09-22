package sharing

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestSnapshotReconstructsDispatcherAllocations(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	dispatcher := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "dispatcher-0", Namespace: "system",
		Labels: map[string]string{ComponentLabel: DispatcherComponent}, Annotations: map[string]string{DispatcherIDAnn: "npu-0", DispatcherEndpointAnn: "unix:///tmp/npu-0.sock", DispatcherSchedulerAnn: "unix:///tmp/npu-0-scheduler.sock"}},
		Spec: corev1.PodSpec{NodeName: "node-a"}, Status: corev1.PodStatus{Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}}}
	workload := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "client", Namespace: "apps", Annotations: map[string]string{
		AssignedNPUAnn: "npu-0", AllocatedShareAnn: "300", WorkloadIDAnn: "uid-a", ModelKeyAnn: "yolo"}}, Spec: corev1.PodSpec{NodeName: "node-a"}, Status: corev1.PodStatus{Phase: corev1.PodRunning}}
	registry := PodRegistry{Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(dispatcher, workload).Build()}
	states, err := registry.Snapshot(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(states) != 1 || states[0].Allocated != 300 || states[0].AvailableCapacity() != 700 || len(states[0].Workloads) != 1 {
		t.Fatalf("unexpected snapshot: %#v", states)
	}
}
