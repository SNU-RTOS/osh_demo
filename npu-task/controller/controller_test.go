package controller

import (
	"context"
	api "github.com/SNU-RTOS/osh_demo/npu-task/api"
	"github.com/SNU-RTOS/osh_demo/npu-task/resource"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"testing"
)

type staticRegistry struct{ states []resource.NPUState }

func (s staticRegistry) Snapshot(context.Context) ([]resource.NPUState, error) { return s.states, nil }

func fixture(t *testing.T) (*Reconciler, *api.NPUTask) {
	t.Helper()
	s := runtime.NewScheme()
	_ = corev1.AddToScheme(s)
	_ = api.AddToScheme(s)
	task := &api.NPUTask{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default", UID: types.UID("unique-task"), Generation: 1}, Spec: api.NPUTaskSpec{Image: "test:v1", Command: []string{"/infer"}, Model: api.Model{Path: "/model.hef"}, NPUCount: 2, Mode: "Batch"}}
	c := fake.NewClientBuilder().WithScheme(s).WithStatusSubresource(&api.NPUTask{}, &corev1.Pod{}).WithObjects(task).Build()
	return &Reconciler{Client: c, Scheme: s}, task
}
func reconcile(t *testing.T, r *Reconciler, task *api.NPUTask) {
	t.Helper()
	if _, err := r.Reconcile(context.Background(), ctrl.Request{NamespacedName: client.ObjectKeyFromObject(task)}); err != nil {
		t.Fatal(err)
	}
	if err := r.Get(context.Background(), client.ObjectKeyFromObject(task), task); err != nil {
		t.Fatal(err)
	}
}
func getPod(t *testing.T, r *Reconciler, task *api.NPUTask) *corev1.Pod {
	t.Helper()
	p := &corev1.Pod{}
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: task.Namespace, Name: podName(task)}, p); err != nil {
		t.Fatal(err)
	}
	return p
}
func updateTask(t *testing.T, r *Reconciler, task *api.NPUTask) {
	t.Helper()
	task.Generation++
	if err := r.Update(context.Background(), task); err != nil {
		t.Fatal(err)
	}
}

func TestResizeWaitsForOldPod(t *testing.T) {
	r, task := fixture(t)
	reconcile(t, r, task)
	p := getPod(t, r, task)
	p.Finalizers = []string{"test/hold"}
	if err := r.Update(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	task.Spec.NPUCount = 4
	updateTask(t, r, task)
	reconcile(t, r, task)
	reconcile(t, r, task)
	old := getPod(t, r, task)
	oldCount := old.Spec.Containers[0].Resources.Limits[api.ResourceName]
	if old.DeletionTimestamp.IsZero() || oldCount.Value() != 2 {
		t.Fatal("replacement started before old pod terminated")
	}
	old.Finalizers = nil
	if err := r.Update(context.Background(), old); err != nil {
		t.Fatal(err)
	}
	reconcile(t, r, task)
	p = getPod(t, r, task)
	q := p.Spec.Containers[0].Resources.Limits[api.ResourceName]
	if q.Value() != 4 {
		t.Fatalf("got allocation %s", q.String())
	}
}

func TestMonitorEnvMountsNodeDirectory(t *testing.T) {
	r, task := fixture(t)
	task.Spec.Env = []corev1.EnvVar{{Name: "HAILO_MONITOR", Value: "1"}}
	updateTask(t, r, task)
	reconcile(t, r, task)
	p := getPod(t, r, task)
	if len(p.Spec.Volumes) != 1 || p.Spec.Volumes[0].HostPath == nil || p.Spec.Volumes[0].HostPath.Path != "/tmp/hmon_files" {
		t.Fatalf("monitor volume missing: %#v", p.Spec.Volumes)
	}
	if len(p.Spec.Containers[0].VolumeMounts) != 1 || p.Spec.Containers[0].VolumeMounts[0].MountPath != "/tmp/hmon_files" {
		t.Fatalf("monitor mount missing: %#v", p.Spec.Containers[0].VolumeMounts)
	}
}

func TestSuspendResumeAndControllerRestart(t *testing.T) {
	r, task := fixture(t)
	reconcile(t, r, task)
	// A fresh reconciler must adopt the deterministic owned pod.
	r = &Reconciler{Client: r.Client, Scheme: r.Scheme}
	reconcile(t, r, task)
	var pods corev1.PodList
	_ = r.List(context.Background(), &pods)
	if len(pods.Items) != 1 {
		t.Fatal("duplicate pod")
	}
	task.Spec.Suspend = true
	updateTask(t, r, task)
	reconcile(t, r, task)
	reconcile(t, r, task)
	if task.Status.Phase != "Suspended" {
		t.Fatal(task.Status)
	}
	task.Spec.Suspend = false
	updateTask(t, r, task)
	reconcile(t, r, task)
	getPod(t, r, task)
}

func TestBatchResultSurvivesPodDeletion(t *testing.T) {
	for _, phase := range []corev1.PodPhase{corev1.PodSucceeded, corev1.PodFailed} {
		t.Run(string(phase), func(t *testing.T) {
			r, task := fixture(t)
			reconcile(t, r, task)
			p := getPod(t, r, task)
			p.Status.Phase = phase
			if err := r.Status().Update(context.Background(), p); err != nil {
				t.Fatal(err)
			}
			reconcile(t, r, task)
			if err := r.Delete(context.Background(), p); err != nil {
				t.Fatal(err)
			}
			reconcile(t, r, task)
			var pods corev1.PodList
			_ = r.List(context.Background(), &pods)
			if len(pods.Items) != 0 || task.Status.Phase != string(phase) {
				t.Fatal("terminal batch retried")
			}
			task.Spec.Args = []string{"--new-run"}
			updateTask(t, r, task)
			reconcile(t, r, task)
			getPod(t, r, task)
		})
	}
}

func TestPendingReasonAndOwnership(t *testing.T) {
	r, task := fixture(t)
	reconcile(t, r, task)
	p := getPod(t, r, task)
	p.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodScheduled, Status: corev1.ConditionFalse, Reason: "Unschedulable", Message: "Insufficient hailo.ai/npu"}}
	if err := r.Status().Update(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	reconcile(t, r, task)
	if task.Status.Conditions[0].Message != "Insufficient hailo.ai/npu" {
		t.Fatal(task.Status)
	}
	p.OwnerReferences = nil
	if err := r.Update(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	task.Spec.Suspend = true
	updateTask(t, r, task)
	reconcile(t, r, task)
	if task.Status.Conditions[0].Reason != "OwnershipConflict" {
		t.Fatal(task.Status)
	}
	getPod(t, r, task)
}

func TestContractAndValidation(t *testing.T) {
	r, task := fixture(t)
	task.Spec.Mode = "Service"
	task.Spec.Model = api.Model{Path: "/models/a.hef", PVC: "models", MountPath: "/models"}
	p, err := desiredPod(task, r.Scheme, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if p.Spec.RestartPolicy != corev1.RestartPolicyAlways || !p.Spec.Containers[0].VolumeMounts[0].ReadOnly {
		t.Fatal("service/PVC contract")
	}
	if err := Validate(task.Spec); err != nil {
		t.Fatal(err)
	}
	for _, count := range []int32{0, 9} {
		s := task.Spec
		s.NPUCount = count
		if Validate(s) == nil {
			t.Fatal("accepted invalid count")
		}
	}
	s := task.Spec
	s.Env = []corev1.EnvVar{{Name: "NPU_COUNT", Value: "8"}}
	if Validate(s) == nil {
		t.Fatal("accepted reserved env")
	}
	s = task.Spec
	s.Model.Path = "/models/../escape.hef"
	if Validate(s) == nil {
		t.Fatal("accepted model outside mount")
	}
}

func TestServiceTerminalPodIsReplaced(t *testing.T) {
	r, task := fixture(t)
	task.Spec.Mode = "Service"
	updateTask(t, r, task)
	reconcile(t, r, task)
	p := getPod(t, r, task)
	p.Status.Phase = corev1.PodFailed
	p.Status.Reason = "Evicted"
	if err := r.Status().Update(context.Background(), p); err != nil {
		t.Fatal(err)
	}
	reconcile(t, r, task)
	if task.Status.Phase != "Stopping" {
		t.Fatal(task.Status)
	}
	reconcile(t, r, task)
	p = getPod(t, r, task)
	if p.Status.Phase == corev1.PodFailed {
		t.Fatal("terminal service pod was not replaced")
	}
}

func TestSharedTaskUsesFirstFitDispatcherWithoutPhysicalResource(t *testing.T) {
	r, task := fixture(t)
	task.Spec.NPUCount = 0
	task.Spec.NPUShare = 300
	task.Spec.Model.Key = "yolo"
	r.Registry = staticRegistry{states: []resource.NPUState{
		{ID: "npu-0", NodeName: "node-a", Endpoint: "unix:///tmp/npu-0.sock", SchedulerEndpoint: "unix:///tmp/npu-0-scheduler.sock", Health: resource.HealthHealthy, Capacity: 1000, Allocated: 800,
			Workloads: []resource.WorkloadAllocation{{WorkloadID: "existing-a", Share: 800}}},
		{ID: "npu-1", NodeName: "node-a", Endpoint: "unix:///tmp/npu-1.sock", SchedulerEndpoint: "unix:///tmp/npu-1-scheduler.sock", Health: resource.HealthHealthy, Capacity: 1000, Allocated: 400,
			Workloads: []resource.WorkloadAllocation{{WorkloadID: "existing-b", Share: 400}}},
	}}
	updateTask(t, r, task)
	reconcile(t, r, task)
	p := getPod(t, r, task)
	if p.Spec.NodeName != "node-a" || p.Annotations["npu.snu-rtos.io/assigned-npu"] != "npu-1" {
		t.Fatalf("unexpected placement: node=%q annotations=%v", p.Spec.NodeName, p.Annotations)
	}
	if _, exists := p.Spec.Containers[0].Resources.Limits[api.ResourceName]; exists {
		t.Fatal("shared client received a physical NPU resource")
	}
	foundEndpoint := false
	foundScheduler := false
	for _, env := range p.Spec.Containers[0].Env {
		if env.Name == "HAILORT_SERVICE_ADDRESS" && env.Value == "unix:///tmp/npu-1.sock" {
			foundEndpoint = true
		}
		if env.Name == "NPU_SCHEDULER_ADDRESS" && env.Value == "unix:///tmp/npu-1-scheduler.sock" {
			foundScheduler = true
		}
	}
	if !foundEndpoint || !foundScheduler {
		t.Fatal("dispatcher endpoints were not injected")
	}
}

func TestSharedTaskWaitsWhenLogicalCapacityIsFull(t *testing.T) {
	r, task := fixture(t)
	task.Spec.NPUCount = 0
	task.Spec.NPUShare = 300
	r.Registry = staticRegistry{states: []resource.NPUState{{ID: "npu-0", NodeName: "node-a", Endpoint: "unix:///tmp/npu-0.sock", SchedulerEndpoint: "unix:///tmp/npu-0-scheduler.sock", Health: resource.HealthHealthy, Capacity: 1000, Allocated: 800,
		Workloads: []resource.WorkloadAllocation{{WorkloadID: "existing", Share: 800}}}}}
	updateTask(t, r, task)
	reconcile(t, r, task)
	if task.Status.Phase != "Pending" || task.Status.Conditions[0].Reason != "InsufficientNPUShare" {
		t.Fatalf("unexpected status: %#v", task.Status)
	}
	var pods corev1.PodList
	if err := r.List(context.Background(), &pods); err != nil {
		t.Fatal(err)
	}
	if len(pods.Items) != 0 {
		t.Fatal("pod created without logical capacity")
	}
}
