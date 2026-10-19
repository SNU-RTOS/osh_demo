package controller

import (
	"context"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/SNU-RTOS/osh_demo/npu-task/admission"
	api "github.com/SNU-RTOS/osh_demo/npu-task/api"
	"github.com/SNU-RTOS/osh_demo/npu-task/dispatcher"
	"github.com/SNU-RTOS/osh_demo/npu-task/resource"
	"github.com/SNU-RTOS/osh_demo/npu-task/sharing"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func bestConfig() admission.Config {
	c := admission.Default()
	c.SharedAdmissionPolicy = admission.BestEffort
	c.Profiles = map[string]admission.Profile{"model": {ModelSHA256: strings.Repeat("a", 64), DeviceArchitecture: "HAILO8", RuntimeVersion: "4.21.0", MeanSeconds: .009, P95Seconds: .01, Samples: 500}}
	return c
}
func dispatcherPod() *corev1.Pod {
	return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "dispatcher", Namespace: "system", Labels: map[string]string{sharing.ComponentLabel: sharing.DispatcherComponent}, Annotations: map[string]string{
		sharing.DispatcherIDAnn: "npu-0", sharing.DispatcherEndpointAnn: "unix:///tmp/hailo", sharing.DispatcherSchedulerAnn: "unix:///tmp/scheduler", sharing.ArchitectureAnn: "HAILO8", sharing.RuntimeAnn: "4.21.0"}}, Spec: corev1.PodSpec{NodeName: "node-a"}, Status: corev1.PodStatus{Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}}}
}

// Uses authoritative pod reconstruction plus injected fresh observations, without hardware.
type observedRegistry struct{ sharing.PodRegistry }

func (r observedRegistry) Snapshot(ctx context.Context) (states []resource.NPUState, err error) {
	states, err = r.PodRegistry.Snapshot(ctx)
	for i := range states {
		states[i].ObservedAt = time.Now()
		states[i].ObservationWindow = 10
		states[i].BrokerOccupancy = 0
	}
	return
}
func TestBestEffortAdmissionReleaseRestartAndSwitch(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	broker := dispatcher.NewBroker()
	go broker.Run(ctx)
	r, base := fixture(t)
	cfg := bestConfig()
	r.Admission = &cfg
	r.APIReader = r.Client
	if err := r.Create(ctx, dispatcherPod()); err != nil {
		t.Fatal(err)
	}
	r.Registry = observedRegistry{sharing.PodRegistry{Client: r.APIReader}}
	tasks := make([]*api.NPUTask, 4)
	for i := range tasks {
		task := base.DeepCopy()
		task.Name = fmt.Sprintf("best-%d", i)
		task.UID = types.UID(task.Name)
		task.ResourceVersion = ""
		task.Spec.NPUCount = 0
		task.Spec.BestEffort = &api.BestEffortRequest{TargetFPS: 20}
		task.Spec.Model.Key = "model"
		task.Spec.Model.SHA256 = strings.Repeat("a", 64)
		if err := r.Create(ctx, task); err != nil {
			t.Fatal(err)
		}
		reconcile(t, r, task)
		tasks[i] = task
	}
	for _, task := range tasks[:3] {
		p := getPod(t, r, task)
		if task.Status.ReservedDemand != 240 || task.Status.ExecutionWeight != 100 || task.Status.AllocatedShare != 0 {
			t.Fatal(task.Status)
		}
		q := p.Spec.Containers[0].Resources.Limits[api.ResourceName]
		if q.Value() != 0 {
			t.Fatal("shared pod owns hardware")
		}
		// Emulate a cooperative client using the real broker after pod admission.
		id := string(task.UID)
		if err := broker.RegisterReservation(id, int(task.Status.ExecutionWeight), int(task.Status.ReservedDemand), admission.BestEffort); err != nil {
			t.Fatal(err)
		}
		if err := broker.Acquire(ctx, id); err != nil {
			t.Fatal(err)
		}
		if err := broker.Complete(id); err != nil {
			t.Fatal(err)
		}
	}
	recorder := httptest.NewRecorder()
	broker.MetricsHandler(recorder, nil)
	for _, metric := range []string{"npu_share_allocated_total 720", "npu_share_active_workloads 3", "npu_broker_completed_grants_total 3"} {
		if !strings.Contains(recorder.Body.String(), metric) {
			t.Fatal(recorder.Body.String())
		}
	}
	if tasks[3].Status.Conditions[0].Reason != "InsufficientBestEffortCapacity" {
		t.Fatal(tasks[3].Status)
	}
	// Restart with the same authoritative store: reservations survive and still block.
	r = &Reconciler{Client: r.Client, APIReader: r.APIReader, Scheme: r.Scheme, Registry: r.Registry, Admission: &cfg}
	reconcile(t, r, tasks[3])
	if tasks[3].Status.Conditions[0].Reason != "InsufficientBestEffortCapacity" {
		t.Fatal(tasks[3].Status)
	}
	old := getPod(t, r, tasks[0])
	old.Finalizers = []string{"test/hold"}
	if err := r.Update(ctx, old); err != nil {
		t.Fatal(err)
	}
	tasks[0].Spec.Suspend = true
	updateTask(t, r, tasks[0])
	reconcile(t, r, tasks[0])
	reconcile(t, r, tasks[3])
	if tasks[3].Status.Conditions[0].Reason != "InsufficientBestEffortCapacity" {
		t.Fatal("terminating reservation released", tasks[3].Status)
	}
	old = getPod(t, r, tasks[0])
	old.Finalizers = nil
	if err := r.Update(ctx, old); err != nil {
		t.Fatal(err)
	}
	reconcile(t, r, tasks[3])
	getPod(t, r, tasks[3])
	broker.Unregister(string(tasks[0].UID))
	if err := broker.RegisterReservation(string(tasks[3].UID), 100, 240, admission.BestEffort); err != nil {
		t.Fatal(err)
	}
	if err := broker.Acquire(ctx, string(tasks[3].UID)); err != nil {
		t.Fatal(err)
	}
	if err := broker.Complete(string(tasks[3].UID)); err != nil {
		t.Fatal(err)
	}
	// Configuration changes cannot mix with any existing shared assignment.
	cfg.Budget = 700
	reason, err := r.admissionGuard(ctx, cfg)
	if err != nil || reason != "AdmissionConfigurationMismatch" {
		t.Fatal(reason, err)
	}
	for _, task := range tasks[1:] {
		task.Spec.Suspend = true
		updateTask(t, r, task)
		reconcile(t, r, task)
	}
	if reason, err = r.admissionGuard(ctx, cfg); err != nil || reason != "" {
		t.Fatal(reason, err)
	}
	cfg = admission.Default()
	tasks[0].Spec.Suspend = false
	tasks[0].Spec.BestEffort = nil
	tasks[0].Spec.NPUShare = 600
	updateTask(t, r, tasks[0])
	reconcile(t, r, tasks[0])
	getPod(t, r, tasks[0])
}

func TestAdmissionGuardUsesAuthoritativeReader(t *testing.T) {
	r, _ := fixture(t)
	cfg := bestConfig()
	p := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "unseen", Annotations: map[string]string{sharing.AssignedNPUAnn: "missing-dispatcher", sharing.FingerprintAnn: "old-config"}}}
	r.APIReader = fake.NewClientBuilder().WithScheme(r.Scheme).WithObjects(p).Build()
	reason, err := r.admissionGuard(context.Background(), cfg)
	if err != nil || reason != "AdmissionConfigurationMismatch" {
		t.Fatal(reason, err)
	}
}

func TestBestEffortPolicyMismatchAndValidation(t *testing.T) {
	r, task := fixture(t)
	task.Spec.NPUCount = 0
	task.Spec.BestEffort = &api.BestEffortRequest{TargetFPS: 20}
	task.Spec.Model.Key = "model"
	task.Spec.Model.SHA256 = strings.Repeat("a", 64)
	if err := Validate(task.Spec); err != nil {
		t.Fatal(err)
	}
	copy := task.DeepCopy()
	copy.Spec.BestEffort.TargetFPS = 30
	if task.Spec.BestEffort.TargetFPS != 20 {
		t.Fatal("shallow copy")
	}
	updateTask(t, r, task)
	reconcile(t, r, task)
	if task.Status.Conditions[0].Reason != "AdmissionPolicyMismatch" {
		t.Fatal(task.Status)
	}
	task.Spec.NPUCount = 1
	if Validate(task.Spec) == nil {
		t.Fatal("mixed requests accepted")
	}
}
