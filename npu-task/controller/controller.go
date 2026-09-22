package controller

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"path"
	"reflect"
	"strings"
	"time"

	api "github.com/SNU-RTOS/osh_demo/npu-task/api"
	"github.com/SNU-RTOS/osh_demo/npu-task/resource"
	"github.com/SNU-RTOS/osh_demo/npu-task/sharing"
	corev1 "k8s.io/api/core/v1"
	errors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	kresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const hashKey = "npu.snu-rtos.io/execution-hash"

type Reconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Registry sharing.Registry
}

func Validate(s api.NPUTaskSpec) error {
	if (s.NPUCount == 0) == (s.NPUShare == 0) {
		return fmt.Errorf("exactly one of npuCount or npuShare is required")
	}
	if s.NPUCount < 0 || s.NPUCount > 8 {
		return fmt.Errorf("npuCount must be between 1 and 8 when used")
	}
	if s.NPUShare < 0 || s.NPUShare > resource.DefaultCapacity {
		return fmt.Errorf("npuShare must be between 1 and %d when used", resource.DefaultCapacity)
	}
	if s.Image == "" || len(s.Command) == 0 || s.Command[0] == "" {
		return fmt.Errorf("image and command are required")
	}
	if s.Mode != "" && s.Mode != "Batch" && s.Mode != "Service" {
		return fmt.Errorf("mode must be Batch or Service")
	}
	if !path.IsAbs(s.Model.Path) {
		return fmt.Errorf("model.path must be absolute")
	}
	if s.Model.PVC != "" && (!path.IsAbs(s.Model.MountPath) || s.Model.MountPath == "/" || !strings.HasPrefix(path.Clean(s.Model.Path), path.Clean(s.Model.MountPath)+"/")) {
		return fmt.Errorf("PVC model.path must be inside an absolute, non-root model.mountPath")
	}
	for _, e := range s.Env {
		if e.Name == "NPU_COUNT" || e.Name == "NPU_SHARE" || e.Name == "MODEL_PATH" || e.Name == "HAILORT_SERVICE_ADDRESS" || e.Name == "NPU_SCHEDULER_ADDRESS" || e.Name == "NPU_WORKLOAD_ID" {
			return fmt.Errorf("%s is reserved", e.Name)
		}
	}
	for _, list := range []corev1.ResourceList{s.Resources.Requests, s.Resources.Limits} {
		for key := range list {
			if key != corev1.ResourceCPU && key != corev1.ResourceMemory {
				return fmt.Errorf("resources only accepts cpu and memory")
			}
		}
	}
	if len(s.Resources.Claims) != 0 {
		return fmt.Errorf("resource claims are not supported")
	}
	return nil
}

func executionHash(s api.NPUTaskSpec) string {
	s.Suspend = false
	if s.Mode == "" {
		s.Mode = "Batch"
	}
	b, _ := json.Marshal(s)
	return fmt.Sprintf("%x", sha256.Sum256(b))
}

func podName(t *api.NPUTask) string { return "nputask-" + string(t.UID) }

func desiredPod(t *api.NPUTask, scheme *runtime.Scheme, allocation *resource.Allocation, state *resource.NPUState) (*corev1.Pod, error) {
	resources := *t.Spec.Resources.DeepCopy()
	if resources.Requests == nil {
		resources.Requests = corev1.ResourceList{}
	}
	if resources.Limits == nil {
		resources.Limits = corev1.ResourceList{}
	}
	env := append([]corev1.EnvVar(nil), t.Spec.Env...)
	env = append(env, corev1.EnvVar{Name: "MODEL_PATH", Value: t.Spec.Model.Path})
	annotations := map[string]string{hashKey: executionHash(t.Spec)}
	if t.Spec.NPUShare == 0 {
		q := *kresource.NewQuantity(int64(t.Spec.NPUCount), kresource.DecimalSI)
		resources.Requests[api.ResourceName] = q
		resources.Limits[api.ResourceName] = q
		env = append(env, corev1.EnvVar{Name: "NPU_COUNT", Value: fmt.Sprint(t.Spec.NPUCount)})
	} else {
		if allocation == nil || state == nil {
			return nil, fmt.Errorf("shared task requires an allocation")
		}
		annotations[sharing.AssignedNPUAnn] = allocation.NPUID
		annotations[sharing.AllocatedShareAnn] = fmt.Sprint(allocation.Share)
		annotations[sharing.WorkloadIDAnn] = allocation.WorkloadID
		annotations[sharing.ModelKeyAnn] = allocation.ModelKey
		env = append(env,
			corev1.EnvVar{Name: "NPU_SHARE", Value: fmt.Sprint(allocation.Share)},
			corev1.EnvVar{Name: "HAILORT_SERVICE_ADDRESS", Value: state.Endpoint},
			corev1.EnvVar{Name: "NPU_SCHEDULER_ADDRESS", Value: state.SchedulerEndpoint},
			corev1.EnvVar{Name: "NPU_WORKLOAD_ID", Value: allocation.WorkloadID})
	}
	grace := int64(30)
	automount := false
	p := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: podName(t), Namespace: t.Namespace,
		Labels:      map[string]string{"npu.snu-rtos.io/task-uid": string(t.UID)},
		Annotations: annotations},
		Spec: corev1.PodSpec{RestartPolicy: corev1.RestartPolicyNever, TerminationGracePeriodSeconds: &grace, AutomountServiceAccountToken: &automount,
			Containers: []corev1.Container{{Name: "inference", Image: t.Spec.Image, ImagePullPolicy: corev1.PullIfNotPresent,
				Command: append([]string(nil), t.Spec.Command...), Args: append([]string(nil), t.Spec.Args...), Env: env, Resources: resources}}}}
	if t.Spec.Mode == "Service" {
		p.Spec.RestartPolicy = corev1.RestartPolicyAlways
	}
	if t.Spec.Model.PVC != "" {
		p.Spec.Volumes = []corev1.Volume{{Name: "model", VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: t.Spec.Model.PVC, ReadOnly: true}}}}
		p.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{{Name: "model", MountPath: t.Spec.Model.MountPath, ReadOnly: true}}
	}
	if t.Spec.NPUShare > 0 {
		p.Spec.NodeName = state.NodeName
		p.Spec.Volumes = append(p.Spec.Volumes,
			corev1.Volume{Name: "dispatcher-uds", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/var/lib/sano/share/uds", Type: hostPathTypePtr(corev1.HostPathDirectoryOrCreate)}}},
			corev1.Volume{Name: "dispatcher-shm", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/var/lib/sano/share/shm", Type: hostPathTypePtr(corev1.HostPathDirectoryOrCreate)}}})
		p.Spec.Containers[0].VolumeMounts = append(p.Spec.Containers[0].VolumeMounts,
			corev1.VolumeMount{Name: "dispatcher-uds", MountPath: "/tmp"},
			corev1.VolumeMount{Name: "dispatcher-shm", MountPath: "/dev/shm"})
	}
	for _, e := range env {
		if e.Name == "HAILO_MONITOR" && e.Value == "1" {
			p.Spec.Volumes = append(p.Spec.Volumes, corev1.Volume{Name: "hailo-monitor", VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/tmp/hmon_files", Type: hostPathTypePtr(corev1.HostPathDirectoryOrCreate)}}})
			p.Spec.Containers[0].VolumeMounts = append(p.Spec.Containers[0].VolumeMounts, corev1.VolumeMount{Name: "hailo-monitor", MountPath: "/tmp/hmon_files"})
			break
		}
	}
	return p, ctrl.SetControllerReference(t, p, scheme)
}

func hostPathTypePtr(v corev1.HostPathType) *corev1.HostPathType { return &v }

func (r *Reconciler) status(ctx context.Context, t *api.NPUTask, phase, reason, message string, p *corev1.Pod, hash string) error {
	before := t.DeepCopy()
	t.Status.ObservedGeneration = t.Generation
	t.Status.Phase = phase
	t.Status.ExecutionHash = hash
	t.Status.PodName = ""
	t.Status.NodeName = ""
	t.Status.NPUID = ""
	t.Status.DispatcherEndpoint = ""
	t.Status.AllocatedShare = 0
	if p != nil {
		t.Status.PodName = p.Name
		t.Status.NodeName = p.Spec.NodeName
		if t.Spec.NPUShare > 0 {
			t.Status.NPUID = p.Annotations[sharing.AssignedNPUAnn]
			t.Status.AllocatedShare = t.Spec.NPUShare
			for _, env := range p.Spec.Containers[0].Env {
				if env.Name == "HAILORT_SERVICE_ADDRESS" {
					t.Status.DispatcherEndpoint = env.Value
				}
			}
		}
	}
	ready := metav1.ConditionFalse
	if phase == "Running" {
		ready = metav1.ConditionTrue
	}
	meta.SetStatusCondition(&t.Status.Conditions, metav1.Condition{Type: "Ready", Status: ready, Reason: reason, Message: message, ObservedGeneration: t.Generation})
	if reflect.DeepEqual(before.Status, t.Status) {
		return nil
	}
	return r.Status().Patch(ctx, t, client.MergeFrom(before))
}

func (r *Reconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	t := &api.NPUTask{}
	if err := r.Get(ctx, req.NamespacedName, t); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !t.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}
	hash := executionHash(t.Spec)
	p := &corev1.Pod{}
	err := r.Get(ctx, types.NamespacedName{Namespace: t.Namespace, Name: podName(t)}, p)
	exists := err == nil
	if err != nil && !errors.IsNotFound(err) {
		return ctrl.Result{}, err
	}
	if exists && !metav1.IsControlledBy(p, t) {
		return ctrl.Result{}, r.status(ctx, t, "Failed", "OwnershipConflict", "Pod name is occupied by an unrelated object", nil, hash)
	}
	// Suspension must remain possible even if an execution setting is invalid.
	validationErr := Validate(t.Spec)
	if t.Spec.Suspend || (validationErr == nil && exists && p.Annotations[hashKey] != hash) {
		if exists {
			if p.DeletionTimestamp.IsZero() {
				uid := p.UID
				if err := r.Delete(ctx, p, &client.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid}}); client.IgnoreNotFound(err) != nil {
					return ctrl.Result{}, err
				}
			}
			return ctrl.Result{RequeueAfter: time.Second}, r.status(ctx, t, "Stopping", "WaitingForTermination", "Waiting for the previous pod to terminate before releasing or replacing it", p, hash)
		}
		if t.Spec.Suspend {
			return ctrl.Result{}, r.status(ctx, t, "Suspended", "UserSuspended", "No task pod is running", nil, "")
		}
	}
	if validationErr != nil {
		return ctrl.Result{}, r.status(ctx, t, "Failed", "InvalidSpec", validationErr.Error(), nil, hash)
	}
	if !exists {
		terminal := meta.FindStatusCondition(t.Status.Conditions, "Ready")
		if t.Spec.Mode != "Service" && t.Status.ExecutionHash == hash && terminal != nil && (terminal.Reason == "Completed" || terminal.Reason == "PodFailed") {
			return ctrl.Result{}, nil
		}
		var allocation *resource.Allocation
		var selected *resource.NPUState
		if t.Spec.NPUShare > 0 {
			if r.Registry == nil {
				return ctrl.Result{RequeueAfter: time.Second}, r.status(ctx, t, "Pending", "NoDispatcherRegistry", "Shared allocation registry is not configured", nil, hash)
			}
			states, snapshotErr := r.Registry.Snapshot(ctx)
			if snapshotErr != nil {
				return ctrl.Result{}, snapshotErr
			}
			manager := resource.NewManager(resource.FirstFitPolicy{})
			for _, state := range states {
				base := state
				base.Allocated, base.Workloads = 0, nil
				if err := manager.RegisterNPU(base); err != nil {
					return ctrl.Result{}, err
				}
				for _, workload := range state.Workloads {
					if err := manager.RestoreAllocation(state.ID, workload); err != nil {
						return ctrl.Result{}, err
					}
				}
			}
			modelKey := t.Spec.Model.Key
			if modelKey == "" {
				modelKey = path.Base(t.Spec.Model.Path)
			}
			placed, allocateErr := manager.Allocate(resource.NPURequest{WorkloadID: string(t.UID), ModelKey: modelKey, Share: int(t.Spec.NPUShare)})
			if allocateErr != nil {
				return ctrl.Result{RequeueAfter: time.Second}, r.status(ctx, t, "Pending", "InsufficientNPUShare", allocateErr.Error(), nil, hash)
			}
			allocation = &placed
			for _, state := range states {
				if state.ID == placed.NPUID {
					copy := state
					selected = &copy
					break
				}
			}
		}
		p, err = desiredPod(t, r.Scheme, allocation, selected)
		if err != nil {
			return ctrl.Result{}, err
		}
		if err = r.Create(ctx, p); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, r.status(ctx, t, "Pending", "PodCreated", "Waiting for scheduling and inference startup", p, hash)
	}
	if !p.DeletionTimestamp.IsZero() {
		return ctrl.Result{RequeueAfter: time.Second}, r.status(ctx, t, "Stopping", "WaitingForTermination", "Task pod is terminating", p, hash)
	}
	if t.Spec.Mode == "Service" && (p.Status.Phase == corev1.PodFailed || p.Status.Phase == corev1.PodSucceeded) {
		uid := p.UID
		if err := r.Delete(ctx, p, &client.DeleteOptions{Preconditions: &metav1.Preconditions{UID: &uid}}); client.IgnoreNotFound(err) != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{RequeueAfter: time.Second}, r.status(ctx, t, "Stopping", "ServiceRestart", "Replacing a terminal service pod", p, hash)
	}
	phase, reason, message := "Pending", "Starting", "Waiting for inference container"
	switch p.Status.Phase {
	case corev1.PodSucceeded:
		phase, reason, message = "Succeeded", "Completed", "Inference completed successfully"
	case corev1.PodFailed:
		phase, reason, message = "Failed", "PodFailed", p.Status.Message
		for _, c := range p.Status.ContainerStatuses {
			if c.State.Terminated != nil {
				exit := c.State.Terminated
				message = fmt.Sprintf("Container %s exited %d (%s): %s", c.Name, exit.ExitCode, exit.Reason, exit.Message)
			}
		}
	default:
		for _, c := range p.Status.Conditions {
			if c.Type == corev1.PodScheduled && c.Status == corev1.ConditionFalse {
				reason, message = c.Reason, c.Message
			}
		}
		for _, c := range p.Status.ContainerStatuses {
			if c.State.Waiting != nil {
				reason, message = c.State.Waiting.Reason, c.State.Waiting.Message
			}
			if c.State.Running != nil && c.Ready {
				phase, reason, message = "Running", "ContainerRunning", "Inference process is running; application readiness is driver-specific"
			}
		}
	}
	return ctrl.Result{}, r.status(ctx, t, phase, reason, message, p, hash)
}

func (r *Reconciler) SetupWithManager(m ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(m).For(&api.NPUTask{}).Owns(&corev1.Pod{}).Complete(r)
}
