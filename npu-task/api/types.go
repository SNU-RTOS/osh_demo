package api

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var GroupVersion = schema.GroupVersion{Group: "npu.snu-rtos.io", Version: "v1alpha1"}

const ResourceName corev1.ResourceName = "hailo.ai/npu"

type Model struct {
	Path      string `json:"path"`
	Key       string `json:"key,omitempty"`
	PVC       string `json:"pvc,omitempty"`
	MountPath string `json:"mountPath,omitempty"`
}

type NPUTaskSpec struct {
	Image     string                      `json:"image"`
	Command   []string                    `json:"command"`
	Args      []string                    `json:"args,omitempty"`
	Env       []corev1.EnvVar             `json:"env,omitempty"`
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`
	Model     Model                       `json:"model"`
	NPUCount  int32                       `json:"npuCount,omitempty"`
	NPUShare  int32                       `json:"npuShare,omitempty"`
	Suspend   bool                        `json:"suspend,omitempty"`
	Mode      string                      `json:"mode,omitempty"`
}

type NPUTaskStatus struct {
	ObservedGeneration int64  `json:"observedGeneration,omitempty"`
	PodName            string `json:"podName,omitempty"`
	NodeName           string `json:"nodeName,omitempty"`
	NPUID              string `json:"npuID,omitempty"`
	DispatcherEndpoint string `json:"dispatcherEndpoint,omitempty"`
	AllocatedShare     int32  `json:"allocatedShare,omitempty"`
	Phase              string `json:"phase,omitempty"`
	// ExecutionHash makes terminal batch results durable across pod garbage collection.
	ExecutionHash string             `json:"executionHash,omitempty"`
	Conditions    []metav1.Condition `json:"conditions,omitempty"`
}

type NPUTask struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              NPUTaskSpec   `json:"spec"`
	Status            NPUTaskStatus `json:"status,omitempty"`
}

type NPUTaskList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []NPUTask `json:"items"`
}

func (in *NPUTask) DeepCopy() *NPUTask {
	if in == nil {
		return nil
	}
	out := new(NPUTask)
	*out = *in
	in.ObjectMeta.DeepCopyInto(&out.ObjectMeta)
	out.Spec.Command = append([]string(nil), in.Spec.Command...)
	out.Spec.Args = append([]string(nil), in.Spec.Args...)
	out.Spec.Env = make([]corev1.EnvVar, len(in.Spec.Env))
	for i := range in.Spec.Env {
		in.Spec.Env[i].DeepCopyInto(&out.Spec.Env[i])
	}
	in.Spec.Resources.DeepCopyInto(&out.Spec.Resources)
	out.Status.Conditions = append([]metav1.Condition(nil), in.Status.Conditions...)
	return out
}
func (in *NPUTask) DeepCopyObject() runtime.Object { return in.DeepCopy() }
func (in *NPUTaskList) DeepCopyObject() runtime.Object {
	out := new(NPUTaskList)
	*out = *in
	in.ListMeta.DeepCopyInto(&out.ListMeta)
	out.Items = make([]NPUTask, len(in.Items))
	for i := range in.Items {
		out.Items[i] = *in.Items[i].DeepCopy()
	}
	return out
}
func AddToScheme(s *runtime.Scheme) error {
	s.AddKnownTypes(GroupVersion, &NPUTask{}, &NPUTaskList{})
	metav1.AddToGroupVersion(s, GroupVersion)
	return nil
}
