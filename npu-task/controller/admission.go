package controller

import (
	"context"
	"github.com/SNU-RTOS/osh_demo/npu-task/admission"
	"github.com/SNU-RTOS/osh_demo/npu-task/sharing"
	corev1 "k8s.io/api/core/v1"
)

// Check every assigned pod, including assignments whose dispatcher is absent.
// Unannotated allocations from earlier releases are compatible only with static.
func (r *Reconciler) admissionGuard(ctx context.Context, cfg admission.Config) (string, error) {
	reader := r.APIReader
	if reader == nil {
		reader = r.Client
	}
	var pods corev1.PodList
	if err := reader.List(ctx, &pods); err != nil {
		return "", err
	}
	for _, p := range pods.Items {
		if p.Annotations[sharing.AssignedNPUAnn] == "" {
			continue
		}
		if p.DeletionTimestamp == nil && (p.Status.Phase == corev1.PodSucceeded || p.Status.Phase == corev1.PodFailed) {
			continue
		}
		fingerprint := p.Annotations[sharing.FingerprintAnn]
		if fingerprint == "" && cfg.SharedAdmissionPolicy == admission.Static {
			continue
		}
		if fingerprint != cfg.Fingerprint() {
			return "AdmissionConfigurationMismatch", nil
		}
	}
	return "", nil
}
