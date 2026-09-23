package controller

import (
	"context"
	"time"

	api "github.com/SNU-RTOS/osh_demo/npu-task/api"
	"github.com/prometheus/client_golang/prometheus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type MetricsCollector struct {
	client          client.Client
	pendingDuration *prometheus.Desc
	allocationAge   *prometheus.Desc
	restarts        *prometheus.Desc
	request         *prometheus.Desc
	phase           *prometheus.Desc
}

func NewMetricsCollector(c client.Client) *MetricsCollector {
	labels := []string{"namespace", "task", "npu"}
	return &MetricsCollector{
		client:          c,
		pendingDuration: prometheus.NewDesc("nputask_pending_duration_seconds", "Seconds the current task has continuously remained Pending.", labels, nil),
		allocationAge:   prometheus.NewDesc("nputask_allocation_age_seconds", "Seconds since the current logical NPU allocation was created.", labels, nil),
		restarts:        prometheus.NewDesc("nputask_container_restart_count", "Container restart count for the current task Pod.", labels, nil),
		request:         prometheus.NewDesc("nputask_share", "Requested and allocated logical NPU share.", append(labels, "kind"), nil),
		phase:           prometheus.NewDesc("nputask_phase_info", "Current task phase as a label.", append(labels, "phase", "reason", "broker_epoch"), nil),
	}
}

func (c *MetricsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.pendingDuration
	ch <- c.allocationAge
	ch <- c.restarts
	ch <- c.request
	ch <- c.phase
}

func (c *MetricsCollector) Collect(ch chan<- prometheus.Metric) {
	var tasks api.NPUTaskList
	if err := c.client.List(context.Background(), &tasks); err != nil {
		return
	}
	now := time.Now()
	for i := range tasks.Items {
		task := &tasks.Items[i]
		labels := []string{task.Namespace, task.Name, task.Status.NPUID}
		reason := ""
		condition := meta.FindStatusCondition(task.Status.Conditions, "Ready")
		if condition != nil {
			reason = condition.Reason
		}
		pending := float64(0)
		if task.Status.Phase == "Pending" && condition != nil {
			pending = now.Sub(condition.LastTransitionTime.Time).Seconds()
			if pending < 0 {
				pending = 0
			}
		}
		age := float64(0)
		if task.Status.AllocationTime != nil {
			age = now.Sub(task.Status.AllocationTime.Time).Seconds()
			if age < 0 {
				age = 0
			}
		}
		restarts := float64(0)
		if task.Status.PodName != "" {
			var pod corev1.Pod
			if err := c.client.Get(context.Background(), types.NamespacedName{Namespace: task.Namespace, Name: task.Status.PodName}, &pod); err == nil {
				for _, status := range pod.Status.ContainerStatuses {
					restarts += float64(status.RestartCount)
				}
			}
		}
		ch <- prometheus.MustNewConstMetric(c.pendingDuration, prometheus.GaugeValue, pending, labels...)
		ch <- prometheus.MustNewConstMetric(c.allocationAge, prometheus.GaugeValue, age, labels...)
		ch <- prometheus.MustNewConstMetric(c.restarts, prometheus.GaugeValue, restarts, labels...)
		ch <- prometheus.MustNewConstMetric(c.request, prometheus.GaugeValue, float64(task.Spec.NPUShare), append(labels, "requested")...)
		ch <- prometheus.MustNewConstMetric(c.request, prometheus.GaugeValue, float64(task.Status.AllocatedShare), append(labels, "allocated")...)
		ch <- prometheus.MustNewConstMetric(c.phase, prometheus.GaugeValue, 1, append(labels, task.Status.Phase, reason, task.Status.BrokerEpoch)...)
	}
}

var _ prometheus.Collector = (*MetricsCollector)(nil)
