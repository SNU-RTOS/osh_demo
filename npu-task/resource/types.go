// Package resource implements logical NPU capacity accounting. It deliberately
// does not interpret measured utilization as allocatable capacity.
package resource

const DefaultCapacity = 1000

type Health string

const (
	HealthHealthy   Health = "Healthy"
	HealthUnhealthy Health = "Unhealthy"
)

type NPURequest struct {
	WorkloadID string
	ModelKey   string
	Share      int
}

type WorkloadAllocation struct {
	WorkloadID string
	ModelKey   string
	Share      int
}

type Allocation struct {
	NPUID      string
	WorkloadID string
	ModelKey   string
	Share      int
}

type NPUState struct {
	ID                  string
	NodeName            string
	Endpoint            string
	SchedulerEndpoint   string
	Health              Health
	Capacity            int
	Allocated           int
	MeasuredUtilization float64
	Workloads           []WorkloadAllocation
}

func (s NPUState) AvailableCapacity() int { return s.Capacity - s.Allocated }
