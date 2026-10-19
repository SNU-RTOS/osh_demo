package admission

import (
	"time"

	"github.com/SNU-RTOS/osh_demo/npu-task/resource"
)

type Request struct {
	WorkloadID, ModelKey, ModelSHA256 string
	Share                             int
	TargetFPS                         float64
}

// Policy separates admission/reservation from the broker's execution scheduler.
type Policy interface {
	Reserve([]resource.NPUState, Request, time.Time) (resource.Allocation, resource.NPUState, string, error)
}
type StaticPolicy struct{}
type UtilizationPolicy struct{ Config Config }

func (c Config) Policy() Policy {
	if c.SharedAdmissionPolicy == BestEffort {
		return UtilizationPolicy{c}
	}
	return StaticPolicy{}
}
func (StaticPolicy) Reserve(states []resource.NPUState, req Request, _ time.Time) (resource.Allocation, resource.NPUState, string, error) {
	manager := resource.NewManager(resource.FirstFitPolicy{})
	for _, n := range states {
		base := n
		base.Allocated = 0
		base.Workloads = nil
		if err := manager.RegisterNPU(base); err != nil {
			return resource.Allocation{}, resource.NPUState{}, "", err
		}
		for _, w := range n.Workloads {
			if err := manager.RestoreAllocation(n.ID, w); err != nil {
				return resource.Allocation{}, resource.NPUState{}, "", err
			}
		}
	}
	allocation, err := manager.Allocate(resource.NPURequest{WorkloadID: req.WorkloadID, ModelKey: req.ModelKey, Share: req.Share})
	if err != nil {
		return allocation, resource.NPUState{}, "InsufficientNPUShare", nil
	}
	for _, n := range states {
		if n.ID == allocation.NPUID {
			return allocation, n, "", nil
		}
	}
	return allocation, resource.NPUState{}, "InsufficientNPUShare", nil
}
func (p UtilizationPolicy) Reserve(states []resource.NPUState, req Request, now time.Time) (resource.Allocation, resource.NPUState, string, error) {
	demand, profile, reason := p.Config.Demand(req.ModelKey, req.ModelSHA256, req.TargetFPS)
	if reason != "" {
		return resource.Allocation{}, resource.NPUState{}, reason, nil
	}
	state, reason := p.Config.Select(states, profile, demand, now)
	return resource.Allocation{NPUID: state.ID, WorkloadID: req.WorkloadID, ModelKey: req.ModelKey, Share: demand}, state, reason, nil
}
