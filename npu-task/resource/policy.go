package resource

import "fmt"

type PlacementPolicy interface {
	Select(npus []NPUState, req NPURequest) (string, error)
}

// FirstFitPolicy preserves the order supplied by the resource manager.
type FirstFitPolicy struct{}

func (FirstFitPolicy) Select(npus []NPUState, req NPURequest) (string, error) {
	for _, npu := range npus {
		if npu.Health == HealthHealthy && npu.AvailableCapacity() >= req.Share {
			return npu.ID, nil
		}
	}
	return "", fmt.Errorf("insufficient NPU capacity for share %d", req.Share)
}
