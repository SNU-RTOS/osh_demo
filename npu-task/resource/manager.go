package resource

import (
	"errors"
	"fmt"
	"sync"
)

var (
	ErrNPUNotFound      = errors.New("NPU not found")
	ErrWorkloadNotFound = errors.New("workload allocation not found")
)

// Manager is an in-memory, concurrency-safe allocation ledger. Persistence and
// dispatcher liveness belong to the control-plane integration milestone.
type Manager struct {
	mu          sync.RWMutex
	policy      PlacementPolicy
	npus        map[string]*NPUState
	order       []string
	allocations map[string]Allocation
}

func NewManager(policy PlacementPolicy) *Manager {
	if policy == nil {
		policy = FirstFitPolicy{}
	}
	return &Manager{policy: policy, npus: make(map[string]*NPUState), allocations: make(map[string]Allocation)}
}

func (m *Manager) RegisterNPU(state NPUState) error {
	if state.ID == "" {
		return errors.New("NPU ID is required")
	}
	if state.Capacity <= 0 {
		return errors.New("NPU capacity must be positive")
	}
	if state.Allocated != 0 || len(state.Workloads) != 0 {
		return errors.New("new NPU must not contain allocations")
	}
	if state.MeasuredUtilization < 0 || state.MeasuredUtilization > 100 {
		return errors.New("measured utilization must be between 0 and 100")
	}
	if state.Health == "" {
		state.Health = HealthHealthy
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.npus[state.ID]; exists {
		return fmt.Errorf("NPU %q is already registered", state.ID)
	}
	state.Workloads = nil
	m.npus[state.ID] = &state
	m.order = append(m.order, state.ID)
	return nil
}

func (m *Manager) UnregisterNPU(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	npu, ok := m.npus[id]
	if !ok {
		return ErrNPUNotFound
	}
	if len(npu.Workloads) != 0 {
		return fmt.Errorf("NPU %q still has %d allocation(s)", id, len(npu.Workloads))
	}
	delete(m.npus, id)
	for i, candidate := range m.order {
		if candidate == id {
			m.order = append(m.order[:i], m.order[i+1:]...)
			break
		}
	}
	return nil
}

func (m *Manager) Allocate(req NPURequest) (Allocation, error) {
	if req.WorkloadID == "" {
		return Allocation{}, errors.New("workload ID is required")
	}
	if req.Share <= 0 || req.Share > DefaultCapacity {
		return Allocation{}, fmt.Errorf("share must be between 1 and %d", DefaultCapacity)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if existing, ok := m.allocations[req.WorkloadID]; ok {
		if existing.ModelKey == req.ModelKey && existing.Share == req.Share {
			return existing, nil
		}
		return Allocation{}, fmt.Errorf("workload %q already has a different allocation", req.WorkloadID)
	}
	states := m.statesLocked()
	id, err := m.policy.Select(states, req)
	if err != nil {
		return Allocation{}, err
	}
	npu, ok := m.npus[id]
	if !ok {
		return Allocation{}, fmt.Errorf("placement policy selected unknown NPU %q", id)
	}
	if npu.Health != HealthHealthy || npu.AvailableCapacity() < req.Share {
		return Allocation{}, fmt.Errorf("placement policy selected unavailable NPU %q", id)
	}
	workload := WorkloadAllocation{WorkloadID: req.WorkloadID, ModelKey: req.ModelKey, Share: req.Share}
	npu.Workloads = append(npu.Workloads, workload)
	npu.Allocated += req.Share
	allocation := Allocation{NPUID: id, WorkloadID: req.WorkloadID, ModelKey: req.ModelKey, Share: req.Share}
	m.allocations[req.WorkloadID] = allocation
	return allocation, nil
}

// RestoreAllocation reconstructs authoritative assignments after a control
// plane restart. It never runs placement and rejects inconsistent snapshots.
func (m *Manager) RestoreAllocation(npuID string, workload WorkloadAllocation) error {
	if workload.WorkloadID == "" || workload.Share <= 0 {
		return errors.New("restored workload ID and positive share are required")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	npu, ok := m.npus[npuID]
	if !ok {
		return ErrNPUNotFound
	}
	if _, duplicate := m.allocations[workload.WorkloadID]; duplicate {
		return fmt.Errorf("workload %q is assigned more than once", workload.WorkloadID)
	}
	if npu.Allocated+workload.Share > npu.Capacity {
		return fmt.Errorf("restored allocation exceeds NPU %q capacity", npuID)
	}
	npu.Workloads = append(npu.Workloads, workload)
	npu.Allocated += workload.Share
	m.allocations[workload.WorkloadID] = Allocation{NPUID: npuID, WorkloadID: workload.WorkloadID, ModelKey: workload.ModelKey, Share: workload.Share}
	return nil
}

func (m *Manager) Release(workloadID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	allocation, ok := m.allocations[workloadID]
	if !ok {
		return ErrWorkloadNotFound
	}
	npu := m.npus[allocation.NPUID]
	for i, workload := range npu.Workloads {
		if workload.WorkloadID == workloadID {
			npu.Workloads = append(npu.Workloads[:i], npu.Workloads[i+1:]...)
			npu.Allocated -= workload.Share
			break
		}
	}
	delete(m.allocations, workloadID)
	return nil
}

func (m *Manager) UpdateObservation(id string, utilization float64) error {
	if utilization < 0 || utilization > 100 {
		return errors.New("measured utilization must be between 0 and 100")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	npu, ok := m.npus[id]
	if !ok {
		return ErrNPUNotFound
	}
	npu.MeasuredUtilization = utilization
	return nil
}

func (m *Manager) SetHealth(id string, health Health) error {
	if health != HealthHealthy && health != HealthUnhealthy {
		return errors.New("invalid NPU health")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	npu, ok := m.npus[id]
	if !ok {
		return ErrNPUNotFound
	}
	npu.Health = health
	return nil
}

func (m *Manager) ListNPUStates() []NPUState {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.statesLocked()
}

func (m *Manager) statesLocked() []NPUState {
	states := make([]NPUState, 0, len(m.order))
	for _, id := range m.order {
		state := *m.npus[id]
		state.Workloads = append([]WorkloadAllocation(nil), state.Workloads...)
		states = append(states, state)
	}
	return states
}
