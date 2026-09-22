// Package dispatcher contains device-local scheduling primitives. Transport,
// model lifecycle, and HailoRT execution are intentionally outside this package.
package dispatcher

import (
	"errors"
	"sync"
)

type workload struct {
	share   int
	current int
	queued  int
}

type QueueState struct {
	WorkloadID string
	Share      int
	Queued     int
}

// WeightedScheduler uses smooth weighted round robin over workloads that have
// queued work. One call to Next consumes one queued execution opportunity.
type WeightedScheduler struct {
	mu        sync.Mutex
	order     []string
	workloads map[string]*workload
}

func NewWeightedScheduler() *WeightedScheduler {
	return &WeightedScheduler{workloads: make(map[string]*workload)}
}

func (s *WeightedScheduler) Register(workloadID string, share int) error {
	if workloadID == "" || share <= 0 {
		return errors.New("workload ID and positive share are required")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.workloads[workloadID]; exists {
		return errors.New("workload is already registered")
	}
	s.workloads[workloadID] = &workload{share: share}
	s.order = append(s.order, workloadID)
	return nil
}

func (s *WeightedScheduler) Unregister(workloadID string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.workloads, workloadID)
	for i, id := range s.order {
		if id == workloadID {
			s.order = append(s.order[:i], s.order[i+1:]...)
			return
		}
	}
}

func (s *WeightedScheduler) Enqueue(workloadID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	w, ok := s.workloads[workloadID]
	if !ok {
		return errors.New("workload is not registered")
	}
	w.queued++
	return nil
}

func (s *WeightedScheduler) Next() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	total := 0
	selected := ""
	for _, id := range s.order {
		w := s.workloads[id]
		if w.queued == 0 {
			continue
		}
		total += w.share
		w.current += w.share
		if selected == "" || w.current > s.workloads[selected].current {
			selected = id
		}
	}
	if selected == "" {
		return "", false
	}
	w := s.workloads[selected]
	w.current -= total
	w.queued--
	return selected, true
}

func (s *WeightedScheduler) QueueStates() []QueueState {
	s.mu.Lock()
	defer s.mu.Unlock()
	states := make([]QueueState, 0, len(s.order))
	for _, id := range s.order {
		w := s.workloads[id]
		states = append(states, QueueState{WorkloadID: id, Share: w.share, Queued: w.queued})
	}
	return states
}
