// Package policy contains the safe, measurement-only part of utilization-aware
// admission. It never revokes an allocation and does not replace kube-scheduler.
package policy

import (
	"sort"
	"time"
)

type Device struct {
	ID          string
	Utilization float64
	ObservedAt  time.Time
	Allocated   bool
}

type Decision struct {
	Accepted bool
	Reason   string
	Devices  []string
}

// Recommend chooses least-utilized fresh, unallocated devices. It is advisory;
// callers must still obtain the authoritative Kubernetes allocation.
func Recommend(devices []Device, count int, maxUtilization float64, now time.Time, maxAge time.Duration) Decision {
	if count < 1 {
		return Decision{Reason: "invalid device count"}
	}
	if maxUtilization < 0 || maxUtilization > 100 {
		return Decision{Reason: "invalid utilization threshold"}
	}
	candidates := make([]Device, 0, len(devices))
	for _, d := range devices {
		if d.ID == "" || d.Allocated || d.ObservedAt.IsZero() || now.Sub(d.ObservedAt) > maxAge || now.Before(d.ObservedAt) || d.Utilization > maxUtilization {
			continue
		}
		candidates = append(candidates, d)
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].Utilization == candidates[j].Utilization {
			return candidates[i].ID < candidates[j].ID
		}
		return candidates[i].Utilization < candidates[j].Utilization
	})
	if len(candidates) < count {
		return Decision{Reason: "insufficient fresh capacity"}
	}
	r := Decision{Accepted: true, Reason: "fresh capacity available"}
	for _, d := range candidates[:count] {
		r.Devices = append(r.Devices, d.ID)
	}
	return r
}
