package policy

import (
	"testing"
	"time"
)

func TestRecommendFreshLeastUtilized(t *testing.T) {
	now := time.Unix(100, 0)
	d := []Device{{ID: "hailo2", Utilization: 80, ObservedAt: now}, {ID: "hailo1", Utilization: 10, ObservedAt: now}, {ID: "hailo0", Utilization: 20, ObservedAt: now, Allocated: true}}
	r := Recommend(d, 1, 90, now, time.Minute)
	if !r.Accepted || len(r.Devices) != 1 || r.Devices[0] != "hailo1" {
		t.Fatalf("unexpected decision: %#v", r)
	}
}

func TestRecommendRejectsStaleAndOverThreshold(t *testing.T) {
	now := time.Unix(100, 0)
	d := []Device{{ID: "stale", Utilization: 0, ObservedAt: now.Add(-2 * time.Minute)}, {ID: "busy", Utilization: 91, ObservedAt: now}}
	r := Recommend(d, 1, 90, now, time.Minute)
	if r.Accepted || r.Reason != "insufficient fresh capacity" {
		t.Fatalf("unexpected decision: %#v", r)
	}
}

func TestRecommendRejectsClockSkew(t *testing.T) {
	now := time.Unix(100, 0)
	r := Recommend([]Device{{ID: "future", ObservedAt: now.Add(time.Second)}}, 1, 90, now, time.Minute)
	if r.Accepted {
		t.Fatalf("accepted future sample: %#v", r)
	}
}
