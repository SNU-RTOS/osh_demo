package admission

import (
	"github.com/SNU-RTOS/osh_demo/npu-task/resource"
	"math"
	"strings"
	"testing"
	"time"
)

func testConfig() Config {
	c := Default()
	c.SharedAdmissionPolicy = BestEffort
	c.Profiles = map[string]Profile{"model": {ModelSHA256: strings.Repeat("a", 64), DeviceArchitecture: "HAILO8", RuntimeVersion: "4.21.0", MeanSeconds: .009, P95Seconds: .01, Samples: 500}}
	return c
}
func TestDemandAndConfiguration(t *testing.T) {
	c := testConfig()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	n, _, reason := c.Demand("model", strings.Repeat("a", 64), 20)
	if n != 240 || reason != "" {
		t.Fatalf("demand=%d reason=%s", n, reason)
	}
	for _, tc := range []struct {
		key, checksum string
		fps           float64
		reason        string
	}{
		{"absent", "", 20, "MissingModelProfile"}, {"model", "bad", 20, "IncompatibleModelProfile"},
		{"model", strings.Repeat("a", 64), 100, "DemandExceedsBudget"}, {"model", strings.Repeat("a", 64), math.Inf(1), "DemandExceedsBudget"},
	} {
		_, _, got := c.Demand(tc.key, tc.checksum, tc.fps)
		if got != tc.reason {
			t.Fatal(got, tc)
		}
	}
	other := c
	other.Budget = 700
	if other.Fingerprint() == c.Fingerprint() {
		t.Fatal("config changes must change fingerprint")
	}
	c.SharedAdmissionPolicy = "typo"
	if c.Validate() == nil {
		t.Fatal("unknown policy accepted")
	}
}
func TestBestEffortPlacementAndFreshness(t *testing.T) {
	c := testConfig()
	p := c.Profiles["model"]
	now := time.Now()
	base := resource.NPUState{ID: "b", Health: resource.HealthHealthy, Capacity: 1000, DeviceArchitecture: p.DeviceArchitecture, RuntimeVersion: p.RuntimeVersion, ObservedAt: now, ObservationWindow: 10, Allocated: 240, BrokerOccupancy: 100}
	a := base
	a.ID = "a"
	selected, reason := c.Select([]resource.NPUState{base, a}, p, 240, now)
	if reason != "" || selected.ID != "a" {
		t.Fatal(selected, reason)
	}
	a.BrokerOccupancy = 700
	selected, reason = c.Select([]resource.NPUState{a, base}, p, 240, now)
	if selected.ID != "b" || reason != "" {
		t.Fatal(selected, reason)
	}
	base.Allocated = 720
	if _, reason = c.Select([]resource.NPUState{base}, p, 240, now); reason != "InsufficientBestEffortCapacity" {
		t.Fatal(reason)
	}
	for _, change := range []func(*resource.NPUState){
		func(n *resource.NPUState) { n.ObservedAt = now.Add(-11 * time.Second) },
		func(n *resource.NPUState) { n.ObservedAt = now.Add(time.Second) },
		func(n *resource.NPUState) { n.ObservationWindow = 9 },
		func(n *resource.NPUState) { n.BrokerOccupancy = math.NaN() },
	} {
		n := base
		change(&n)
		if _, r := c.Select([]resource.NPUState{n}, p, 1, now); r != "StaleBrokerObservation" {
			t.Fatal(r)
		}
	}
	base.RuntimeVersion = "different"
	if _, r := c.Select([]resource.NPUState{base}, p, 1, now); r != "IncompatibleDispatcher" {
		t.Fatal(r)
	}
}
