package dispatcher

import (
	"math"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRollingOccupancyIncludingInFlight(t *testing.T) {
	now := time.Unix(100, 0)
	b := NewBroker()
	b.clock = func() time.Time { return now }
	b.started = now
	units, window := b.occupancyLocked(now)
	if units != 0 || window != 0 {
		t.Fatal(units, window)
	}
	now = now.Add(5 * time.Second)
	b.busy = true
	b.busySince = now
	now = now.Add(5 * time.Second)
	units, window = b.occupancyLocked(now)
	if units != 500 || window != 10 {
		t.Fatal(units, window)
	}
	b.finishLocked(true)
	if b.completed != 1 {
		t.Fatal(b.completed)
	}
	now = now.Add(5 * time.Second)
	units, _ = b.occupancyLocked(now)
	if units != 500 {
		t.Fatal(units)
	}
	now = now.Add(5 * time.Second)
	units, _ = b.occupancyLocked(now)
	if units != 0 || len(b.intervals) != 0 {
		t.Fatal(units, b.intervals)
	}
	b.busy = true
	b.busySince = now
	now = now.Add(15 * time.Second)
	units, _ = b.occupancyLocked(now)
	if math.Abs(units-1000) > .001 {
		t.Fatal(units)
	}
	b.finishLocked(false)
	if b.completed != 1 {
		t.Fatal("expiry counted as completion")
	}
}
func TestReservationMetricsSeparateWeight(t *testing.T) {
	b := NewBroker()
	if err := b.RegisterReservation("client", 100, 240, "utilization-best-effort"); err != nil {
		t.Fatal(err)
	}
	if err := b.RegisterReservation("client", 100, 300, "utilization-best-effort"); err == nil {
		t.Fatal("reservation changed silently")
	}
	w := httptest.NewRecorder()
	b.MetricsHandler(w, nil)
	for _, value := range []string{"npu_share_allocated_total 240", `npu_share_execution_weight{workload="client"} 100`, "npu_broker_observation_window_seconds"} {
		if !strings.Contains(w.Body.String(), value) {
			t.Fatal(w.Body.String())
		}
	}
}
