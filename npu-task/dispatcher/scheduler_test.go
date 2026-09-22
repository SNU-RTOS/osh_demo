package dispatcher

import "testing"

func TestWeightedRoundRobinRatio(t *testing.T) {
	scheduler := NewWeightedScheduler()
	shares := map[string]int{"a": 300, "b": 500, "c": 200}
	for _, id := range []string{"a", "b", "c"} {
		if err := scheduler.Register(id, shares[id]); err != nil {
			t.Fatal(err)
		}
		for i := 0; i < shares[id]/100; i++ {
			if err := scheduler.Enqueue(id); err != nil {
				t.Fatal(err)
			}
		}
	}
	counts := map[string]int{}
	for i := 0; i < 10; i++ {
		id, ok := scheduler.Next()
		if !ok {
			t.Fatalf("queue empty after %d selections", i)
		}
		counts[id]++
	}
	if counts["a"] != 3 || counts["b"] != 5 || counts["c"] != 2 {
		t.Fatalf("unexpected ratio: %#v", counts)
	}
	if _, ok := scheduler.Next(); ok {
		t.Fatal("scheduler selected an empty workload")
	}
}

func TestOnlyReadyWorkloadsAreSelected(t *testing.T) {
	scheduler := NewWeightedScheduler()
	_ = scheduler.Register("high", 900)
	_ = scheduler.Register("ready", 100)
	_ = scheduler.Enqueue("ready")
	if id, ok := scheduler.Next(); !ok || id != "ready" {
		t.Fatalf("selected %q, %v", id, ok)
	}
}
