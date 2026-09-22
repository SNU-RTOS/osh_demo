package resource

import "testing"

func register(t *testing.T, manager *Manager, ids ...string) {
	t.Helper()
	for _, id := range ids {
		if err := manager.RegisterNPU(NPUState{ID: id, Capacity: DefaultCapacity}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestFirstFitAllocateReleaseAndObservation(t *testing.T) {
	manager := NewManager(FirstFitPolicy{})
	register(t, manager, "npu-0", "npu-1")
	requests := []NPURequest{{"a", "yolo", 300}, {"b", "resnet", 300}, {"c", "depth", 400}, {"d", "pose", 400}}
	want := []string{"npu-0", "npu-0", "npu-0", "npu-1"}
	for i, request := range requests {
		allocation, err := manager.Allocate(request)
		if err != nil {
			t.Fatal(err)
		}
		if allocation.NPUID != want[i] {
			t.Fatalf("request %s placed on %s, want %s", request.WorkloadID, allocation.NPUID, want[i])
		}
	}
	if err := manager.UpdateObservation("npu-0", 17); err != nil {
		t.Fatal(err)
	}
	states := manager.ListNPUStates()
	if states[0].Allocated != 1000 || states[0].AvailableCapacity() != 0 || states[0].MeasuredUtilization != 17 {
		t.Fatalf("accounting and observation were not kept separate: %#v", states[0])
	}
	if err := manager.Release("b"); err != nil {
		t.Fatal(err)
	}
	states = manager.ListNPUStates()
	if states[0].Allocated != 700 || states[0].MeasuredUtilization != 17 {
		t.Fatalf("unexpected state after release: %#v", states[0])
	}
}

func TestAdmissionHealthAndIdempotency(t *testing.T) {
	manager := NewManager(nil)
	register(t, manager, "npu-0")
	request := NPURequest{WorkloadID: "a", ModelKey: "model", Share: 700}
	first, err := manager.Allocate(request)
	if err != nil {
		t.Fatal(err)
	}
	second, err := manager.Allocate(request)
	if err != nil || first != second {
		t.Fatalf("idempotent allocation failed: %#v %v", second, err)
	}
	if _, err := manager.Allocate(NPURequest{WorkloadID: "b", Share: 301}); err == nil {
		t.Fatal("over-capacity request was admitted")
	}
	if err := manager.Release("a"); err != nil {
		t.Fatal(err)
	}
	if err := manager.SetHealth("npu-0", HealthUnhealthy); err != nil {
		t.Fatal(err)
	}
	if _, err := manager.Allocate(NPURequest{WorkloadID: "b", Share: 1}); err == nil {
		t.Fatal("unhealthy NPU was selected")
	}
}

func TestSnapshotsCannotMutateLedger(t *testing.T) {
	manager := NewManager(nil)
	register(t, manager, "npu-0")
	if _, err := manager.Allocate(NPURequest{WorkloadID: "a", Share: 100}); err != nil {
		t.Fatal(err)
	}
	states := manager.ListNPUStates()
	states[0].Workloads[0].Share = 999
	if got := manager.ListNPUStates()[0].Workloads[0].Share; got != 100 {
		t.Fatalf("snapshot mutated ledger: %d", got)
	}
}

func TestRestorePreservesPhysicalAssignment(t *testing.T) {
	manager := NewManager(nil)
	register(t, manager, "npu-0", "npu-1")
	if err := manager.RestoreAllocation("npu-1", WorkloadAllocation{WorkloadID: "existing", Share: 800}); err != nil {
		t.Fatal(err)
	}
	allocation, err := manager.Allocate(NPURequest{WorkloadID: "new", Share: 300})
	if err != nil {
		t.Fatal(err)
	}
	if allocation.NPUID != "npu-0" {
		t.Fatalf("selected %q", allocation.NPUID)
	}
}
