package dispatcher

import (
	"context"
	"io"
	"net"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func brokerCommand(t *testing.T, socket, request string) string {
	t.Helper()
	connection, err := net.Dial("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	if _, err := io.WriteString(connection, request+"\n"); err != nil {
		t.Fatal(err)
	}
	response, err := io.ReadAll(connection)
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(response))
}

func TestBrokerProtocolAndMetrics(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	broker := NewBroker()
	go broker.Run(ctx)
	socket := filepath.Join(t.TempDir(), "broker.sock")
	done := make(chan error, 1)
	go func() { done <- broker.ServeUnix(ctx, socket) }()
	deadline := time.Now().Add(time.Second)
	for {
		connection, err := net.Dial("unix", socket)
		if err == nil {
			connection.Close()
			break
		}
		if time.Now().After(deadline) {
			t.Fatal(err)
		}
		time.Sleep(time.Millisecond)
	}
	if got := brokerCommand(t, socket, "REGISTER workload-a 300"); !strings.HasPrefix(got, "OK ") {
		t.Fatal(got)
	}
	if got := brokerCommand(t, socket, "HEARTBEAT workload-a"); got != "OK" {
		t.Fatal(got)
	}
	if got := brokerCommand(t, socket, "ACQUIRE workload-a"); got != "OK" {
		t.Fatal(got)
	}
	if got := brokerCommand(t, socket, "COMPLETE workload-a"); got != "OK" {
		t.Fatal(got)
	}
	recorder := httptest.NewRecorder()
	broker.MetricsHandler(recorder, httptest.NewRequest("GET", "/metrics", nil))
	body := recorder.Body.String()
	for _, expected := range []string{`npu_share_active_workloads 1`, `npu_share_capacity 1000`, `npu_share_allocated_total 300`, `npu_share_available 700`, `npu_share_broker_info{epoch="`, `npu_share_allocated{workload="workload-a"} 300`, `npu_share_session_age_seconds{workload="workload-a"}`, `npu_share_grants_total{workload="workload-a"} 1`, `npu_share_grant_wait_seconds_count{workload="workload-a"} 1`, `npu_share_grant_wait_seconds_max{workload="workload-a"}`} {
		if !strings.Contains(body, expected) {
			t.Fatalf("metrics missing %q:\n%s", expected, body)
		}
	}
	cancel()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestBrokerExpiresStaleSession(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	broker := newBrokerWithTTL(time.Second, 20*time.Millisecond)
	go broker.Run(ctx)
	if err := broker.Register("abandoned", 400); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		recorder := httptest.NewRecorder()
		broker.MetricsHandler(recorder, httptest.NewRequest("GET", "/metrics", nil))
		if strings.Contains(recorder.Body.String(), "npu_share_active_workloads 0") {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("stale session was not expired")
}

func TestBrokerRecoversExpiredExecutionToken(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	broker := newBroker(10 * time.Millisecond)
	go broker.Run(ctx)
	if err := broker.Register("stuck", 900); err != nil {
		t.Fatal(err)
	}
	if err := broker.Register("next", 100); err != nil {
		t.Fatal(err)
	}
	if err := broker.Acquire(ctx, "stuck"); err != nil {
		t.Fatal(err)
	}
	deadline, stop := context.WithTimeout(ctx, time.Second)
	defer stop()
	if err := broker.Acquire(deadline, "next"); err != nil {
		t.Fatal(err)
	}
}
