package dispatcher

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Broker turns concurrent ACQUIRE requests into execution grants selected by
// the weighted scheduler. Clients acquire one grant before each HailoRT call.
type Broker struct {
	mu         sync.Mutex
	scheduler  *WeightedScheduler
	waiters    map[string][]chan error
	grants     map[string]uint64
	sessions   map[string]session
	epoch      string
	wake       chan struct{}
	busy       bool
	inFlight   string
	token      uint64
	tokenTTL   time.Duration
	sessionTTL time.Duration
}

type session struct {
	share    int
	lastSeen time.Time
}

const logicalCapacity = 1000

func NewBroker() *Broker {
	return newBrokerWithTTL(35*time.Second, 45*time.Second)
}

func newBroker(tokenTTL time.Duration) *Broker {
	return newBrokerWithTTL(tokenTTL, 45*time.Second)

}

func newBrokerWithTTL(tokenTTL, sessionTTL time.Duration) *Broker {
	bytes := make([]byte, 8)
	if _, err := rand.Read(bytes); err != nil {
		panic(fmt.Sprintf("create broker epoch: %v", err))
	}
	return &Broker{scheduler: NewWeightedScheduler(), waiters: make(map[string][]chan error), grants: make(map[string]uint64), sessions: make(map[string]session), epoch: hex.EncodeToString(bytes), wake: make(chan struct{}, 1), tokenTTL: tokenTTL, sessionTTL: sessionTTL}
}

func (b *Broker) Register(id string, share int) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if existing, ok := b.sessions[id]; ok {
		if existing.share != share {
			return fmt.Errorf("workload %q is already registered with share %d", id, existing.share)
		}
		existing.lastSeen = time.Now()
		b.sessions[id] = existing
		return nil
	}
	if err := b.scheduler.Register(id, share); err != nil {
		return err
	}
	b.sessions[id] = session{share: share, lastSeen: time.Now()}
	return nil
}

func (b *Broker) Unregister(id string) {
	b.mu.Lock()
	wasInFlight := b.unregisterLocked(id, fmt.Errorf("workload %q was unregistered", id))
	b.mu.Unlock()
	if wasInFlight {
		b.notify()
	}
}

func (b *Broker) unregisterLocked(id string, waiterErr error) bool {
	b.scheduler.Unregister(id)
	delete(b.sessions, id)
	for _, waiter := range b.waiters[id] {
		waiter <- waiterErr
		close(waiter)
	}
	delete(b.waiters, id)
	wasInFlight := b.busy && b.inFlight == id
	if wasInFlight {
		b.busy, b.inFlight = false, ""
	}
	return wasInFlight
}

func (b *Broker) notify() {
	select {
	case b.wake <- struct{}{}:
	default:
	}
}

func (b *Broker) touchLocked(id string) error {
	current, ok := b.sessions[id]
	if !ok {
		return fmt.Errorf("workload %q is not registered", id)
	}
	current.lastSeen = time.Now()
	b.sessions[id] = current
	return nil
}

func (b *Broker) Heartbeat(id string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.touchLocked(id)
}

func (b *Broker) Acquire(ctx context.Context, id string) error {
	granted := make(chan error, 1)
	b.mu.Lock()
	if err := b.touchLocked(id); err != nil {
		b.mu.Unlock()
		return err
	}
	if err := b.scheduler.Enqueue(id); err != nil {
		b.mu.Unlock()
		return err
	}
	b.waiters[id] = append(b.waiters[id], granted)
	b.mu.Unlock()
	b.notify()
	select {
	case err := <-granted:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *Broker) Complete(id string) error {
	b.mu.Lock()
	if err := b.touchLocked(id); err != nil {
		b.mu.Unlock()
		return err
	}
	if !b.busy || b.inFlight != id {
		b.mu.Unlock()
		return fmt.Errorf("workload %q does not own the execution token", id)
	}
	b.busy, b.inFlight = false, ""
	b.mu.Unlock()
	b.notify()
	return nil
}

// Next atomically completes the current execution and queues the caller's next
// request before another workload is selected.
func (b *Broker) Next(ctx context.Context, id string) error {
	granted := make(chan error, 1)
	b.mu.Lock()
	if err := b.touchLocked(id); err != nil {
		b.mu.Unlock()
		return err
	}
	if !b.busy || b.inFlight != id {
		b.mu.Unlock()
		return fmt.Errorf("workload %q does not own the execution token", id)
	}
	if err := b.scheduler.Enqueue(id); err != nil {
		b.mu.Unlock()
		return err
	}
	b.waiters[id] = append(b.waiters[id], granted)
	b.busy, b.inFlight = false, ""
	b.mu.Unlock()
	b.notify()
	select {
	case err := <-granted:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (b *Broker) Run(ctx context.Context) {
	interval := b.sessionTTL / 3
	if interval <= 0 {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			b.mu.Lock()
			wake := false
			for id, current := range b.sessions {
				if now.Sub(current.lastSeen) > b.sessionTTL {
					wake = b.unregisterLocked(id, fmt.Errorf("workload %q session expired", id)) || wake
				}
			}
			b.mu.Unlock()
			if wake {
				b.notify()
			}
		case <-b.wake:
			b.mu.Lock()
			if b.busy {
				b.mu.Unlock()
				continue
			}
			id, ok := b.scheduler.Next()
			if !ok {
				b.mu.Unlock()
				continue
			}
			waiter := b.waiters[id][0]
			b.waiters[id] = b.waiters[id][1:]
			b.grants[id]++
			b.busy, b.inFlight = true, id
			b.token++
			token := b.token
			waiter <- nil
			close(waiter)
			b.mu.Unlock()
			time.AfterFunc(b.tokenTTL, func() {
				b.mu.Lock()
				expired := b.busy && b.inFlight == id && b.token == token
				if expired {
					b.busy, b.inFlight = false, ""
				}
				b.mu.Unlock()
				if expired {
					select {
					case b.wake <- struct{}{}:
					default:
					}
				}
			})
		}
	}
}

func (b *Broker) ServeUnix(ctx context.Context, path string) error {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	listener, err := net.Listen("unix", path)
	if err != nil {
		return err
	}
	defer listener.Close()
	go func() { <-ctx.Done(); listener.Close() }()
	for {
		connection, err := listener.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		go b.handle(ctx, connection)
	}
}

func (b *Broker) handle(ctx context.Context, connection net.Conn) {
	defer connection.Close()
	line, err := bufio.NewReader(io.LimitReader(connection, 4096)).ReadString('\n')
	if err != nil {
		fmt.Fprintln(connection, "ERROR malformed request")
		return
	}
	fields := strings.Fields(line)
	if len(fields) < 2 {
		fmt.Fprintln(connection, "ERROR malformed request")
		return
	}
	switch fields[0] {
	case "REGISTER":
		if len(fields) != 3 {
			fmt.Fprintln(connection, "ERROR malformed REGISTER")
			return
		}
		share, err := strconv.Atoi(fields[2])
		if err != nil {
			fmt.Fprintln(connection, "ERROR invalid share")
			return
		}
		err = b.Register(fields[1], share)
		if err != nil {
			fmt.Fprintln(connection, "ERROR "+err.Error())
			return
		}
		fmt.Fprintln(connection, "OK "+b.epoch)
		return
	case "HEARTBEAT":
		if len(fields) != 2 {
			fmt.Fprintln(connection, "ERROR malformed HEARTBEAT")
			return
		}
		if err := b.Heartbeat(fields[1]); err != nil {
			fmt.Fprintln(connection, "ERROR "+err.Error())
			return
		}
	case "ACQUIRE":
		if len(fields) != 2 {
			fmt.Fprintln(connection, "ERROR malformed ACQUIRE")
			return
		}
		if err := b.Acquire(ctx, fields[1]); err != nil {
			fmt.Fprintln(connection, "ERROR "+err.Error())
			return
		}
	case "COMPLETE":
		if len(fields) != 2 {
			fmt.Fprintln(connection, "ERROR malformed COMPLETE")
			return
		}
		if err := b.Complete(fields[1]); err != nil {
			fmt.Fprintln(connection, "ERROR "+err.Error())
			return
		}
	case "NEXT":
		if len(fields) != 2 {
			fmt.Fprintln(connection, "ERROR malformed NEXT")
			return
		}
		if err := b.Next(ctx, fields[1]); err != nil {
			fmt.Fprintln(connection, "ERROR "+err.Error())
			return
		}
	case "UNREGISTER":
		b.Unregister(fields[1])
	default:
		fmt.Fprintln(connection, "ERROR unknown command")
		return
	}
	fmt.Fprintln(connection, "OK")
}

func (b *Broker) MetricsHandler(w http.ResponseWriter, _ *http.Request) {
	b.mu.Lock()
	defer b.mu.Unlock()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	states := b.scheduler.QueueStates()
	now := time.Now()
	allocated := 0
	for _, state := range states {
		allocated += state.Share
	}
	fmt.Fprintf(w, "# HELP npu_share_active_workloads Registered workloads.\n# TYPE npu_share_active_workloads gauge\nnpu_share_active_workloads %d\n", len(states))
	fmt.Fprintf(w, "npu_share_capacity %d\n", logicalCapacity)
	fmt.Fprintf(w, "npu_share_allocated_total %d\n", allocated)
	fmt.Fprintf(w, "npu_share_available %d\n", logicalCapacity-allocated)
	fmt.Fprintf(w, "npu_share_broker_info{epoch=\"%s\"} 1\n", b.epoch)
	for _, state := range states {
		id := strings.NewReplacer("\\", "\\\\", "\"", "\\\"").Replace(state.WorkloadID)
		fmt.Fprintf(w, "npu_share_allocated{workload=\"%s\"} %d\n", id, state.Share)
		fmt.Fprintf(w, "npu_share_queue_depth{workload=\"%s\"} %d\n", id, state.Queued)
		current := b.sessions[state.WorkloadID]
		fmt.Fprintf(w, "npu_share_session_last_seen_seconds{workload=\"%s\"} %.3f\n", id, float64(current.lastSeen.UnixNano())/1e9)
		fmt.Fprintf(w, "npu_share_session_age_seconds{workload=\"%s\"} %.3f\n", id, now.Sub(current.lastSeen).Seconds())
	}
	for workloadID, grants := range b.grants {
		id := strings.NewReplacer("\\", "\\\\", "\"", "\\\"").Replace(workloadID)
		fmt.Fprintf(w, "npu_share_grants_total{workload=\"%s\"} %d\n", id, grants)
	}
}
