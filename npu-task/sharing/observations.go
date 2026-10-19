package sharing

import (
	"context"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Poll independently of arrivals; reservations still come from authoritative API
// reads. Cached samples use the broker's timestamp; failures never look idle.
type BrokerObserver struct {
	Reader  client.Reader
	mu      sync.RWMutex
	samples map[string]string
}

func (o *BrokerObserver) Metrics(p *corev1.Pod) string {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.samples[string(p.UID)]
}
func (o *BrokerObserver) poll(ctx context.Context) {
	var pods corev1.PodList
	if o.Reader.List(ctx, &pods, client.MatchingLabels{ComponentLabel: DispatcherComponent}) != nil {
		return
	}
	samples := make(map[string]string)
	for i := range pods.Items {
		p := &pods.Items[i]
		if podReady(p) && p.DeletionTimestamp == nil {
			samples[string(p.UID)] = brokerMetrics(p)
		}
	}
	o.mu.Lock()
	o.samples = samples
	o.mu.Unlock()
}
func (o *BrokerObserver) Start(ctx context.Context) error {
	o.poll(ctx)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			o.poll(ctx)
		}
	}
}
