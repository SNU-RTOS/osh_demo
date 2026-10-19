package dispatcher

import "time"

type busyInterval struct{ start, end time.Time }

const occupancyWindow = 10 * time.Second

// Called under the broker lock, for every way an execution token is released.
func (b *Broker) finishLocked(completed bool) {
	if !b.busy {
		return
	}
	now := b.clock()
	b.intervals = append(b.intervals, busyInterval{b.busySince, now})
	if completed {
		b.completed++
	}
	b.busy, b.inFlight = false, ""
	b.pruneLocked(now)
}
func (b *Broker) pruneLocked(now time.Time) {
	cutoff := now.Add(-occupancyWindow)
	first := 0
	for first < len(b.intervals) && !b.intervals[first].end.After(cutoff) {
		first++
	}
	b.intervals = b.intervals[first:]
}
func (b *Broker) occupancyLocked(now time.Time) (float64, float64) {
	b.pruneLocked(now)
	start := now.Add(-occupancyWindow)
	if b.started.After(start) {
		start = b.started
	}
	duration := now.Sub(start)
	if duration <= 0 {
		return 0, 0
	}
	busy := time.Duration(0)
	for _, i := range b.intervals {
		s := i.start
		if s.Before(start) {
			s = start
		}
		if i.end.After(s) {
			busy += i.end.Sub(s)
		}
	}
	if b.busy {
		s := b.busySince
		if s.Before(start) {
			s = start
		}
		busy += now.Sub(s)
	}
	return 1000 * float64(busy) / float64(duration), duration.Seconds()
}
