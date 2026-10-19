// Package admission implements restart-configured shared admission policies.
package admission

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/SNU-RTOS/osh_demo/npu-task/resource"
)

const Static = "static-share"
const BestEffort = "utilization-best-effort"

type Profile struct {
	ModelSHA256        string  `json:"modelSHA256"`
	DeviceArchitecture string  `json:"deviceArchitecture"`
	RuntimeVersion     string  `json:"runtimeVersion"`
	P95Seconds         float64 `json:"p95Seconds"`
	MeanSeconds        float64 `json:"meanSeconds"`
	Samples            int     `json:"samples"`
}

type Config struct {
	SharedAdmissionPolicy string             `json:"sharedAdmissionPolicy"`
	Budget                int                `json:"budget"`
	ProfileMargin         float64            `json:"profileMargin"`
	Profiles              map[string]Profile `json:"profiles"`
}

func Default() Config {
	return Config{SharedAdmissionPolicy: Static, Budget: 800, ProfileMargin: 1.2, Profiles: map[string]Profile{}}
}
func Load(path string) (Config, error) {
	c := Default()
	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			return c, err
		}
		if err = json.Unmarshal(data, &c); err != nil {
			return c, err
		}
	}
	if c.Profiles == nil {
		c.Profiles = map[string]Profile{}
	}
	return c, c.Validate()
}
func (c Config) Validate() error {
	if c.SharedAdmissionPolicy != Static && c.SharedAdmissionPolicy != BestEffort {
		return fmt.Errorf("unknown sharedAdmissionPolicy %q", c.SharedAdmissionPolicy)
	}
	if c.Budget < 1 || c.Budget > 1000 || !finite(c.ProfileMargin) || c.ProfileMargin < 1 {
		return fmt.Errorf("budget must be 1–1000 and profileMargin >= 1")
	}
	for key, p := range c.Profiles {
		digest, err := hex.DecodeString(p.ModelSHA256)
		if key == "" || err != nil || len(digest) != 32 || p.DeviceArchitecture == "" || p.RuntimeVersion == "" || !finite(p.P95Seconds) || p.P95Seconds <= 0 || !finite(p.MeanSeconds) || p.MeanSeconds <= 0 || p.Samples < 500 {
			return fmt.Errorf("invalid profile %q: checksum, compatibility, positive costs and >=500 samples required", key)
		}
	}
	return nil
}
func (c Config) Fingerprint() string {
	b, _ := json.Marshal(c)
	return fmt.Sprintf("%x", sha256.Sum256(b))
}
func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func (c Config) ProfileRevision(key string) string {
	b, _ := json.Marshal(c.Profiles[key])
	return fmt.Sprintf("%x", sha256.Sum256(b))
}
func (c Config) Demand(key, checksum string, fps float64) (int, Profile, string) {
	p, ok := c.Profiles[key]
	if !ok {
		return 0, p, "MissingModelProfile"
	}
	if checksum != p.ModelSHA256 {
		return 0, p, "IncompatibleModelProfile"
	}
	value := math.Ceil(1000 * fps * p.P95Seconds * c.ProfileMargin)
	if !finite(value) || fps <= 0 || value < 1 || value > float64(c.Budget) {
		return 0, p, "DemandExceedsBudget"
	}
	return int(value), p, ""
}

func LoadUnits(n resource.NPUState) int {
	if !finite(n.BrokerOccupancy) {
		return n.Allocated
	}
	return max(n.Allocated, int(math.Ceil(n.BrokerOccupancy)))
}

// Select conservatively accounts for reservations even before clients start.
func (c Config) Select(states []resource.NPUState, p Profile, demand int, now time.Time) (resource.NPUState, string) {
	var chosen resource.NPUState
	fresh, compatible := false, false
	for _, n := range states {
		if n.Health != resource.HealthHealthy || n.DeviceArchitecture != p.DeviceArchitecture || n.RuntimeVersion != p.RuntimeVersion {
			continue
		}
		compatible = true
		age := now.Sub(n.ObservedAt)
		if n.ObservedAt.IsZero() || age < 0 || age > 10*time.Second || !finite(n.ObservationWindow) || n.ObservationWindow < 10 || !finite(n.BrokerOccupancy) || n.BrokerOccupancy < 0 || n.BrokerOccupancy > 1000 {
			continue
		}
		fresh = true
		load := LoadUnits(n)
		if load+demand > min(c.Budget, n.Capacity) {
			continue
		}
		if chosen.ID == "" || load < LoadUnits(chosen) || (load == LoadUnits(chosen) && n.ID < chosen.ID) {
			chosen = n
		}
	}
	if chosen.ID != "" {
		return chosen, ""
	}
	if !compatible {
		return chosen, "IncompatibleDispatcher"
	}
	if !fresh {
		return chosen, "StaleBrokerObservation"
	}
	return chosen, "InsufficientBestEffortCapacity"
}
