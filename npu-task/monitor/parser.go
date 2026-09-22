package monitor

import (
	"bufio"
	"regexp"
	"strconv"
	"strings"
)

// Sample is one HailoRT monitor measurement. Utilization and FPS are percent
// and frames per second, respectively.
type Sample struct {
	Kind         string
	Device       string
	Architecture string
	Model        string
	Utilization  float64
	FPS          float64
	PID          string
}

var deviceLine = regexp.MustCompile(`^\s*(\S+)\s+([0-9]+(?:\.[0-9]+)?)\s+(HAILO\S+)\s*$`)
var modelLine = regexp.MustCompile(`^\s*(.+?)\s{2,}([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+)\s*$`)
var ansiCSI = regexp.MustCompile(`\x1b\[[0-9;?]*[ -/]*[@-~]`)

// Parse converts the human-readable output of `hailortcli monitor` into
// samples. Unknown/header lines are ignored so terminal control sequences and
// future HailoRT additions do not break collection.
func Parse(text string) []Sample {
	var out []Sample
	section := ""
	s := bufio.NewScanner(strings.NewReader(text))
	for s.Scan() {
		line := strings.TrimSpace(ansiCSI.ReplaceAllString(s.Text(), ""))
		if strings.HasPrefix(line, "Device ID") && strings.Contains(line, "Utilization") {
			section = "device"
			continue
		}
		if strings.HasPrefix(line, "Model") && strings.Contains(line, "Stream") {
			section = "stream"
			continue
		}
		if strings.HasPrefix(line, "Model") && strings.Contains(line, "Utilization") {
			section = "model"
			continue
		}
		if section == "device" {
			if m := deviceLine.FindStringSubmatch(line); m != nil {
				u, _ := strconv.ParseFloat(m[2], 64)
				out = append(out, Sample{Kind: "device", Device: m[1], Architecture: m[3], Utilization: u})
			}
			continue
		}
		if section == "model" {
			if m := modelLine.FindStringSubmatch(line); m != nil {
				u, _ := strconv.ParseFloat(m[2], 64)
				fps, _ := strconv.ParseFloat(m[3], 64)
				out = append(out, Sample{Kind: "model", Model: strings.TrimSpace(m[1]), Utilization: u, FPS: fps, PID: m[4]})
			}
		}
	}
	return out
}
