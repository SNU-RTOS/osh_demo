package monitor

import "testing"

func TestParseMonitorTables(t *testing.T) {
	input := `Device ID                                                   Utilization (%)         Architecture
0000:03:00.0                                               42.5                     HAILO8
Model                                                        Utilization (%)         FPS            PID
yolov10s                                                     37.5                     30.0           812
`
	samples := Parse(input)
	if len(samples) != 2 {
		t.Fatalf("got %d samples: %#v", len(samples), samples)
	}
	if samples[0].Kind != "device" || samples[0].Device != "0000:03:00.0" || samples[0].Utilization != 42.5 {
		t.Fatalf("bad device sample: %#v", samples[0])
	}
	if samples[1].Kind != "model" || samples[1].Model != "yolov10s" || samples[1].FPS != 30 || samples[1].PID != "812" {
		t.Fatalf("bad model sample: %#v", samples[1])
	}
}

func TestParseIgnoresWarnings(t *testing.T) {
	if got := Parse("Monitor did not retrieve any files\n"); len(got) != 0 {
		t.Fatalf("unexpected samples: %#v", got)
	}
}
