package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/SNU-RTOS/osh_demo/npu-task/monitor"
)

type store struct {
	sync.RWMutex
	samples []monitor.Sample
	last    time.Time
}

func (s *store) replace(samples []monitor.Sample) {
	s.Lock()
	defer s.Unlock()
	s.samples, s.last = samples, time.Now()
}

func (s *store) metrics(w http.ResponseWriter, _ *http.Request) {
	s.RLock()
	defer s.RUnlock()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintln(w, "# HELP hailo_monitor_up Whether hailortcli monitor is producing samples.")
	fmt.Fprintln(w, "# TYPE hailo_monitor_up gauge")
	up := 0
	if len(s.samples) > 0 {
		up = 1
	}
	fmt.Fprintf(w, "hailo_monitor_up %d\n", up)
	if !s.last.IsZero() {
		fmt.Fprintf(w, "hailo_monitor_last_sample_timestamp_seconds %.3f\n", float64(s.last.UnixNano())/1e9)
	}
	for _, v := range s.samples {
		if v.Kind == "device" {
			fmt.Fprintf(w, "hailo_device_utilization_percent{device=\"%s\",architecture=\"%s\"} %.4f\n", escape(v.Device), escape(v.Architecture), v.Utilization)
		} else {
			fmt.Fprintf(w, "hailo_model_utilization_percent{model=\"%s\",pid=\"%s\"} %.4f\n", escape(v.Model), escape(v.PID), v.Utilization)
			fmt.Fprintf(w, "hailo_model_fps{model=\"%s\",pid=\"%s\"} %.4f\n", escape(v.Model), escape(v.PID), v.FPS)
		}
	}
}

func escape(v string) string { return strings.ReplaceAll(strings.ReplaceAll(v, `\`, `\\`), `"`, `\"`) }

func collect(ctx context.Context, path string, out *store) error {
	cmd := exec.CommandContext(ctx, path, "monitor")
	cmd.Env = append(os.Environ(), "HAILO_MONITOR=1")
	pipe, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	scanner := bufio.NewScanner(pipe)
	var block strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		block.WriteString(line)
		block.WriteByte('\n')
		// monitor redraws periodically; publish after each device/model table.
		if strings.Contains(line, "Frames Queue") {
			out.replace(monitor.Parse(block.String()))
			block.Reset()
		}
	}
	return cmd.Wait()
}

func main() {
	var cli, listen string
	flag.StringVar(&cli, "hailortcli", "hailortcli", "path to hailortcli")
	flag.StringVar(&listen, "listen", ":9788", "metrics listen address")
	flag.Parse()
	s := &store{}
	http.HandleFunc("/metrics", s.metrics)
	http.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok\n")) })
	go func() {
		for {
			ctx, cancel := context.WithCancel(context.Background())
			err := collect(ctx, cli, s)
			cancel()
			if err != nil {
				log.Printf("hailortcli monitor stopped: %v", err)
			}
			time.Sleep(time.Second)
		}
	}()
	log.Printf("listening on %s", listen)
	log.Fatal(http.ListenAndServe(listen, nil))
}
