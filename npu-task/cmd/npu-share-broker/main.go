package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os/signal"
	"syscall"

	"github.com/SNU-RTOS/osh_demo/npu-task/dispatcher"
)

func main() {
	socket := flag.String("socket", "/tmp/npu-share-scheduler.sock", "Unix socket path")
	metrics := flag.String("metrics", ":9790", "metrics listen address")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	broker := dispatcher.NewBroker()
	go broker.Run(ctx)
	server := &http.Server{Addr: *metrics, Handler: http.HandlerFunc(broker.MetricsHandler)}
	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("metrics: %v", err)
			stop()
		}
	}()
	if err := broker.ServeUnix(ctx, *socket); err != nil {
		log.Fatal(err)
	}
	_ = server.Shutdown(context.Background())
}
