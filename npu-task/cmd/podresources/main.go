// podresources reads the node-local kubelet device assignment inventory.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	podresources "k8s.io/kubelet/pkg/apis/podresources/v1"
	"net"
	"os"
	"time"
)

func main() {
	socket := flag.String("socket", "/var/lib/kubelet/pod-resources/kubelet.sock", "kubelet PodResources Unix socket")
	flag.Parse()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	conn, err := grpc.NewClient("passthrough:///kubelet", grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "unix", *socket)
	}))
	check(err)
	defer conn.Close()
	result, err := podresources.NewPodResourcesListerClient(conn).List(ctx, &podresources.ListPodResourcesRequest{})
	check(err)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	check(enc.Encode(result))
}
func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
