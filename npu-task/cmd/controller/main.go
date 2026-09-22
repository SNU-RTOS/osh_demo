package main

import (
	api "github.com/SNU-RTOS/osh_demo/npu-task/api"
	"github.com/SNU-RTOS/osh_demo/npu-task/controller"
	"github.com/SNU-RTOS/osh_demo/npu-task/sharing"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"os"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metrics "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

func main() {
	ctrl.SetLogger(zap.New())
	scheme := runtime.NewScheme()
	must(corev1.AddToScheme(scheme))
	must(api.AddToScheme(scheme))
	m, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{Scheme: scheme, LeaderElection: true,
		LeaderElectionID: "nputask.npu.snu-rtos.io", LeaderElectionNamespace: os.Getenv("POD_NAMESPACE"),
		HealthProbeBindAddress: ":8081", Metrics: metrics.Options{BindAddress: ":8080"}})
	must(err)
	must((&controller.Reconciler{Client: m.GetClient(), Scheme: scheme, Registry: sharing.PodRegistry{Client: m.GetClient()}}).SetupWithManager(m))
	must(m.AddHealthzCheck("healthz", healthz.Ping))
	must(m.AddReadyzCheck("readyz", healthz.Ping))
	must(m.Start(ctrl.SetupSignalHandler()))
}
func must(err error) {
	if err != nil {
		ctrl.Log.Error(err, "controller stopped")
		os.Exit(1)
	}
}
