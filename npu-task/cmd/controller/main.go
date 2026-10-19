package main

import (
	"flag"
	"os"

	"github.com/SNU-RTOS/osh_demo/npu-task/admission"
	api "github.com/SNU-RTOS/osh_demo/npu-task/api"
	"github.com/SNU-RTOS/osh_demo/npu-task/controller"
	"github.com/SNU-RTOS/osh_demo/npu-task/sharing"
	"github.com/prometheus/client_golang/prometheus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

func main() {
	ctrl.SetLogger(zap.New())
	configPath := flag.String("admission-config", "", "Startup admission JSON file (restart to change)")
	flag.Parse()
	cfg, configErr := admission.Load(*configPath)
	must(configErr)
	scheme := runtime.NewScheme()
	must(corev1.AddToScheme(scheme))
	must(api.AddToScheme(scheme))
	m, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{Scheme: scheme, LeaderElection: true,
		LeaderElectionID: "nputask.npu.snu-rtos.io", LeaderElectionNamespace: os.Getenv("POD_NAMESPACE"),
		HealthProbeBindAddress: ":8081", Metrics: metricsserver.Options{BindAddress: ":8080"}})
	must(err)
	observer := &sharing.BrokerObserver{Reader: m.GetAPIReader()}
	registry := sharing.PodRegistry{Client: m.GetAPIReader(), Observer: observer}
	must(m.Add(observer))
	collector := controller.NewMetricsCollector(m.GetClient())
	collector.Registry = registry
	collector.Admission = &cfg
	ctrlmetrics.Registry.MustRegister(collector)
	ctrlmetrics.Registry.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{Name: "npu_admission_configuration_info", Help: "Active startup admission policy and configuration.", ConstLabels: prometheus.Labels{"policy": cfg.SharedAdmissionPolicy, "fingerprint": cfg.Fingerprint()}}, func() float64 { return 1 }))
	must((&controller.Reconciler{Client: m.GetClient(), APIReader: m.GetAPIReader(), Admission: &cfg, Scheme: scheme, Registry: registry}).SetupWithManager(m))
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
