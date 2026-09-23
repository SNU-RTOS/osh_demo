# Multi-NPU sharing: next engineering steps

The current milestone proves two physical Hailo-8 dispatchers, fractional
admission, First Fit placement, per-NPU weighted execution, release, and
separate utilization observation. The next work should improve failure handling
and operational confidence before adding smarter placement algorithms.

## P0 — recovery and state consistency

1. Exercise dispatcher restart while Service workloads are active. A client
   must restart, re-register, and receive new execution grants without manual
   intervention. `shared/realistic_scenario.py` now covers this path.
2. Persist dispatcher epoch in control-plane status. Broker epoch, client
   heartbeat/session age, and stale-session expiry are now exposed at runtime;
   the controller does not yet copy the observed epoch into `NPUTask.status`.
3. Make allocation creation atomic for future multi-replica controllers. The
   current single controller worker serializes reconciliation; leader election
   prevents two active controller processes. A durable Allocation CR or compare
   and swap ledger is required before enabling concurrent reconcile workers.
4. Authenticate session heartbeats. The broker now expires a registration after
   45 seconds without commands or heartbeat, but workload identity is still
   trusted from the local socket protocol.
5. Define Batch retry/checkpoint semantics. A dispatcher loss during one
   inference surfaces the Batch as Failed; only Service mode is automatically
   restarted today. Production jobs need an explicit at-least-once policy or
   application-level checkpoint before automatic retry is safe.

## P1 — scheduling and QoS evidence

1. Measure weighted fairness over fixed model pairs and equal-cost synthetic
   requests. Raw grant ratios validate WRR; model throughput ratios do not equal
   share ratios when model execution costs differ.
2. Add maximum queue wait, grant wait histogram, service restart count, and
   dispatcher epoch metrics.
3. Profile model load/residency cost. Repeated model configuration currently
   affects latency and should inform a later ModelAware policy.
4. Add controlled oversubscription only after admission and measured utilization
   remain clearly separate in APIs and dashboards.

## P1 — observability and identity

1. Publish one controller endpoint that joins logical NPU IDs, dispatcher Pods,
   kubelet device-plugin IDs, and Hailo PCI IDs. The CLI currently reports both
   authoritative views without guessing a mapping that HailoRT does not expose.
2. Export NPUTask phase, requested share, assigned NPU, pending reason, and
   allocation age as Prometheus metrics instead of relying only on CLI joins.
3. Add Grafana panels and alert rules for over-allocation, missing dispatchers,
   prolonged Pending, zero grants, repeated client restarts, and stale monitor
   samples.

## P2 — security and multi-node operation

1. Replace broad hostPath UDS/shm mounts with a node-local transport and scoped
   credentials. Authenticate workload IDs before accepting broker registration.
2. Test multiple Kubernetes nodes. Placement must first select a node/NPU pair,
   and clients must bind only to a dispatcher reachable on that node.
3. Add network policy, Pod Security settings, resource limits, and upgrade/
   rollback procedures for production deployment.

## Exit criteria for the next milestone

- The realistic mixed-arrival scenario passes repeatedly without leaked share.
- Dispatcher restart recovers every Service workload within a measured bound.
- Pending duration, restart count, grant progress, and allocation age are visible.
- A broker or client crash cannot leave permanent allocated capacity.
- Weighted grant ratios are reported with confidence intervals for at least two
  model combinations.
