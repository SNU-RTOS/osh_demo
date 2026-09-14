# Hailo device-plugin fixes

`milestone-1.patch` applies to SNU-RTOS/hailo-device-plugin commit
`7770bf450b3c06db9016b210b1aa0fb904d87762` and includes regression tests.
The plugin remains a separate repository; this patch is the reproducible source
of the locally deployed `hailo-device-plugin:nputask-m1` image.

## Reproduced failure

With the original plugin, a three-device inference pod succeeded, but a second
five-device pod failed HailoRT discovery with `HAILO_DRIVER_INVALID_RESPONSE(85)`
while opening an unassigned device's `board_location`. Both containers used the
same host backing directory for `/sys/class/hailo_chardev`. Mountpoint directories
from one container were visible in the other. The post-stop hook then removed
entries belonging to the surviving container.

The patch mounts a private tmpfs at that path, overlays only allocated device
directories, and removes the CDI post-stop hook. The mount `type` field is part
of the [CDI specification](https://github.com/cncf-tags/container-device-interface/blob/main/specs-go/config.go).
Existing device-node cgroup restrictions remain the access boundary.

Also included: atomic CDI file publication, rejection of unknown/duplicate
allocation IDs, correct empty discovery parsing, and repair of an obsolete test
fixture. Allocation stays idempotent; no independent allocation ledger or
`Deallocate` method is added.

## Apply and build

In a clean checkout of the plugin at the commit above:

```sh
git apply /data/osh_demo/npu-task/plugin/milestone-1.patch
go test -race ./...
docker build -t hailo-device-plugin:nputask-m1 .
docker save hailo-device-plugin:nputask-m1 | k3s ctr images import -
```

Terminate existing NPU consumers before migrating from the shared-directory
version. Change the DaemonSet's container image to the built tag and its pull
policy to `IfNotPresent`; wait for rollout and verify all eight devices return.
Old running containers retain their old CDI mounts and hooks until terminated.
The patch does not remove host files referenced by those old hooks.

For rollback, terminate NPU consumers, restore the previous DaemonSet image and
pull policy, and wait for inventory. The prior deployed digest was
`ghcr.io/snu-rtos/hailo-device-plugin@sha256:3befb24fb3432a4ec966f3271d7e3078e71ff0afbf9f72df3842dc65f2d61757`.
Rollback restores the known concurrent-discovery bug as well.
