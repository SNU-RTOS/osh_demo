
---

# Execution Guide

This document describes the step-by-step procedure to build and run the OSH demo with K3s, Hailo NPU, and camera overlay.

---

## 1. Upgrade K3s Version

First, check the installed K3s version.

```bash
$ k3s --version
```

If your K3s version is **lower than**:

* Version: `v1.33.6+k3s1`
* Go version: `go1.24.9`

upgrade K3s as follows.

```bash
$ systemctl stop k3s
$ curl -sfL https://get.k3s.io | sh -
```

---

## 2. Remove Existing CNI Configuration

Remove the default containerd CNI configuration to avoid conflicts.

```bash
$ rm /etc/cni/net.d/cni-containerd-net.conf
```

---

## 3. Clone the Git Repository

Clone the OSH demo repository.

```bash
$ git clone https://github.com/SNU-RTOS/osh_demo
```

---

## 4. Build Binaries

Build the required binaries using CMake.

```bash
$ cd osh-demo
$ docker run --rm -it -v "$PWD":/workspace -w /workspace ghcr.io/snu-rtos/osh-compile /bin/bash
$ mkdir build && cd build
$ cmake ..
$ make -j8
$ exit
```

---

## 5. Deploy Hailo Device Plugin

Deploy the Hailo device plugin to expose the NPU resource to Kubernetes.

```bash
$ kubectl create -f https://raw.githubusercontent.com/SNU-RTOS/hailo-device-plugin/main/deploy/hailo-device-plugin.yaml
```

For more details, refer to:
[https://github.com/SNU-RTOS/hailo-device-plugin](https://github.com/SNU-RTOS/hailo-device-plugin)

---

## 6. Run the Inference Driver Pod

Start the inference driver pod and execute the inference binary.

```bash
$ kubectl apply -f inference_driver.yaml
$ kubectl exec -it inference-driver -- /bin/bash
$ cd build/inference
$ ./inference_driver ../../yolov10s.hef
```

---

## 7. Run Camera Overlay Binary

Run the camera overlay binary on a system with a display connected to the board.

**Note**

* Use a separate terminal from the inference driver pod.
* Make sure a display is connected to the target board.

```bash
$ cd build/camera
$ ./camera_overlay
```

---

## 8. Run Inference Driver without Hailo Device Plugin

Verify that our Hailo Device Plugin enables access of Hailo NPUs from containers.

```bash
$ kubectl apply -f inference_driver_without_npu.yaml
$ kubectl exec -it inference-driver-wo-npu -- /bin/bash
$ cd build/inference
$ ./inference_driver ../../yolov10s.hef
```

---