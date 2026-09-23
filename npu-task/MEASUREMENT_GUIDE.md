# Multi-NPU 측정·평가·검증 가이드

현재 구현은 기능 개발과 실환경 smoke test를 마쳤으며, 반복 측정을 시작할 수
있는 상태다. 아래 세 단계는 목적이 다르므로 결과를 섞지 않는다.

- **측정**은 원시 관측값을 같은 조건에서 반복 수집한다.
- **평가**는 원시 관측값을 비교 가능한 지표와 신뢰구간으로 변환한다.
- **검증**은 결과가 사전에 정한 acceptance gate를 통과하는지 판정한다.

`npuShare`는 admission quantity와 WRR execution weight이다. 측정 결과가 share와
비례하더라도 strict throughput guarantee로 해석하지 않는다.

## 측정 전 조건

1. 전용 single-node K3s 환경과 Hailo-8 두 개 이상을 준비한다.
2. Hailo driver와 HailoRT userspace를 4.21.0으로 맞춘다.
3. CPU governor, thermal 상태, background workload, model HEF, container image를
   campaign 전체에서 고정한다.
4. 다른 shared `NPUTask`가 없어야 한다. 측정 스크립트는 이를 검사하고 거부한다.
5. `TINY_YOLOV4_HEF`에 실제 HEF 경로를 지정한다.
6. Git commit, 노드 이름, 커널, Hailo device ID, driver/firmware 버전을 실험 노트에
   기록한다.

권장 정식 campaign은 recovery 3회, fairness profile별 5회, fairness 측정 구간
30초다. 빠른 smoke run은 가능하지만 정식 검증 결과로 사용하지 않는다.

```bash
export TINY_YOLOV4_HEF=/absolute/path/to/tiny_yolov4.hef
RECOVERY_TRIALS=3 FAIRNESS_TRIALS=5 FAIRNESS_DURATION=30 \
  bash npu-task/shared/run_measurements.sh
```

## 1. 측정 방법

### Exclusive와 shared 비교

`reproduce.sh`가 정확히 두 개의 NPU만 필요한 방식으로 측정한다. Exclusive는
두 workload씩 두 wave로 실행하며, dispatcher를 배치하기 전에 수행한다. Shared는
두 dispatcher에 네 workload를 동시에 배치한다.

수집값:

- admitted/completed workload 수
- workload별 physical/logical NPU placement
- throughput FPS
- average, P95, maximum inference latency
- allocated/available share
- physical NPU 및 model utilization

출력: `npu-task/results/multi-npu-share.json`

### 복구와 pending 동작

`realistic_scenario.py`를 독립적으로 세 번 실행한다. 각 실행은 capacity가 가득 찬
상태에서 pending 요청을 만들고 capacity release 후 admission 시간을 기록한다.
그 다음 활성 Service가 있는 dispatcher를 교체하고 새 epoch 등록과 grant 재개까지
걸린 시간을 측정한다.

수집값:

- pending 감지 시간
- release 이후 admission 시간
- dispatcher recovery 시간
- 재시작 횟수와 복구 후 grant 수
- broker epoch와 allocation time의 연속성

출력: `npu-task/results/recovery-1.json`부터 `recovery-3.json`

### Weighted fairness

`fairness_benchmark.py`는 두 workload를 한 NPU에 배치하고 warm-up 이후 고정된
30초 구간의 broker grant delta를 측정한다.

- `same-model-20-80`: 같은 YOLO 모델로 실행 cost를 통제하고 WRR 비율을 확인한다.
- `mixed-model-30-70`: YOLO와 Tiny YOLOv4를 함께 실행해 서로 다른 execution cost에서도
  grant weight가 유지되는지 확인한다.

각 trial마다 grant rate, observed grant fraction, average grant wait, share로
정규화한 Jain fairness index를 기록한다.

출력: `npu-task/results/fairness.json`

## 2. 평가 방법

Exclusive 네 workload는 두 개씩 실행한 두 wave의 system throughput 평균을 사용한다.
네 workload의 FPS를 단순 합산하면 물리 NPU 네 개를 사용한 것처럼 과대평가되므로
사용하지 않는다. Shared는 네 workload가 같은 시간에 두 NPU에서 실행되므로 네 FPS의
합을 system throughput으로 사용한다.

Latency는 workload별 average/P95/max를 유지하고, summary에는 workload average와 가장
큰 P95를 함께 기록한다. 서로 다른 모델의 latency를 하나의 대표값으로만 해석하지 않는다.

Fairness는 다음 두 값을 profile별로 계산한다.

```text
grant_fraction_i = grants_i / sum(grants)
share_fraction_i = share_i / sum(shares)
fraction_error   = max(abs(grant_fraction_i - share_fraction_i))

normalized_rate_i = grants_i / share_i
Jain = sum(normalized_rate)^2 / (n * sum(normalized_rate^2))
```

5회 반복 결과에는 Student-t 기반 양측 95% 신뢰구간을 붙인다. 같은 모델 결과는 WRR
구현의 fairness 근거로 사용하고, mixed-model 결과의 FPS 차이는 모델 실행 cost가 포함된
관측값으로 해석한다.

## 3. 검증 방법

측정 종료 후 `validate_results.py`가 독립 산출물을 읽어 gate를 판정한다.

```bash
python3 npu-task/shared/validate_results.py
```

필수 gate:

- exclusive와 shared에서 각각 네 workload가 정상 inference를 완료한다.
- 두 방식 모두 비교 대상으로 물리 NPU 두 개를 사용한다.
- First Fit 결과가 `A+C -> npu-0`, `B+D -> npu-1`이다.
- throughput과 P95 latency가 모든 workload에서 양수다.
- mixed-arrival/recovery scenario가 최소 세 번 통과한다.
- 각 recovery가 180초 안에 끝나고 epoch/status/metrics 검사를 통과한다.
- fairness 두 profile이 각각 최소 다섯 번 실행되고 95% CI가 존재한다.
- fairness campaign 자체 기준은 fraction-error CI 상한 `<= 0.03`, normalized Jain CI
  하한 `>= 0.99`이다.

최종 출력 `npu-task/results/validation.json`의 `passed`가 `true`여야 정식 campaign을
통과한 것으로 판정한다. 실패 시 `failures` 항목을 원인으로 사용하고, threshold를
결과에 맞춰 사후 변경하지 않는다.

코드 회귀 검증도 별도로 통과해야 한다.

```bash
docker run --rm -v "$PWD/npu-task:/src" -w /src golang:1.24 go test -race ./...
python3 npu-task/shared/test_evaluation.py
python3 -m unittest discover -s npu-task/tools -p 'test_*.py'
bash -n npu-task/shared/*.sh
git diff --check
```

## 결과 보존

`npu-task/results/`는 Git에서 제외된다. 정식 결과를 별도 보관소에 복사하고 다음 metadata를
함께 보존한다.

- Git commit SHA와 실험 시작/종료 시각
- node, kernel, Kubernetes/K3s 버전
- Hailo PCI/device ID, driver, firmware, HailoRT 버전
- HEF checksum과 container image digest
- 각 JSON 원본과 console log

이 metadata가 다르면 동일 campaign의 반복으로 합치지 않는다.
