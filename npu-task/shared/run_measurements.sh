#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
recovery_trials="${RECOVERY_TRIALS:-3}"
fairness_trials="${FAIRNESS_TRIALS:-5}"
fairness_duration="${FAIRNESS_DURATION:-30}"

cd "${repo_dir}"
for stale in npu-task/results/recovery-*.json; do
  [[ -e "${stale}" ]] && rm -f "${stale}"
done
RESULT_FILE="${repo_dir}/npu-task/results/multi-npu-share.json" \
  bash npu-task/shared/reproduce.sh
python3 npu-task/shared/record_environment.py npu-task/results/environment.json

for trial in $(seq 1 "${recovery_trials}"); do
  python3 npu-task/shared/realistic_scenario.py \
    --output "npu-task/results/recovery-${trial}.json"
done

python3 npu-task/shared/fairness_benchmark.py \
  --trials "${fairness_trials}" \
  --duration "${fairness_duration}" \
  --output npu-task/results/fairness.json

python3 npu-task/shared/validate_results.py \
  --comparison npu-task/results/multi-npu-share.json \
  --recovery-glob 'npu-task/results/recovery-*.json' \
  --fairness npu-task/results/fairness.json \
  --environment npu-task/results/environment.json \
  --output npu-task/results/validation.json
