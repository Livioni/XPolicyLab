#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

# Supply calibrated cameras for both the standard local and remote entry points.
# Scheduler arguments follow the default so explicit task_config values win.
fourrc_eval_args_file="$(mktemp /tmp/fourrc-eval-args-XXXXXX)"
trap 'rm -f "$fourrc_eval_args_file"' EXIT
printf '%s\n' --task_config 4rc_clean > "$fourrc_eval_args_file"
if [[ -n "${ROBOTWIN_EVAL_ARGS_FILE:-}" ]]; then
    cat -- "$ROBOTWIN_EVAL_ARGS_FILE" >> "$fourrc_eval_args_file"
fi
export ROBOTWIN_EVAL_ARGS_FILE="$fourrc_eval_args_file"

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${BENCH_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
