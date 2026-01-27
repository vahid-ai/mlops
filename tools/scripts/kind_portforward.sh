#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-start}"

CLUSTER_NAME="${CLUSTER_NAME:-dfp-kind}"
NAMESPACE="${NAMESPACE:-dfp}"
PORT_FORWARD_ADDRESS="${PORT_FORWARD_ADDRESS:-127.0.0.1,::1}"

LAKEFS_LOCAL_PORT="${LAKEFS_LOCAL_PORT:-8000}"
MINIO_API_LOCAL_PORT="${MINIO_API_LOCAL_PORT:-19000}"
MINIO_CONSOLE_LOCAL_PORT="${MINIO_CONSOLE_LOCAL_PORT:-19001}"
MLFLOW_LOCAL_PORT="${MLFLOW_LOCAL_PORT:-5050}"
REDIS_LOCAL_PORT="${REDIS_LOCAL_PORT:-16379}"
KFP_LOCAL_PORT="${KFP_LOCAL_PORT:-8080}"
KFP_UI_LOCAL_PORT="${KFP_UI_LOCAL_PORT:-8081}"

PF_DIR="${ROOT_DIR}/.task/port-forwards/${CLUSTER_NAME}/${NAMESPACE}"

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "${cmd} is required." >&2
    exit 1
  fi
}

is_pid_running() {
  local pid="$1"
  if [ -z "${pid}" ]; then
    return 1
  fi
  kill -0 "${pid}" >/dev/null 2>&1
}

cleanup_legacy_pidfile() {
  local legacy_label="$1"
  local pid_file="${PF_DIR}/${legacy_label}.pid"

  if [ ! -f "${pid_file}" ]; then
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  rm -f "${pid_file}"

  if is_pid_running "${pid}"; then
    kill "${pid}" >/dev/null 2>&1 || true
  fi
}

cleanup_legacy() {
  cleanup_legacy_pidfile lakefs
  cleanup_legacy_pidfile minio
  cleanup_legacy_pidfile mlflow
  cleanup_legacy_pidfile redis
}

start_one() {
  local label="$1"
  local resource="$2"
  local local_port="$3"
  local remote_port="$4"
  local ns="${5:-$NAMESPACE}"

  local pid_file="${PF_DIR}/${label}.pid"
  local log_file="${PF_DIR}/${label}.log"

  mkdir -p "${PF_DIR}"

  if [ -f "${pid_file}" ]; then
    local existing_pid
    existing_pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if is_pid_running "${existing_pid}"; then
      echo "port-forward already running for ${label} (pid ${existing_pid})"
      return 0
    fi
    rm -f "${pid_file}"
  fi

  nohup kubectl -n "${ns}" port-forward "${resource}" "${local_port}:${remote_port}" --address "${PORT_FORWARD_ADDRESS}" >"${log_file}" 2>&1 &
  local pid="$!"
  echo "${pid}" >"${pid_file}"

  sleep 0.3
  if ! is_pid_running "${pid}"; then
    echo "failed to start port-forward for ${label} (see ${log_file})" >&2
    rm -f "${pid_file}"
    return 1
  fi

  echo "port-forward started for ${label} on http://localhost:${local_port} (pid ${pid})"
}

stop_one() {
  local label="$1"
  local pid_file="${PF_DIR}/${label}.pid"

  if [ ! -f "${pid_file}" ]; then
    echo "port-forward not running for ${label}"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  rm -f "${pid_file}"

  if ! is_pid_running "${pid}"; then
    echo "port-forward already stopped for ${label}"
    return 0
  fi

  kill "${pid}" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    if ! is_pid_running "${pid}"; then
      echo "port-forward stopped for ${label}"
      return 0
    fi
    sleep 0.1
  done

  echo "port-forward for ${label} did not stop cleanly (pid ${pid})" >&2
  return 1
}

status_one() {
  local label="$1"
  local pid_file="${PF_DIR}/${label}.pid"
  local log_file="${PF_DIR}/${label}.log"

  if [ ! -f "${pid_file}" ]; then
    echo "${label}: stopped"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if is_pid_running "${pid}"; then
    echo "${label}: running (pid ${pid}, log ${log_file})"
  else
    echo "${label}: stale pid file (log ${log_file})"
    rm -f "${pid_file}"
  fi
}

require_cmd kubectl
mkdir -p "${PF_DIR}"

case "${ACTION}" in
  start)
    cleanup_legacy
    failures=0
    if ! start_one lakefs svc/lakefs "${LAKEFS_LOCAL_PORT}" 8000; then failures=1; fi
    if ! start_one minio-api svc/minio "${MINIO_API_LOCAL_PORT}" 9000; then failures=1; fi
    if ! start_one minio-console svc/minio "${MINIO_CONSOLE_LOCAL_PORT}" 9001; then failures=1; fi
    if ! start_one mlflow svc/mlflow "${MLFLOW_LOCAL_PORT}" 5000; then failures=1; fi
    if ! start_one redis svc/redis "${REDIS_LOCAL_PORT}" 6379; then failures=1; fi
    if ! start_one kfp svc/ml-pipeline "${KFP_LOCAL_PORT}" 8888 kubeflow; then failures=1; fi
    if ! start_one kfp-ui svc/ml-pipeline-ui "${KFP_UI_LOCAL_PORT}" 80 kubeflow; then failures=1; fi
    exit "${failures}"
    ;;
  stop)
    cleanup_legacy
    failures=0
    if ! stop_one lakefs; then failures=1; fi
    if ! stop_one minio-api; then failures=1; fi
    if ! stop_one minio-console; then failures=1; fi
    if ! stop_one mlflow; then failures=1; fi
    if ! stop_one redis; then failures=1; fi
    if ! stop_one kfp; then failures=1; fi
    if ! stop_one kfp-ui; then failures=1; fi
    exit "${failures}"
    ;;
  status)
    status_one lakefs
    status_one minio-api
    status_one minio-console
    status_one mlflow
    status_one redis
    status_one kfp
    status_one kfp-ui
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
