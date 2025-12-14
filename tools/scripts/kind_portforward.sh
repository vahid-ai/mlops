#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACTION="${1:-start}"

CLUSTER_NAME="${CLUSTER_NAME:-dfp-kind}"
NAMESPACE="${NAMESPACE:-dfp}"

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

start_one() {
  local name="$1"
  local local_port="$2"
  local remote_port="$3"

  local pid_file="${PF_DIR}/${name}.pid"
  local log_file="${PF_DIR}/${name}.log"

  mkdir -p "${PF_DIR}"

  if [ -f "${pid_file}" ]; then
    local existing_pid
    existing_pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if is_pid_running "${existing_pid}"; then
      echo "port-forward already running for ${name} (pid ${existing_pid})"
      return 0
    fi
    rm -f "${pid_file}"
  fi

  nohup kubectl -n "${NAMESPACE}" port-forward "svc/${name}" "${local_port}:${remote_port}" --address 127.0.0.1 >"${log_file}" 2>&1 &
  local pid="$!"
  echo "${pid}" >"${pid_file}"

  sleep 0.3
  if ! is_pid_running "${pid}"; then
    echo "failed to start port-forward for ${name} (see ${log_file})" >&2
    rm -f "${pid_file}"
    return 1
  fi

  echo "port-forward started for ${name} on http://localhost:${local_port} (pid ${pid})"
}

stop_one() {
  local name="$1"
  local pid_file="${PF_DIR}/${name}.pid"

  if [ ! -f "${pid_file}" ]; then
    echo "port-forward not running for ${name}"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  rm -f "${pid_file}"

  if ! is_pid_running "${pid}"; then
    echo "port-forward already stopped for ${name}"
    return 0
  fi

  kill "${pid}" >/dev/null 2>&1 || true
  for _ in {1..20}; do
    if ! is_pid_running "${pid}"; then
      echo "port-forward stopped for ${name}"
      return 0
    fi
    sleep 0.1
  done

  echo "port-forward for ${name} did not stop cleanly (pid ${pid})" >&2
  return 1
}

status_one() {
  local name="$1"
  local pid_file="${PF_DIR}/${name}.pid"
  local log_file="${PF_DIR}/${name}.log"

  if [ ! -f "${pid_file}" ]; then
    echo "${name}: stopped"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if is_pid_running "${pid}"; then
    echo "${name}: running (pid ${pid}, log ${log_file})"
  else
    echo "${name}: stale pid file (log ${log_file})"
    rm -f "${pid_file}"
  fi
}

require_cmd kubectl
mkdir -p "${PF_DIR}"

case "${ACTION}" in
  start)
    failures=0
    if ! start_one lakefs 8000 8000; then failures=1; fi
    if ! start_one minio 9000 9000; then failures=1; fi
    if ! start_one mlflow 5050 5000; then failures=1; fi
    if ! start_one redis 6379 6379; then failures=1; fi
    exit "${failures}"
    ;;
  stop)
    failures=0
    if ! stop_one lakefs; then failures=1; fi
    if ! stop_one minio; then failures=1; fi
    if ! stop_one mlflow; then failures=1; fi
    if ! stop_one redis; then failures=1; fi
    exit "${failures}"
    ;;
  status)
    status_one lakefs
    status_one minio
    status_one mlflow
    status_one redis
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
