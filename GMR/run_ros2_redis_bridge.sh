#!/usr/bin/env bash
set -euo pipefail

# Keep ROS/GMR envs separate:
# - Run teleop producer in GMR env (python3.10): scripts/retarget_teleop.sh
# - Run this bridge in ROS env (python3.12): this script

if [ -f /opt/ros/jazzy/setup.bash ]; then
  # ROS setup scripts can reference unset vars; avoid nounset failure here.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
fi

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
ROBOT="${ROBOT:-robros_igris_c_v2}"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
TOPIC_PREFIX="${TOPIC_PREFIX:-/gmr/teleop}"
ROS_NODE_NAME="${ROS_NODE_NAME:-gmr_redis_ros2_bridge}"
ROS_QOS_DEPTH="${ROS_QOS_DEPTH:-1}"
POLL_HZ="${POLL_HZ:-100}"
STALE_WARN_MS="${STALE_WARN_MS:-150}"
DEBUG_LOG_SEC="${DEBUG_LOG_SEC:-2.0}"
DEBUG_PREVIEW_DIM="${DEBUG_PREVIEW_DIM:-6}"

"${PYTHON_BIN}" - <<'PY'
import sys
import rclpy  # noqa: F401
import rclpy._rclpy_pybind11  # noqa: F401
print(f"[INFO] ROS python ok: {sys.executable}")
PY

"${PYTHON_BIN}" scripts/redis_to_ros2_bridge.py \
  --robot "${ROBOT}" \
  --redis_host "${REDIS_HOST}" \
  --redis_port "${REDIS_PORT}" \
  --topic_prefix "${TOPIC_PREFIX}" \
  --ros_node_name "${ROS_NODE_NAME}" \
  --ros_qos_depth "${ROS_QOS_DEPTH}" \
  --poll_hz "${POLL_HZ}" \
  --stale_warn_ms "${STALE_WARN_MS}" \
  --debug_log_sec "${DEBUG_LOG_SEC}" \
  --debug_preview_dim "${DEBUG_PREVIEW_DIM}"
