"""
Bridge retargeted teleop data from Redis to ROS2 topics.

Design goal:
- Keep GMR (Python 3.10) untouched.
- Run this bridge with ROS Python (usually 3.12 for Jazzy).
- Avoid extra Python dependencies (no redis-py).
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import pathlib
import socket
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _preview(data: list[float] | None, n: int) -> str:
    if data is None:
        return "None"
    if len(data) == 0:
        return "[]"
    shown = ", ".join(f"{v:.3f}" for v in data[:n])
    suffix = ", ..." if len(data) > n else ""
    return f"[{shown}{suffix}] (len={len(data)})"


def _first_frame_full(data: Any) -> str:
    if not isinstance(data, list) or len(data) == 0:
        return "None"
    frame0 = data[0]
    return _fmt_float_tree(frame0)


def _fmt_float_tree(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_fmt_float_tree(v) for v in value) + "]"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


class RedisRespClient:
    """Tiny Redis RESP client for GET/MGET commands."""

    def __init__(self, host: str, port: int, timeout_sec: float):
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self.sock: socket.socket | None = None
        self.reader = None

    def connect(self) -> None:
        self.close()
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
        self.sock.settimeout(self.timeout_sec)
        self.reader = self.sock.makefile("rb")

    def close(self) -> None:
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def command(self, *parts: str) -> Any:
        if self.sock is None or self.reader is None:
            self.connect()
        payload = self._encode(parts)
        assert self.sock is not None
        self.sock.sendall(payload)
        return self._read_reply()

    @staticmethod
    def _encode(parts: tuple[str, ...]) -> bytes:
        out = [f"*{len(parts)}\r\n".encode()]
        for part in parts:
            b = part.encode()
            out.append(f"${len(b)}\r\n".encode())
            out.append(b + b"\r\n")
        return b"".join(out)

    def _readline(self) -> bytes:
        if self.reader is None:
            raise ConnectionError("Redis reader is not initialized")
        line = self.reader.readline()
        if not line:
            raise ConnectionError("Redis connection closed")
        return line

    def _read_reply(self) -> Any:
        line = self._readline()
        prefix = line[:1]
        payload = line[1:-2]  # strip marker + CRLF

        if prefix == b"+":
            return payload.decode()
        if prefix == b":":
            return int(payload)
        if prefix == b"$":
            n = int(payload)
            if n == -1:
                return None
            if self.reader is None:
                raise ConnectionError("Redis reader is not initialized")
            data = self.reader.read(n + 2)
            if data is None or len(data) < n + 2:
                raise ConnectionError("Failed to read bulk string from Redis")
            return data[:-2]
        if prefix == b"*":
            n = int(payload)
            if n == -1:
                return None
            return [self._read_reply() for _ in range(n)]
        if prefix == b"-":
            raise RuntimeError(f"Redis error: {payload.decode(errors='replace')}")
        raise RuntimeError(f"Unknown Redis RESP prefix: {prefix!r}")


class RedisToRos2Bridge(Node):
    def __init__(self, args):
        super().__init__(args.ros_node_name)
        self.args = args
        self.client = RedisRespClient(args.redis_host, args.redis_port, args.redis_timeout_sec)

        suffix = self._resolve_action_suffix(args.robot)
        self.redis_keys = {
            "dof_pos": f"action_dof_pos_{suffix}",
            "qpos": f"action_qpos_{suffix}",
            "retarget_frame": f"action_retarget_frame_{suffix}",
            "t_action": "t_action",
        }
        self.default_qpos = self._build_default_qpos(args.robot)
        self.default_retarget_frame = self._build_fallback_retarget_frame(self.default_qpos)
        self.default_retarget_frame["record_meta"] = {
            "source": "redis_to_ros2_bridge_default",
            "robot": args.robot,
            "unix_ms": int(time.time() * 1000),
            "default": True,
        }

        qos_depth = max(1, int(args.ros_qos_depth))
        self.pub_retarget_frame = self.create_publisher(
            String, self._topic(args.topic_prefix, "retarget_frame"), qos_depth
        )

        self._last_error_log_time = 0.0
        self._last_stale_log_time = 0.0
        self._last_debug_log_time = 0.0
        self._last_default_seed_time = 0.0
        period = 1.0 / max(1.0, float(args.poll_hz))
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            "Redis->ROS2 bridge started "
            f"(redis={args.redis_host}:{args.redis_port}, robot={args.robot}, "
            f"topic_prefix={args.topic_prefix}, poll_hz={args.poll_hz})"
        )
        self.get_logger().info(
            "Redis keys: "
            + ", ".join(f"{k}={v}" for k, v in self.redis_keys.items())
        )
        self._seed_default_redis_frame(force=True)

    @staticmethod
    def _resolve_action_suffix(robot_name: str) -> str:
        if robot_name in ["unitree_g1", "unitree_g1_with_hands"]:
            return "unitree_g1_with_hands"
        return robot_name

    @staticmethod
    def _topic(prefix: str, name: str) -> str:
        p = prefix.strip() or "/gmr/teleop"
        if not p.startswith("/"):
            p = f"/{p}"
        return f"{p.rstrip('/')}/{name}"

    @staticmethod
    def _decode_bulk(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    def _tick(self) -> None:
        key_order = [
            self.redis_keys["qpos"],
            self.redis_keys["retarget_frame"],
            self.redis_keys["t_action"],
        ]
        try:
            values = self.client.command("MGET", *key_order)
            if values is None:
                return
            if not isinstance(values, list) or len(values) != len(key_order):
                raise RuntimeError(f"Unexpected MGET reply: {type(values)}")
            self._publish_values(values)
        except Exception as exc:
            self.client.close()
            now = time.time()
            if now - self._last_error_log_time > 2.0:
                self.get_logger().warning(f"Redis read failed: {exc}")
                self._last_error_log_time = now

    def _publish_values(self, values: list[Any]) -> None:
        qpos_raw, frame_raw, t_raw = values
        qpos = self._parse_json_list(qpos_raw)
        retarget_frame = self._parse_json_dict(frame_raw)
        t_action = self._parse_int(t_raw)

        if t_action is not None:
            self._check_staleness(t_action)

        if self._is_stale_retarget_frame(retarget_frame, t_action):
            self._seed_default_redis_frame()
            retarget_frame = self.default_retarget_frame
        elif retarget_frame is None and qpos is not None and len(qpos) >= 7:
            retarget_frame = self._build_fallback_retarget_frame(qpos)
        if retarget_frame is not None and "root_height" not in retarget_frame:
            root_height = self._extract_root_height(retarget_frame)
            if root_height is not None:
                retarget_frame["root_height"] = root_height
        if retarget_frame is not None:
            msg = String()
            msg.data = self._to_dict_literal_string(retarget_frame)
            self.pub_retarget_frame.publish(msg)

        self._debug_log(retarget_frame, t_action)

    def _debug_log(
        self,
        retarget_frame: dict[str, Any] | None,
        t_action_ms: int | None,
    ) -> None:
        if self.args.debug_log_sec <= 0:
            return
        now = time.time()
        if now - self._last_debug_log_time < self.args.debug_log_sec:
            return

        age_ms = None
        if t_action_ms is not None:
            age_ms = int(time.time() * 1000) - t_action_ms

        if retarget_frame is None:
            self.get_logger().info(f"[RETARGET]\n- retarget_frame: None\n- t_action_age_ms: {age_ms}\n")
            self._last_debug_log_time = now
            return

        dof_pos = _first_frame_full(retarget_frame.get("dof_pos"))
        dof_vel = _first_frame_full(retarget_frame.get("dof_vel"))
        root_rot = _first_frame_full(retarget_frame.get("root_rot"))
        root_height = retarget_frame.get("root_height")
        try:
            root_height_str = f"{float(root_height):.3f}" if root_height is not None else "None"
        except Exception:
            root_height_str = "None"
        root_vel = _first_frame_full(retarget_frame.get("root_vel"))
        root_angvel = _first_frame_full(retarget_frame.get("root_angvel"))
        keybody_pos_local = _first_frame_full(retarget_frame.get("keybody_pos_local"))

        self.get_logger().info(
            "[RETARGET]\n"
            f"* ref_dof_pos      : {dof_pos}\n"
            f"* ref_dof_vel      : {dof_vel}\n"
            f"* ref_root_rot     : {root_rot}\n"
            f"* root_height      : {root_height_str}\n"
            f"* ref_local_lin_vel: {root_vel}\n"
            f"* ref_local_ang_vel: {root_angvel}\n"
            f"* local_kb_pos     : {keybody_pos_local}\n"
            f"* t_action_age_ms  : {age_ms}\n"
        )
        self._last_debug_log_time = now

    def _check_staleness(self, t_action_ms: int) -> None:
        now_ms = int(time.time() * 1000)
        age = now_ms - t_action_ms
        if age <= self.args.stale_warn_ms:
            return
        now = time.time()
        if now - self._last_stale_log_time > 2.0:
            self.get_logger().warning(
                f"Stale teleop data detected: age_ms={age}, threshold_ms={self.args.stale_warn_ms}"
            )
            self._last_stale_log_time = now

    @staticmethod
    def _parse_json_list(raw: Any) -> list[float] | None:
        s = RedisToRos2Bridge._decode_bulk(raw)
        if s is None:
            return None
        try:
            val = json.loads(s)
        except Exception:
            return None
        if not isinstance(val, list):
            return None
        out = []
        for x in val:
            try:
                out.append(float(x))
            except Exception:
                return None
        return out

    @staticmethod
    def _parse_json_dict(raw: Any) -> dict[str, Any] | None:
        s = RedisToRos2Bridge._decode_bulk(raw)
        if s is None:
            return None
        try:
            val = json.loads(s)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        try:
            val = ast.literal_eval(s)
            if isinstance(val, dict):
                return val
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_first_frame_vec(
        payload: dict[str, Any] | None, key: str, expected_dim: int
    ) -> list[float] | None:
        if payload is None:
            return None
        value = payload.get(key)
        if not isinstance(value, list) or len(value) == 0:
            return None
        frame0 = value[0]
        if isinstance(frame0, list):
            vec = frame0
        else:
            vec = value
        if len(vec) < expected_dim:
            return None
        out = []
        for i in range(expected_dim):
            try:
                out.append(float(vec[i]))
            except Exception:
                return None
        return out

    @staticmethod
    def _extract_root_height(payload: dict[str, Any] | None) -> float | None:
        root_pos = RedisToRos2Bridge._extract_first_frame_vec(payload, "root_pos", 3)
        if root_pos is None:
            return None
        return float(root_pos[2])

    @staticmethod
    def _to_dict_literal_string(payload: dict[str, Any]) -> str:
        # Publish as Python dict literal style:
        # data: {'fps': ..., 'root_pos': ...}
        return repr(payload)

    def _build_fallback_retarget_frame(self, qpos: list[float]) -> dict[str, Any]:
        dof = qpos[7:]
        zero_dof = [0.0] * len(dof)
        frame = {
            "fps": float(self.args.poll_hz),
            "root_pos": [qpos[:3]],
            "root_rot": [qpos[3:7]],
            "root_height": float(qpos[2]),
            "root_vel": [[0.0, 0.0, 0.0]],
            "root_angvel": [[0.0, 0.0, 0.0]],
            "dof_pos": [dof],
            "dof_vel": [zero_dof],
            "keybody_pos_world": [[]],
            "keybody_pos_local": [[]],
            "keybody_rot_world": [[]],
            "keybody_rot_local": [[]],
            "keybody_pos": [[]],
            "link_body_list": [],
            "local_body_link_body_list": None,
            "local_body_pos": None,
        }
        return frame

    def _seed_default_redis_frame(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_default_seed_time < 1.0:
            return

        now_ms = int(time.time() * 1000)
        self.default_retarget_frame["record_meta"]["unix_ms"] = now_ms
        try:
            self.client.command(
                "DEL",
                self.redis_keys["qpos"],
                self.redis_keys["dof_pos"],
                self.redis_keys["retarget_frame"],
                self.redis_keys["t_action"],
            )
            self.client.command("SET", self.redis_keys["qpos"], json.dumps(self.default_qpos))
            self.client.command("SET", self.redis_keys["dof_pos"], json.dumps(self.default_qpos[7:]))
            self.client.command(
                "SET",
                self.redis_keys["retarget_frame"],
                json.dumps(self.default_retarget_frame, separators=(",", ":")),
            )
            self.client.command("SET", self.redis_keys["t_action"], str(now_ms))
            self._last_default_seed_time = now
            if force:
                self.get_logger().info("Seeded Redis retarget keys with default frame")
        except Exception as exc:
            self.client.close()
            self.get_logger().warning(f"Failed to seed default Redis frame: {exc}")

    @staticmethod
    def _build_default_qpos(robot_name: str) -> list[float]:
        params_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "general_motion_retargeting"
            / "utils"
            / "params.py"
        )
        spec = importlib.util.spec_from_file_location("gmr_default_params", params_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load default params from {params_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        default_obs = module.DEFAULT_MIMIC_OBS[robot_name].tolist()
        root_z = float(default_obs[2])
        dof = [float(v) for v in default_obs[6:]]
        return [0.0, 0.0, root_z, 1.0, 0.0, 0.0, 0.0] + dof

    def _is_stale_retarget_frame(
        self,
        retarget_frame: dict[str, Any] | None,
        t_action_ms: int | None,
    ) -> bool:
        if retarget_frame is None or t_action_ms is None:
            return False
        meta = retarget_frame.get("record_meta")
        if not isinstance(meta, dict) or meta.get("default"):
            return False
        frame_unix_ms = self._parse_int(meta.get("unix_ms"))
        if frame_unix_ms is None:
            return False
        return abs(t_action_ms - frame_unix_ms) > self.args.stale_warn_ms

    @staticmethod
    def _parse_int(raw: Any) -> int | None:
        s = RedisToRos2Bridge._decode_bulk(raw)
        if s is None:
            return None
        try:
            return int(s)
        except Exception:
            return None

    def destroy_node(self) -> bool:
        self.client.close()
        return super().destroy_node()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "robros_igris_c_v2"],
        default="robros_igris_c_v2",
    )
    parser.add_argument("--redis_host", type=str, default="127.0.0.1")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument("--redis_timeout_sec", type=float, default=0.2)

    parser.add_argument("--topic_prefix", type=str, default="/gmr/teleop")
    parser.add_argument("--ros_node_name", type=str, default="gmr_redis_ros2_bridge")
    parser.add_argument("--ros_qos_depth", type=int, default=1)
    parser.add_argument("--poll_hz", type=float, default=100.0)
    parser.add_argument("--stale_warn_ms", type=int, default=150)
    parser.add_argument(
        "--debug_log_sec",
        type=float,
        default=2.0,
        help="Periodic debug print interval in seconds (<=0 to disable).",
    )
    parser.add_argument(
        "--debug_preview_dim",
        type=int,
        default=6,
        help="How many leading values to print for each topic payload preview.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = RedisToRos2Bridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
