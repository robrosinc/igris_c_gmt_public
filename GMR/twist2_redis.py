import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


TWIST2_CANONICAL_SUFFIX = "unitree_g1_with_hands"


def resolve_action_suffix(robot_name: str) -> str:
    """Return the Redis action suffix used by TWIST2-compatible consumers."""
    if robot_name in ("unitree_g1", "unitree_g1_with_hands"):
        return TWIST2_CANONICAL_SUFFIX
    return robot_name


@dataclass(frozen=True)
class Twist2RedisKeys:
    suffix: str

    @property
    def action_body(self) -> str:
        return f"action_body_{self.suffix}"

    @property
    def action_hand_left(self) -> str:
        return f"action_hand_left_{self.suffix}"

    @property
    def action_hand_right(self) -> str:
        return f"action_hand_right_{self.suffix}"

    @property
    def action_neck(self) -> str:
        return f"action_neck_{self.suffix}"

    @property
    def action_qpos(self) -> str:
        return f"action_qpos_{self.suffix}"

    @property
    def action_dof_pos(self) -> str:
        return f"action_dof_pos_{self.suffix}"

    @property
    def action_retarget_frame(self) -> str:
        return f"action_retarget_frame_{self.suffix}"

    @property
    def state_body(self) -> str:
        return f"state_body_{self.suffix}"

    @property
    def state_hand_left(self) -> str:
        return f"state_hand_left_{self.suffix}"

    @property
    def state_hand_right(self) -> str:
        return f"state_hand_right_{self.suffix}"

    @property
    def state_neck(self) -> str:
        return f"state_neck_{self.suffix}"


def make_redis_client(host: str, port: int = 6379, db: int = 0):
    import redis

    client = redis.Redis(host=host, port=port, db=db)
    client.ping()
    return client


def json_dumps(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    return json.dumps(value, separators=(",", ":"))


def json_loads(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


class Twist2RedisPublisher:
    """
    TWIST2-compatible Redis publisher.

    Body, hand, neck, and timestamp keys match the TWIST2 producer/consumer
    contract. qpos/dof/retarget_frame are GMR extensions used by ROS bridges and
    motion debugging; they are published in the same pipeline so timestamps stay
    aligned with the action frame.
    """

    def __init__(self, host: str, robot_name: str, port: int = 6379, db: int = 0):
        self.robot_name = robot_name
        self.keys = Twist2RedisKeys(resolve_action_suffix(robot_name))
        self.client = make_redis_client(host=host, port=port, db=db)
        self.pipeline = self.client.pipeline()

    def publish_controller_data(self, controller_data: Any) -> None:
        if controller_data is None:
            return
        self.client.set("controller_data", json_dumps(controller_data))

    def publish_action(
        self,
        body: np.ndarray,
        hand_left: np.ndarray,
        hand_right: np.ndarray,
        neck: list[float] | None,
        qpos: np.ndarray | None = None,
        retarget_frame: dict[str, Any] | None = None,
    ) -> None:
        self.pipeline.set(self.keys.action_body, json_dumps(body))
        self.pipeline.set(self.keys.action_hand_left, json_dumps(hand_left))
        self.pipeline.set(self.keys.action_hand_right, json_dumps(hand_right))

        if neck is not None:
            self.pipeline.set(self.keys.action_neck, json_dumps(neck))

        if qpos is not None:
            self.pipeline.set(self.keys.action_qpos, json_dumps(qpos))
            self.pipeline.set(self.keys.action_dof_pos, json_dumps(qpos[7:]))

        if retarget_frame is not None:
            self.pipeline.set(self.keys.action_retarget_frame, json_dumps(retarget_frame))

        self.pipeline.set("t_action", int(time.time() * 1000))
        self.pipeline.execute()


class Twist2RedisReader:
    def __init__(self, host: str, robot_name: str, port: int = 6379, db: int = 0):
        import redis

        self.keys = Twist2RedisKeys(resolve_action_suffix(robot_name))
        pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            max_connections=10,
            retry_on_timeout=True,
            socket_timeout=0.1,
            socket_connect_timeout=0.1,
        )
        self.client = redis.Redis(connection_pool=pool)
        self.pipeline = self.client.pipeline()
        self.client.ping()

    def read_controller_data(self) -> Any:
        return json_loads(self.client.get("controller_data"))

    def read_record_frame(self) -> dict[str, Any]:
        redis_keys = [
            self.keys.state_body,
            self.keys.state_hand_left,
            self.keys.state_hand_right,
            self.keys.state_neck,
            "t_state",
            self.keys.action_body,
            self.keys.action_hand_left,
            self.keys.action_hand_right,
            self.keys.action_neck,
            "t_action",
            self.keys.action_qpos,
            self.keys.action_dof_pos,
            self.keys.action_retarget_frame,
        ]
        data_keys = [
            "state_body",
            "state_hand_left",
            "state_hand_right",
            "state_neck",
            "t_state",
            "action_body",
            "action_hand_left",
            "action_hand_right",
            "action_neck",
            "t_action",
            "retarget_qpos",
            "retarget_dof_pos",
            "retarget_data",
        ]

        for key in redis_keys:
            self.pipeline.get(key)
        results = self.pipeline.execute()

        out = {}
        for redis_key, data_key, raw in zip(redis_keys, data_keys, results):
            try:
                out[data_key] = json_loads(raw)
            except json.JSONDecodeError:
                out[data_key] = None
                print(f"Warning: failed to decode Redis key {redis_key}")
        return out
