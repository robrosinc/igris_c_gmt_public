import numpy as np
from scipy.spatial.transform import Rotation as R


def finite_difference(data: np.ndarray, dt: float) -> np.ndarray:
    if len(data) < 2:
        return np.zeros_like(data)
    vel = np.zeros_like(data)
    vel[1:-1] = (data[2:] - data[:-2]) / (2.0 * dt)
    vel[0] = (data[1] - data[0]) / dt
    vel[-1] = (data[-1] - data[-2]) / dt
    return vel


def quat_wxyz_to_scipy(quat: np.ndarray) -> np.ndarray:
    return quat[..., [1, 2, 3, 0]]


def quat_scipy_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def quaternion_angular_velocity(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    if len(quats_wxyz) < 2:
        return np.zeros((len(quats_wxyz), 3))
    quats_scipy = quat_wxyz_to_scipy(quats_wxyz)
    rotations = R.from_quat(quats_scipy)
    ang_vel = np.zeros((len(quats_wxyz), 3))
    for i in range(1, len(quats_wxyz) - 1):
        rel = rotations[i - 1].inv() * rotations[i + 1]
        ang_vel[i] = rel.as_rotvec() / (2.0 * dt)
    ang_vel[0] = (rotations[0].inv() * rotations[1]).as_rotvec() / dt
    ang_vel[-1] = (rotations[-2].inv() * rotations[-1]).as_rotvec() / dt
    return ang_vel


def get_default_keybody_names(robot_name: str) -> list[str]:
    if robot_name in ("robros_igris_c", "robros_igris_c_v2"):
        return [
            "Link_Waist_Yaw",
            "Link_Waist_Roll",
            "Link_Waist_Pitch",
            "Link_Hip_Pitch_Left",
            "Link_Hip_Roll_Left",
            "Link_Hip_Yaw_Left",
            "Link_Knee_Pitch_Left",
            "Link_Ankle_Pitch_Left",
            "Link_Ankle_Roll_Left",
            "Link_Hip_Pitch_Right",
            "Link_Hip_Roll_Right",
            "Link_Hip_Yaw_Right",
            "Link_Knee_Pitch_Right",
            "Link_Ankle_Pitch_Right",
            "Link_Ankle_Roll_Right",
            "Link_Shoulder_Pitch_Left",
            "Link_Shoulder_Roll_Left",
            "Link_Shoulder_Yaw_Left",
            "Link_Elbow_Pitch_Left",
            "Link_Wrist_Yaw_Left",
            "Link_Wrist_Roll_Left",
            "Link_Wrist_Pitch_Left",
            "Left_Hand",
            "Link_Shoulder_Pitch_Right",
            "Link_Shoulder_Roll_Right",
            "Link_Shoulder_Yaw_Right",
            "Link_Elbow_Pitch_Right",
            "Link_Wrist_Yaw_Right",
            "Link_Wrist_Roll_Right",
            "Link_Wrist_Pitch_Right",
            "Right_Hand",
            "Link_Neck_Yaw",
            "Link_Neck_Pitch",
        ]
    if robot_name == "robros_igris_max":
        return [
            "Left_Arm_Wrist_Roll",
            "Right_Arm_Wrist_Roll",
            "Left_Leg_Ankle_Roll_Foot",
            "Right_Leg_Ankle_Roll_Foot",
            "Left_Arm_Shoulder_Pitch",
            "Right_Arm_Shoulder_Pitch",
            "Left_Leg_Hip_Pitch",
            "Right_Leg_Hip_Pitch",
            "Left_Arm_Elbow",
            "Right_Arm_Elbow",
            "Left_Leg_Knee",
            "Right_Leg_Knee",
        ]
    return []


def compute_root_local_keybody(
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    keybody_pos_world: np.ndarray,
    keybody_rot_world_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames = root_pos.shape[0]
    num_keybodies = keybody_pos_world.shape[1]
    if num_keybodies == 0:
        return (
            np.zeros((num_frames, 0, 3), dtype=np.float64),
            np.zeros((num_frames, 0, 4), dtype=np.float64),
        )

    rel_world = keybody_pos_world - root_pos[:, None, :]
    root_rot_scipy = quat_wxyz_to_scipy(root_rot_wxyz)
    root_inv = R.from_quat(np.repeat(root_rot_scipy, num_keybodies, axis=0)).inv()
    rel_local = root_inv.apply(rel_world.reshape(-1, 3)).reshape(num_frames, num_keybodies, 3)

    keybody_rot_world_scipy = quat_wxyz_to_scipy(keybody_rot_world_wxyz)
    key_rot_world = R.from_quat(keybody_rot_world_scipy.reshape(-1, 4))
    key_rot_local_scipy = (root_inv * key_rot_world).as_quat().reshape(num_frames, num_keybodies, 4)
    key_rot_local_wxyz = quat_scipy_to_wxyz(key_rot_local_scipy)
    return rel_local, key_rot_local_wxyz


def build_motion_data(
    aligned_fps: float,
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    keybody_pos_world: np.ndarray | None,
    keybody_rot_world_wxyz: np.ndarray | None,
    keybody_names: list[str] | None,
    local_body_pos: np.ndarray | None = None,
    local_body_link_body_list: list[str] | None = None,
) -> dict:
    num_frames = root_pos.shape[0]
    if keybody_pos_world is None:
        keybody_pos_world = np.zeros((num_frames, 0, 3))
    if keybody_rot_world_wxyz is None:
        keybody_rot_world_wxyz = np.zeros((num_frames, 0, 4))
    if keybody_names is None:
        keybody_names = []

    dt = 1.0 / aligned_fps
    root_vel = finite_difference(root_pos, dt)
    dof_vel = finite_difference(dof_pos, dt)
    root_angvel = quaternion_angular_velocity(root_rot_wxyz, dt)

    keybody_pos_local, keybody_rot_local = compute_root_local_keybody(
        root_pos=root_pos,
        root_rot_wxyz=root_rot_wxyz,
        keybody_pos_world=keybody_pos_world,
        keybody_rot_world_wxyz=keybody_rot_world_wxyz,
    )

    return {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot_wxyz,
        "root_vel": root_vel,
        "root_angvel": root_angvel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "keybody_pos_world": keybody_pos_world,
        "keybody_pos_local": keybody_pos_local,
        "keybody_rot_world": keybody_rot_world_wxyz,
        "keybody_rot_local": keybody_rot_local,
        "keybody_pos": keybody_pos_world,
        "link_body_list": keybody_names,
        "local_body_link_body_list": local_body_link_body_list,
        "local_body_pos": local_body_pos,
    }
