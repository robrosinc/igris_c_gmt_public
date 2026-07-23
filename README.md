# public_inference_module

External low-level inference package for IGRIS-C teleoperation.

## Scope

- Reads robot state through `igris_c_sdk` over CycloneDDS.
- Runs a policy ONNX model in a control loop shaped after `TrackingModuleV1`.
- Publishes `igris_c_sdk::LowCmd` only.
- Loads `kp`, `kd`, and `kinematic_modes` from this package's YAML config.
- Includes an example teleoperation ONNX model under `models/wbc_teleop/0721/policy.onnx`.
- Does not switch robot modes automatically.
- Does not depend on internal `igris_c` controller types.

## Current Motion Ingress

The package accepts motion input through one of these source types:

- `ros2`: subscribes to a ROS 2 topic from the PICO-side module.
- `onnx_replay`: replays a recorded motion ONNX file with `time_step` input and motion tensor outputs.
- `csv_replay`: replays retargeted motion frames from a CSV file.
- `null`: no upstream motion data. The node publishes a hold command from the latest `LowState`.

`ros2` is the intended public teleoperation path. `onnx_replay` and `csv_replay` are useful for recorded-motion bring-up.

## Motion Data Buffer

The ROS 2 callback does not write directly into mutable inference state. It writes an immutable `MotionDataSample` snapshot into an atomic latest-sample buffer owned by `InferenceModule`.

Each sample carries:

- `seq`
- `stamp_ns`
- `values[]`

The control loop atomically loads one snapshot per cycle. That avoids reading the motion data while a ROS callback is writing it.

## Observation Construction

Observation construction is separated from ROS transport.

- ROS transport stores raw numeric payloads in `MotionDataSample.values`.
- The inference module reads one coherent snapshot from the motion buffer.
- The observation constructor decodes that snapshot according to `motion_source.layout`.

Supported layouts:

- `general_motion_tracking_v1`
- `reference_tracking_v1`
- `q23_dq23_quatwxyz4`

`general_motion_tracking_v1` is the default DAgger-PPO student-policy path and decodes:

- `joint_position[23]`
- `motion_body_pos_b[39]`
- `motion_anchor_linear_velocity_b[3]`
- `motion_anchor_angular_velocity_b[3]`
- `motion_anchor_height[1]`
- `anchor_quaternion_wxyz[4]` from the sample metadata or optional payload suffix

`reference_tracking_v1` is the WBC policy path and decodes:

- `joint_position[23]`
- `joint_velocity[23]`
- `root_position_z[1]`
- `root_state[6]`
- `body_position[42]`
- `root_linear_velocity[3]`
- `root_angular_velocity[3]`

`q23_dq23_quatwxyz4` remains available for older 1210-observation policies and decodes:

- `joint_position[23]`
- `joint_velocity[23]`
- `anchor_quaternion_wxyz[4]`

The general motion tracking observation contains 10-step history of:

- reference joint position
- reference body positions in the motion anchor frame
- reference anchor linear and angular velocity in the motion anchor frame
- reference anchor orientation relative to the live robot anchor
- reference anchor height
- robot base angular velocity, projected gravity, relative joint position, joint velocity, and last actions

The WBC reference-tracking observation contains:

- history of robot joint position, joint velocity, projected gravity, and base angular velocity
- last policy actions
- reference joint position and velocity
- reference root state
- reference body positions
- reference root linear and angular velocity

If the customer later needs a different payload layout or a richer observation, they only need to change the decode path inside the inference module. The ROS 2 subscriber and atomic buffer can remain unchanged.

## ROS 2 Topic Type

The ROS 2 receiver subscribes to:

- `std_msgs/msg/String`

The payload is parsed for the same GMR keys used by the internal WBC teleop path: `dof_pos`, `dof_vel`, `keybody_pos_local` or
`keybody_pos_world`, `link_body_list`, `root_pos` or `root_height`, `root_rot`, `root_vel_b` or `root_vel`, and `root_angvel_b` or
`root_angvel`.

## CSV Format

Each non-comment row in `csv_replay` for `reference_tracking_v1` must contain exactly:

```text
seq,stamp_ns,q0,...,q22,dq0,...,dq22,root_z,root_state0,...,root_state5,body_pos0,...,body_pos41,root_lin_vel0,...,root_lin_vel2,root_ang_vel0,...,root_ang_vel2
```

The sample file in `data/csv/sample_motion_frames.csv` shows the layout.

## ONNX Motion Replay

The default config replays:

```text
data/onnx/motion.onnx
```

The motion ONNX must have one `int64` input named `time_step`, outputs named `joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_pos_b`, `body_lin_vel_b`, and `body_ang_vel_b`, and metadata keys `motion_fps`, `motion_num_frames`, and `body_names`.

Use `motion_source.loop: true` if you want the recorded clip to repeat. With `loop: false`, playback holds the final motion frame after the clip ends.

## Run

The default config uses this package's bundled example model:

```text
models/wbc_teleop/0721/policy.onnx
```

We included one WBC model trained on our side, so the default config should run as-is.
You may freely change the MDP setup, observation/action layout, and model checkpoint as needed, as long as the ONNX model and C++ inference layout stay consistent.

Build from the Isaac workspace:

```bash
cd /home/robros/isaac_ws
source /home/robros/workspace/install/setup.bash
colcon build --packages-select public_inference_module
```

Run recorded ONNX motion replay:

```bash
source /home/robros/workspace/install/setup.bash
source /home/robros/isaac_ws/install/setup.bash
ros2 run public_inference_module public_inference_module_node \
  --config /home/robros/isaac_ws/public_inference_module/config/params.yaml \
  --namespace igris_c_<your namespace>
```

Overall execution order:

1. Run the IGRIS-C bridge/controller process so it publishes `rt/lowstate` and forwards DDS `rt/lowcmd` into the controller command buffer.
2. Run `public_inference_module`.
3. Switch the robot to LOW_LEVEL mode when ready.

For live ROS 2 teleoperation, set `motion_source.type: "ros2"`, run the GMR teleop process, then run the Redis bridge before `public_inference_module`.

## Notes

- The robot must be switched to LOW_LEVEL mode before the published `LowCmd` is applied.
- If the motion sample is missing or stale, the node falls back to a hold command from the latest `LowState`.
- The package is intentionally conservative on safety: no hidden mode changes and no dependency on core-only command paths.
