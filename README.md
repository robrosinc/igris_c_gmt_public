# igris_c_gmt_public

External low-level inference package for IGRIS-C teleoperation.

## Scope

- Reads robot state through `igris_c_sdk` over CycloneDDS.
- Runs a policy ONNX model in a control loop.
- Publishes `igris_c_sdk::LowCmd`.
- Loads observation layout from `config/obs.yaml` and action settings from `config/action.yaml`.
- Includes an example teleoperation ONNX model under `models/wbc_teleop/0721/policy.onnx`.
- Does not switch robot modes automatically.
- Does not depend on internal `igris_c` controller types.

## Current Motion Ingress

The package accepts motion input through one of these source types:

- `ros2`: subscribes to a ROS 2 topic from the PICO-side module.
- `onnx_replay`: replays a recorded motion ONNX file with `time_step` input and motion tensor outputs.
- `csv_replay`: replays retargeted motion frames from a CSV file.
- `null`: no upstream motion data. The worker does not run policy inference until a valid motion source is configured.

`ros2` is the intended public teleoperation path. `onnx_replay` and `csv_replay` are useful for recorded-motion bring-up.

## Modular Pipeline

The node is split into small modules:

- `StateHandler`: reads `LowState`, applies the configured filters, builds joint/base observations, and stores the latest processed state.
- `MotionHandler`: owns ROS motion input, recorded-motion replay, freshness checks, mode transitions, and the motion-step clock.
- `ObservationBuilder`: reads `config/obs.yaml`, calls observation functions from `include/utils/obs_functions.h`, and builds ONNX input groups.
- `OnnxRunner`: executes the policy ONNX model from `policy.onnx_path`.
- `ActionBuilder`: reads `config/action.yaml`, converts raw network actions into desired joint positions, applies joint limits, and builds `LowCmd`.

`main.cpp` wires those modules into two loops:

1. The main loop runs at `loop.control_hz`.
2. It reads the newest DDS `LowState` from `RobotIo`.
3. When the state sequence changes, it updates `StateHandler`.
4. The worker thread wakes at `loop.policy_hz`.
5. The worker reads the latest processed state.
6. `MotionHandler` supplies either the latest live ROS motion frame or the replay frame at the current motion step.
7. `ObservationBuilder` builds the ONNX input groups from the configured observation terms.
8. `OnnxRunner` runs the policy and returns the raw 23-element action vector.
9. `ActionBuilder` converts the raw action into desired joint positions and `LowCmd` fields.
10. The worker writes the command into the action buffer.
11. The main loop publishes the command once when the action buffer sequence changes.
12. `MotionHandler::advance()` increments the motion replay clock only after a command is successfully built.

If robot state is not changing, motion is missing/stale, observation building fails, ONNX inference fails, or action building fails, no new command is published. This is intentional: the package waits instead of silently publishing commands from incomplete data.

ROS 2 motion callbacks write immutable `MotionFrame` snapshots into an atomic latest-frame buffer. Each frame stores:

- `seq`
- `joint_position[23]`
- `joint_velocity[23]`
- `body_names[]`
- `body_position[]` as xyz-packed local retarget body positions from `keybody_pos_local`
- `body_orientation[]` as wxyz-packed local retarget body orientations from `keybody_rot_local`, when present
- `body_linear_velocity[]` and `body_angular_velocity[]` as xyz-packed local/body-frame body velocities, when present
- `anchor_position[3]`
- `anchor_linear_velocity[3]` and `anchor_angular_velocity[3]` in the world frame
- `anchor_linear_velocity_b[3]` and `anchor_angular_velocity_b[3]` in the anchor/body frame
- `anchor_quaternion_wxyz[4]`

The worker atomically loads one snapshot per cycle. That avoids reading motion data while a ROS callback is writing it.

## Where To Customize

Most policy-specific changes should not require editing `main.cpp`.

- Robot/network settings: edit `config/params.yaml`, section `robot`, or pass `--domain-id`, `--namespace`, or `--cyclonedds-xml`.
- Loop rates and filters: edit `config/params.yaml`, sections `loop` and `filters`.
- Motion source: edit `config/params.yaml`, section `motion_source`.
- ROS motion topic/QoS: edit `config/ros.yaml`.
- Policy ONNX path: edit `config/params.yaml`, section `policy`.
- Observation order/history/input groups: edit `config/obs.yaml`.
- New observation terms: add a function and registry entry in `include/utils/obs_functions.h`, then reference its name from `config/obs.yaml`.
- Body-position subset for motion body observations: edit `kMotionBodyNames` near the top of `include/utils/obs_functions.h`.
- Action scaling, residual mode, limits, gains, and kinematic modes: edit `config/action.yaml`.
- New action semantics: edit `src/modules/action_builder.cpp`.
- New motion source type: implement it in `src/motion/frame_source.cpp` and add it to `CreateMotionFrameSource`.

The 23-entry policy/action joint order used by `obs.yaml` and most `action.yaml` arrays is:

```text
[LSP, RSP, WY, LSR, RSR, WR, LSY, RSY, WP, LEP, REP,
 LHP, RHP, LHR, RHR, LHY, RHY, LKP, RKP, LAP, RAP, LAR, RAR]
```

Full motor vectors such as 31-entry `kp` and `kd` use the SDK motor order documented in `config/action.yaml`.

## Observation Construction

Observation construction is configured in `config/obs.yaml`.

- ROS/replay motion sources store data in the same canonical `MotionFrame` shape.
- Observation term order, group name, and per-term history length come from `obs.yaml`.
- `q_default` for `joint_pos_rel` comes from `obs.yaml`.
- Custom observation functions can be added to `include/utils/obs_functions.h` and referenced by function name in `obs.yaml`.

The ROS retarget payload format is the GMR payload. The receiver requires `dof_pos`, `dof_vel`, `keybody_pos_local`, `link_body_list`, `root_pos`, `root_rot`, `root_vel`, and `root_angvel`. Missing fields make the motion frame invalid. The receiver stores `root_vel` and `root_angvel` as world-frame anchor velocities, then also stores converted anchor/body-frame copies in the `_b` fields.

Observation functions decide how to use the frame:

- `motion_body_pos` selects the bodies named in `kMotionBodyNames` at the top of `include/utils/obs_functions.h`.
- `motion_body_orientation` selects the matching local body quaternions when the upstream frame contains them.
- `motion_body_lin_vel` and `motion_body_ang_vel` select matching local/body-frame body velocities when the upstream frame contains them.
- `motion_anchor_lin_vel` and `motion_anchor_ang_vel` return world-frame anchor velocities.
- `motion_anchor_lin_vel_b` and `motion_anchor_ang_vel_b` return anchor/body-frame anchor velocities.
- `motion_anchor_orientation` and `motion_root_state` derive 6D orientation terms from `anchor_quaternion_wxyz` and the live robot state.
- `motion_anchor_height` extracts z from `anchor_position`.

Body selection is strict after ROS payload normalization. The ROS receiver inserts `base_link` with zero local position and identity local orientation if GMR omits it. If any other requested body is missing, duplicated, or malformed, observation construction fails.

The general motion tracking observation contains 10-step history of:

- reference joint position
- reference body positions in the motion anchor frame
- reference anchor linear and angular velocity in the motion anchor frame
- reference anchor orientation relative to the live robot anchor
- reference anchor height
- robot base angular velocity, projected gravity, relative joint position, joint velocity, and last actions

If a policy needs a different observation order or history length, edit `config/obs.yaml`. If it needs a new term, add the function in `include/utils/obs_functions.h` and reference it in `obs.yaml`.

## ROS 2 Topic Type

The ROS 2 receiver subscribes to:

- `std_msgs/msg/String`

The payload is parsed for these GMR keys: `dof_pos`, `dof_vel`, `keybody_pos_local`, `link_body_list`, `root_pos`, `root_rot`,
`root_vel`, `root_angvel`, and optionally `keybody_rot_local`. Body velocity arrays are optional and may be named
`keybody_lin_vel_local` / `keybody_ang_vel_local` or `body_lin_vel_b` / `body_ang_vel_b`.

## CSV Format

CSV replay accepts the current general-motion-tracking row format:

```text
seq,unused_stamp_ns,q0,...,q22,body_pos0,...,body_pos38,anchor_lin_vel0,...,anchor_lin_vel2,anchor_ang_vel0,...,anchor_ang_vel2,anchor_height,anchor_quat_w,anchor_quat_x,anchor_quat_y,anchor_quat_z
```

The second CSV column is accepted for compatibility with older exports but is not stored in `MotionFrame`.
CSV body positions are interpreted in the exact `kMotionBodyNames` order. CSV replay treats the legacy anchor velocity columns as anchor/body-frame values, derives world-frame anchor velocities from the anchor quaternion, and also accepts rows with optional `body_orientation0,...` values after `body_pos38` and full `anchor_pos0,anchor_pos1,anchor_pos2` in place of the single `anchor_height` column.

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

Build from the ROS workspace:

```bash
cd /home/robros/workspace
source /home/robros/workspace/install/setup.bash
colcon build --packages-select igris_c_gmt_public
```

Run recorded ONNX motion replay:

```bash
source /home/robros/workspace/install/setup.bash
ros2 run igris_c_gmt_public igris_c_gmt_public_node \
  --config "$(ros2 pkg prefix igris_c_gmt_public)/share/igris_c_gmt_public/config/params.yaml" \
  --namespace igris_c_<your namespace>
```

Run dds bridge
```bash
source /home/robros/workspace/install/setup.bash
ros2 run igris_c_gmt_public gmr_control_mode_dds_bridge --ros-args -p dds_domain:=<domain_id> -p robot_namespace:=<ns>
```

Overall execution order:

1. Run the IGRIS-C bridge/controller process so it publishes `rt/lowstate` and forwards DDS `rt/lowcmd` into the controller command buffer.
2. Run `igris_c_gmt_public`.
3. Switch the robot to LOW_LEVEL mode when ready.

For live ROS 2 teleoperation, set `motion_source.type: "ros2"`, run the GMR teleop process, then run the Redis bridge before `igris_c_gmt_public`.

## Notes

- The robot must be switched to LOW_LEVEL mode before the published `LowCmd` is applied.
- If the motion sample is missing or stale, the worker skips policy inference and does not publish a new action.
- The package is intentionally conservative on safety: no hidden mode changes and no dependency on core-only command paths.
