# GMR Redis ROS2 Bridge

TWIST2 코드를 참고해 만든 GMR 프로젝트용 브릿지입니다.

PICO 입력을 받아 GMR에서 리타게팅한 결과를 Redis에 쓰고, Redis 데이터를 ROS2 topic으로 전달합니다. `public_inference_module`은 이 ROS2 topic을 받아 policy observation으로 사용합니다.

## 적용 위치

대상 GMR 프로젝트에 아래 파일을 추가하거나 기존 파일을 교체합니다.

```text
GMR/scripts/xrobot_teleop_to_robot_w_hand.py
GMR/scripts/redis_to_ros2_bridge.py
GMR/scripts/retarget_teleop.sh
GMR/scripts/run_ros2_redis_bridge.sh
```

아래 유틸 파일은 GMR retargeting package 내부에 추가하거나 교체합니다.

```text
GMR/general_motion_retargeting/utils/twist2_redis.py
GMR/general_motion_retargeting/utils/motion_utils.py
GMR/general_motion_retargeting/utils/params.py
```

## 실행 방법

GMR 리타게팅 실행:

```bash
bash scripts/retarget_teleop.sh
```

Redis to ROS2 브릿지 실행:

```bash
bash scripts/run_ros2_redis_bridge.sh
```

## Topic

브릿지는 기본적으로 아래 ROS2 topic으로 retarget frame을 발행합니다.

```text
/gmr/teleop/retarget_frame
```

`public_inference_module`의 기본 설정도 이 topic을 구독합니다.

## 참고

PICO 데이터가 아직 들어오지 않으면 브릿지는 기본 자세에 대한 리타게팅 값을 Redis에 쓰고, 같은 값을 ROS2 topic으로 발행합니다. 이 동작은 downstream inference node가 초기 입력 없이 바로 멈추는 상황을 줄이기 위한 기본값입니다.
