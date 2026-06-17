# localhost if you are using laptop to verify sim2sim or sim2real
redis_ip="localhost"

# the height (empirically) should be smaller than the actual human height, due to inaccuracy of the PICO estimation.
actual_human_height=1.5
python scripts/xrobot_teleop_to_robot_w_hand.py --robot robros_igris_c_v2 \
             --actual_human_height $actual_human_height \
             --redis_ip $redis_ip \
             --target_fps 100 \
             --measure_fps 1 \
             --save_pkl_enabled false \
             --save_pkl_toggle_with_right_key_one \
             --pinch_mode \
            #  --smooth \
