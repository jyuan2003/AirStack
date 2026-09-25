# 2026-09-25 — Onboard policy, state from mocap (replaces OpenVINS)

Placeholders: `<GROUND_IP>` ground PC, `<MOTIVE_IP>` Motive PC, `<DRONE_IP>` VOXL.

## 1. Ground PC — host shell

```bash
cd ~/AirStack
git switch yikuan/SVG_ground_control
./airstack.sh image-build robot-desktop
./airstack.sh up robot-desktop
adb push robot/ros_ws/src/svg_ground_control/scripts/voxl_setup_real_drone.sh /usr/bin/
adb push ~/superfly_rgb_nav/deploy/onboard/board/fly_indoor.sh /data/superfly/
```

## 2. Ground PC — robot container, shell A (build + uXRCE agent)

```bash
cd ~/AirStack && ./airstack.sh connect robot --command=bash
bws --packages-select px4_msgs natnet_ros2 svg_ground_control && sws
MicroXRCEAgent udp4 -p 8888 -v4
```

## 3. VOXL — `adb shell` (one-time)

```bash
chmod +x /usr/bin/voxl_setup_real_drone.sh
voxl_setup_real_drone.sh drone_1 <GROUND_IP> 1 8888
px4-microdds_client status

sed -i 's/"en_vio":.*true/"en_vio": false/' /etc/modalai/voxl-vision-hub.conf
systemctl restart voxl-vision-hub
systemctl disable --now voxl-open-vins-server voxl-qvio-server

px4-param show > /data/superfly/params_backup_0925_before_mocap.txt
px4-param set EKF2_EV_CTRL 11
px4-param set EKF2_HGT_REF 3
px4-param set EKF2_EV_DELAY 50
px4-param set EKF2_GPS_CTRL 0
px4-param set EKF2_MAG_TYPE 5
px4-param set SYS_HAS_MAG 0
px4-param save
systemctl restart voxl-px4
```

## 4. Ground PC — robot container, shell B (mocap in)

```bash
cd ~/AirStack && ./airstack.sh connect robot --command=bash
sws
ros2 launch natnet_ros2 natnet_ros2.launch.py serverIP:=<MOTIVE_IP> clientIP:=<GROUND_IP>
```

## 5. Ground PC — robot container, shell C (mocap → drone)

```bash
cd ~/AirStack && ./airstack.sh connect robot --command=bash
sws
ros2 run svg_ground_control mocap_bridge --ros-args \
  --params-file $(ros2 pkg prefix svg_ground_control)/share/svg_ground_control/config/swarm_real.yaml \
  -p drone_names:="['drone_1']"
```

## 6. Check — ground PC, robot container, shell D

```bash
cd ~/AirStack && ./airstack.sh connect robot --command=bash
ros2 topic hz /drone_1/pose
ros2 topic hz /drone_1/fmu/in/vehicle_visual_odometry
ros2 topic echo /drone_1/fmu/out/vehicle_odometry --once \
  --qos-reliability best_effort --qos-durability volatile
```

## 7. Check — VOXL `adb shell`

```bash
px4-listener vehicle_visual_odometry
px4-listener estimator_status_flags
px4-listener vehicle_local_position
```

## 8. Policy — VOXL `adb shell`

```bash
bash /data/superfly/fly_indoor.sh test
bash /data/superfly/fly_indoor.sh
```

## 9. Goal — ground PC, host shell

```bash
python3 ~/superfly_rgb_nav/deploy/onboard/ground/send_goal.py --host <DRONE_IP> --port 15021 \
  --duration 180 --at 1 SET_GOAL <X> <Y> <Z> --at 3 TAKEOFF
```

## 10. Back to OpenVINS — VOXL `adb shell`

```bash
px4-param set EKF2_EV_CTRL 15
px4-param set EKF2_HGT_REF 3
px4-param set EKF2_EV_DELAY 0.0
px4-param set EKF2_GPS_CTRL 7
px4-param set EKF2_MAG_TYPE 0
px4-param set SYS_HAS_MAG 1
px4-param save
sed -i 's/"en_vio":.*false/"en_vio": true/' /etc/modalai/voxl-vision-hub.conf
systemctl enable --now voxl-open-vins-server
systemctl restart voxl-vision-hub voxl-px4
```

---

# Function 2 — policy on the ground PC

Prerequisite: sections 1 and 3 above done (image, uXRCE link, VIO off, EKF2 params). Check with sections 6–7 once F2-4 runs. Do NOT run `fly_indoor.sh`.

## F2-1. Ground PC — host shell (one-time)

```bash
cd ~/AirStack
adb push robot/ros_ws/superfly_onboard/rgb_bridge_voxl.py /data/superfly/
```

## F2-2. Ground PC — robot container, shell A (build + uXRCE agent)

```bash
cd ~/AirStack && ./airstack.sh connect robot --command=bash
bws --packages-select px4_msgs natnet_ros2 svg_ground_control && sws
MicroXRCEAgent udp4 -p 8888 -v4
```

## F2-3. Ground PC — robot container, shell B (mocap in)

```bash
cd ~/AirStack && ./airstack.sh connect robot --command=bash
sws
ros2 launch natnet_ros2 natnet_ros2.launch.py serverIP:=<MOTIVE_IP> clientIP:=<GROUND_IP>
```

## F2-4. Ground PC — robot container, shell C (mocap → drone)

```bash
cd ~/AirStack && ./airstack.sh connect robot --command=bash
sws
ros2 run svg_ground_control mocap_bridge --ros-args \
  --params-file $(ros2 pkg prefix svg_ground_control)/share/svg_ground_control/config/rgb_policy_real.yaml
```

## F2-5. Ground PC — robot container, shell D (px4_interface)

```bash
cd ~/AirStack && ./airstack.sh connect robot --command=bash
sws
ros2 launch svg_ground_control real_interfaces.launch.py drones:=drone_1 target_systems:=<MAV_SYS_ID>
```

## F2-6. VOXL — `adb shell` (camera → ground)

```bash
python3 /data/superfly/rgb_bridge_voxl.py --source mpa --pipe hires_front_small_color \
  --lens rectilinear87 --fps 15 --host <GROUND_IP> --port 15003 --stats
```

## F2-7. Ground PC — robot container, shell E (policy)

```bash
cd ~/AirStack && ./airstack.sh connect robot --command=bash
pip3 install --break-system-packages ai-edge-litert
sws
CFG=$(ros2 pkg prefix svg_ground_control)/share/svg_ground_control/config/rgb_policy_real.yaml

# DINO (depthnav):
ros2 run svg_ground_control swarm_commander --ros-args --params-file $CFG \
  -p rgb_policy_variant:=dnav-rgb-dinov3-cnxt-tiny-tok-vel-starling-orew-ts2-4-v0

# or agile:
ros2 run svg_ground_control swarm_commander --ros-args --params-file $CFG \
  -p rgb_policy_variant:=agile-rgb-cl4nav-r50-int8ptdense-vel-tartanair-v1
```

## F2-8. Ground PC — robot container, shell F (fly)

```bash
cd ~/AirStack && ./airstack.sh connect robot --command=bash
ros2 service call /swarm_commander/takeoff std_srvs/srv/Trigger
ros2 service call /swarm_commander/start std_srvs/srv/Trigger
ros2 service call /swarm_commander/hold std_srvs/srv/Trigger
ros2 service call /swarm_commander/land std_srvs/srv/Trigger
```
