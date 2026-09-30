# 2026-09-25 — Onboard policy, state from mocap (replaces OpenVINS)

Network: everything on the lab router `192.168.50.1` (ASUS). Data path: Motive → ground PC → drone.

| Machine | IP | Notes |
|---|---|---|
| Motive PC (Windows) | `192.168.50.5` | NatNet 4.5, multicast `239.255.42.99`, ~124 Hz. Needs its network cable in the router (not just the camera switch). |
| Ground PC | `192.168.50.28` | USB-Ethernet `enxc8a36215bc73`, connection `svg`. Robot container uses `network_mode: host`. |
| Drone `m0054` (VOXL) | `192.168.50.205` | `wlan0`, `MAV_SYS_ID` 1. Tailscale `100.69.138.88`. Motive rigid body `drone_bw` → `/drone_bw/pose`; PX4 topics are `/drone_1/fmu/...`. |

IPs come from the router's DHCP. If `ip -4 -br addr` shows a different ground IP, update this file.

One terminal per step. Container steps: enter the container, then paste the inside block. The
prompt must be `[...]root@deli:~/AirStack/robot/ros_ws#` before pasting — `ros2`, `bws`, `sws`
only exist in the container. `sg docker -c` gives the command docker-group permission, so it
works even in a terminal opened before you were added to the group (where plain `connect` fails
with `Docker daemon is not running`).

`export ROS_DOMAIN_ID=1`: matches the drone's `XRCE_DDS_DOM_ID=1` (set by
`voxl_setup_real_drone.sh`). It only takes effect after a drone reboot; a PX4 that has not
picked it up yet publishes on domain 0. If `ros2 topic list | grep drone_1/fmu` shows only
`vehicle_visual_odometry`, the drone is on the other domain — check with
`ROS_DOMAIN_ID=0 ros2 topic list | grep -c fmu`. Drone commands go through `adb shell` (USB)
or `ssh root@192.168.50.205`.

## 0. Network check

```bash
ping -c 3 192.168.50.5 && ping -c 3 192.168.50.205
```

If `192.168.50.5` doesn't answer, the Motive PC isn't on the router: on it, check that
**Settings → Network & Internet** shows an IPv4 address of `192.168.50.5`, and that Motive's
**View → Data Streaming Pane** has **Broadcast Frame Data** on with **Local Interface** `192.168.50.5`.

## 1. Image, container, files to drone (one-time)

```bash
cd ~/AirStack && git switch yikuan/SVG_ground_control && \
./airstack.sh image-build robot-desktop && ./airstack.sh up robot-desktop && \
adb push robot/ros_ws/src/svg_ground_control/scripts/voxl_setup_real_drone.sh /usr/bin/ && \
adb push ~/superfly_rgb_nav/deploy/onboard/board/fly_indoor.sh /data/superfly/
```

Policy runner — push after every change to `superfly_rgb_nav/deploy/onboard/board/`.
`fly_indoor.sh` launches `/data/superfly/o4runtime/onboard_policy_runner.py`; push both together
(the script's flags, e.g. `--profile --save-frames`, must exist in the runner). Backs up the
drone's copies first. Over Wi-Fi (ssh, root password `oelinux123` unless a key is set up with
`ssh-copy-id root@192.168.50.205`):

```bash
ssh root@192.168.50.205 'cp /data/superfly/o4runtime/onboard_policy_runner.py /data/superfly/o4runtime/onboard_policy_runner.py.bak_$(date +%Y%m%d_%H%M) && cp /data/superfly/fly_indoor.sh /data/superfly/fly_indoor.sh.bak_$(date +%Y%m%d_%H%M)'
scp ~/superfly_rgb_nav/deploy/onboard/board/onboard_policy_runner.py root@192.168.50.205:/data/superfly/o4runtime/
scp ~/superfly_rgb_nav/deploy/onboard/board/fly_indoor.sh root@192.168.50.205:/data/superfly/
```

## 2. Build (after code changes)

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
bws --packages-up-to svg_ground_control natnet_ros2
sws
```

## 3. Drone setup (one-time)

```bash
adb shell 'chmod +x /usr/bin/voxl_setup_real_drone.sh && voxl_setup_real_drone.sh drone_1 192.168.50.28 1 8888'
adb shell 'sed -i "s/\"en_vio\":.*true/\"en_vio\": false/" /etc/modalai/voxl-vision-hub.conf && systemctl restart voxl-vision-hub && systemctl disable --now voxl-open-vins-server voxl-qvio-server'
adb shell 'px4-param set EKF2_EV_CTRL 11; px4-param set EKF2_HGT_REF 3; px4-param set EKF2_EV_DELAY 50; px4-param set EKF2_GPS_CTRL 0; px4-param set EKF2_MAG_TYPE 5; px4-param set SYS_HAS_MAG 0; px4-param save; systemctl restart voxl-px4'
```

Verified on `m0054` (2026-09-29): the EKF2 params, OpenVINS/QVIO disabled, and the uXRCE target
are already set. Only re-run after a reflash.

## 4. Terminal A — uXRCE agent (drone link)

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
MicroXRCEAgent udp4 -p 8888 -v4
```

Check the drone connected:

```bash
adb shell 'px4-microdds_client status'   # Running, connected / Agent IP: 192.168.50.28 / port 8888
```

## 5. Terminal B — mocap in

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
ros2 launch natnet_ros2 natnet_ros2.launch.py serverIP:=192.168.50.5 clientIP:=192.168.50.28
```

## 6. Terminal C — mocap → drone

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
ros2 run svg_ground_control mocap_bridge --ros-args --params-file $(ros2 pkg prefix svg_ground_control)/share/svg_ground_control/config/swarm_real.yaml -p drone_names:="[drone_1]" -p mocap_topic_template:=/drone_bw/pose
```

## 7. Check — ground

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
timeout 5 ros2 topic hz /drone_bw/pose
timeout 5 ros2 topic hz /drone_1/fmu/in/vehicle_visual_odometry
ros2 topic echo /drone_1/fmu/out/vehicle_odometry --once --qos-reliability best_effort --qos-durability volatile
```

## 8. Check — drone

```bash
adb shell 'px4-listener vehicle_visual_odometry; px4-listener estimator_status_flags; px4-listener vehicle_local_position'
```

## 9. Policy runner — drone (ssh)

Dry run (prints setpoints, sends nothing to PX4; Ctrl+C to stop). First line shows CPU/GPU
temperature — cool the board below 65 C before flying:

```bash
ssh -t root@192.168.50.205 'bash /data/superfly/fly_indoor.sh test'
```

Real run — leave it running for the flight; Ctrl+C after landing:

```bash
ssh -t root@192.168.50.205 'bash /data/superfly/fly_indoor.sh'
```

`-t` makes Ctrl+C stop the runner cleanly. **Restart the runner for every trial**: it accepts one
`TAKEOFF` per process (a second is rejected with `takeoff already … (one per run)`), and each
start writes a new log folder `/data/superfly/runs/<NNNNN>_<time>_<policy>/`. Arrival radius is
`--goal-radius` in `fly_indoor.sh` (0.3 m; the policy was trained with `success_radius: 0.1`).

## 10. Goal

```bash
python3 ~/superfly_rgb_nav/deploy/onboard/ground/send_goal.py --host 192.168.50.205 --port 15021 --duration 180 --at 1 SET_GOAL 2 0 1 --at 3 TAKEOFF
```

Keep it on one line: without the `--` before `at 3`, `TAKEOFF` is silently dropped and the drone
never arms. While hovering after `REACHED` (same flight, no restart), another leg is just a goal
without `TAKEOFF`, e.g. back over the takeoff spot:

```bash
python3 ~/superfly_rgb_nav/deploy/onboard/ground/send_goal.py --host 192.168.50.205 --port 15021 --duration 60 --at 1 SET_GOAL 0 0 1
```

`fly_indoor.sh` (section 9, real run) must already be running on the drone. Goal frame is
`takeoff-flu` (not NED, despite `send_goal.py`'s docstring): `X` forward along the nose at
takeoff, `Y` left, `Z` up, metres from the takeoff spot; `Z` ≥ 0.5. `TAKEOFF` makes the runner
arm and switch to OFFBOARD itself — put the RC mode switch in **Manual** before `TAKEOFF` and
don't touch it until the climb. Take over by flipping to **Position** (or moving a stick >30%);
the runner stops requesting OFFBOARD. Land by holding throttle down. Keep the goal inside the
mocap cage — out of camera view, NatNet keeps publishing the last pose.

## 10b. Logs → video (ground)

Pull the newest run and render `vis/dashboard.mp4`, `policy_frames.mp4`, `trajectory_3d.png`,
`profile.png` (dashboard traces cover the POLICY phase only, with the success radius drawn):

```bash
RUN=$(ssh root@192.168.50.205 'ls -1t /data/superfly/runs | head -1') && mkdir -p ~/drone_runs && scp -r root@192.168.50.205:/data/superfly/runs/$RUN ~/drone_runs/ && python3 ~/superfly_rgb_nav/deploy/onboard/ground/vis_run.py ~/drone_runs/$RUN && xdg-open ~/drone_runs/$RUN/vis/dashboard.mp4
```

## 11. Back to OpenVINS (drone)

```bash
adb shell 'px4-param set EKF2_EV_CTRL 15; px4-param set EKF2_HGT_REF 3; px4-param set EKF2_EV_DELAY 0.0; px4-param set EKF2_GPS_CTRL 7; px4-param set EKF2_MAG_TYPE 0; px4-param set SYS_HAS_MAG 1; px4-param save'
adb shell 'sed -i "s/\"en_vio\":.*false/\"en_vio\": true/" /etc/modalai/voxl-vision-hub.conf && systemctl enable --now voxl-open-vins-server && systemctl restart voxl-vision-hub voxl-px4'
```

---

# Function 2 — policy on the ground PC

Prerequisite: sections 1–4 above (image, build, drone setup, agent running). Check with sections
7–8 once F2-2 runs. Do NOT run `fly_indoor.sh`.

## F2-0. Copy camera bridge to drone (one-time)

```bash
adb push ~/AirStack/robot/ros_ws/superfly_onboard/rgb_bridge_voxl.py /data/superfly/
```

## F2-1. Terminal B — mocap in

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
ros2 launch natnet_ros2 natnet_ros2.launch.py serverIP:=192.168.50.5 clientIP:=192.168.50.28
```

## F2-2. Terminal C — mocap → drone

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
ros2 run svg_ground_control mocap_bridge --ros-args --params-file $(ros2 pkg prefix svg_ground_control)/share/svg_ground_control/config/rgb_policy_real.yaml -p mocap_topic_template:=/drone_bw/pose
```

## F2-3. Terminal D — px4_interface

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
ros2 launch svg_ground_control real_interfaces.launch.py drones:=drone_1 target_systems:=1
```

## F2-4. Terminal E — camera → ground (drone)

```bash
adb shell 'python3 /data/superfly/rgb_bridge_voxl.py --source mpa --pipe hires_front_small_color --lens rectilinear87 --fps 15 --host 192.168.50.28 --port 15003 --stats'
```

## F2-5. Terminal F — policy

DINO (depthnav):

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
pip3 install -q --break-system-packages ai-edge-litert
ros2 run svg_ground_control swarm_commander --ros-args --params-file $(ros2 pkg prefix svg_ground_control)/share/svg_ground_control/config/rgb_policy_real.yaml -p rgb_policy_variant:=dnav-rgb-dinov3-cnxt-tiny-tok-vel-starling-orew-ts2-4-v0
```

or agile:

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
pip3 install -q --break-system-packages ai-edge-litert
ros2 run svg_ground_control swarm_commander --ros-args --params-file $(ros2 pkg prefix svg_ground_control)/share/svg_ground_control/config/rgb_policy_real.yaml -p rgb_policy_variant:=agile-rgb-cl4nav-r50-int8ptdense-vel-tartanair-v1
```

## F2-6. Fly (one command at a time)

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
ros2 service call /swarm_commander/takeoff std_srvs/srv/Trigger
```

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
ros2 service call /swarm_commander/start std_srvs/srv/Trigger
```

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
ros2 service call /swarm_commander/hold std_srvs/srv/Trigger
```

Host terminal (enter the container):

```bash
cd ~/AirStack && sg docker -c "./airstack.sh connect robot --command=bash"
```

Inside the container:

```bash
export ROS_DOMAIN_ID=1
sws
ros2 service call /swarm_commander/land std_srvs/srv/Trigger
```
