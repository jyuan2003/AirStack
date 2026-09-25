#!/usr/bin/env python3
"""The REAL `agile_rgb` policy tick, running ON the VOXL2, with numpy + tflite only.

What this is
------------
`scripts/tier25_policy_runner.py` puts the shipped `AgilePolicy` in the 4.3
seam on a **ground** box: TensorFlow for the PlaNet head, onnxruntime/LiteRT
for the encoder, `import superfly` for everything else. None of that exists on
the board -- and `superfly.policies.agile.core` cannot even be imported there,
because it imports scipy (`minimum_filter`, `Rotation`) at module scope while
`superfly.common` is numpy+scipy+pymavlink by charter.

So this file is the onboard twin of that runner: the same two learned stages
and the same NumPy tail, with every superfly import replaced by a **vendored
copy of the exact function it would have called**. Its only dependencies are
`numpy` and a TFLite runtime (`tflite_runtime`, or `tensorflow.lite` when it is
desk-tested on x86). No scipy, no TF, no ROS, no `superfly` package, no acados.

Everything the vehicle needs is read and written by this one process: two MPA
pipes in, one MAVLink stream out, and a JSON goal port for the operator. There
is no bridge, no adapter and no intermediate wire format on the board.

```
 /run/mpa/hires_front  ---MPA--> [ THIS ]
 (voxl-camera-server)              encoder_int8_ptdense.tflite
                                   head_fp32.tflite
 /run/mpa/mavlink_onboard ---MPA-> NumPy tail (docs/1.2)
 (voxl-mavlink-server)              --state synthetic: dead-reckoned
   ^                                --state mpa:       the EKF2 estimate
   |                                       |
   |  LOCAL_POSITION_NED                   |  SET_POSITION_TARGET_LOCAL_NED
   |  ATTITUDE_QUATERNION                  |  (velocity + yaw rate, world NED)
   |  ESTIMATOR_STATUS                     v
 PX4 (voxl-px4) <-------------- udpout 127.0.0.1:14556

 operator's laptop --JSON/UDP 15021--> SET_GOAL / HOLD / RESUME / STATUS
```

Bundle, plan and numbers: `docs/4.5-onboard-policy-runtime.md`,
`results/4.5-onboard-policy-runtime/`.

Where every vendored block comes from (canonical source stays authoritative)
---------------------------------------------------------------------------
| block | origin |
|---|---|
| `yaw_rate_toward`, `TaskCommandPort`, `apply_goal_command` | `src/superfly/common/vel_command_transport.py` (verbatim; only `X | Y` annotations rewritten so the file also parses under 3.6/3.8) |
| `MPAPipeClient`, `MPASource`, `parse_meta`, the YUV/RAW8/RGB converters, `resize_bilinear` | `scripts/rgb_bridge_voxl.py` (verbatim; the same reader that ran on this very board -- only the pipe handshake is split into a base class so the state reader can share it) |
| `MavlinkVelPublisher` | `scripts/sfvc_mavlink_adapter.py` (verbatim; its desk loopback test proved the message content) |
| `fit_trajectory`, `reference_velocity` | `src/superfly/policies/agile/mpc.py` (verbatim) |
| int8 `ptdense` encoder pre/post | `src/superfly/policies/agile/cl4nav_encoder.py::FrozenCL4NavTFLiteEncoder` + `checkpoints/AgileAutonomy/rgb_tartanair_v1/ENCODERS.md` |
| `TFLiteHead`, `sort_modes` | `scripts/export_agile_head_tflite.py` (verbatim) |
| `_state_to_model_input`, `_goal_dir`, `_scale_body_plan`, `_select_mode`, `_adopt_plan`, `_velocity_cmd` | `src/superfly/policies/agile/core.py::AgilePolicy` (verbatim arithmetic, scipy-free) |
| fixed hover state + dead reckoning, the seam behaviour | `scripts/tier25_policy_runner.py` |
| `R_enu_from_ned_frd_quat`, `yaw_ned_from_R_enu` (the PX4 attitude -> the ENU the policy consumes) | `src/superfly/common/px4_offboard.py::DroneState.update_from_attitude` + `src/superfly/common/frames.py` (same composition, written out as matrices because there is no scipy here) |
| `mavlink_message_t` layout, the three payload layouts | `mavlink/v2.0/mavlink_types.h` + `message_definitions/common.xml` -- quoted in section 2b; **verified on the board 2026-09-02** (0 undecodable records; `ESTIMATOR_STATUS.flags` matches `px4-listener` bit for bit) |

Where this SIMPLIFIES the ground runner, deliberately
-----------------------------------------------------
1. **No frame hop.** tier25 subscribes to the RGB bridge's compressed UDP
   stream; here the camera's MPA pipe is read in-process (the bridge's own
   reader, vendored), so there is no compress/fragment/decompress round trip on
   the board. There is no UDP frame input: one camera reader, in one process.
2. **Velocity output only.** No attitude mode, no MPC, no keep-out memory, no
   depth path -- the 2026-08-29 decision in `docs/4-deploy-onboard.md`.
   `_select_mode` is therefore literally `return 0`, as upstream.
3. **No `--net-thread`.** The net runs inline on the control tick. At the
   measured 32.6 ms encoder (results/2.1, run `s1-mai-int8-cpu`) a 15 Hz net
   inside a 15 Hz control loop has no tick to steal from; if the leader raises
   `--rate` above `--net-rate`, the net tick is the long one and the loop
   catches up by skipping its sleep (same as tier25).
4. **State is the same stand-in tier25 uses, by default** -- level attitude at
   the dead-reckoned heading, zero body rates, velocity = last commanded,
   position = its integral (`tier25_policy_runner.py`, "THE STAND-IN, STATED UP
   FRONT"). `--state mpa` replaces it with the real EKF2 estimate, decoded
   in-process off voxl-mavlink-server's pipe (section 2b;
   `docs/4.7-onboard-standalone-flight.md`): position, velocity, the full
   measured attitude AND the body rates, with a validity+staleness gate that
   holds at zero velocity rather than steering on a pose nobody measured.
   `--imu-fixed` instead freezes the whole 21-vector to
   `handheld_viz.hover_imu_state`, for a hand-held check.
5. **Frames never touch the disk -- unless `--save-frames N` is typed** (doc
   15.11.6.5, off by default): then every N-th network input is queued (a
   reference, never a copy on the control path) to a background writer that
   stores chunked .npz files in the run directory, dropping (and counting)
   whatever does not fit its queue. Without the flag no image is ever written.

Usage
-----
    # on the board, the flight shape: real camera, real EKF2, MAVLink out
    ./python/bin/python3 onboard_policy_runner.py \
        --encoder /data/superfly/encoder_int8_ptdense.tflite \
        --head    /data/superfly/head_fp32.tflite \
        --pipe hires_front --rate 15 --net-rate 15 \
        --state mpa --max-vel 1.0 --planar
    # (no --silent-until-tasked: PX4 will not ENTER offboard without a
    #  live setpoint stream, so the zero-velocity stream must start at
    #  boot -- the pilot's POSITION mode ignores it until the switch.)

    # on the board, no camera: how fast does a tick run here?
    ./python/bin/python3 onboard_policy_runner.py --bench 200 \
        --encoder ... --head ...

    # desk, x86, TF's LiteRT: identical tick against recorded frames
    SUPERFLY_AGILE_RUNTIME=docker ./scripts/agile_python.sh \
        scripts/onboard_policy_runner.py --bench 20 --frames-npz tmp/frames.npz
"""

import argparse
import collections
import errno
import json
import math
import os
import random
import re
import select
import signal
import socket
import struct
import sys
import threading
import time

import numpy as np

try:                                    # optional, desk only; see rgb_bridge_voxl
    import cv2                          # noqa: F401
except Exception:                       # pragma: no cover - depends on host
    cv2 = None


# ===========================================================================
# 0. TFLite runtime
# ===========================================================================
# Order is the reverse of cl4nav_encoder._tflite_interpreter_class: onboard the
# standalone runtime is the only thing installed, and it is what the bundle
# ships, so it is tried first. TensorFlow is the desk fallback that lets this
# same file be diffed against tier25/handheld_viz inside superfly-agile:acados.

def tflite_interpreter_class():
    errors = []
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter, "tflite_runtime"
    except Exception as exc:            # noqa: BLE001 - report every try
        errors.append("tflite_runtime: %r" % (exc,))
    try:
        import tensorflow as _tf
        return _tf.lite.Interpreter, "tensorflow.lite"
    except Exception as exc:            # noqa: BLE001
        errors.append("tensorflow: %r" % (exc,))
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter, "ai_edge_litert"
    except Exception as exc:            # noqa: BLE001
        errors.append("ai_edge_litert: %r" % (exc,))
    raise RuntimeError(
        "No TFLite runtime. Tried: " + "; ".join(errors)
        + ". On the board this means the bundle's site-packages is not on "
          "sys.path -- run the bundled ./python/bin/python3, not the system one.")


# The python Interpreter applies exactly ONE delegate by itself: XNNPACK, via
# the lazy delegate providers in tensorflow/lite/core/interpreter_builder.cc.
# Anything else -- NNAPI on this board, i.e. the DSP/GPU that ModalAI's
# benchmark_model_mai reaches -- has to come in through the external-delegate
# ABI: a .so exporting tflite_plugin_create_delegate(), loaded here.
# See results/4.5-onboard-policy-runtime/ (wheel-v2 half) for the .so we build.

def load_tflite_delegate(spec, options=None):
    """`spec` is a path to an external delegate .so (or None). Returns a
    delegate object to hand to `experimental_delegates`, or None."""
    if not spec:
        return None
    try:
        from tflite_runtime.interpreter import load_delegate
    except Exception:                   # noqa: BLE001 - desk fallback
        from tensorflow.lite.python.interpreter import load_delegate
    opts = dict(options or {})
    return load_delegate(spec, opts)


def parse_delegate_options(pairs):
    out = {}
    for kv in pairs or []:
        if "=" not in kv:
            raise ValueError("--delegate-option wants key=value, got %r" % (kv,))
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def delegated_node_count(interp):
    """How many graph nodes a delegate swallowed -- the only honest way to tell
    from python whether the delegate actually took the model or silently
    declined. Returns (n_delegate_nodes, n_nodes) or (None, None)."""
    try:
        ops = interp._get_ops_details()          # noqa: SLF001 - no public API
    except Exception:                            # noqa: BLE001
        return None, None
    n = sum(1 for o in ops if "elegate" in str(o.get("op_name", "")))
    return n, len(ops)


# ===========================================================================
# 1. The command seam: MAVLink setpoints out, JSON goals in
# ===========================================================================
# The velocity setpoint leaves this process as MAVLink and nothing else: the
# runner is the only writer of the control stream, so an intermediate datagram
# format between it and the autopilot bought nothing but a hop to lose.
#
# The goal port stays, and stays JSON over UDP. It is the one link that has to
# cross Wi-Fi from the operator's laptop, which no in-board pipe can carry, and
# it is a handful of human-readable verbs rather than a packed struct -- see
# `TaskCommandPort` below and ground/send_goal.py.
#
# `yaw_rate_toward` and `TaskCommandPort` remain verbatim from
# src/superfly/common/vel_command_transport.py; only the binary velocity
# datagram it also defined is gone.

CMD_PORT = 15021
DEFAULT_CMD_HOST = "127.0.0.1"


def yaw_rate_toward(yaw_cmd, yaw_meas, dt, yaw_rate_max):
    """Heading setpoint -> yaw RATE (vel_command_transport.yaw_rate_toward)."""
    err = math.atan2(math.sin(yaw_cmd - yaw_meas), math.cos(yaw_cmd - yaw_meas))
    if dt <= 0.0:
        return 0.0
    rate = err / dt
    return float(np.clip(rate, -abs(yaw_rate_max), abs(yaw_rate_max)))


def wrap_pi(angle):
    """Wrap an angle into (-pi, pi] -- audit defect 8 (`swap_yaw` has no wrap).

    Deliberately a SEPARATE helper rather than a wrap folded into `swap_yaw`
    itself.  `swap_yaw` is vendored (section 6) and the 43200-case equivalence
    test pins `swap_yaw(swap_yaw(y)) == y` exactly; a wrap inside it turns
    y = -pi into +pi and that check hard-fails.  Every consumer on the yaw
    RATE path is 2*pi-periodic, so the wrap is only actually needed where an
    absolute yaw ANGLE goes on the wire (`--yaw-out angle`), and it is applied
    exactly there.
    """
    a = float(angle)
    a = a - 2.0 * math.pi * math.floor((a + math.pi) / (2.0 * math.pi))
    if a <= -math.pi:                       # floor lands on [-pi, pi)
        a += 2.0 * math.pi
    return a


class MavlinkVelPublisher:
    """The control stream: one setpoint per tick, straight to the autopilot.

    The docs/4.7 all-MAVLink ruling: the runner speaks straight to the PX4
    onboard mavlink instance at udpout:127.0.0.1:14556 (voxl-px4-start), one
    SET_POSITION_TARGET_LOCAL_NED per tick, velocity + yaw-rate mask 0x07C7,
    world NED -- the frame the policy tail already produces, so nothing is
    converted on the way out. Vendored from
    scripts/sfvc_mavlink_adapter.py, whose desk loopback test proved the
    message content; a 1 Hz onboard-computer heartbeat rides along. Pure
    translator semantics are preserved: if the caller stops calling send(),
    nothing goes out and PX4's COM_OF_LOSS_T failsafe owns the aircraft
    (stage (viii)/(xi) verified) -- no keepalives are synthesized here.
    """

    MASK_VEL_YAWRATE = 0x07C7
    # --yaw-out angle (audit defect 2): the mask the SIM has always used --
    # superfly.common.px4_offboard.send_velocity_target_ned, IGNORE_POS(1|2|4)
    # | IGNORE_ACC(64|128|256) | IGNORE_YAW_RATE(2048) = 0x09C7, so PX4 runs
    # its own yaw controller on the absolute setpoint instead of executing a
    # bang-bang rate this process differentiated at the nominal dt.
    MASK_VEL_YAW = 0x09C7

    def __init__(self, host="127.0.0.1", port=14556,
                 source_system=1, source_component=191, yaw_mode="rate"):
        from pymavlink import mavutil          # in the o4runtime bundle
        self._mavutil = mavutil
        self.yaw_mode = str(yaw_mode)
        self.addr = (host, int(port))
        self._mav = mavutil.mavlink_connection(
            "udpout:%s:%d" % (host, int(port)),
            source_system=int(source_system),
            source_component=int(source_component))
        self._seq = 0
        self.sent = 0
        self.send_errors = 0
        self._last_hb = 0.0
        self._boot = time.time()

    @property
    def seq(self):
        return self._seq

    def send(self, vx, vy, vz, yaw_rate, flags=0, yaw=0.0):
        seq = self._seq
        now = time.time()
        # The field mapping is send_velocity_target_ned's, verbatim: the two
        # trailing floats are (yaw, yaw_rate), and exactly one of them is live.
        if self.yaw_mode == "angle":
            mask, yaw_f, rate_f = self.MASK_VEL_YAW, float(yaw), 0.0
        else:
            mask, yaw_f, rate_f = self.MASK_VEL_YAWRATE, 0.0, float(yaw_rate)
        try:
            if now - self._last_hb >= 1.0:
                self._mav.mav.heartbeat_send(
                    self._mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                    self._mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                self._last_hb = now
            self._mav.mav.set_position_target_local_ned_send(
                int((now - self._boot) * 1000) & 0xFFFFFFFF,
                1, 1,                                  # target: the autopilot
                self._mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                mask,
                0.0, 0.0, 0.0,
                float(vx), float(vy), float(vz),
                0.0, 0.0, 0.0,
                yaw_f, rate_f)
            self.sent += 1
        except OSError as exc:
            self.send_errors += 1
            if self.send_errors < 3 or self.send_errors % 100 == 0:
                print("[mavlink_out] send failed (%s); %d setpoints dropped "
                      "so far." % (exc, self.send_errors),
                      file=sys.stderr, flush=True)
        self._seq = (seq + 1) & 0xFFFFFFFF
        return seq

    # --auto-takeoff only (doc 15.11.6.4): the three COMMAND_LONGs of
    # superfly.common.px4_offboard (set_offboard_mode / arm / disarm), verbatim.
    def request(self, what):
        m = self._mavutil.mavlink
        try:
            if what == "offboard":
                self._mav.mav.command_long_send(
                    1, 1, m.MAV_CMD_DO_SET_MODE, 0,
                    m.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    PX4_CUSTOM_MAIN_MODE_OFFBOARD, 0, 0, 0, 0, 0)
            elif what in ("arm", "disarm"):
                self._mav.mav.command_long_send(
                    1, 1, m.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                    1 if what == "arm" else 0, 0, 0, 0, 0, 0, 0)
            else:
                raise ValueError(what)
        except OSError as exc:
            self.send_errors += 1
            print("[mavlink_out] %s request failed (%s)" % (what, exc),
                  file=sys.stderr, flush=True)

    def close(self):
        try:
            self._mav.close()
        except Exception:
            pass


class PrintOnlyPublisher:
    """--enable_test: print the setpoint, send nothing. The same duck-type as
    MavlinkVelPublisher, so the flight loop cannot tell the difference.

    The point of the smoke test is that the vehicle is provably untouched, and
    the way that is guaranteed here is by absence: when this stands in for the
    real publisher, `main()` never constructs a MavlinkVelPublisher, so no
    MAVLink connection is opened, no heartbeat goes out, and there is no socket
    for a setpoint to escape through even if something upstream misbehaves.
    """

    def __init__(self, every=1, yaw_mode="rate"):
        self.addr = ("<print>", 0)
        self.every = max(1, int(every))
        self.yaw_mode = str(yaw_mode)
        self._seq = 0
        self.sent = 0
        self.send_errors = 0

    @property
    def seq(self):
        return self._seq

    def send(self, vx, vy, vz, yaw_rate, flags=0, yaw=0.0):
        seq = self._seq
        if seq % self.every == 0:
            if self.yaw_mode == "angle":
                tail = ("yaw_sp=%+7.2f deg (mask 0x09C7)"
                        % math.degrees(float(yaw)))
            else:
                tail = "yaw_rate=%+6.3f rad/s" % yaw_rate
            print("[test] seq=%-6d v_ned=(%+6.3f, %+6.3f, %+6.3f) m/s  "
                  "|v_xy|=%5.3f  %s   NOT SENT"
                  % (seq, vx, vy, vz, math.hypot(vx, vy), tail),
                  flush=True)
        self.sent += 1
        self._seq = (seq + 1) & 0xFFFFFFFF
        return seq

    def request(self, what):
        print("[test] WOULD request %s -- NOT SENT (no MAVLink connection "
              "exists under --enable_test)" % what.upper(), flush=True)

    def close(self):
        pass


class TaskCommandPort:
    """SET_GOAL / HOLD / RESUME / STATUS in, REACHED out (the goal port).

    Vendored from vel_command_transport.TaskCommandPort with its threading
    contract intact: a daemon thread owns the socket, the flight loop calls
    `take_goal()` once per tick (which is what makes a mid-flight SET_GOAL
    atomic with respect to a tick), and nothing here can raise into the loop.
    """

    def __init__(self, host=DEFAULT_CMD_HOST, port=CMD_PORT, start=True,
                 config=None):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, int(port)))
        self.port = self._sock.getsockname()[1]
        self._lock = threading.Lock()
        self._pending_goal = None
        self._takeoff_req = False
        self._hold = False
        self._seq = 0
        self._commander = None
        # Static for the run: the caps, lookahead and model names the BOARD is
        # actually flying. The ground dashboard prints these rather than its own
        # argparse defaults -- a HUD showing laptop settings beside a board
        # flying different ones is a quiet lie.
        self.config = dict(config or {})
        self._status = {"phase": None, "dist_to_goal": None,
                        "reached": False, "awaiting_goal": False,
                        "goal": None}
        # The newest SET_GOAL the runner refused (--goal-z-mode current/hold).
        # It rides in the STATUS reply so an operator who missed the UDP event
        # still sees WHY the vehicle did not move.
        self._last_reject = None
        self._stop = threading.Event()
        self._thread = None
        self.commands = 0
        self.errors = 0
        if start:
            self.start()

    def start(self):
        if self._thread is not None:
            return
        self._sock.settimeout(0.2)
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="task-cmd-port")
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self._sock.close()
        except OSError:
            pass

    @property
    def hold(self):
        with self._lock:
            return self._hold

    @property
    def seq(self):
        with self._lock:
            return self._seq

    @property
    def commander(self):
        with self._lock:
            return self._commander

    def take_goal(self, default_z=None):
        """Pop the pending SET_GOAL as a world-NED xyz array, or None."""
        with self._lock:
            pending = self._pending_goal
            self._pending_goal = None
        if pending is None:
            return None
        x, y, z = pending
        if z is None:
            z = 0.0 if default_z is None else float(default_z)
        return np.array([float(x), float(y), float(z)], dtype=np.float64)

    def force_hold(self):
        """The runner itself asks for a HOLD (estimator jump); an operator
        RESUME clears it exactly like an operator HOLD."""
        with self._lock:
            self._hold = True
            self._seq += 1

    def take_takeoff(self):
        """Pop a pending TAKEOFF request -> bool."""
        with self._lock:
            req, self._takeoff_req = self._takeoff_req, False
        return req

    def take_goal_raw(self):
        """Pop the pending SET_GOAL as `(x, y, z_or_None)`, or None.

        Additive: `take_goal` above stays verbatim from vel_command_transport.
        The goal-z modes have to know whether the operator actually TYPED a z,
        which `take_goal` has already thrown away by the time it returns.
        """
        with self._lock:
            pending = self._pending_goal
            self._pending_goal = None
        return pending

    def note_reject(self, info):
        """Record a refused SET_GOAL for the next STATUS reply."""
        with self._lock:
            self._last_reject = dict(info)

    def publish_status(self, phase, dist_to_goal=None, reached=False,
                       goal=None, awaiting_goal=False, goal_dir_body=None,
                       vel_body=None, hold_reason=None, viz=None):
        """Snapshot for the next STATUS reply. Called once per tick.

        `goal_dir_body` and `vel_body` are the ONLY telemetry that leaves this
        process besides the setpoint, and they leave through a socket that
        already exists and already replies -- no new port, no new protocol, and
        nothing here can command the vehicle.

        They are published in BODY FLU, not world NED, because the only
        consumer is a camera overlay: the ground viewer projects them straight
        through the training pinhole (ground/viz_primitives.py) and never
        touches a frame conversion. The runner already holds both vectors in
        exactly this frame -- `_state_to_model_input` computes `R_enu.T @ ...`
        for the net -- so this is a copy, not a computation, and the ground
        cannot get the rotation wrong because it never does one.
        """
        with self._lock:
            self._status = {
                "phase": phase,
                "dist_to_goal": (None if dist_to_goal is None
                                 else float(dist_to_goal)),
                "reached": bool(reached),
                "awaiting_goal": bool(awaiting_goal),
                "goal": (None if goal is None
                         else [float(v) for v in np.asarray(goal).ravel()[:3]]),
                # Body FLU (x fwd, y left, z up); null when there is nothing
                # honest to draw. A viewer that draws null draws nothing.
                "goal_dir_body": (None if goal_dir_body is None
                                  else [float(v) for v in
                                        np.asarray(goal_dir_body).ravel()[:3]]),
                "vel_body": (None if vel_body is None
                             else [float(v) for v in
                                   np.asarray(vel_body).ravel()[:3]]),
                "hold_reason": hold_reason,
                # The `out` dict handheld_viz's renderer draws, verbatim in
                # shape: modes/mode_idx/alphas/vel/yaw_rate/enc_ms/head_ms.
                # Upstream builds it by running the net on the ground station;
                # the board ran it already, so it reports the same thing and
                # the ground draws the identical picture. Null while holding.
                "viz": viz,
            }

    def send_event(self, event, **fields):
        with self._lock:
            addr = self._commander
            payload = {"event": event, "seq": self._seq}
        if addr is None:
            return False
        payload.update(fields)
        return self._send_json(payload, addr)

    def send_reached(self, goal, pos, dist_to_goal, phase="POLICY"):
        return self.send_event(
            "REACHED",
            goal=[float(v) for v in np.asarray(goal).ravel()[:3]],
            pos=[float(v) for v in np.asarray(pos).ravel()[:3]],
            dist_to_goal=float(dist_to_goal),
            phase=phase,
        )

    def _send_json(self, payload, addr):
        try:
            self._sock.sendto((json.dumps(payload) + "\n").encode("utf-8"), addr)
            return True
        except OSError:
            return False

    def _serve(self):
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
            except (socket.timeout, BlockingIOError):
                continue
            except OSError:
                break
            with self._lock:
                self._commander = addr
            for line in data.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    reply = self._handle(line)
                except Exception as exc:                # pragma: no cover
                    self.errors += 1
                    reply = {"ok": False,
                             "error": "%s: %s" % (type(exc).__name__, exc)}
                self._send_json(reply, addr)

    def _handle(self, line):
        try:
            msg = json.loads(line)
        except ValueError as exc:
            self.errors += 1
            return {"ok": False, "error": "bad JSON: %s" % (exc,)}
        if not isinstance(msg, dict):
            self.errors += 1
            return {"ok": False, "error": "command must be a JSON object"}
        cmd = str(msg.get("cmd", "")).upper()

        if cmd == "STATUS":
            with self._lock:
                out = {"ok": True, "cmd": "STATUS", "seq": self._seq,
                       "hold": self._hold, "config": self.config,
                       "last_reject": self._last_reject}
                out.update(self._status)
            return out

        if cmd == "SET_GOAL":
            try:
                x = float(msg["x"])
                y = float(msg["y"])
            except (KeyError, TypeError, ValueError):
                self.errors += 1
                return {"ok": False, "cmd": cmd,
                        "error": "SET_GOAL needs numeric x and y"}
            z = msg.get("z", None)
            if z is not None:
                try:
                    z = float(z)
                except (TypeError, ValueError):
                    self.errors += 1
                    return {"ok": False, "cmd": cmd,
                            "error": "SET_GOAL z must be numeric or absent"}
            if not all(math.isfinite(v) for v in (x, y, z) if v is not None):
                self.errors += 1
                return {"ok": False, "cmd": cmd,
                        "error": "SET_GOAL x/y/z must be finite"}
            with self._lock:
                self._pending_goal = (x, y, z)
                self._seq += 1
                seq = self._seq
                # Retire the arrival flags AT ACCEPTANCE, not at the next tick
                # (vel_command_transport.py carries the full reasoning: a
                # STATUS landing in the gap otherwise reports the PREVIOUS
                # goal's arrival against the NEW seq, and the executor
                # completes a leg that was never flown).
                self._status["reached"] = False
                self._status["awaiting_goal"] = False
                self._status["dist_to_goal"] = None
            self.commands += 1
            return {"ok": True, "cmd": cmd, "seq": seq, "goal": [x, y, z]}

        if cmd == "TAKEOFF":
            # --auto-takeoff (doc 15.11.6.4): the ONLY thing that makes the
            # runner ask PX4 to arm. Accepted here = queued; the runner's
            # preflight decides, and answers with a TAKEOFF_ACCEPTED /
            # TAKEOFF_REJECTED event.
            with self._lock:
                self._takeoff_req = True
                self._seq += 1
                seq = self._seq
            self.commands += 1
            return {"ok": True, "cmd": cmd, "seq": seq, "queued": True}

        if cmd in ("HOLD", "RESUME"):
            with self._lock:
                self._hold = (cmd == "HOLD")
                self._seq += 1
                seq, hold = self._seq, self._hold
            self.commands += 1
            return {"ok": True, "cmd": cmd, "seq": seq, "hold": hold}

        self.errors += 1
        return {"ok": False, "error": "unknown cmd %r" % (cmd,)}


def apply_goal_command(port, goal_enu, pos_enu, retarget=None, default_z=None):
    """Apply at most one queued SET_GOAL, between ticks. -> (goal, changed)."""
    if port is None:
        return goal_enu, False
    new_goal = port.take_goal(default_z=default_z)
    if new_goal is None:
        return goal_enu, False
    if retarget is not None:
        retarget(new_goal, pos_enu)
    return new_goal, True


GOAL_Z_MODES = ("absolute", "current", "hold")


def apply_goal_command_z(port, goal_ned, pos_ned, retarget=None,
                         cruise_d=None, mode="absolute", max_delta=3.0,
                         on_reject=None):
    """`apply_goal_command` plus the goal-z reference frame -- audit defect 1.

    `absolute` delegates to `apply_goal_command` unchanged, so the flight
    command that has been flown keeps its exact behaviour: the operator's z is
    an EKF-local NED absolute, and an omitted z falls back to the `cruise_d`
    first-sample latch.

    `current` keeps the operator's z only when it is within `max_delta` of the
    altitude the EKF reports RIGHT NOW; further than that and the whole
    SET_GOAL is refused (the runner keeps its previous goal and its awaiting
    state).  An omitted z takes the current EKF z, not the latch.

    `hold` never uses the operator's z at all: the reference line is forced
    horizontal, which is the only thing that means anything under --planar.
    There is nothing to sanity-check in that mode, so nothing is refused.

    -> (goal_ned, changed)
    """
    if mode == "absolute":
        return apply_goal_command(port, goal_ned, pos_ned, retarget=retarget,
                                  default_z=cruise_d)
    if port is None:
        return goal_ned, False
    pending = port.take_goal_raw()
    if pending is None:
        return goal_ned, False
    x, y, z = pending
    z_now = None if pos_ned is None else float(np.asarray(pos_ned).ravel()[2])
    if z_now is None:                       # no state yet: nothing to refer to
        return goal_ned, False
    if mode == "hold" or z is None:
        z_use = z_now
    else:
        delta = abs(float(z) - z_now)
        if delta > float(max_delta):
            if on_reject is not None:
                on_reject(dict(x=float(x), y=float(y), z=float(z),
                               pos_z=z_now, delta=delta,
                               max_delta=float(max_delta), mode=mode))
            return goal_ned, False
        z_use = float(z)
    new_goal = np.array([float(x), float(y), float(z_use)], dtype=np.float64)
    if retarget is not None:
        retarget(new_goal, pos_ned)
    return new_goal, True


def altitude_vz_ned(ref_d, pos_d, vel_d, kp=1.0, kd=0.0, max_vz=0.5):
    """NED vz command that holds `ref_d` -- audit fix 4.

    Mirrors `scripts/diffaero_vel_offboard.py::altitude_vz_ned` exactly; that
    one is written in ENU (`alt_err = cruise_alt - pos_up`,
    `vz = -kp*alt_err - kd*vz_meas_ned`), and substituting pos_up = -pos_d,
    cruise_alt = -ref_d gives `alt_err = pos_d - ref_d`, i.e.

        vz_d = -kp * (pos_d - ref_d) - kd * vel_d

    Sign check, which is the whole point of the function: above the reference
    means pos_d < ref_d in NED, so (pos_d - ref_d) < 0 and vz_d comes out
    POSITIVE = down = back toward the reference.
    """
    err_d = float(pos_d) - float(ref_d)
    vz = -float(kp) * err_d - float(kd) * float(vel_d)
    return float(np.clip(vz, -abs(max_vz), abs(max_vz)))


RESET_KEYS = ("xy", "z", "vxy", "vz", "heading")


def parse_vehicle_local_position(text):
    """`px4-listener vehicle_local_position` text -> dict(counters{}, delta_xy,
    delta_z, delta_heading), or None if the counters are not in it."""
    t = re.sub(r"\x1b\[[0-9;]*m", "", text or "")
    counters = {}
    for k in RESET_KEYS:
        m = re.search(r"\b%s_reset_counter: (\d+)" % k, t)
        if m is None:
            return None
        counters[k] = int(m.group(1))
    m = re.search(r"\bdelta_xy: \[([-\d.e+]+), ([-\d.e+]+)\]", t)
    mz = re.search(r"\bdelta_z: ([-\d.e+]+)", t)
    mh = re.search(r"\bdelta_heading: ([-\d.e+]+)", t)
    return dict(counters=counters,
                delta_xy=(float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0),
                delta_z=float(mz.group(1)) if mz else 0.0,
                delta_heading=float(mh.group(1)) if mh else 0.0)


def query_vehicle_local_position(cmd=None):
    """One read of the EKF2's vehicle_local_position (~40 ms on the VOXL 2).
    $SUPERFLY_RESET_QUERY_CMD replaces the command (desk tests)."""
    import shlex
    import subprocess
    cmd = cmd or os.environ.get("SUPERFLY_RESET_QUERY_CMD",
                                "px4-listener vehicle_local_position")
    try:
        out = subprocess.run(shlex.split(cmd), capture_output=True, text=True,
                             timeout=2.0).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_vehicle_local_position(out)


class EstimatorResetWatch:
    """EKF2 reset watch (doc 15.11.6.4), replacing any kinematic guess.

    Detection: ODOMETRY.reset_counter on the state pipe (30 Hz; PX4 1.14 sums
    the five vehicle_local_position reset counters into it). Size: on a change,
    ONE query of vehicle_local_position gives the EKF2's own delta_xy / delta_z
    / delta_heading of its most recent reset; a delta is used only for the kinds
    whose counter actually went up since the last snapshot. If a kind went up by
    more than one (several resets between two queries) its delta covers only the
    last one: the result is flagged `exact=False`.

    -> None, or dict(dp[3], dyaw, p_pred[3], p_new[3], exact, kinds, counters)
       in the form reanchor_ned() takes: p_new = the post-reset estimate,
       p_pred = the same instant in the pre-reset frame (p_new - delta)."""

    def __init__(self, query=query_vehicle_local_position):
        self.query = query
        self._counter = None
        self._snap = None
        self.resets = 0

    def prime(self):
        """Per-kind baseline before flight (None if the query does not work)."""
        q = self.query()
        self._snap = None if q is None else dict(q["counters"])
        return q

    def update(self, counter, pos_ned):
        if counter is None:
            return None
        counter = int(counter)
        prev, self._counter = self._counter, counter
        if prev is None or counter == prev:
            return None
        self.resets += 1
        q = self.query()
        p_new = np.asarray(pos_ned, dtype=np.float64).copy()
        if q is None:
            return dict(dp=np.zeros(3), dyaw=0.0, p_pred=p_new, p_new=p_new,
                        exact=False, kinds=[], counters=None,
                        why="vehicle_local_position unreadable")
        had_base = self._snap is not None
        base = self._snap or {}
        up = {k: q["counters"][k] - base.get(k, q["counters"][k] - 1)
              for k in RESET_KEYS}
        self._snap = dict(q["counters"])
        kinds = [k for k in RESET_KEYS if up[k] > 0]
        dx, dy = q["delta_xy"] if "xy" in kinds else (0.0, 0.0)
        dz = q["delta_z"] if "z" in kinds else 0.0
        dyaw = q["delta_heading"] if "heading" in kinds else 0.0
        dp = np.array([dx, dy, dz], dtype=np.float64)
        exact = had_base and all(up[k] <= 1 for k in RESET_KEYS)
        why = (None if exact else "no per-kind baseline (every delta applied)"
               if not had_base else "several resets since the last query")
        return dict(dp=dp, dyaw=float(dyaw), p_pred=p_new - dp, p_new=p_new,
                    exact=exact, kinds=kinds, counters=dict(q["counters"]), why=why)


def reanchor_ned(x_ned, jump):
    """A point expressed in the PRE-reset EKF frame -> the same physical point in
    the POST-reset frame, taking the reset as a pure frame shift about the
    vehicle (it did not move; its estimate did): rotate about the vehicle's
    pre-reset position by dyaw (NED yaw, clockwise from above), then translate
    to its post-reset estimate."""
    x = np.asarray(x_ned, dtype=np.float64)
    rel = x - jump["p_pred"]
    c, s = math.cos(jump["dyaw"]), math.sin(jump["dyaw"])
    rot = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
    return jump["p_new"] + rot


class TakeoffFrame:
    """--goal-frame takeoff-flu (doc 15.11.6.4): goals relative to the aircraft
    as it stood when TAKEOFF was accepted.

        origin  the EKF2 position at that instant (on the ground)
        x       forward along the heading it had then     [m]
        y       to its LEFT                               [m]
        z       UP, height above that ground point        [m]

    Heading only (roll/pitch ignored), so the axes are level. Latched ONCE per
    run; every later SET_GOAL is read in the same frame. The conversion uses
    the EKF2's own measured pose, so it is exact in the EKF2's local NED
    frame whatever that frame's absolute orientation is -- on this aircraft
    the EKF2 fuses VIO position and yaw (EKF2_EV_CTRL 15), so its "north" is
    wherever VIO initialised, which an operator cannot know. Everything
    downstream (climb target, ALIGN bearing, policy target, arrival) uses the
    converted EKF NED goal.

        N = p0_n + x cos(psi) + y sin(psi)
        E = p0_e + x sin(psi) - y cos(psi)
        D = p0_d - z
    """

    def __init__(self, p0_ned, yaw_ned):
        self.p0 = np.asarray(p0_ned, dtype=np.float64).copy()
        self.yaw = float(yaw_ned)

    def to_ned(self, x, y, z):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([self.p0[0] + x * c + y * s,
                         self.p0[1] + x * s - y * c,
                         self.p0[2] - z], dtype=np.float64)

    def from_ned(self, g):
        d = np.asarray(g, dtype=np.float64) - self.p0
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([d[0] * c + d[1] * s, d[0] * s - d[1] * c, -d[2]])

    def reanchor(self, jump):
        """Follow an EKF2 reset (EstimatorResetWatch) so later SET_GOALs
        still mean the same physical point."""
        self.p0 = reanchor_ned(self.p0, jump)
        self.yaw = wrap_pi(self.yaw + jump["dyaw"])

    def describe(self):
        return ("takeoff frame: origin EKF NED (%+.2f, %+.2f, %+.2f), x = "
                "heading %.1f deg (EKF NED yaw), y = left, z = up"
                % (self.p0[0], self.p0[1], self.p0[2], math.degrees(self.yaw)))


class AutoTakeoff:
    """--auto-takeoff: ARM + OFFBOARD + climb on an explicit TAKEOFF command,
    then hand over to the ordinary ALIGN -> POLICY path (doc 15.11.6.4).

    Mirrors the SITL offboard's CLIMB phase (depthnav_vel_offboard.py: re-send
    OFFBOARD + arm until PX4 has both, climb at --climb-rate straight up with
    the heading held, done when within --arrive-tol of --climb-alt and slower
    than --settle-speed). Pure: `step()` returns what to send, the runner sends
    it, so the whole machine is testable without a vehicle.

    Pilot authority (user, 2026-09-23: "the program should never compete with
    the pilot"): the runner asks PX4 for OFFBOARD / arm ONLY between an
    accepted TAKEOFF and the first OFFBOARD entry, and only while nobody else
    has touched the aircraft. ANY of these makes the machine PILOT for the rest
    of the run, after which it never asks for anything again:
      - OFFBOARD was reached and then left (mode switch, stick override);
      - while arming, PX4's main mode became anything other than the mode it
        had when TAKEOFF was accepted or OFFBOARD (the pilot flipped a switch);
      - PX4 was armed and then disarmed (the pilot, or a kill switch);
      - disarmed during the climb.
    The runner never sends DISARM: an aborted arming is left to PX4's own
    COM_DISARM_PRFLT (20 s on this aircraft). The pilot re-entering OFFBOARD
    goes through the normal hand-over latch (ALIGN, new START frame, fresh GRU
    state).

      IDLE     nothing requested; the runner behaves exactly as without the flag
      ARMING   zero velocity, heading held; OFFBOARD + ARM re-requested every
               retry_s until both hold, or arm_timeout_s -> ABORTED
      CLIMB    vz up at climb_rate to the GOAL's altitude (target_d, given
               with the request), settle, then HANDED (the latch may fire)
      HANDED   normal operation (ALIGN -> POLICY)
      PILOT    OFFBOARD was left (or the vehicle disarmed) -> never request again
      ABORTED  arming did not complete in arm_timeout_s; requests stop, never
               retried; no DISARM is sent (PX4's COM_DISARM_PRFLT disarms)
    """

    IDLE, ARMING, CLIMB, HANDED, PILOT, ABORTED = (
        "IDLE", "ARMING", "CLIMB", "HANDED", "PILOT", "ABORTED")

    def __init__(self, climb_rate=1.0, arrive_tol=0.3,
                 settle_speed=0.2, arm_timeout_s=10.0, retry_s=1.0,
                 climb_timeout_s=30.0):
        self.climb_rate = float(climb_rate)
        self.arrive_tol = float(arrive_tol)
        self.settle_speed = float(settle_speed)
        self.arm_timeout_s = float(arm_timeout_s)
        self.retry_s = float(retry_s)
        self.climb_timeout_s = float(climb_timeout_s)
        self.phase = self.IDLE
        self._t_req = None
        self._t_try = None
        self._t_climb = None
        self._yaw_hold = None
        self._z_ground = None
        self._target_d = None
        self._was_offboard = False
        self._was_armed = False
        self._mode_at_request = None
        self.reason = None

    def reanchor(self, jump):
        """Follow an estimator jump: the climb target's D moves with the frame."""
        if self._target_d is not None:
            self._target_d += float(jump["p_new"][2] - jump["p_pred"][2])
        if self._z_ground is not None:
            self._z_ground += float(jump["p_new"][2] - jump["p_pred"][2])

    @property
    def active(self):
        """True while the takeoff owns the setpoint (ARMING / CLIMB)."""
        return self.phase in (self.ARMING, self.CLIMB)

    def request(self, now, checks, yaw_ned, target_d, main_mode=None):
        """An operator TAKEOFF. `checks`: name -> bool, all must hold;
        `target_d`: the altitude to climb to, EKF NED down (the goal's).
        -> (accepted, reason)."""
        if self.phase != self.IDLE:
            return False, "takeoff already %s (one per run)" % self.phase
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            return False, "preflight: " + ", ".join(failed)
        self.phase = self.ARMING
        self._t_req = now
        self._t_try = None
        self._yaw_hold = float(yaw_ned)
        self._target_d = float(target_d)
        self._mode_at_request = main_mode
        return True, "arming"

    def step(self, now, armed, offboard, pos_ned, vel_ned, main_mode=None):
        """-> dict(phase, override, v_ned, yaw_sp_ned, send, log, allow_latch)."""
        out = dict(override=False, v_ned=None, yaw_sp_ned=self._yaw_hold,
                   send=[], log=[], allow_latch=True)
        # Pilot authority first: any sign that someone else is handling the
        # aircraft ends every automatic request for good.
        why = None
        if self.phase in (self.ARMING, self.CLIMB):
            if self._was_offboard and not offboard:
                why = "PX4 left OFFBOARD"
            elif self._was_armed and not armed:
                why = "PX4 was disarmed"
            elif (self.phase == self.ARMING and main_mode is not None
                  and not offboard and self._mode_at_request is not None
                  and main_mode != self._mode_at_request):
                why = ("PX4's mode changed %s -> %s while arming"
                       % (self._mode_at_request, main_mode))
        if offboard:
            self._was_offboard = True
        if armed:
            self._was_armed = True
        if why is not None:
            during = self.phase
            self.phase = self.PILOT
            out["log"].append("[takeoff] PILOT has the aircraft (%s during %s); "
                              "the runner will not request OFFBOARD or arm again."
                              % (why, during))
        if self.phase == self.HANDED and not offboard:
            self.phase = self.PILOT
            out["log"].append("[takeoff] PILOT has the aircraft; no further "
                              "automatic requests.")
        if self.phase == self.ARMING:
            out.update(override=True, v_ned=np.zeros(3), allow_latch=False)
            if armed and offboard:
                self.phase = self.CLIMB
                self._t_climb = now
                self._z_ground = float(pos_ned[2])
                out["log"].append("[takeoff] ARMED + OFFBOARD; climbing from NED z "
                                  "%+.2f to the goal's %+.2f (%.1f m up) at %.1f m/s."
                                  % (self._z_ground, self._target_d,
                                     self._z_ground - self._target_d, self.climb_rate))
            elif now - self._t_req > self.arm_timeout_s:
                self.phase = self.ABORTED
                self.reason = ("no ARMED+OFFBOARD within %.0f s (armed=%s "
                               "offboard=%s)" % (self.arm_timeout_s, armed, offboard))
                out["log"].append("[takeoff] ABORTED: %s. No further requests; "
                                  "the runner sends no DISARM (PX4's own "
                                  "COM_DISARM_PRFLT disarms an idle armed "
                                  "vehicle)." % self.reason)
                out.update(override=False, allow_latch=True)
            elif self._t_try is None or now - self._t_try >= self.retry_s:
                self._t_try = now
                if not offboard:
                    out["send"].append("offboard")
                if not armed:
                    out["send"].append("arm")
        if self.phase == self.CLIMB:
            target_d = self._target_d
            pos_d = float(pos_ned[2])
            below = pos_d > target_d + self.arrive_tol
            speed = float(np.linalg.norm(vel_ned))
            v = np.array([0.0, 0.0, -self.climb_rate if below else 0.0])
            out.update(override=True, v_ned=v, allow_latch=False)
            if not below and speed < self.settle_speed:
                self.phase = self.HANDED
                out["log"].append("[takeoff] at %.2f m above ground, |v| %.2f m/s: "
                                  "handing over to ALIGN -> POLICY."
                                  % (self._z_ground - pos_d, speed))
                out.update(override=False, allow_latch=True)
            elif now - self._t_climb > self.climb_timeout_s:
                # Not a failure of control authority: stop climbing, hold, and
                # hand over where we are rather than hang in the takeoff.
                self.phase = self.HANDED
                out["log"].append("[takeoff] climb TIMEOUT after %.0f s at %.2f m; "
                                  "handing over where we are."
                                  % (self.climb_timeout_s, self._z_ground - pos_d))
                out.update(override=False, allow_latch=True)
        out["phase"] = self.phase
        return out


# ===========================================================================
# 2. Frames -- vendored from scripts/rgb_bridge_voxl.py
# ===========================================================================
# CANONICAL SOURCE: scripts/rgb_bridge_voxl.py sections 2-4 (the MPA client, the
# colour conversion and the resize). That file is the reader that has actually
# run on this board; it is copied rather than imported because the bundle may be
# the only superfly artefact on the board, and rgb_bridge_voxl.py may not be
# next to it. The struct layout, the magic number and the ITU-R BT.601
# fixed-point constants are quoted from libmodal-pipe / OpenCV there, with the
# provenance comments; they are not re-derived here.

MPA_BASE_DIR = "/run/mpa/"
CAMERA_MAGIC_NUMBER = 0x564F584C            # "VOXL", magic_number.h
_CAM_META = struct.Struct("<Iqihhiiihhhh")  # packed camera_image_metadata_t
assert _CAM_META.size == 40, "camera_image_metadata_t must be 40 packed bytes"
_META_FIELDS = ("magic_number", "timestamp_ns", "frame_id", "width", "height",
                "size_bytes", "stride", "exposure_ns", "gain", "format",
                "framerate", "reserved")

IMAGE_FORMAT_RAW8 = 0
IMAGE_FORMAT_NV12 = 1
IMAGE_FORMAT_NV21 = 6
IMAGE_FORMAT_YUV420 = 9
IMAGE_FORMAT_RGB = 10
FORMAT_NAMES = {
    IMAGE_FORMAT_RAW8: "RAW8", IMAGE_FORMAT_NV12: "NV12",
    IMAGE_FORMAT_NV21: "NV21", IMAGE_FORMAT_YUV420: "YUV420",
    IMAGE_FORMAT_RGB: "RGB",
}

NET_SIZE = 224                              # AG_NET_SIZE, px4_sim.py


def _log(msg):
    print("[onboard] %s" % (msg,), file=sys.stderr, flush=True)


def parse_meta(raw):
    """Unpack a packed camera_image_metadata_t into a dict."""
    return dict(zip(_META_FIELDS, _CAM_META.unpack(raw)))


class MPAPipeClient:
    """libmodal-pipe client: the request handshake plus non-blocking refills.

    Factored out of the camera reader so the state reader can join
    voxl-mavlink-server's pipe by exactly the same protocol. Framing is the
    subclass's business; this class only keeps `self._buf` fed and reconnects
    when the server goes away.
    """

    name = "mpa"

    def __init__(self, pipe, client_name="onboard_policy", reconnect_s=0.5):
        pipe = str(pipe)
        self.pipe_dir = pipe if pipe.startswith("/") else MPA_BASE_DIR + pipe
        if not self.pipe_dir.endswith("/"):
            self.pipe_dir += "/"
        self.req_path = self.pipe_dir + "request"
        self.client_name = client_name
        self.reconnect_s = float(reconnect_s)
        self._fd = None
        self._data_path = None
        self._buf = bytearray()
        self._next_try = 0.0
        self.reconnects = 0

    @property
    def connected(self):
        return self._fd is not None

    def connect(self):
        now = time.monotonic()
        if now < self._next_try:
            return False
        self._next_try = now + self.reconnect_s
        if not os.path.exists(self.req_path):
            return False                       # server not running yet
        newname = "%s%08d" % (self.client_name[:24], random.randint(0, 99999999))
        data_path = self.pipe_dir + newname
        if os.path.exists(data_path):
            return False
        try:
            req_fd = os.open(self.req_path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno != errno.ENXIO:       # ENXIO = server left stale pipes
                _log("cannot open request pipe %s: %s" % (self.req_path, exc))
            return False
        try:
            os.write(req_fd, newname.encode() + b"\x00")
        except OSError as exc:
            _log("request write failed: %s" % (exc,))
            os.close(req_fd)
            return False
        os.close(req_fd)
        for _ in range(500):                   # the server creates our FIFO
            if os.path.exists(data_path):
                break
            time.sleep(0.002)
        try:
            fd = os.open(data_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            _log("cannot open data pipe %s: %s" % (data_path, exc))
            return False
        self._fd = fd
        self._data_path = data_path
        self._buf = bytearray()
        _log("connected to %s as %s" % (self.pipe_dir, newname))
        return True

    def disconnect(self, why=""):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        if self._data_path:
            try:
                os.remove(self._data_path)     # client.c:1426 does the same
            except OSError:
                pass
        if self._fd is not None:
            self.reconnects += 1
            _log("disconnected%s; will retry" % ((": " + why) if why else ""))
        self._fd = None
        self._data_path = None
        self._buf = bytearray()

    close = disconnect

    def _fill(self, timeout):
        try:
            r, _, _ = select.select([self._fd], [], [], timeout)
        except (OSError, ValueError):
            self.disconnect("select failed")
            return False
        if not r:
            return True                        # just idle
        try:
            data = os.read(self._fd, 1 << 20)
        except BlockingIOError:
            return True
        except OSError as exc:
            self.disconnect("read failed (%s)" % (exc,))
            return False
        if not data:                           # writer closed => server gone
            self.disconnect("pipe closed by server")
            return False
        self._buf += data
        return True


class MPASource(MPAPipeClient):
    """libmodal-pipe camera client, raw protocol, auto-reconnecting.

    `read()` returns `(meta, payload_bytes)` or None if nothing arrived within
    the timeout. It never raises on a camera-server restart.
    """

    def __init__(self, pipe, client_name="onboard_policy", reconnect_s=0.5,
                 max_frame_bytes=64 << 20):
        MPAPipeClient.__init__(self, pipe, client_name=client_name,
                               reconnect_s=reconnect_s)
        self.max_frame_bytes = int(max_frame_bytes)

    def _resync(self):
        magic = struct.pack("<I", CAMERA_MAGIC_NUMBER)
        idx = self._buf.find(magic, 1)
        if idx < 0:
            keep = 3                           # a magic may straddle the edge
            del self._buf[:max(0, len(self._buf) - keep)]
            return False
        _log("resynced, dropped %d bytes (pipe overflow / fell behind)" % idx)
        del self._buf[:idx]
        return True

    def read(self, timeout=0.2):
        """Return (meta, payload) or None. Never raises."""
        if self._fd is None:
            self.connect()
            if self._fd is None:
                time.sleep(min(timeout, self.reconnect_s))
                return None
        deadline = time.monotonic() + timeout
        while True:
            if len(self._buf) >= _CAM_META.size:
                meta = parse_meta(bytes(self._buf[:_CAM_META.size]))
                if meta["magic_number"] != CAMERA_MAGIC_NUMBER:
                    if not self._resync():
                        if time.monotonic() >= deadline:
                            return None
                        if not self._fill(max(0.0, deadline - time.monotonic())):
                            return None
                    continue
                n = meta["size_bytes"]
                if n <= 0 or n > self.max_frame_bytes:
                    self.disconnect("implausible size_bytes=%d" % n)
                    return None
                need = _CAM_META.size + n
                if len(self._buf) >= need:
                    payload = bytes(self._buf[_CAM_META.size:need])
                    del self._buf[:need]
                    return meta, payload
            # The fill attempt must come BEFORE the deadline check: with
            # read(timeout=0.0) -- the main loop's non-blocking drain -- the
            # deadline is already past on entry, so checking it first meant
            # _fill() never ran and the pipe was never read at all. One
            # non-blocking fill per pass, and we exit only when it brings
            # no new bytes past the deadline.
            before = len(self._buf)
            if not self._fill(max(0.0, deadline - time.monotonic())):
                return None
            if time.monotonic() >= deadline and len(self._buf) == before:
                return None


# -- colour conversion, ITU-R BT.601 video range, bit-identical to OpenCV -----
_ITUR_SHIFT = 20
_ITUR_CY = 1220542
_ITUR_CUB = 2116026
_ITUR_CUG = -409993
_ITUR_CVG = -852492
_ITUR_CVR = 1673527


def yuv_to_rgb(y, u, v):
    yy = np.maximum(y.astype(np.int32) - 16, 0) * _ITUR_CY
    ui = u.astype(np.int32) - 128
    vi = v.astype(np.int32) - 128
    half = 1 << (_ITUR_SHIFT - 1)
    out = np.empty(y.shape + (3,), np.uint8)
    out[..., 0] = np.clip((yy + _ITUR_CVR * vi + half) >> _ITUR_SHIFT, 0, 255)
    out[..., 1] = np.clip((yy + _ITUR_CVG * vi + _ITUR_CUG * ui + half) >> _ITUR_SHIFT, 0, 255)
    out[..., 2] = np.clip((yy + _ITUR_CUB * ui + half) >> _ITUR_SHIFT, 0, 255)
    return out


def _upsample2x(c, h, w):
    return np.repeat(np.repeat(c, 2, axis=0), 2, axis=1)[:h, :w]


def nv12_to_rgb(buf, width, height, stride=None, swap_uv=False):
    stride = int(stride or width)
    a = np.frombuffer(buf, dtype=np.uint8)
    ch = height // 2
    need = stride * (height + ch)
    if a.size < need:
        raise ValueError("NV12 payload too small: %d < %d" % (a.size, need))
    y = a[:stride * height].reshape(height, stride)[:, :width]
    uv = a[stride * height:need].reshape(ch, stride)[:, :width]
    u = uv[:, 0::2]
    v = uv[:, 1::2]
    if swap_uv:                                # NV21 is V first
        u, v = v, u
    return yuv_to_rgb(y, _upsample2x(u, height, width),
                      _upsample2x(v, height, width))


def yuv420_to_rgb(buf, width, height, stride=None):
    stride = int(stride or width)
    cstride = max(1, stride // 2)
    a = np.frombuffer(buf, dtype=np.uint8)
    ch, cw = height // 2, width // 2
    y_end = stride * height
    u_end = y_end + cstride * ch
    need = u_end + cstride * ch
    if a.size < need:
        raise ValueError("YUV420 payload too small: %d < %d" % (a.size, need))
    y = a[:y_end].reshape(height, stride)[:, :width]
    u = a[y_end:u_end].reshape(ch, cstride)[:, :cw]
    v = a[u_end:need].reshape(ch, cstride)[:, :cw]
    return yuv_to_rgb(y, _upsample2x(u, height, width),
                      _upsample2x(v, height, width))


def rgb_to_rgb(buf, width, height, stride=None):
    stride = int(stride or width * 3)
    a = np.frombuffer(buf, dtype=np.uint8)
    need = stride * height
    if a.size < need:
        raise ValueError("RGB payload too small: %d < %d" % (a.size, need))
    return a[:need].reshape(height, stride)[:, :width * 3].reshape(height, width, 3)


def raw8_to_rgb(buf, width, height, stride=None):
    stride = int(stride or width)
    a = np.frombuffer(buf, dtype=np.uint8)
    need = stride * height
    if a.size < need:
        raise ValueError("RAW8 payload too small: %d < %d" % (a.size, need))
    gray = a[:need].reshape(height, stride)[:, :width]
    return np.repeat(gray[:, :, None], 3, axis=2)


def payload_to_rgb(payload, meta):
    """Dispatch on meta['format']; returns a contiguous (h,w,3) uint8 array."""
    if isinstance(payload, np.ndarray):
        return np.ascontiguousarray(payload)
    w, h = int(meta["width"]), int(meta["height"])
    stride = int(meta["stride"]) or None
    fmt = int(meta["format"])
    if fmt == IMAGE_FORMAT_NV12:
        rgb = nv12_to_rgb(payload, w, h, stride)
    elif fmt == IMAGE_FORMAT_NV21:
        rgb = nv12_to_rgb(payload, w, h, stride, swap_uv=True)
    elif fmt == IMAGE_FORMAT_YUV420:
        rgb = yuv420_to_rgb(payload, w, h, stride)
    elif fmt == IMAGE_FORMAT_RGB:
        rgb = rgb_to_rgb(payload, w, h, stride)
    elif fmt == IMAGE_FORMAT_RAW8:
        rgb = raw8_to_rgb(payload, w, h, stride)
    else:
        raise ValueError(
            "unsupported MPA image format %d (%s); reconfigure "
            "voxl-camera-server to publish NV12/YUV420/RGB/RAW8"
            % (fmt, FORMAT_NAMES.get(fmt, "unknown")))
    return np.ascontiguousarray(rgb)


def resize_bilinear(img, out_w, out_h):
    """cv2.INTER_LINEAR when cv2 is present, else the numpy bilinear twin.

    Anisotropic on purpose: training squashed 640x480 -> 224x224 with no crop
    (POLICY_RGB_PUBLISH, px4_sim.py), so a square crop would rescale every
    object horizontally. The board normally has no python3-opencv, so the
    fallback is the onboard path -- within ~1-2 LSB of cv2 (docs/rgb_bridge_voxl.md).
    """
    if cv2 is not None:
        return cv2.resize(img, (int(out_w), int(out_h)),
                          interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    x0, x1, ax, y0, y1, ay = _bilinear_taps(w, h, out_w, out_h)
    return _bilinear_apply(img, x0, x1, ax, y0, y1, ay)


def _bilinear_taps(w, h, out_w, out_h):
    """Source columns/rows and weights of the numpy bilinear twin."""
    out_w, out_h = int(out_w), int(out_h)
    xs = (np.arange(out_w) + 0.5) * (w / out_w) - 0.5
    ys = (np.arange(out_h) + 0.5) * (h / out_h) - 0.5
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    ax = (xs - x0)[None, :, None]
    ay = (ys - y0)[:, None, None]
    return x0, x1, ax, y0, y1, ay


def _bilinear_apply(img, x0, x1, ax, y0, y1, ay):
    f = img.astype(np.float32)
    top = f[y0][:, x0] * (1.0 - ax) + f[y0][:, x1] * ax
    bot = f[y1][:, x0] * (1.0 - ax) + f[y1][:, x1] * ax
    return np.clip(top * (1.0 - ay) + bot * ay + 0.5, 0, 255).astype(np.uint8)


def nv12_to_net_frame(buf, width, height, stride=None, swap_uv=False,
                      out_size=NET_SIZE):
    """`resize_bilinear(nv12_to_rgb(...))` on the numpy path, BIT-IDENTICAL, ~4x cheaper.

    The bilinear resample reads at most 2*out_size source rows and columns,
    and the YUV->RGB conversion (clip included) is per pixel, with the chroma
    of pixel (r, c) at (r//2, c//2) exactly as `_upsample2x` puts it -- so
    converting only that sub-grid and resampling it with the same taps
    (re-indexed) and the same float expressions gives the same bytes.  On the
    board 1024x768 -> 224 converts 448x448 instead of 786 432 pixels, which
    also stops the conversion from eating the memory bandwidth the GPU
    encoder shares (doc 15.11.6.4).
    """
    stride = int(stride or width)
    a = np.frombuffer(buf, dtype=np.uint8)
    ch = height // 2
    need = stride * (height + ch)
    if a.size < need:
        raise ValueError("NV12 payload too small: %d < %d" % (a.size, need))
    x0, x1, ax, y0, y1, ay = _bilinear_taps(width, height, out_size, out_size)
    rows = np.unique(np.concatenate([y0, y1]))
    cols = np.unique(np.concatenate([x0, x1]))
    y = a[:stride * height].reshape(height, stride)
    uv = a[stride * height:need].reshape(ch, stride)
    u = uv[:, 0:width:2]
    v = uv[:, 1:width:2]
    if swap_uv:
        u, v = v, u
    sub = yuv_to_rgb(y[rows][:, cols], u[rows // 2][:, cols // 2],
                     v[rows // 2][:, cols // 2])
    return _bilinear_apply(sub, np.searchsorted(cols, x0), np.searchsorted(cols, x1), ax,
                           np.searchsorted(rows, y0), np.searchsorted(rows, y1), ay)


def payload_to_net_frame(payload, meta, out_size=NET_SIZE):
    """`to_net_frame(payload_to_rgb(payload, meta))`, via the sub-grid fast path
    when it applies (numpy resampler, NV12/NV21 payload bytes)."""
    if (cv2 is None and not isinstance(payload, np.ndarray)
            and int(meta["format"]) in (IMAGE_FORMAT_NV12, IMAGE_FORMAT_NV21)):
        return np.ascontiguousarray(nv12_to_net_frame(
            payload, int(meta["width"]), int(meta["height"]),
            int(meta["stride"]) or None,
            swap_uv=int(meta["format"]) == IMAGE_FORMAT_NV21, out_size=out_size))
    return to_net_frame(payload_to_rgb(payload, meta), out_size)


def to_net_frame(rgb, out_size=NET_SIZE):
    """`rgb_bridge_voxl.Geometry` with `--lens none`: resize, nothing else."""
    if rgb.shape[0] != out_size or rgb.shape[1] != out_size:
        rgb = resize_bilinear(rgb, out_size, out_size)
    return np.ascontiguousarray(rgb)


class NpzSource:
    """Desk stand-in: recorded 224x224 frames from an .npz, paced like a camera.

    Same `(meta, payload)` contract as MPASource, so the tick sees no
    difference. Bench/desk only -- there is no video decoder onboard.
    """

    name = "npz"

    def __init__(self, path, key=None, fps=15.0, loop=True):
        data = np.load(str(path))
        if key is None:
            key = "frames" if "frames" in data else list(data.keys())[0]
        self.frames = np.asarray(data[key])
        if self.frames.ndim != 4 or self.frames.shape[-1] != 3:
            raise ValueError("frames must be [N,H,W,3] uint8, got %s"
                             % (self.frames.shape,))
        self.loop = bool(loop)
        self.fps = float(fps)
        self._period = 1.0 / self.fps if self.fps > 0 else 0.0
        self._next = None
        self._i = 0
        self.ended = False
        self.reconnects = 0

    @property
    def connected(self):
        return not self.ended

    def read(self, timeout=0.2):
        now = time.monotonic()
        if self._next is None:
            self._next = now
        if now < self._next:
            time.sleep(min(timeout, self._next - now))
            return None
        if self._i >= len(self.frames):
            if not self.loop:
                self.ended = True
                return None
            self._i = 0
        frame = self.frames[self._i]
        self._i += 1
        self._next += self._period
        if self._next < time.monotonic():
            self._next = time.monotonic() + self._period
        h, w = frame.shape[:2]
        meta = dict(magic_number=CAMERA_MAGIC_NUMBER,
                    timestamp_ns=int(time.time() * 1e9), frame_id=self._i,
                    width=w, height=h, size_bytes=int(frame.size),
                    stride=w * 3, exposure_ns=0, gain=0,
                    format=IMAGE_FORMAT_RGB, framerate=int(self.fps),
                    reserved=0)
        return meta, np.ascontiguousarray(frame)

    def disconnect(self, why=""):
        pass

    close = disconnect


# ===========================================================================
# 2b. State: the EKF2 estimate, read straight off voxl-mavlink-server's pipe
# ===========================================================================
# There is no state adapter and no state socket. `voxl-mavlink-server` already
# terminates the PX4 link and republishes every message it receives on an MPA
# pipe, so the runner joins that pipe as a second client exactly the way it
# joins the camera pipe, and decodes the three messages it needs in-process.
#
# Why the pipe and not UDP: stage (viii) measured that voxl-mavlink-server only
# pushes UDP toward the configured GCS IP, so an on-board UDP listener never
# sees the telemetry at all without reconfiguring the server. The pipe is the
# supported on-board consumer path (docs/4.7-onboard-standalone-flight.md).
#
# What reading the pipe buys over the retired state datagram, beyond one
# fewer process:
#   * body rates. ATTITUDE_QUATERNION carries rollspeed/pitchspeed/yawspeed,
#     so the 21-vector's three rate slots finally hold measured values instead
#     of the zeros the old 64-byte state datagram had no field to fill.
#   * one clock. Staleness is measured on arrival, as before, but position and
#     attitude now carry INDEPENDENT arrival times and the older of the two
#     gates the tick -- a stalled attitude stream can no longer ride along on a
#     fresh position sample.
#   * no re-encode. The floats the EKF2 published are the floats the policy
#     consumes; nothing is quantised into a 64-byte frame on the way.
#
# The pipe carries packed `mavlink_message_t` records back to back (libmodal-
# pipe's mavlink_io.h hands consumers whole structs, not a byte stream), so the
# reader splits on the fixed record size rather than running a MAVLink parser.
# Only the header is decoded generically; three payloads are unpacked by hand.
#
# VERIFIED on the VOXL2 2026-09-02: /run/mpa/mavlink_onboard carries exactly
# these 291-byte records (2683 decoded, 0 undecodable, 0 resyncs in 12 s), and
# the ESTIMATOR_STATUS word decoded here equalled `px4-listener
# estimator_status` -> solution_status_flags (0x033e) on the same bench.

# mavlink_types.h, MAVPACKED -- 291 bytes, no padding anywhere.
#   0   checksum        uint16
#   2   magic           uint8    0xFD (v2) or 0xFE (v1)
#   3   len             uint8    payload length AFTER v2 trailing-zero trimming
#   4   incompat_flags  uint8
#   5   compat_flags    uint8
#   6   seq             uint8
#   7   sysid           uint8
#   8   compid          uint8
#   9   msgid           uint24   little-endian, a 3-byte bitfield
#   12  payload64       uint64[33]   = 264 bytes
#   276 ck              uint8[2]
#   278 signature       uint8[13]
_MAV_RECORD = struct.Struct("<HBBBBBBB3s264s2s13s")
MAV_RECORD_SIZE = _MAV_RECORD.size          # 291
assert MAV_RECORD_SIZE == 291, "mavlink_message_t must be 291 packed bytes"
_MAV_MAGIC_V2 = 0xFD
_MAV_MAGIC_V1 = 0xFE

MSGID_HEARTBEAT = 0
MSGID_ATTITUDE = 30
MSGID_ATTITUDE_QUATERNION = 31
MSGID_LOCAL_POSITION_NED = 32
MSGID_ESTIMATOR_STATUS = 230
MSGID_ODOMETRY = 331

# Payload layouts. MAVLink serialises fields in DESCENDING field-size order,
# not declaration order; for the two attitude/position messages every field is
# 4 bytes wide so the two orders coincide, and ESTIMATOR_STATUS is written out
# in its wire order (uint64, then the floats, then the uint16) below.
_PL_LOCAL_POSITION_NED = struct.Struct("<Iffffff")     # t, x,y,z, vx,vy,vz
_PL_ATTITUDE_QUATERNION = struct.Struct("<Ifffffff")   # t, q1..q4, p,q,r
_PL_ATTITUDE = struct.Struct("<Iffffff")               # t, roll,pitch,yaw, p,q,r
_PL_ESTIMATOR_STATUS = struct.Struct("<QffffffffH")    # t_us, 8 ratios, flags
# ODOMETRY (331): only the extension byte this runner reads. Wire order: t_us(8),
# x,y,z, q[4], vx,vy,vz, rates[3], pose_cov[21], vel_cov[21] (55 floats),
# frame_id, child_frame_id, then the extensions reset_counter, estimator_type,
# quality. PX4 1.14 fills reset_counter with the SUM of vehicle_local_position's
# xy/z/vxy/vz/heading reset counters (checked on the board 2026-09-23: 10 = 5 x 2).
_ODOM_RESET_OFFSET = 8 + 4 * 55 + 2       # = 230
# HEARTBEAT, common.xml. Descending field-size order puts the uint32
# custom_mode FIRST, ahead of the five uint8s -- NOT the declaration order in
# the XML (type, autopilot, base_mode, custom_mode, system_status, version).
_PL_HEARTBEAT = struct.Struct("<IBBBBB")   # custom_mode, type, autopilot,
                                           # base_mode, system_status, version

# Cross-check against the generated MAVLINK_MSG_ID_*_LEN constants: a struct
# that is the right total width has the right number of fields of the right
# sizes, which is the half of the layout a desk test cannot otherwise reach.
# ATTITUDE_QUATERNION's v2 extension (repr_offset_q, float[4]) is appended
# AFTER these fields, so a 48-byte payload from a newer PX4 still decodes --
# `_pad` truncates it back to the base 32.
assert _PL_LOCAL_POSITION_NED.size == 28, "LOCAL_POSITION_NED is 28 bytes"
assert _PL_ATTITUDE_QUATERNION.size == 32, "ATTITUDE_QUATERNION base is 32 bytes"
assert _PL_ATTITUDE.size == 28, "ATTITUDE is 28 bytes"
assert _PL_ESTIMATOR_STATUS.size == 42, "ESTIMATOR_STATUS is 42 bytes"
assert _PL_HEARTBEAT.size == 9, "HEARTBEAT is 9 bytes"

STATE_PIPE = "mavlink_onboard"
STATE_STALE_S = 0.3

# --- Is PX4 actually in OFFBOARD? (results/7.10-runner-final-ablation-field-
# --- sequence, the field-sequence defect) ------------------------------------
# In the field the goal is sent while the aircraft is ON THE GROUND, the pilot
# then takes off in POSITION, and only afterwards flips OFFBOARD. The runner
# had no way to know that, so it entered POLICY the instant the goal landed:
# it streamed full steering commands that PX4 discarded (it was in POSITION),
# and --align-first's 8 s timeout expired on the ground -- F2/F3 measured
# "[align] TIMEOUT after 8.00 s ... policy ENGAGED anyway" twelve seconds
# BEFORE the OFFBOARD switch, handing over with an 88 deg heading error.
#
# The missing fact is on the same pipe already being read: PX4's own
# HEARTBEAT. base_mode carries the arm bit, custom_mode carries the PX4 mode
# triple, and the main mode is byte 2 of it.
MAV_COMP_ID_AUTOPILOT1 = 1          # only the autopilot's own HEARTBEAT counts
MAV_MODE_FLAG_SAFETY_ARMED = 128    # base_mode & this = armed
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6   # (custom_mode >> 16) & 0xFF
# HEARTBEAT is 1 Hz on every PX4 link, so 2 s is "two beats missed": long
# enough not to flap on one dropped packet, short enough that a dead link
# stops the policy well inside a leg. Unknown is NOT offboard.
HEARTBEAT_STALE_S = 2.0


def px4_main_mode(custom_mode):
    """PX4's main mode out of a HEARTBEAT custom_mode (px4_custom_mode.h)."""
    return (int(custom_mode) >> 16) & 0xFF

# The same five gates the retired state datagram's flag byte carried, now
# sourced from
# ESTIMATOR_STATUS (common.xml ESTIMATOR_STATUS_FLAGS). The local position the
# policy steers on is EKF-relative, so POS_HORIZ_REL -- not _ABS -- is the bit
# that says "this xy is usable".
STATE_FLAG_XY_VALID = 0x01
STATE_FLAG_Z_VALID = 0x02
STATE_FLAG_V_XY_VALID = 0x04
STATE_FLAG_V_Z_VALID = 0x08
STATE_FLAG_ATTITUDE_VALID = 0x10

_EST_ATTITUDE = 1 << 0
_EST_VELOCITY_HORIZ = 1 << 1
_EST_VELOCITY_VERT = 1 << 2
_EST_POS_HORIZ_REL = 1 << 3
_EST_POS_VERT_ABS = 1 << 5


def state_flags_from_estimator(est_flags):
    """ESTIMATOR_STATUS.flags -> the runner's five-bit validity byte."""
    est = int(est_flags)
    out = 0
    if est & _EST_POS_HORIZ_REL:
        out |= STATE_FLAG_XY_VALID
    if est & _EST_POS_VERT_ABS:
        out |= STATE_FLAG_Z_VALID
    if est & _EST_VELOCITY_HORIZ:
        out |= STATE_FLAG_V_XY_VALID
    if est & _EST_VELOCITY_VERT:
        out |= STATE_FLAG_V_Z_VALID
    if est & _EST_ATTITUDE:
        out |= STATE_FLAG_ATTITUDE_VALID
    return out


# What the policy actually steers on: a horizontal position (goal distance and
# REACHED), a horizontal velocity (the 21-vector) and an attitude (the
# 21-vector and the ENU plan mapping). z_valid / v_z_valid are reported and
# logged but NOT required -- the vertical command is either the net's own
# (which the horizontal state already gates) or zeroed by --planar, and a
# barometric z that is momentarily flagged invalid must not ground the leg.
STATE_FLAGS_REQUIRED = (STATE_FLAG_XY_VALID | STATE_FLAG_V_XY_VALID
                        | STATE_FLAG_ATTITUDE_VALID)
_STATE_FLAG_NAMES = (
    (STATE_FLAG_XY_VALID, "xy_valid"),
    (STATE_FLAG_Z_VALID, "z_valid"),
    (STATE_FLAG_V_XY_VALID, "v_xy_valid"),
    (STATE_FLAG_V_Z_VALID, "v_z_valid"),
    (STATE_FLAG_ATTITUDE_VALID, "attitude_valid"),
)


def state_flags_ok(flags):
    return (int(flags) & STATE_FLAGS_REQUIRED) == STATE_FLAGS_REQUIRED


def missing_state_flags(flags):
    """"attitude_valid" / "xy_valid+v_xy_valid" -- for the hold log line."""
    out = [name for bit, name in _STATE_FLAG_NAMES
           if (STATE_FLAGS_REQUIRED & bit) and not (int(flags) & bit)]
    return "+".join(out) if out else "(none)"


def unpack_mav_record(rec):
    """One 291-byte mavlink_message_t -> (msgid, compid, payload), or None.

    The payload is returned at its full declared width: MAVLink v2 trims
    trailing zero bytes on the wire and `len` reflects that, so a short payload
    is zero-extended here rather than in each decoder.

    `compid` is returned because HEARTBEAT is the one message on this pipe that
    several components emit -- the autopilot, a companion, a GCS -- and only
    the autopilot's says anything about the flight mode.
    """
    if len(rec) != MAV_RECORD_SIZE:
        return None
    (_ck, magic, length, _incompat, _compat, _seq, _sysid, compid,
     msgid_b, payload, _ck2, _sig) = _MAV_RECORD.unpack(rec)
    if magic not in (_MAV_MAGIC_V2, _MAV_MAGIC_V1):
        return None
    if length > 255:
        return None
    msgid = msgid_b[0] | (msgid_b[1] << 8) | (msgid_b[2] << 16)
    return msgid, compid, payload[:length]


def _pad(payload, n):
    """v2 trailing-zero trimming undone: widen a payload to n bytes."""
    if len(payload) >= n:
        return payload[:n]
    return payload + b"\x00" * (n - len(payload))


def quat_from_euler_ned(roll, pitch, yaw):
    """ATTITUDE's Euler triple -> the Hamilton [w,x,y,z] NED->FRD quaternion.

    Only used when the autopilot streams ATTITUDE but not
    ATTITUDE_QUATERNION; PX4's onboard stream normally carries both.
    """
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def first_sample_line(sample, saw_estimator_status):
    """The one log line that proves the pipe format on a new board.

    A module-level function, not an inline print, so the desk tests can run
    it: the first version of this line was only reachable on a live pipe and
    crashed there on a malformed format tuple -- after every other part of the
    chain had already come up.
    """
    x, y, z = (float(v) for v in sample["pos_ned"])
    vx, vy, vz = (float(v) for v in sample["vel_ned"])
    gate = ("ESTIMATOR_STATUS" if saw_estimator_status
            else "no ESTIMATOR_STATUS on this stream: validity gate is "
                 "staleness only")
    return ("[state] first sample: seq=%d NED pos=(%.2f, %.2f, %.2f) "
            "vel=(%.2f, %.2f, %.2f) flags=0x%02x (%s)"
            % (int(sample["seq"]), x, y, z, vx, vy, vz,
               int(sample["flags"]), gate))


class MavlinkStateSource(MPAPipeClient):
    """EKF2 state off the voxl-mavlink-server MPA pipe. Never raises.

    `drain()` follows the camera's discipline exactly -- consume everything
    queued, keep only the newest of each message -- because a 50 Hz telemetry
    stream feeding a 15 Hz control loop must never build a backlog of poses the
    vehicle has already left.

    A sample is only produced once BOTH a position and an attitude message have
    been seen. Each carries its own arrival time; `age` is taken from the older
    of the two, so neither stream can hide the other going quiet.
    """

    name = "mavlink-mpa"

    def __init__(self, pipe=STATE_PIPE, client_name="onboard_state",
                 reconnect_s=0.5):
        MPAPipeClient.__init__(self, pipe, client_name=client_name,
                               reconnect_s=reconnect_s)
        self.received = 0          # records that decoded as a MAVLink header
        self.rejected = 0          # records whose header did not decode
        self.used = 0              # records that were one of ours
        self.gaps = 0              # kept for the stats line; resync events
        self.dropped = 0           # samples superseded within one tick
        self.seq = 0               # our own counter, one per emitted sample
        self._pos = None           # (pos_ned, vel_ned, t_arrival)
        self._att = None           # (q_ned_frd, rates_frd, t_arrival)
        self._est_flags = None     # newest ESTIMATOR_STATUS-derived byte
        self.est_raw = None        # ... and the raw uint16 it came from
        self.saw_estimator_status = False
        self.saw_odometry = False
        self.reset_counter = None         # ODOMETRY.reset_counter (uint8)
        self.saw_attitude_quaternion = False
        self._hb = None            # (base_mode, custom_mode, t_arrival)
        self.saw_heartbeat = False

    # -- record framing ----------------------------------------------------
    def _resync(self):
        """Realign on a plausible record boundary after a pipe overflow.

        A mavlink_message_t has no leading magic, but byte 2 of every record is
        the frame magic (0xFD/0xFE), so a candidate offset is one whose byte 2
        looks right and whose declared length fits. Scanning from 1 guarantees
        forward progress.
        """
        for off in range(1, len(self._buf) - 3):
            if (self._buf[off + 2] in (_MAV_MAGIC_V2, _MAV_MAGIC_V1)
                    and self._buf[off + 3] <= 255):
                if off:
                    _log("state pipe resynced, dropped %d bytes" % off)
                    self.gaps += 1
                del self._buf[:off]
                return True
        keep = MAV_RECORD_SIZE - 1        # a record may straddle the edge
        del self._buf[:max(0, len(self._buf) - keep)]
        return False

    def _consume(self, rec, now):
        got = unpack_mav_record(rec)
        if got is None:
            self.rejected += 1
            return False
        self.received += 1
        msgid, compid, payload = got
        if msgid == MSGID_LOCAL_POSITION_NED:
            _t, x, y, z, vx, vy, vz = _PL_LOCAL_POSITION_NED.unpack(
                _pad(payload, _PL_LOCAL_POSITION_NED.size))
            if self._pos is not None and self._pos[2] == now:
                self.dropped += 1       # superseded inside one tick
            self._pos = ((x, y, z), (vx, vy, vz), now)
        elif msgid == MSGID_ATTITUDE_QUATERNION:
            (_t, qw, qx, qy, qz,
             p, q, r) = _PL_ATTITUDE_QUATERNION.unpack(
                _pad(payload, _PL_ATTITUDE_QUATERNION.size))
            self._att = ((qw, qx, qy, qz), (p, q, r), now)
            self.saw_attitude_quaternion = True
        elif msgid == MSGID_ATTITUDE:
            # Fallback only: ATTITUDE_QUATERNION is the same data without a
            # Euler round trip, so it wins whenever the autopilot streams it.
            if self.saw_attitude_quaternion:
                return False
            _t, roll, pitch, yaw, p, q, r = _PL_ATTITUDE.unpack(
                _pad(payload, _PL_ATTITUDE.size))
            self._att = (quat_from_euler_ned(roll, pitch, yaw), (p, q, r), now)
        elif msgid == MSGID_HEARTBEAT:
            # Only the autopilot's own HEARTBEAT says what mode the vehicle is
            # in. voxl-mavlink-server republishes everything it sees, so the
            # companion's and the GCS's beats are on this pipe too and would
            # otherwise be read as "PX4 left OFFBOARD".
            if int(compid) != MAV_COMP_ID_AUTOPILOT1:
                return False
            (custom_mode, _type, _autopilot, base_mode, _status,
             _ver) = _PL_HEARTBEAT.unpack(_pad(payload, _PL_HEARTBEAT.size))
            self._hb = (int(base_mode), int(custom_mode), now)
            self.saw_heartbeat = True
        elif msgid == MSGID_ESTIMATOR_STATUS:
            fields = _PL_ESTIMATOR_STATUS.unpack(
                _pad(payload, _PL_ESTIMATOR_STATUS.size))
            self.est_raw = int(fields[-1])
            self._est_flags = state_flags_from_estimator(self.est_raw)
            self.saw_estimator_status = True
        elif msgid == MSGID_ODOMETRY:
            # doc 15.11.6.4: the EKF2's own reset counter, for the estimator
            # reset watch. Only the autopilot's odometry counts (voxl-vision-hub
            # also sends ODOMETRY *to* PX4 with the VIO pose).
            if int(compid) != MAV_COMP_ID_AUTOPILOT1:
                return False
            self.reset_counter = _pad(payload, _ODOM_RESET_OFFSET + 1)[_ODOM_RESET_OFFSET]
            self.saw_odometry = True
        else:
            return False
        self.used += 1
        return True

    # -- the flight-loop entry point ---------------------------------------
    def drain(self):
        """Newest complete sample, or None. Never blocks, never raises."""
        if self._fd is None:
            self.connect()
            if self._fd is None:
                return None
        for _ in range(8):              # bounded; one read is normally enough
            before = len(self._buf)
            if not self._fill(0.0):     # disconnected -- buf was reset
                return None
            if len(self._buf) == before:
                break                   # kernel queue empty
        now = time.time()
        fresh = False
        while len(self._buf) >= MAV_RECORD_SIZE:
            if self._buf[2] not in (_MAV_MAGIC_V2, _MAV_MAGIC_V1):
                if not self._resync():  # trims the buffer, so this terminates
                    break
                continue
            rec = bytes(self._buf[:MAV_RECORD_SIZE])
            del self._buf[:MAV_RECORD_SIZE]
            fresh = self._consume(rec, now) or fresh
        if not fresh:
            return None
        return self.sample()

    def sample(self):
        """Assemble the current best estimate, or None if half of it is missing."""
        if self._pos is None or self._att is None:
            return None
        pos_ned, vel_ned, t_pos = self._pos
        q_ned_frd, rates_frd, t_att = self._att
        if self._est_flags is None:
            # No ESTIMATOR_STATUS on this stream: the arrival of both messages
            # is the only validity signal there is. Staleness still gates the
            # leg, and the banner says the gate is degraded.
            flags = STATE_FLAGS_REQUIRED | STATE_FLAG_Z_VALID | STATE_FLAG_V_Z_VALID
        else:
            flags = self._est_flags
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        return dict(seq=self.seq, flags=flags,
                    pos_ned=pos_ned, vel_ned=vel_ned,
                    q_ned_frd=q_ned_frd, rates_frd=rates_frd,
                    t_pos=t_pos, t_att=t_att, t_arrival=min(t_pos, t_att),
                    reset_counter=self.reset_counter)

    def offboard_state(self, now=None, stale_s=HEARTBEAT_STALE_S):
        """(offboard, armed, main_mode, age_s) from the newest autopilot beat.

        `offboard` and `armed` are None when there is nothing fresh to say --
        no HEARTBEAT yet, or the newest one is older than `stale_s`. The caller
        treats None as NOT offboard: the same discipline the position and
        attitude streams get, for the same reason (a link that went quiet must
        never leave the policy engaged on a stale fact).
        """
        if self._hb is None:
            return None, None, None, None
        base_mode, custom_mode, t = self._hb
        age = max(0.0, (time.time() if now is None else now) - t)
        main = px4_main_mode(custom_mode)
        if age > stale_s:
            return None, None, main, age
        return (main == PX4_CUSTOM_MAIN_MODE_OFFBOARD,
                bool(base_mode & MAV_MODE_FLAG_SAFETY_ARMED), main, age)


# ===========================================================================
# 3. The reference velocity -- vendored from policies/agile/mpc.py
# ===========================================================================
# CANONICAL SOURCE: src/superfly/policies/agile/mpc.py (fit_trajectory,
# reference_velocity), verbatim. Importing that module does NOT import acados
# (the solver is built only in MPC.__init__) -- but it does import
# superfly.common, so onboard it is copied instead.

WAYPOINT_DT = 0.1                 # core.WAYPOINT_DT: the net's waypoint spacing


def fit_trajectory(world_pts, dt_wp=0.1):
    """Cubic fit of the net's world waypoints, per axis. -> (t_max, d1, d2)."""
    wp = np.asarray(world_pts, dtype=np.float64)
    t = np.arange(len(wp)) * dt_wp
    coeffs = [np.polyfit(t, wp[:, k], 3) for k in range(3)]
    d1 = tuple(np.polyder(c, 1) for c in coeffs)
    d2 = tuple(np.polyder(c, 2) for c in coeffs)
    return float(t[-1]), d1, d2


def reference_velocity(world_pts, t_query, dt_wp=0.1):
    """World-ENU velocity of the cubic fit at t_query [s] (clamped to the fit
    horizon). Uncapped -- the caller applies its own max_vel_xy/max_vel_z."""
    t_max, d1, _ = fit_trajectory(world_pts, dt_wp)
    ti = float(min(max(float(t_query), 0.0), t_max))
    return np.array([float(np.polyval(d, ti)) for d in d1])


# ===========================================================================
# 4. The two learned stages
# ===========================================================================
# Encoder: CANONICAL SOURCE cl4nav_encoder.FrozenCL4NavTFLiteEncoder, restricted
# to the one file that flies. Head: CANONICAL SOURCE
# scripts/export_agile_head_tflite.py::TFLiteHead + sort_modes, verbatim.

FEATURE_DIM = 128
RAW_STATE_DIM = 21                # LoquercioModelConfig.raw_state_dim
MODES = 3
STATE_DIM = 3
OUT_SEQ_LEN = 10
NATIVE_PLAN_SPEED = 7.0           # core.NATIVE_PLAN_SPEED (test_time_velocity)
REF_LOOKAHEAD_S = 5.0             # core.REF_LOOKAHEAD_S
VEL_LOOKAHEAD_S = 0.3             # core.VEL_LOOKAHEAD_S
VEL_YAW_MIN_SPEED = 0.3           # core.VEL_YAW_MIN_SPEED
VEL_YAW_RATE_MAX = 1.0            # core.VEL_YAW_RATE_MAX
RGB_STALE_S = 0.25                # rgb_transport.RGB_STALE_S

INT8_INPUT_SCALE = 1.0 / 255.0
INT8_INPUT_ZERO_POINT = -128
QUANT_SCALE_TOL = 1e-6


class OnboardEncoder:
    """`encoder_int8_ptdense.tflite`, with the two conversions the graph drops.

    The ENCODERS.md contract for this file, restated because the loader is the
    only place it exists:

        input   int8  [1,224,224,3] NHWC, quant (1/255, -128)
                -> (camera_bytes.astype(int16) - 128).astype(int8)
        output  int8  [1,128],      quant (0.001436328049749136, 5)
                -> (q.astype(float32) - 5) * 1.436328049749136e-3

    Both are exact: the input shift only relabels the same 256 codes, and the
    output map is the affine one the file itself carries. So the head still
    receives the RAW (not L2-normalised) float32 feature it was trained on.

    Unlike FrozenCL4NavTFLiteEncoder this takes the uint8 frame straight to
    int8 instead of going through float [0,1] and back; for a uint8 input those
    are the same 256 codes (`x/255*255` then `rint` is the identity), and the
    desk verification checks the features byte-for-byte against the shipped
    loader. float32 and uint8 encoder files are also accepted so the same
    runner can be pointed at `encoder_fp32.tflite` for a bench comparison; any
    other quantisation is REFUSED at load, exactly as the shipped loader does.
    """

    def __init__(self, model_path, interpreter_class, num_threads=None,
                 feature_dim=FEATURE_DIM, delegates=None):
        kwargs = {} if num_threads is None else {"num_threads": int(num_threads)}
        if delegates:
            kwargs["experimental_delegates"] = list(delegates)
        self.model_path = str(model_path)
        self.feature_dim = int(feature_dim)
        # Allocate ONCE: allocation is milliseconds, this runs at net rate.
        self.interp = interpreter_class(model_path=self.model_path, **kwargs)
        self.interp.allocate_tensors()
        ins = self.interp.get_input_details()
        outs = self.interp.get_output_details()
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError("encoder must have one input and one output, got "
                             "%d / %d: %s" % (len(ins), len(outs), self.model_path))
        self._in, self._out = ins[0], outs[0]
        self.input_dtype = np.dtype(self._in["dtype"])
        self.output_dtype = np.dtype(self._out["dtype"])
        self.input_scale = float(self._in["quantization"][0])
        self.input_zero_point = int(self._in["quantization"][1])
        self.output_scale = float(self._out["quantization"][0])
        self.output_zero_point = int(self._out["quantization"][1])
        shape = [int(d) for d in self._in["shape"]]
        if shape != [1, NET_SIZE, NET_SIZE, 3]:
            raise ValueError("encoder input must be [1,224,224,3] NHWC, got %s: %s"
                             % (shape, self.model_path))
        oshape = [int(d) for d in self._out["shape"]]
        if oshape != [1, self.feature_dim]:
            raise ValueError("encoder output must be [1,%d], got %s: %s"
                             % (self.feature_dim, oshape, self.model_path))
        if self.input_dtype == np.int8:
            if (abs(self.input_scale - INT8_INPUT_SCALE) > QUANT_SCALE_TOL
                    or self.input_zero_point != INT8_INPUT_ZERO_POINT):
                raise ValueError(
                    "int8 encoder input quantisation must be (1/255, -128) -- "
                    "the only int8 input that is exactly the camera bytes minus "
                    "128 (encoder_int8_ptdense.tflite). Got (%r, %d): %s"
                    % (self.input_scale, self.input_zero_point, self.model_path))
        elif self.input_dtype == np.uint8:
            if (abs(self.input_scale - INT8_INPUT_SCALE) > QUANT_SCALE_TOL
                    or self.input_zero_point != 0):
                raise ValueError(
                    "uint8 encoder input quantisation must be (1/255, 0): %s"
                    % self.model_path)
        elif self.input_dtype != np.float32:
            raise ValueError("encoder input dtype must be int8, uint8 or float32,"
                             " got %s: %s" % (self.input_dtype, self.model_path))
        if self.output_dtype == np.int8 and not self.output_scale > 0.0:
            raise ValueError("int8 encoder output needs a positive scale: %s"
                             % self.model_path)
        if self.output_dtype not in (np.int8, np.float32):
            raise ValueError("encoder output must be float32 or int8, got %s: %s"
                             % (self.output_dtype, self.model_path))

    def _preprocess(self, rgb):
        frame = np.asarray(rgb)
        if frame.shape != (NET_SIZE, NET_SIZE, 3):
            raise ValueError("frame must be 224x224x3, got %s" % (frame.shape,))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if self.input_dtype == np.int8:
            # int16 first so 128..255 does not wrap on the way (ENCODERS.md).
            return (frame.astype(np.int16) - 128).astype(np.int8)[None, ...]
        if self.input_dtype == np.uint8:
            return frame[None, ...]
        return (frame.astype(np.float32) / 255.0)[None, ...]

    def __call__(self, rgb):
        """224x224x3 uint8 -> [1, 1, 128] float32 (the head's `visual_features`)."""
        self.interp.set_tensor(self._in["index"],
                               np.ascontiguousarray(self._preprocess(rgb)))
        self.interp.invoke()
        raw = self.interp.get_tensor(self._out["index"])[0]
        if self.output_dtype == np.int8:
            feature = (raw.astype(np.float32)
                       - self.output_zero_point) * self.output_scale
        else:
            feature = raw.astype(np.float32)
        if not np.all(np.isfinite(feature)):
            raise ValueError("encoder output contains NaN or Inf")
        return feature.reshape(1, 1, self.feature_dim)


class ShimEncoder:
    """`encoder_int8_ptdense.tflite` through the BOARD'S OWN TFLite runtime.

    Drives `libtflite_shim.so` (tmp/board_shim/tflite_shim.cc, ~100 lines of C
    shim compiled ON the VOXL2 against Qualcomm's /usr/lib64 static TFLite
    2.8 build) via ctypes. This is the runtime `benchmark_model_mai` embeds --
    the one that measured 32.4 ms CPU / 15.5 ms NNAPI on this encoder -- so
    this backend exists to reuse those proven kernels, not to rebuild them.

    Hardcodes the ptdense contract (ENCODERS.md): int8 in (bytes-128), int8
    out dequantised (q-5)*1.436328049749136e-3. Refuses any model whose tensor
    byte sizes do not match that contract.
    """

    IN_BYTES = NET_SIZE * NET_SIZE * 3     # int8 [1,224,224,3]
    OUT_ZERO_POINT = 5
    OUT_SCALE = 1.436328049749136e-3

    def __init__(self, model_path, shim_lib, num_threads=4, use_nnapi=True,
                 feature_dim=FEATURE_DIM):
        import ctypes
        self.feature_dim = int(feature_dim)
        self.model_path = str(model_path)
        self._lib = ctypes.CDLL(str(shim_lib))
        self._lib.shim_create.restype = ctypes.c_void_p
        self._lib.shim_create.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                          ctypes.c_int]
        self._lib.shim_invoke.restype = ctypes.c_int
        self._lib.shim_invoke.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_int, ctypes.c_void_p,
                                          ctypes.c_int]
        self._lib.shim_input_bytes.argtypes = [ctypes.c_void_p]
        self._lib.shim_output_bytes.argtypes = [ctypes.c_void_p]
        self._lib.shim_last_error.restype = ctypes.c_char_p
        self._h = self._lib.shim_create(self.model_path.encode(),
                                        int(num_threads), 1 if use_nnapi else 0)
        if not self._h:
            raise RuntimeError("shim_create failed: %s"
                               % self._lib.shim_last_error().decode())
        n_in = self._lib.shim_input_bytes(self._h)
        n_out = self._lib.shim_output_bytes(self._h)
        if n_in != self.IN_BYTES or n_out != self.feature_dim:
            raise ValueError(
                "shim tensors do not match the ptdense contract: in %d B "
                "(want %d), out %d B (want %d): %s"
                % (n_in, self.IN_BYTES, n_out, self.feature_dim,
                   self.model_path))
        self._ctypes = ctypes
        self._out_buf = np.zeros(self.feature_dim, dtype=np.int8)
        self.backend = "shim/%s" % ("nnapi" if use_nnapi else "cpu")

    def __call__(self, rgb):
        """224x224x3 uint8 -> [1, 1, 128] float32 (same contract as
        OnboardEncoder.__call__)."""
        frame = np.asarray(rgb)
        if frame.shape != (NET_SIZE, NET_SIZE, 3):
            raise ValueError("frame must be 224x224x3, got %s" % (frame.shape,))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        buf_in = np.ascontiguousarray(
            (frame.astype(np.int16) - 128).astype(np.int8))
        rc = self._lib.shim_invoke(
            self._h,
            buf_in.ctypes.data_as(self._ctypes.c_void_p), buf_in.nbytes,
            self._out_buf.ctypes.data_as(self._ctypes.c_void_p),
            self._out_buf.nbytes)
        if rc != 0:
            raise RuntimeError("shim_invoke rc=%d: %s"
                               % (rc, self._lib.shim_last_error().decode()))
        feature = (self._out_buf.astype(np.float32)
                   - self.OUT_ZERO_POINT) * self.OUT_SCALE
        if not np.all(np.isfinite(feature)):
            raise ValueError("encoder output contains NaN or Inf")
        return feature.reshape(1, 1, self.feature_dim)


class TFLiteHead:
    """`head_fp32.tflite` with the loader-side work the graph does not do.

    Vendored verbatim from scripts/export_agile_head_tflite.py::TFLiteHead:
    quantise/dequantise for an integer-IO file, and nothing else; returns the
    RAW [1,3,31] prediction, exactly like `net.model(inputs)`. The mode sort
    stays OUTSIDE the graph (HEADS.md, trap 1) and is `sort_modes` below.
    """

    def __init__(self, path, interpreter_class, num_threads=None):
        kwargs = {} if num_threads is None else {"num_threads": int(num_threads)}
        self.interp = interpreter_class(model_path=str(path), **kwargs)
        self.interp.allocate_tensors()
        details = self.interp.get_input_details()
        self.visual = next(d for d in details if d["shape"][-1] == FEATURE_DIM)
        self.imu = next(d for d in details if d["shape"][-1] != FEATURE_DIM)
        # 18 means the file wants the position slice already applied.
        self.state_in = int(self.imu["shape"][-1])
        self.out = self.interp.get_output_details()[0]

    @staticmethod
    def _quantize(detail, x):
        scale, zero = detail["quantization"]
        if not scale:
            return x.astype(detail["dtype"], copy=False)
        q = np.round(x / scale) + zero
        info = np.iinfo(detail["dtype"])
        return np.clip(q, info.min, info.max).astype(detail["dtype"])

    def __call__(self, feature, state):
        state = state[:, :, -self.state_in:]
        self.interp.set_tensor(self.visual["index"],
                               self._quantize(self.visual, feature))
        self.interp.set_tensor(self.imu["index"], self._quantize(self.imu, state))
        self.interp.invoke()
        raw = self.interp.get_tensor(self.out["index"])
        scale, zero = self.out["quantization"]
        if scale:
            raw = (raw.astype(np.float32) - zero) * scale
        return raw.astype(np.float32)


def sort_modes(pred):
    """TensorFlowLoquercioBackend.infer's NumPy tail: |alpha| ascending."""
    order = np.abs(pred[0, :, 0]).argsort()
    return np.abs(pred[0, order, 0]), pred[0, order, 1:]


# ===========================================================================
# 4b. The Policy protocol (doc 12 s7)
# ===========================================================================
# Defined here, ahead of its first implementation in section 5, and used by
# `main()`'s CommandPublisher registry in section 6c.

DEFAULT_COMMAND_TYPE = "velocity_yaw"


class Policy:
    """What the flight loop is allowed to know about a policy (doc 12 s7).

    Duck-typed, like every other seam in this file: `OnboardAgilePolicy` below
    is the first implementation and today the only one. A DART-Direct
    implementation (DepthART TFLite front end + an exported DepthNav policy) is
    planned and is deliberately NOT in this file yet.

      command_type                       key into main()'s `publishers`
                                         registry; a policy whose type has no
                                         publisher is refused at start-up
      max_vel_xy / max_vel_z             reported in the task-port config block
      enc_ms / head_ms / tail_ms / passes  per-tick timing, for the CSV
      retarget(goal_enu, pos_enu=None)   swap the mission goal mid-flight
      engage(pos_enu, R_enu, goal_enu)   control has just changed hands
      disengage()                        it has just been handed back
      compute(...) -> dict               the command for this tick; the keys
                                         the loop needs are vel_enu, yaw,
                                         net_pass, enc_ms, head_ms, tail_ms,
                                         plus the type's own payload

    `engage`/`disengage` are no-ops here and no-ops for the agile policy, which
    carries no recurrent state and re-latches its straight reference line
    through `retarget()`. They exist for the policy that comes next: a
    recurrent one MUST clear its hidden state and freeze its START frame at the
    hand-over instant, and the only place that instant is known is the runner.
    """

    command_type = DEFAULT_COMMAND_TYPE

    def retarget(self, goal_enu, pos_enu=None):
        raise NotImplementedError

    def engage(self, pos_enu=None, R_enu=None, goal_enu=None):
        """Control has just been handed to this policy. Default: nothing."""

    def disengage(self):
        """Control has just been taken away. Default: nothing."""

    def compute(self, pos_enu, vel_enu, R_enu, angular_body, goal_enu, rgb,
                imu_override=None):
        raise NotImplementedError


# ===========================================================================
# 5. The policy tail -- vendored from policies/agile/core.py::AgilePolicy
# ===========================================================================

def _unit(v, fallback=(1.0, 0.0, 0.0)):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n > 1e-6:
        return v / n
    return np.asarray(fallback, dtype=np.float64)


def hover_imu_state(speed=3.0):
    """The fixed synthetic 21-vector of `handheld_viz.hover_imu_state`:
    level, de-yawed identity attitude, `speed` m/s straight ahead, no rates,
    goal dead ahead. Used by --imu-fixed (the hand-held check)."""
    state = np.zeros((1, 1, RAW_STATE_DIM), dtype=np.float32)
    state[:, :] = np.concatenate([
        np.array([0.0, 0.0, 2.0], np.float32),           # position (ENU)
        np.eye(3, dtype=np.float32).reshape(-1),         # de-yawed attitude
        np.array([float(speed), 0.0, 0.0], np.float32),  # body velocity
        np.zeros(3, np.float32),                         # body rates
        np.array([1.0, 0.0, 0.0], np.float32),           # body goal direction
    ])
    return state


class OnboardAgilePolicy(Policy):
    """`AgilePolicy` with `visual_input="cl4nav_rgb"`, `output_mode="velocity"`,
    the two learned stages as TFLite and every scipy import gone.

    Every method below is the arithmetic of the same-named method in
    `superfly.policies.agile.core.AgilePolicy`. What is NOT here, because
    velocity mode never reaches it: the acados MPC, the PD fallback, the
    attitude low-pass, the thrust PD, the depth frontend, the keep-out
    obstacle memory, `--depth-inflate-px` and `--net-thread`.

    doc 12 s7: the first `Policy` implementation. `command_type` names the
    CommandPublisher the runner must have for it (`VelocityYawPublisher`), and
    `engage`/`disengage` are inherited no-ops -- this policy is stateless
    across the hand-over except for the straight reference line, which the
    runner re-latches through `retarget()` at exactly the same instant.
    """

    command_type = "velocity_yaw"

    def __init__(self, encoder, head, max_vel=2.0, control_hz=15.0,
                 ref_lookahead_s=REF_LOOKAHEAD_S,
                 vel_lookahead_s=VEL_LOOKAHEAD_S, max_vel_xy=None,
                 max_vel_z=None, planar=False, yaw_rate_max=VEL_YAW_RATE_MAX,
                 net_every=1):
        self.encoder = encoder
        self.head = head
        self.max_vel = float(max_vel)
        self.max_vel_xy = float(max_vel if max_vel_xy is None else max_vel_xy)
        self.max_vel_z = float(max_vel if max_vel_z is None else max_vel_z)
        self.vel_lookahead_s = float(vel_lookahead_s)
        self.planar = bool(planar)
        self.yaw_rate_max = float(yaw_rate_max)
        self.ref_lookahead_s = float(ref_lookahead_s)
        self.control_dt = 1.0 / float(control_hz)
        self.net_every = max(1, int(net_every))
        self.enc_ms = float("nan")
        self.head_ms = float("nan")
        self.tail_ms = float("nan")
        self.passes = 0
        self.reset()

    def reset(self):
        self._tick = -1
        self._world_points = None
        self._world_points_per_mode = None
        self._alphas = np.zeros(MODES, dtype=np.float32)
        self._mode_idx = 0
        self._sort_order = np.arange(MODES, dtype=np.int64)   # --log-plans only
        self._ref_start = None
        self._ref_goal = None
        self._cruise_alt = None
        self._yaw_cmd = None
        # doc 15.11.6.5, logging only (read by raw_output(); nothing in the
        # control path reads them): the tail's velocity BEFORE --planar and the
        # caps, and the attitude the adopted plan was made at.
        self._raw_vel_enu = None
        self._plan_R_enu = None

    # doc 15.11.6.5: --save-frames. The loop sets `keep_net_input` (and, per
    # tick, `tick_tag` = (tick, wall_s)) only when frames are being saved;
    # compute() then leaves the frame the network consumed in
    # `last_net_record` = (frame, tick_tag, time.time() at the encoder call).
    keep_net_input = False
    tick_tag = None
    last_net_record = None

    def raw_output(self):
        """The head's own velocity in ITS frame, for ticks.csv (doc 15.11.6.5):
        the reference velocity of the tracked plan in the body FLU frame the
        plan was made in, at the net's native 7 m/s (i.e. before
        _scale_body_plan, --planar and the |v| caps). Agile emits no yaw.
        -> (v_flu[3], None) or None before the first plan."""
        v, R = self._raw_vel_enu, self._plan_R_enu
        if v is None or R is None:
            return None
        s = (1.0 if (self.max_vel <= 0.0 or self.max_vel >= NATIVE_PLAN_SPEED)
             else self.max_vel / NATIVE_PLAN_SPEED)
        return (np.asarray(R).T @ np.asarray(v)) / s, None

    # -- core.AgilePolicy.retarget ----------------------------------------
    def retarget(self, goal_enu, pos_enu=None):
        """Swap the mission goal mid-flight WITHOUT restarting the policy:
        only the straight start->goal reference line is re-latched, so the
        commanded trajectory stays continuous across the swap."""
        goal = np.asarray(goal_enu, np.float64).copy()
        if pos_enu is None:
            if self._ref_start is None:
                self._ref_goal = goal
                return
        else:
            self._ref_start = np.asarray(pos_enu, np.float64).copy()
        self._ref_goal = goal

    # -- core.AgilePolicy._state_to_model_input ---------------------------
    def _state_to_model_input(self, pos_enu, R_enu, vel_enu, angular_body,
                              goal_dir_world):
        local_velocity = R_enu.T @ vel_enu
        local_goal = R_enu.T @ goal_dir_world
        # De-yaw the rotation-matrix input: the checkpoint was trained along
        # world +x and reads absolute yaw in R as an error to correct. Body
        # velocity/rates/goal are yaw-invariant already, and the plan is mapped
        # back with the FULL R_enu below, so only the net input is de-yawed.
        yaw = math.atan2(R_enu[1, 0], R_enu[0, 0])
        cz, sz = math.cos(-yaw), math.sin(-yaw)
        R_deyaw = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]) @ R_enu
        state = np.concatenate([
            np.asarray(pos_enu, np.float32),
            np.asarray(R_deyaw, np.float32).reshape(-1),
            local_velocity, np.asarray(angular_body, np.float32),
            local_goal,
        ]).astype(np.float32)
        return state.reshape((1, 1, RAW_STATE_DIM))

    # -- core.AgilePolicy._goal_dir ---------------------------------------
    def _goal_dir(self, pos_enu, goal_enu):
        if self._ref_start is None:
            self._ref_start = np.asarray(pos_enu, np.float64).copy()
            self._ref_goal = np.asarray(goal_enu, np.float64).copy()
        line = self._ref_goal - self._ref_start
        line_len = float(np.linalg.norm(line))
        if self.ref_lookahead_s <= 0.0 or line_len < 1e-3:
            return _unit(goal_enu - pos_enu)
        line_dir = line / line_len
        progress = float(np.clip(np.dot(pos_enu - self._ref_start, line_dir),
                                 0.0, line_len))
        lookahead_m = self.ref_lookahead_s * self.max_vel
        target_s = min(progress + lookahead_m, line_len)
        return _unit(self._ref_start + target_s * line_dir - pos_enu,
                     fallback=line_dir)

    # -- core.AgilePolicy._scale_body_plan --------------------------------
    def _scale_body_plan(self, local_xyz):
        if self.max_vel <= 0.0 or self.max_vel >= NATIVE_PLAN_SPEED:
            return local_xyz
        return local_xyz * (self.max_vel / NATIVE_PLAN_SPEED)

    # -- core.AgilePolicy._select_mode ------------------------------------
    @staticmethod
    def _select_mode(local_xyz_per_mode):
        """Upstream agile_autonomy always tracks mode 0 (lowest alpha)."""
        return 0

    # -- core.AgilePolicy._adopt_plan --------------------------------------
    def _adopt_plan(self, alphas, trajectories, pos, R_enu):
        local_per_mode = [t.reshape(STATE_DIM, OUT_SEQ_LEN) for t in trajectories]
        self._mode_idx = self._select_mode(local_per_mode)
        per_mode = []
        for local_xyz in local_per_mode:
            local_xyz = self._scale_body_plan(local_xyz)
            per_mode.append(pos[None, :] + (R_enu @ local_xyz).T)
        self._world_points_per_mode = np.stack(per_mode, axis=0)  # (modes,T,3)
        self._world_points = self._world_points_per_mode[self._mode_idx]
        self._alphas = alphas
        self._plan_R_enu = np.array(R_enu, dtype=np.float64)   # logging only

    # -- core.AgilePolicy._velocity_cmd ------------------------------------
    def _velocity_cmd(self, world_points, R_enu):
        v = reference_velocity(world_points, self.vel_lookahead_s,
                               dt_wp=WAYPOINT_DT)
        self._raw_vel_enu = v.copy()          # logging only (raw_output)
        if self.planar:
            v[2] = 0.0                        # Starling planar loop: PX4 holds z
        speed_xy = float(np.hypot(v[0], v[1]))
        if speed_xy > self.max_vel_xy and speed_xy > 1e-9:
            v[:2] *= self.max_vel_xy / speed_xy
            speed_xy = self.max_vel_xy
        v[2] = float(np.clip(v[2], -self.max_vel_z, self.max_vel_z))
        if self._yaw_cmd is None:
            self._yaw_cmd = math.atan2(float(R_enu[1, 0]), float(R_enu[0, 0]))
        if speed_xy > VEL_YAW_MIN_SPEED:
            desired = math.atan2(float(v[1]), float(v[0]))
            err = math.atan2(math.sin(desired - self._yaw_cmd),
                             math.cos(desired - self._yaw_cmd))
            step = self.yaw_rate_max * self.control_dt
            yaw = self._yaw_cmd + float(np.clip(err, -step, step))
            self._yaw_cmd = math.atan2(math.sin(yaw), math.cos(yaw))
        return dict(vel_enu=v, yaw=float(self._yaw_cmd),
                    mode_idx=self._mode_idx, alphas=self._alphas)

    # -- core.AgilePolicy.compute (velocity branch) ------------------------
    def compute(self, pos_enu, vel_enu, R_enu, angular_body, goal_enu, rgb,
                imu_override=None):
        self._tick += 1
        pos = np.asarray(pos_enu, np.float64)
        vel = np.asarray(vel_enu, np.float64)
        R_enu = np.asarray(R_enu, np.float64)
        goal = np.asarray(goal_enu, np.float64)
        # DEAD, deliberately kept (audit defect 9): `_cruise_alt` is latched
        # here and read by nothing -- velocity mode never reaches upstream's
        # altitude branch. It stays so this tail is byte-identical to the
        # audited/equivalence-tested version; the altitude hold that DOES fly
        # lives in the runner loop (--alt-hold / altitude_vz_ned), not here.
        if self._cruise_alt is None:
            self._cruise_alt = float(pos[2])
        goal_dir = self._goal_dir(pos, goal)

        if self._world_points is None or (self._tick % self.net_every) == 0:
            if rgb is None:
                raise RuntimeError("no camera frame; compute() cannot run")
            state_in = (self._state_to_model_input(
                pos, R_enu, vel, angular_body, goal_dir)
                if imu_override is None else imu_override)
            if self.keep_net_input:            # --save-frames only
                self.last_net_record = (rgb, self.tick_tag, time.time())
            t0 = time.perf_counter()
            features = self.encoder(rgb)
            t1 = time.perf_counter()
            pred = self.head(features, state_in)
            t2 = time.perf_counter()
            alphas, trajectories = sort_modes(pred)
            # Which RAW head row each sorted slot came from. sort_modes()
            # throws the order away (it is the sim's arithmetic, verbatim, and
            # must stay that way), so it is recomputed here -- 3 elements --
            # rather than changing that function's contract. Read by
            # --log-plans and by nothing in the flight path.
            self._sort_order = np.abs(pred[0, :, 0]).argsort().astype(np.int64)
            self._adopt_plan(alphas, trajectories, pos, R_enu)
            self.enc_ms = (t1 - t0) * 1e3
            self.head_ms = (t2 - t1) * 1e3
            self.passes += 1
            net_pass = True
        else:
            net_pass = False
        t3 = time.perf_counter()
        cmd = self._velocity_cmd(self._world_points, R_enu)
        self.tail_ms = (time.perf_counter() - t3) * 1e3
        cmd["net_pass"] = net_pass
        cmd["enc_ms"] = self.enc_ms
        cmd["head_ms"] = self.head_ms
        cmd["tail_ms"] = self.tail_ms
        # doc 12 s7.1: the command carries its own type, and main() picks the
        # publisher by it. Purely additive -- the velocity payload above is
        # untouched, and a reader that does not know about the key (the
        # equivalence test's `board_wire_cmd`) is unaffected.
        cmd["command_type"] = self.command_type
        return cmd


# ===========================================================================
# 5b. Token-front-end DepthNav policies (doc 15.11.6.4) -- the second Policy
# ===========================================================================
# `dnav-rgb-dinov3-cnxt-tiny-tok-*` as two TFLite files: the frozen DINOv3
# ConvNeXt-T encoder (RGB [0,1] NHWC -> tokens [1,7,7,768], ImageNet
# normalisation inside the graph) and the exported post-front-end policy
# (`export_depthnav_tflite.py --image-input tokens`: TokenNeck + GRU-192 +
# velocity_bounded_yaw; inputs state[1,7], target[1,4], tokens, latent[1,192];
# outputs action[1,4], new_latent[1,192]).
#
# Observation and action are `bench/src/superfly/policies/depthnav.py::
# DepthNavPolicy.step()` -- the SITL row this policy was benchmarked under --
# restated without torch/scipy, statement for statement:
#   START frame  R_ws = the full attitude at the hand-over (after ALIGN)
#   state(7)     [quat(START<-body) wxyz with w >= 0, R_sw @ vel]
#   target(4)    [R_sw @ clamp_norm(1.5 * (goal - pos), target_speed),
#                 1 / max(|goal - pos|, 0.5)]
#   action       vel_world = R_ws @ a[:3];  yaw_world = yaw(START x) + a[3],
#                sent as an absolute heading, NOT rate-limited: training has
#                `yaw_rate_limit_rad_s: null` / `yaw_tau_s: null`, and every
#                SITL run of these variants passed `--no-yaw-rate-limit`
#                (runs 15.11-sitl-*-20260918a; 15.5 #2). The committed
#                bench offboard's 60 deg/s `slew_yaw` default is NOT what was
#                benchmarked. `--yaw-slew-deg D` re-adds it as an optional
#                deploy-side safety clamp.
#
# The encoder is ~100 ms on the VOXL 2 GPU (fp32, SUSTAINED), so the policy is
# PIPELINED (default): an inference thread steps the network back to back on
# the newest state, and a preprocessing thread converts the next camera frame
# just before the running inference ends, so the NV12 conversion overlaps the
# GPU instead of adding to it (doc 15.11.6.4: 9.7 Hz vs 6.9 Hz sequential,
# bit-identical actions). The flight loop keeps streaming setpoints at --rate
# and picks up whichever result is newest. Every inference is ONE recurrent
# step; nothing advances while compute() is not being called (hold, not
# OFFBOARD), and engage()/disengage() drop anything in flight.

TOKEN_POLICY_TOKENS_SHAPE = (1, 7, 7, 768)
TOKEN_POLICY_LATENT_DIM = 192
TOKEN_POLICY_YAW_SLEW_DEG = None          # no limit: training and the SITL benchmark (see above)
TOKEN_POLICY_ACTIVE_S = 0.3               # compute() silent this long -> pipeline idles


def quat_wxyz_from_R(R):
    """3x3 rotation -> [w, x, y, z] with w >= 0, float32 (depthnav._quat_wxyz_from_R
    without scipy: the largest-pivot (Shepperd) extraction, normalised)."""
    m = np.asarray(R, np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, np.float64)
    q /= np.linalg.norm(q)
    q = q.astype(np.float32)
    if q[0] < 0:
        q = -q
    return q


def depthnav_observation(pos_enu, vel_enu, R_enu, goal_enu, R_ws, target_speed):
    """(state[1,7], target[1,4]) exactly as DepthNavPolicy.step() builds them."""
    R_sw = np.asarray(R_ws, float).T
    pos = np.asarray(pos_enu, float)
    vel = np.asarray(vel_enu, float)
    R_sb = R_sw @ np.asarray(R_enu, float)
    state = np.concatenate([quat_wxyz_from_R(R_sb), R_sw @ vel]).astype(np.float32)
    target_vec = np.asarray(goal_enu, float) - pos
    dist = np.linalg.norm(target_vec)
    des_v = 1.5 * target_vec
    des_v_norm = np.linalg.norm(des_v)
    if des_v_norm > 1e-9:
        des_v = des_v / des_v_norm * min(des_v_norm, float(target_speed))
    else:
        des_v = np.zeros(3)
    inv_dist = 1.0 / max(dist, 0.5)
    target = np.concatenate([R_sw @ des_v, [inv_dist]]).astype(np.float32)
    return state.reshape(1, 7), target.reshape(1, 4)


def depthnav_decode(action, R_ws):
    """action[4] (START frame) -> (vel_world_enu[3], yaw_world_enu)."""
    a = np.asarray(action, np.float64).reshape(-1)
    R_ws = np.asarray(R_ws, float)
    start_yaw = float(np.arctan2(R_ws[1, 0], R_ws[0, 0]))
    return R_ws @ a[:3], start_yaw + float(a[3])


def slew_yaw_enu(yaw_cmd, yaw_target, max_rate_deg, dt):
    """depthnav_vel_offboard.slew_yaw: step toward the target by <= rate*dt."""
    err = wrap_pi(yaw_target - yaw_cmd)
    step = math.radians(float(max_rate_deg)) * max(float(dt), 0.0)
    return wrap_pi(yaw_cmd + float(np.clip(err, -step, step)))


class RawFrame:
    """The newest camera payload, NOT yet converted: the pipelined policy
    converts it itself, just in time (the loop only has to keep the newest)."""
    __slots__ = ("meta", "payload")

    def __init__(self, meta, payload):
        self.meta, self.payload = meta, payload


def net_frame_of(rgb):
    """RawFrame or an already-224 uint8 frame -> float32 [1,224,224,3] in [0,1]."""
    if isinstance(rgb, RawFrame):
        rgb = payload_to_net_frame(rgb.payload, rgb.meta)
    x = np.asarray(rgb)
    if x.shape != (NET_SIZE, NET_SIZE, 3):
        raise ValueError("net frame must be %dx%dx3, got %s"
                         % (NET_SIZE, NET_SIZE, x.shape))
    if np.issubdtype(x.dtype, np.floating):
        return np.ascontiguousarray(x, np.float32)[None]
    return (x.astype(np.float32) / 255.0)[None]


class ShimTokenEncoder:
    """DINOv3 encoder through libtflite_shim.so's `shim_create_ex` (backend 2 =
    the board's GPU V2 delegate; gpu_flags bit 0 fp16 arithmetic, bit 1
    SUSTAINED_SPEED). Default flags 2 = fp32 arithmetic + SUSTAINED: the one
    configuration that is both exact on the board (relL2 <= 8e-6) and fastest
    (doc 15.11.6.4). ctypes drops the GIL for the call, which is what lets the
    preprocessing thread run during it."""

    def __init__(self, model_path, shim_lib, gpu_flags=2, num_threads=4):
        import ctypes
        self._ct = ctypes
        self.model_path = str(model_path)
        lib = ctypes.CDLL(str(shim_lib))
        if not hasattr(lib, "shim_create_ex"):
            raise RuntimeError("%s has no shim_create_ex: rebuild it from this "
                               "repo's tflite_shim.cc with -DSHIM_WITH_GPU "
                               "-lgpu_delegate" % shim_lib)
        lib.shim_create_ex.restype = ctypes.c_void_p
        lib.shim_create_ex.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int]
        lib.shim_invoke.restype = ctypes.c_int
        lib.shim_invoke.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_int]
        lib.shim_input_bytes.argtypes = [ctypes.c_void_p]
        lib.shim_output_bytes.argtypes = [ctypes.c_void_p]
        lib.shim_last_error.restype = ctypes.c_char_p
        self._lib = lib
        self._h = lib.shim_create_ex(self.model_path.encode(), int(num_threads),
                                     2, int(gpu_flags))
        if not self._h:
            raise RuntimeError("shim_create_ex(GPU) failed: %s"
                               % lib.shim_last_error().decode())
        n_in, n_out = lib.shim_input_bytes(self._h), lib.shim_output_bytes(self._h)
        if n_in != 4 * NET_SIZE * NET_SIZE * 3 or n_out != 4 * int(np.prod(TOKEN_POLICY_TOKENS_SHAPE)):
            raise ValueError("encoder tensors are %d / %d B; expected float32 "
                             "[1,224,224,3] -> [1,7,7,768]" % (n_in, n_out))
        self._out = np.empty(TOKEN_POLICY_TOKENS_SHAPE, np.float32)
        self.backend = "shim/gpu(flags=%d)" % int(gpu_flags)

    def __call__(self, x):
        x = np.ascontiguousarray(x, np.float32)
        out = np.empty(TOKEN_POLICY_TOKENS_SHAPE, np.float32)
        c = self._ct
        rc = self._lib.shim_invoke(self._h, x.ctypes.data_as(c.c_void_p), x.nbytes,
                                   out.ctypes.data_as(c.c_void_p), out.nbytes)
        if rc != 0:
            raise RuntimeError("shim_invoke rc=%d: %s"
                               % (rc, self._lib.shim_last_error().decode()))
        return out


class TFLiteTokenEncoder:
    """The same file on a plain TFLite interpreter (desk / CPU fallback)."""

    def __init__(self, model_path, interpreter_class, num_threads=None):
        kwargs = {} if num_threads is None else {"num_threads": int(num_threads)}
        self.interp = interpreter_class(model_path=str(model_path), **kwargs)
        self.interp.allocate_tensors()
        self._in = self.interp.get_input_details()[0]
        self._out = self.interp.get_output_details()[0]
        if [int(d) for d in self._out["shape"]] != list(TOKEN_POLICY_TOKENS_SHAPE):
            raise ValueError("encoder output %s, expected %s"
                             % (list(self._out["shape"]), TOKEN_POLICY_TOKENS_SHAPE))
        self.backend = "tflite/cpu"

    def __call__(self, x):
        self.interp.set_tensor(self._in["index"], np.asarray(x, np.float32))
        self.interp.invoke()
        return self.interp.get_tensor(self._out["index"]).copy()


class TokenPolicyHead:
    """The exported TokenNeck + GRU + head; inputs matched by name, outputs by
    size (onnx2tf names them positionally)."""

    def __init__(self, path, interpreter_class, num_threads=1):
        self.interp = interpreter_class(model_path=str(path),
                                        num_threads=int(num_threads))
        self.interp.allocate_tensors()
        base = lambda d: d["name"].split(":")[0].replace("serving_default_", "")  # noqa: E731
        self.inp = {base(d): d for d in self.interp.get_input_details()}
        if set(self.inp) != {"state", "target", "tokens", "latent"}:
            raise ValueError("head inputs %s; expected state/target/tokens/latent "
                             "(export_depthnav_tflite.py --image-input tokens)"
                             % sorted(self.inp))
        self.out = {}
        for d in self.interp.get_output_details():
            n = int(np.prod(d["shape"]))
            self.out["action" if n == 4 else "new_latent"] = d
        if set(self.out) != {"action", "new_latent"}:
            raise ValueError("head outputs are not action[4] + latent[192]")

    def __call__(self, state, target, tokens, latent):
        s = self.interp
        s.set_tensor(self.inp["state"]["index"], np.asarray(state, np.float32))
        s.set_tensor(self.inp["target"]["index"], np.asarray(target, np.float32))
        s.set_tensor(self.inp["tokens"]["index"], np.asarray(tokens, np.float32))
        s.set_tensor(self.inp["latent"]["index"], np.asarray(latent, np.float32))
        s.invoke()
        return (s.get_tensor(self.out["action"]["index"]).reshape(4).copy(),
                s.get_tensor(self.out["new_latent"]["index"]).reshape(1, TOKEN_POLICY_LATENT_DIM).copy())


class OnboardTokenPolicy(Policy):
    """doc 15.11.6.4: a token-front-end DepthNav policy (selected by its variant_id,
    ONBOARD_VARIANTS) behind the `Policy` seam.

    `compute()` never blocks on the network when `pipeline=True`: it publishes
    the newest inputs, returns the newest finished command, and leaves the
    stepping to two threads. `pipeline=False` is the plain synchronous version
    (convert + encode + head inside compute()), kept for A/B and for the desk
    equivalence test. `step_log`, when a list, receives one record per network
    step with the exact inputs it used, so a pipelined run can be replayed
    synchronously and compared bit for bit.
    """

    command_type = "velocity_yaw"
    takes_raw_frames = True

    def __init__(self, encoder, head, target_speed, max_vel_xy=2.0, max_vel_z=1.0,
                 planar=False, yaw_slew_deg=TOKEN_POLICY_YAW_SLEW_DEG, pipeline=True,
                 step_log=None):
        self.encoder, self.head = encoder, head
        self.target_speed = float(target_speed)
        self.max_vel_xy, self.max_vel_z = float(max_vel_xy), float(max_vel_z)
        self.planar = bool(planar)
        self.yaw_slew_deg = None if yaw_slew_deg is None else float(yaw_slew_deg)
        self.pipeline = bool(pipeline)
        self.step_log = step_log
        self.enc_ms = self.head_ms = self.pre_ms = float("nan")
        self.tail_ms = 0.0
        self.passes = 0
        self._lock = threading.Lock()
        self._gen = 0
        self._need_latch = True
        self._R_ws = None
        self._latent = np.zeros((1, TOKEN_POLICY_LATENT_DIM), np.float32)
        self._yaw_cmd = None
        self._cmd = None                   # the standing command
        self._last_step_t = None
        self._inputs = None                # newest (gen, pos, vel, R, goal, frame, t)
        self._active_until = 0.0
        self._result = None                # newest finished step, not yet adopted
        # doc 15.11.6.5 (logging only; nothing in the control path reads them):
        # the action[4] last adopted (START-frame FLU velocity + yaw offset), and,
        # with --save-frames (`keep_net_input`), the network input that produced
        # it: `last_net_record` = (x, tick_tag of the tick whose inputs the
        # frame came from, time.time() at the encoder call). `_result_rec`
        # travels with `_result` under the same lock, so a saved frame is
        # always paired with the action computed from THAT frame.
        self.last_action = None
        self.keep_net_input = False
        self.tick_tag = None
        self.last_net_record = None
        self._result_rec = None
        if self.pipeline:
            self._ema = {"infer": 0.0, "pre": 0.0}
            self._frame_req = threading.Event()
            self._frame_ready = threading.Event()
            self._prepared = None
            self._stop = False
            self._wake = threading.Event()
            threading.Thread(target=self._pre_loop, name="token-policy-pre", daemon=True).start()
            threading.Thread(target=self._infer_loop, name="token-policy-infer", daemon=True).start()

    # -- the seam ----------------------------------------------------------
    def retarget(self, goal_enu, pos_enu=None):
        """Nothing to re-latch: the target channel is rebuilt from the goal on
        every step, exactly as in DepthNavPolicy.step()."""

    def engage(self, pos_enu=None, R_enu=None, goal_enu=None):
        """A new hand-over. The START frame is latched at the FIRST compute()
        after this, not here: the runner calls engage() when the goal lands /
        OFFBOARD is entered and only THEN turns to the goal (--align-first),
        while SITL (and training) latch START already facing the goal."""
        with self._lock:
            self._gen += 1
            self._need_latch = True
            self._result = None
            self._inputs = None

    def disengage(self):
        with self._lock:
            self._gen += 1
            self._active_until = 0.0
            self._result = None
            self._inputs = None

    def close(self):
        if self.pipeline:
            self._stop = True
            self._wake.set()
            self._frame_req.set()

    # -- one network step (both modes) --------------------------------------
    def _step(self, gen, pos, vel, R_enu, goal, x, R_ws, latent):
        state, target = depthnav_observation(pos, vel, R_enu, goal, R_ws, self.target_speed)
        t0 = time.perf_counter()
        tokens = self.encoder(x)
        t1 = time.perf_counter()
        action, new_latent = self.head(state, target, tokens, latent)
        t2 = time.perf_counter()
        if self.step_log is not None:
            self.step_log.append(dict(gen=gen, pos=np.array(pos), vel=np.array(vel),
                                      R=np.array(R_enu), goal=np.array(goal),
                                      x=x, R_ws=np.array(R_ws), latent=latent.copy(),
                                      action=action.copy()))
        return action, new_latent, (t1 - t0) * 1e3, (t2 - t1) * 1e3

    def _adopt(self, action, R_ws, now):
        vel_world, yaw_target = depthnav_decode(action, R_ws)
        if self.planar:
            vel_world = vel_world.copy()
            vel_world[2] = 0.0
        vel_world[2] = float(np.clip(vel_world[2], -self.max_vel_z, self.max_vel_z))
        dt = 0.0 if self._last_step_t is None else now - self._last_step_t
        self._last_step_t = now
        if self.yaw_slew_deg is None:
            self._yaw_cmd = wrap_pi(yaw_target)
        else:
            self._yaw_cmd = slew_yaw_enu(self._yaw_cmd, yaw_target, self.yaw_slew_deg, dt)
        self._cmd = np.asarray(vel_world, np.float64)
        self.last_action = action              # logging only (raw_output)
        self.passes += 1

    def raw_output(self):
        """The head's action in ITS frame, for ticks.csv (doc 15.11.6.5): the
        START-frame FLU velocity a[:3] and the yaw offset a[3] from the START
        heading [rad] -- before R_ws, --planar, the caps and the slew.
        -> (v[3], yaw) or None until the first step of an episode is adopted."""
        a = self.last_action
        if a is None:
            return None
        a = np.asarray(a, np.float64).reshape(-1)
        return a[:3], float(a[3])

    # -- the pipelined threads ---------------------------------------------
    def _pre_loop(self):
        while not self._stop:
            self._frame_req.wait()
            self._frame_req.clear()
            if self._stop:
                return
            with self._lock:
                inp = self._inputs
            if inp is None:
                self._prepared = None
            else:
                t0 = time.perf_counter()
                # [2] = the tick the frame was handed in at (--save-frames only)
                self._prepared = (inp["gen"], net_frame_of(inp["frame"]),
                                  inp.get("tag"))
                dt = time.perf_counter() - t0
                self.pre_ms = dt * 1e3
                e = self._ema
                e["pre"] = dt if e["pre"] == 0.0 else 0.8 * e["pre"] + 0.2 * dt
            self._frame_ready.set()

    def _prepare_now(self):
        self._frame_ready.clear()
        self._frame_req.set()
        self._frame_ready.wait()
        return self._prepared

    def _infer_loop(self):
        prepared = None
        while not self._stop:
            with self._lock:
                active = time.time() < self._active_until and self._inputs is not None
            if not active:
                prepared = None
                self._wake.wait(0.02)
                self._wake.clear()
                continue
            if prepared is None:
                prepared = self._prepare_now()
            with self._lock:
                inp = self._inputs
                gen = self._gen
                if inp is None or prepared is None or prepared[0] != gen or inp["gen"] != gen:
                    prepared = None
                    continue
                R_ws, latent = self._R_ws, self._latent
            t_start = time.perf_counter()
            # JIT: ask for the next frame so that it is ready as this step ends
            e = self._ema
            timer = threading.Timer(max(0.0, e["infer"] - e["pre"] - 0.004),
                                    lambda: (self._frame_ready.clear(), self._frame_req.set()))
            timer.start()
            rec = (prepared[1], prepared[2], time.time()) if self.keep_net_input else None
            action, new_latent, enc_ms, head_ms = self._step(
                gen, inp["pos"], inp["vel"], inp["R"], inp["goal"], prepared[1], R_ws, latent)
            dt = time.perf_counter() - t_start
            e["infer"] = dt if e["infer"] == 0.0 else 0.8 * e["infer"] + 0.2 * dt
            with self._lock:
                if gen == self._gen:
                    self._latent = new_latent
                    self._result = (action, R_ws, enc_ms, head_ms)
                    self._result_rec = rec
            timer.join()
            self._frame_ready.wait()
            prepared = self._prepared

    # -- compute -----------------------------------------------------------
    def compute(self, pos_enu, vel_enu, R_enu, angular_body, goal_enu, rgb,
                imu_override=None):
        if rgb is None:
            raise RuntimeError("no camera frame; compute() cannot run")
        now = time.time()
        pos = np.asarray(pos_enu, np.float64)
        R_enu = np.asarray(R_enu, np.float64)
        net_pass = False
        with self._lock:
            if self._need_latch:
                # THE hand-over: START frame, fresh hidden state, heading.
                self._need_latch = False
                self._R_ws = R_enu.copy()
                self._latent = np.zeros((1, TOKEN_POLICY_LATENT_DIM), np.float32)
                self._yaw_cmd = math.atan2(R_enu[1, 0], R_enu[0, 0])
                self._cmd = np.zeros(3)
                self._last_step_t = None
                self._result = None
                self.last_action = None            # logging only
            gen = self._gen
            if self.pipeline:
                self._inputs = dict(gen=gen, pos=pos, vel=np.asarray(vel_enu, np.float64),
                                    R=R_enu, goal=np.asarray(goal_enu, np.float64),
                                    frame=rgb)
                if self.keep_net_input:            # --save-frames only
                    self._inputs["tag"] = self.tick_tag
                self._active_until = now + TOKEN_POLICY_ACTIVE_S
                res, self._result = self._result, None
                rec, self._result_rec = self._result_rec, None
            R_ws, latent = self._R_ws, self._latent
        if self.pipeline:
            self._wake.set()
            if res is not None:
                action, R_ws_used, self.enc_ms, self.head_ms = res
                self._adopt(action, R_ws_used, now)
                if self.keep_net_input:
                    self.last_net_record = rec
                net_pass = True
        else:
            t0 = time.perf_counter()
            x = net_frame_of(rgb)
            self.pre_ms = (time.perf_counter() - t0) * 1e3
            if self.keep_net_input:                # --save-frames only
                self.last_net_record = (x, self.tick_tag, time.time())
            action, new_latent, self.enc_ms, self.head_ms = self._step(
                gen, pos, vel_enu, R_enu, goal_enu, x, R_ws, latent)
            with self._lock:
                self._latent = new_latent
            self._adopt(action, R_ws, time.time())
            net_pass = True
        return dict(vel_enu=self._cmd.copy(), yaw=float(self._yaw_cmd),
                    net_pass=net_pass, enc_ms=self.enc_ms, head_ms=self.head_ms,
                    tail_ms=self.tail_ms, command_type=self.command_type)



# The onboard variants, keyed by the FULL variant_id (CLAUDE.md constraint 7:
# a model is only ever named by its variant_id). `--policy <variant_id>`
# selects one; the files are looked up in --model-dir unless --encoder/--head
# name them. `target_speed_band` is the training band (Uniform(mean, half) ->
# [mean - half/2, mean + half/2]); the head bounds are the experiment
# contract's, NOT a flight clearance -- the |v_xy| cap is --max-vel-xy /
# --max-vel.
AGILE_VARIANT = "agile-rgb-cl4nav-r50-int8ptdense-vel-tartanair-v1"

ONBOARD_VARIANTS = {
    # The agile stack (sections 4-5): CL4Nav ResNet-50 encoder, int8 ptdense
    # (77/78 ops on the NNAPI accelerator, 16.8 ms; superfly_deployment 2.1),
    # + the agile head, velocity output; checkpoint AgileAutonomy/rgb_tartanair_v1.
    AGILE_VARIANT: dict(
        kind="agile",
        checkpoint="superfly_deployment/checkpoints/AgileAutonomy/rgb_tartanair_v1",
        encoder="encoder_int8_ptdense.tflite",
        head="head_fp32.tflite"),
    "dnav-rgb-dinov3-cnxt-tiny-tok-vel-starling-orew-ts2-4-v0": dict(
        kind="token",
        checkpoint="bench/checkpoints/DepthNav/rgb_dinov3_cnxt_tiny_tok_vel_starling_orew_ts2_4_v0/level1_1.pth",
        checkpoint_sha256="090ac8a4f148ab77f2d3f3707acaa10939a8e83c99235b04e99a3dcf6c260005",
        encoder="dinov3-cnxt-tiny-tflite-fp16-gpufix-v0.tflite",
        head="dnav-rgb-dinov3-cnxt-tiny-tok-vel-starling-orew-ts2-4-v0-head.tflite",
        target_speed_band=(2.0, 4.0),
        head_max_vel_xy=4.0, head_max_vel_z=2.4,
        trained_control_hz=15.0),
}


# ===========================================================================
# 6. Frames: NED <-> ENU (superfly.common.frames, vendored)
# ===========================================================================
# AgilePolicy is ENU-native; both wire protocols are world NED. The swap is
# done at the policy boundary and nowhere else, exactly as
# tier25_policy_runner.py does it. (x,y,z)->(y,x,-z) is its own inverse.

def swap_ne(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([v[1], v[0], -v[2]], dtype=np.float64)


def swap_yaw(yaw):
    """ENU heading <-> NED compass heading; pi/2 - x is its own inverse."""
    return math.pi / 2.0 - yaw


# --- PX4 attitude quaternion (NED/FRD) -> R_enu (ENU/FLU), scipy-free -------
# ORIGIN, quoted rather than re-derived:
#
#   src/superfly/common/px4_offboard.py:110-118  DroneState.update_from_attitude
#       q_ned_frd   = Rotation.from_quat([msg.q2, msg.q3, msg.q4, msg.q1])
#                     # MAVLink/PX4 [w,x,y,z] -> scipy [x,y,z,w]
#       rot_enu_flu = ROT_ENU_TO_NED.inv() * q_ned_frd * ROT_FLU_TO_FRD.inv()
#       self.R_enu  = rot_enu_flu.as_matrix()
#       self.yaw    = atan2(R_enu[1, 0], R_enu[0, 0])      # body-x into ENU xy
#
#   src/superfly/common/frames.py:17-20
#       ROT_ENU_TO_NED = Rotation.from_quat([0.70711, 0.70711, 0.0, 0.0])
#       ROT_FLU_TO_FRD = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])
#
# That R_enu is what scripts/agile_offboard.py:619 takes out of
# `state.get_full()` and hands to the policy at :765-768, so reproducing it
# exactly is what makes the onboard 21-vector the same vector the ground
# deployment builds. This file has no scipy, so the two fixed rotations are
# written out as matrices instead of constructed:
#
#   ROT_ENU_TO_NED is 180 deg about (1,1,0)/sqrt(2)  ->  M = 2nn^T - I
#                  = [[0,1,0],[1,0,0],[0,0,-1]]      (E,N,U) -> (N,E,-U)
#   ROT_FLU_TO_FRD is 180 deg about x                ->  F = diag(1,-1,-1)
#
# Both are involutions, so `.inv()` is the matrix itself and the composition
# collapses to  R_enu = M @ R(q_ned_frd) @ F.
_M_ENU_NED = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0]], dtype=np.float64)
_F_FLU_FRD = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def quat_wxyz_to_matrix(q_wxyz):
    """Hamilton [w,x,y,z] -> 3x3 rotation matrix (scipy's active convention)."""
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def R_enu_from_ned_frd_quat(q_wxyz):
    """The PX4 attitude quaternion -> the ENU/FLU R the policy consumes.

    Sanity, and the unit check in tests/test_sfst_state.py:
      level, heading NORTH (q = 1,0,0,0) -> R_enu = [[0,-1,0],[1,0,0],[0,0,1]]
          (body-x = ENU +y = North, and swap_yaw(atan2(1,0)) = 0 rad NED)
      level, heading EAST  (yaw_ned = +pi/2) -> R_enu = I
          (body-x = ENU +x = East, and swap_yaw(atan2(0,1)) = pi/2 rad NED)
    """
    return _M_ENU_NED @ quat_wxyz_to_matrix(q_wxyz) @ _F_FLU_FRD


def yaw_ned_from_R_enu(R_enu):
    """NED compass heading of an ENU/FLU rotation matrix.

    px4_offboard.py:117-118 takes the ENU yaw as the heading of body-x
    projected into the ENU xy plane; `swap_yaw` is this file's copy of
    frames.yaw_enu_to_ned.
    """
    return swap_yaw(math.atan2(float(R_enu[1, 0]), float(R_enu[0, 0])))


# ===========================================================================
# 6c. The command-type seam: Policy -> CommandPublisher (doc 12 s7 / s7.1)
# ===========================================================================
# `main()` used to own the whole velocity-specific downstream itself: swap the
# policy's ENU command into world NED, clamp |v_xy|, overwrite vz for
# --alt-hold, differentiate the commanded heading into a yaw rate (or pass the
# absolute angle through under --yaw-out angle), then hand the result to
# MavlinkVelPublisher. That is ONE command type's encoding, not a property of
# the flight loop, so it is collected here: one CommandPublisher per command
# type, and `main()` only dispatches on the type the Policy declares.
#
# Nothing in `VelocityYawPublisher.encode()` is new. It is the block that used
# to follow `policy.compute()` in `main()`, moved verbatim; the main()-locals
# it read became attributes of the publisher that owns them:
#
#     policy.max_vel_xy -> self.max_vel_xy      cap_violations -> self.cap_violations
#     alt_hold_on       -> self.alt_hold        alt_hold_warned -> self.alt_hold_warned
#     args.alt_hold_*   -> self.alt_hold_*      yaw_mode / args.yaw_rate_max -> self.*
#
# The wire publisher (MavlinkVelPublisher, or PrintOnlyPublisher under
# --enable_test) is NOT changed and is held here as `self.wire`; the runner's
# "nothing is sent" guarantee still rests on which object main() constructs.
#
# Rule (2) of doc 12 s7.1: safety that does not depend on the command type
# (the engage gate, hold, the failsafe by silence) stays in `main()`; safety
# that does (the |v_xy| clamp, the vertical channel) lives in the publisher.

class CommandPublisher:
    """One command type's encoding, plus the wire it leaves on.

    `encode()` turns a policy command into whatever the setpoint message for
    this command type carries; `send()` puts it out. Splitting the two is what
    lets `main()` keep the hold/failsafe path -- which has no policy command to
    encode -- while still leaving through the same object.
    """

    command_type = None

    def __init__(self, wire):
        self.wire = wire

    def encode(self, cmd, tick):
        raise NotImplementedError

    def send(self, vx, vy, vz, yaw_rate, flags=0, yaw=0.0):
        return self.wire.send(vx, vy, vz, yaw_rate, flags=flags, yaw=yaw)


class VelocityYawPublisher(CommandPublisher):
    """`velocity_yaw`: SET_POSITION_TARGET_LOCAL_NED, velocity + yaw.

    The only command type registered today. `thrust_rates`
    (SET_ATTITUDE_TARGET, body rates + collective thrust) is the one doc 12
    s7.1 reserves the shape for; it is not implemented.

    `tick` carries the loop measurements the encoding needs and that the policy
    command does not: pos_ned, vel_ned, yaw_ned, dt, alt_ref_d, state_src,
    state_last.
    """

    command_type = "velocity_yaw"

    def __init__(self, wire, max_vel_xy, yaw_rate_max, yaw_mode="rate",
                 alt_hold=False, alt_hold_kp=1.0, alt_hold_kd=0.0,
                 alt_hold_max_vz=0.5):
        CommandPublisher.__init__(self, wire)
        self.max_vel_xy = float(max_vel_xy)
        self.yaw_rate_max = float(yaw_rate_max)
        self.yaw_mode = str(yaw_mode)
        self.alt_hold = bool(alt_hold)
        self.alt_hold_kp = float(alt_hold_kp)
        self.alt_hold_kd = float(alt_hold_kd)
        self.alt_hold_max_vz = float(alt_hold_max_vz)
        self.cap_violations = 0
        self.alt_hold_warned = False     # the "EKF vertical not valid" one-shot

    def encode(self, cmd, tick):
        pos_ned = tick["pos_ned"]
        vel_ned = tick["vel_ned"]
        yaw_ned = tick["yaw_ned"]
        dt = tick["dt"]
        alt_ref_d = tick["alt_ref_d"]
        state_src = tick["state_src"]
        state_last = tick["state_last"]
        vx_n, vy_e, vz_d = swap_ne(cmd["vel_enu"])
        yaw_out_ned = swap_yaw(cmd["yaw"])
        vxy = math.hypot(vx_n, vy_e)
        if vxy > self.max_vel_xy + 1e-6:
            self.cap_violations += 1
            scale = self.max_vel_xy / vxy
            vx_n *= scale
            vy_e *= scale
        # --alt-hold (audit defect 5): PX4 v1.14 runs NO altitude lock
        # in pure-velocity OFFBOARD, so under --planar the vertical
        # channel is whatever the EKF's vz bias says. Close it here.
        if self.alt_hold and alt_ref_d is not None:
            vertical_ok = True
            if state_src is not None:
                fl = 0 if state_last is None else int(state_last["flags"])
                vertical_ok = bool(
                    fl & STATE_FLAG_Z_VALID) and bool(
                    fl & STATE_FLAG_V_Z_VALID)
            if vertical_ok:
                vz_d = altitude_vz_ned(
                    alt_ref_d, float(pos_ned[2]), float(vel_ned[2]),
                    kp=self.alt_hold_kp, kd=self.alt_hold_kd,
                    max_vz=self.alt_hold_max_vz)
            else:
                vz_d = 0.0
                if not self.alt_hold_warned:
                    self.alt_hold_warned = True
                    print("[alt-hold] the EKF does not report "
                          "z_valid+v_z_valid (flags 0x%02x); vz stays "
                          "0 and PX4 owns the vertical channel."
                          % (0 if state_last is None
                             else int(state_last["flags"])),
                          flush=True)
        v_ned = np.array([vx_n, vy_e, vz_d], dtype=np.float64)
        # SET_POSITION_TARGET_LOCAL_NED carries a yaw RATE, so the
        # commanded heading is differentiated against the measured one.
        yaw_rate_cmd = yaw_rate_toward(yaw_out_ned, yaw_ned, dt,
                                       self.yaw_rate_max)
        # ... unless --yaw-out angle, where the absolute setpoint goes
        # out unchanged and PX4's own yaw controller converges on it.
        yaw_sp_ned = wrap_pi(yaw_out_ned)
        if self.yaw_mode == "angle":
            yaw_rate_cmd = 0.0
        return v_ned, yaw_rate_cmd, yaw_sp_ned


# ===========================================================================
# 6b. Per-network-update plan log (--log-plans)
# ===========================================================================
# Vendored to match scripts/agile_offboard.py::PlanLogger field-for-field, the
# same way the policy tail above is vendored from core.py. The point of the
# duplication is that ONE analysis script
# (results/7.4-starling-sway-sanity-checks/analyze_plans.py) reads a board file
# and a sim file without knowing which is which -- so the field names, the
# shapes AND the frame must be identical, not merely equivalent.
#
# The board's wire protocol is world NED and its policy tail is ENU-native
# (section 6): `OnboardAgilePolicy._world_points_per_mode` is ALREADY world
# ENU, so the candidate plans are logged as-is; the loop's own pos/vel/goal
# are NED and go through swap_ne() at the call site. `frame` in the npz spells
# this out so a reader never has to infer it.


class PlanLogger:
    """One buffered row per NETWORK PASS (not per control tick).

    Fields (N = number of net passes, T = OUT_SEQ_LEN):

      t_wall          (N,)        unix wall clock of the tick
      t_since_policy  (N,)        seconds since the runner's t0
      pos_enu         (N,3)       vehicle position, world ENU
      yaw_enu         (N,)        atan2(R_enu[1,0], R_enu[0,0]), rad
      R_enu           (N,9)       body-FLU -> world-ENU rotation, row-major
      vel_enu_meas    (N,3)       measured velocity, world ENU
      goal_dir_world  (N,3)       _goal_dir unit vector, world ENU
      alphas_sorted   (N,3)       |alpha| ascending -- the policy's own order
      modes_world     (N,3,T,3)   the three candidate plans, world ENU, same
                                  sorted order
      sort_order      (N,3)       argsort indices: sort_order[i,k] is the RAW
                                  head row sorted slot k came from
      mode_idx        (N,)        which sorted slot is tracked (always 0)
      vel_cmd_enu     (N,3)       the velocity command that went out this tick,
                                  world ENU, AFTER the speed cap and --alt-hold
      yaw_cmd         (N,)        the commanded heading, world ENU, rad
      frame           scalar str  frame convention, spelled out
      source          scalar str  which program wrote the file

    Rows are buffered in lists and written once with np.savez at close(),
    which is idempotent and safe from a finally block.
    """

    FRAME = ("world ENU (x=east, y=north, z=up); yaw/heading = "
             "atan2(R_enu[1,0], R_enu[0,0]) = atan2(north, east)")
    SOURCE = "board:onboard_policy_runner.py"

    _COLS = ("t_wall", "t_since_policy", "pos_enu", "yaw_enu", "R_enu",
             "vel_enu_meas", "goal_dir_world", "alphas_sorted", "modes_world",
             "sort_order", "mode_idx", "vel_cmd_enu", "yaw_cmd")

    def __init__(self, path):
        self.path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        self.rows = 0
        self._closed = False
        self._buf = dict((k, []) for k in self._COLS)

    def log(self, t_wall, t_since_policy, pos_enu, R_enu, vel_enu_meas,
            goal_dir_world, alphas_sorted, modes_world, sort_order, mode_idx,
            vel_cmd_enu, yaw_cmd):
        R = np.asarray(R_enu, dtype=np.float64).reshape(3, 3)
        b = self._buf
        b["t_wall"].append(float(t_wall))
        b["t_since_policy"].append(float(t_since_policy))
        b["pos_enu"].append(np.asarray(pos_enu, np.float64).reshape(3).copy())
        b["yaw_enu"].append(math.atan2(float(R[1, 0]), float(R[0, 0])))
        b["R_enu"].append(R.reshape(9).copy())
        b["vel_enu_meas"].append(
            np.asarray(vel_enu_meas, np.float64).reshape(3).copy())
        b["goal_dir_world"].append(
            np.asarray(goal_dir_world, np.float64).reshape(3).copy())
        b["alphas_sorted"].append(
            np.asarray(alphas_sorted, np.float64).reshape(-1).copy())
        b["modes_world"].append(np.asarray(modes_world, np.float64).copy())
        if sort_order is None:
            sort_order = np.full(MODES, -1, dtype=np.int64)
        b["sort_order"].append(
            np.asarray(sort_order, np.int64).reshape(-1).copy())
        b["mode_idx"].append(int(mode_idx))
        b["vel_cmd_enu"].append(
            np.asarray(vel_cmd_enu, np.float64).reshape(3).copy())
        b["yaw_cmd"].append(float(yaw_cmd))
        self.rows += 1

    def close(self):
        if self._closed:
            return
        self._closed = True
        out = dict((k, np.asarray(v)) for k, v in self._buf.items())
        out["frame"] = np.asarray(self.FRAME)
        out["source"] = np.asarray(self.SOURCE)
        np.savez(self.path, **out)


# ===========================================================================
# 7. CLI
# ===========================================================================

def _boolish(text):
    """`--enable_test false` and friends. Anything ambiguous is an error, not
    a guess: a typo here would silently arm the vehicle."""
    s = str(text).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(
        "expected true/false, got %r" % (text,))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)

    mdl = p.add_argument_group("models")
    mdl.add_argument("--policy", required=True,
                     choices=sorted(ONBOARD_VARIANTS),
                     help="REQUIRED: the full variant_id of a registered "
                          "policy (ONBOARD_VARIANTS). No default: the name of "
                          "what flies is always written on the command line.")
    mdl.add_argument("--model-dir", default="/data/superfly",
                     help="where the --policy's encoder/head files are, "
                          "unless --encoder/--head name them")
    mdl.add_argument("--target-speed", type=float, default=None,
                     help="--policy <variant_id>: the policy's target-speed "
                          "input [m/s]; required, and must lie in the "
                          "variant's training band")
    mdl.add_argument("--gpu-flags", type=int, default=2,
                     help="--policy <variant_id> with --encoder-backend shim: "
                          "shim_create_ex GPU flags (bit0 fp16 arithmetic, "
                          "bit1 SUSTAINED). Default 2 = fp32 + SUSTAINED, the "
                          "only exact-and-fast setting (doc 15.11.6.4)")
    mdl.add_argument("--no-pipeline", dest="pipeline", action="store_false",
                     help="--policy <variant_id>: run convert+encode+head "
                          "synchronously inside the tick (A/B only; ~8 Hz vs "
                          "~9.7 Hz pipelined)")
    mdl.add_argument("--yaw-slew-deg", type=float,
                     default=TOKEN_POLICY_YAW_SLEW_DEG,
                     help="--policy <variant_id>: OPTIONAL heading-command slew "
                          "limit [deg/s]. Default: none -- training and the "
                          "SITL benchmark (--no-yaw-rate-limit) have no yaw "
                          "rate limit")
    mdl.add_argument("--encoder", default=None,
                     help="encoder .tflite (default: <--model-dir>/<the "
                          "variant's encoder file>)")
    mdl.add_argument("--head", default=None,
                     help="head .tflite (default: <--model-dir>/<the "
                          "variant's head file>)")
    mdl.add_argument("--encoder-backend", choices=("tflite", "shim"),
                     default="tflite",
                     help="'shim' drives the ENCODER through the board's own "
                          "Qualcomm TFLite build via libtflite_shim.so -- the "
                          "proven 15.5 ms NNAPI / 30 ms CPU kernels (head "
                          "stays on tflite_runtime; it is 0.3 ms)")
    mdl.add_argument("--shim-lib", default="/data/superfly/libtflite_shim.so",
                     help="path to libtflite_shim.so (default %(default)s)")
    mdl.add_argument("--shim-cpu", action="store_true",
                     help="shim backend without NNAPI (CPU kernels, ~30 ms)")
    mdl.add_argument("--encoder-delegate", default=None, metavar="SO",
                     help="external delegate .so applied to the ENCODER only "
                          "(the head is 0.3 ms, it stays on CPU). On the board: "
                          "/data/superfly/o4runtime/libnnapi_external_delegate.so")
    mdl.add_argument("--delegate-option", action="append", default=[],
                     metavar="K=V",
                     help="repeatable option for --encoder-delegate, e.g. "
                          "accelerator_name=qti-dsp, disallow_nnapi_cpu=1")
    mdl.add_argument("--threads", type=int, default=None,
                     help="TFLite interpreter threads (default: runtime default)")
    mdl.add_argument("--max-vel", type=float, default=2.0,
                     help="cruise cap [m/s]; also scales the net's body plan "
                          "below the 7 m/s training speed (default %(default)s)")
    mdl.add_argument("--max-vel-xy", type=float, default=None)
    mdl.add_argument("--max-vel-z", type=float, default=None)
    mdl.add_argument("--vel-lookahead-s", type=float, default=VEL_LOOKAHEAD_S)
    mdl.add_argument("--ref-lookahead-s", type=float, default=REF_LOOKAHEAD_S)
    mdl.add_argument("--yaw-rate-max", type=float, default=VEL_YAW_RATE_MAX)
    mdl.add_argument("--planar", action="store_true",
                     help="zero the commanded vz (the Starling planar loop)")

    cam = p.add_argument_group("camera")
    cam.add_argument("--pipe", default="hires_front",
                     help="MPA pipe name or path (default %(default)s)")
    cam.add_argument("--frames-npz", default=None,
                     help="DESK ONLY: replay [N,224,224,3] uint8 frames from an "
                          ".npz instead of the MPA pipe")
    cam.add_argument("--frame-stale-hold-s", type=float, default=1.0,
                     help="hold at zero velocity after frames have been stale "
                          "(older than %.2f s) for this long; negative disables "
                          "(default %%(default)s)" % RGB_STALE_S)

    seam = p.add_argument_group("seam (MAVLink setpoints out / goals in)")
    seam.add_argument("--mav-host", default="127.0.0.1",
                      help="PX4 onboard mavlink instance host (default "
                           "%(default)s)")
    seam.add_argument("--mav-port", type=int, default=14556,
                      help="PX4 onboard mavlink instance port (default "
                           "%(default)s, voxl-px4-start)")
    seam.add_argument("--cmd-bind", default="0.0.0.0",
                      help="goal port bind address (default %(default)s so the "
                           "operator's laptop can reach it)")
    seam.add_argument("--cmd-port", type=int, default=CMD_PORT)
    seam.add_argument("--enable_test", "--enable-test", dest="enable_test",
                      nargs="?", const=True, default=False, type=_boolish,
                      metavar="BOOL",
                      help="SMOKE TEST: compute the setpoint every tick and "
                           "PRINT it instead of sending it. No MAVLink "
                           "connection is opened at all, so the vehicle cannot "
                           "be commanded. Default false = fly (setpoints go to "
                           "PX4). Accepts --enable_test, --enable_test true or "
                           "--enable_test false")
    seam.add_argument("--test-print-every", type=int, default=1,
                      help="with --enable_test, print one line every N ticks "
                           "(default %(default)s = every tick, 15 lines/s at "
                           "the default rate)")
    seam.add_argument("--silent-until-tasked", action="store_true",
                      help="send NO setpoint until the first SET_GOAL. NOT for "
                           "the flight script: PX4 refuses to ENTER offboard "
                           "without a live setpoint stream, so the pilot's "
                           "OFFBOARD switch will be rejected until a goal has "
                           "been sent. Use it only when the runner must not "
                           "touch the vehicle at all")

    st = p.add_argument_group("state (the stand-in, or the real EKF2 estimate)")
    st.add_argument("--state", choices=("synthetic", "mpa"), default="synthetic",
                    help="synthetic (default, unchanged): dead-reckon the pose "
                         "from the commanded velocity at a level attitude. "
                         "mpa: take position/velocity/attitude/body rates from "
                         "the EKF2 estimate, decoded in-process off "
                         "voxl-mavlink-server's MPA pipe (see "
                         "docs/4.7-onboard-standalone-flight.md) -- dead "
                         "reckoning is then off entirely")
    st.add_argument("--state-pipe", default=STATE_PIPE,
                    help="voxl-mavlink-server MPA pipe carrying the messages "
                         "from the flight controller (default %(default)s; an "
                         "absolute path is taken as-is)")
    st.add_argument("--body-rates", choices=("mavlink", "zero"),
                    default="mavlink",
                    help="fill the 21-vector's three body-rate slots from "
                         "ATTITUDE_QUATERNION (default) or leave them zero. "
                         "'zero' reproduces every run made before the rates "
                         "were available; --state synthetic is always zero")
    st.add_argument("--state-stale-s", type=float, default=STATE_STALE_S,
                    help="hold at zero velocity once the older of the newest "
                         "position/attitude messages is older than this [s] "
                         "(default %(default)s)")
    st.add_argument("--start", nargs=3, type=float, metavar=("N", "E", "D"),
                    default=(0.0, 0.0, -2.0),
                    help="dead-reckoning start, world NED (default %(default)s); "
                         "--state mpa ignores it except as the placeholder "
                         "printed before the first sample arrives")
    st.add_argument("--start-yaw-deg", type=float, default=0.0)
    st.add_argument("--freeze-pose", action="store_true",
                    help="--state synthetic only: do NOT dead-reckon the pose "
                         "from the commanded velocity; hold it at --start / "
                         "--start-yaw-deg for the whole run. For the bench and "
                         "hand-held visualisation, where the vehicle is not "
                         "actually moving: without this the fictional pose "
                         "'arrives' at the goal after dist/max_vel seconds, "
                         "the runner enters the post-arrival hold, and the "
                         "policy (and the dashboard overlay) stops. Ignored "
                         "with --state mpa")
    st.add_argument("--goal", nargs=3, type=float, metavar=("N", "E", "D"),
                    default=None,
                    help="optional initial goal, world NED. Omit for the "
                         "ready-for-tasking posture")
    st.add_argument("--goal-radius", type=float, default=1.5,
                    help="HORIZONTAL arrival radius [m] (default %(default)s)")
    st.add_argument("--imu-fixed", action="store_true",
                    help="freeze the 21-vector to handheld_viz.hover_imu_state "
                         "(the hand-held check: no dead reckoning in the net "
                         "input; the tail still runs)")
    st.add_argument("--imu-speed", type=float, default=3.0,
                    help="--imu-fixed forward speed [m/s] (default %(default)s)")
    st.add_argument("--rate", type=float, default=15.0,
                    help="control rate [Hz] (default %(default)s)")
    st.add_argument("--net-rate", type=float, default=15.0,
                    help="net forward-pass rate [Hz] (default %(default)s)")
    st.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many seconds; 0 = until interrupted")

    # -- results/4.7.2-integration-audit fixes ------------------------------
    # EVERY default in this group reproduces the behaviour that has been
    # flown, so adding the group changes nothing until a flag is typed.
    fix = p.add_argument_group(
        "integration fixes (results/4.7.2-integration-audit; defaults = "
        "today's behaviour)")
    fix.add_argument("--goal-z-mode", choices=GOAL_Z_MODES, default="absolute",
                     help="how a SET_GOAL's z is interpreted (audit defect 1). "
                          "absolute (default, unchanged): an EKF-local NED "
                          "absolute, omitted z falls back to the cruise-alt "
                          "first-sample latch, no sanity check. current: the "
                          "operator's z is kept only within "
                          "--goal-z-max-delta of the CURRENT EKF altitude and "
                          "the goal is REFUSED beyond that; omitted z takes "
                          "the current EKF z. hold: z is ALWAYS the current "
                          "EKF z, so the reference line is horizontal -- the "
                          "right thing under --planar")
    fix.add_argument("--goal-z-max-delta", type=float, default=3.0,
                     help="--goal-z-mode current: refuse a SET_GOAL whose z is "
                          "further than this [m] from the current altitude "
                          "(default %(default)s)")
    fix.add_argument("--yaw-out", choices=("rate", "angle"), default="rate",
                     help="what the setpoint carries (audit defect 2). rate "
                          "(default, unchanged): mask 0x07C7, a yaw RATE this "
                          "process differentiates. angle: mask 0x09C7, the "
                          "absolute NED yaw setpoint PX4 converges on itself "
                          "-- what the sim has always sent")
    fix.add_argument("--align-first", action="store_true",
                     help="after a goal is applied, stream ZERO velocity while "
                          "turning to the goal bearing, and engage the policy "
                          "only once the heading is within --align-tol-deg or "
                          "--align-timeout-s has passed (audit defect 4; the "
                          "sim's YAW phase)")
    fix.add_argument("--align-tol-deg", type=float, default=5.0,
                     help="--align-first heading tolerance [deg] (default "
                          "%(default)s)")
    fix.add_argument("--align-timeout-s", type=float, default=8.0,
                     help="--align-first gives up and engages after this many "
                          "seconds (default %(default)s)")
    fix.add_argument("--alt-hold", action="store_true",
                     help="close a vertical loop on the runner side instead of "
                          "leaving vz to PX4, which in v1.14 pure-velocity "
                          "OFFBOARD has no altitude lock at all (audit defect "
                          "5). Only meaningful with --planar, and only active "
                          "while the EKF reports z_valid AND v_z_valid")
    fix.add_argument("--alt-hold-kp", type=float, default=2.0,
                     help="--alt-hold proportional gain (default %(default)s)")
    fix.add_argument("--alt-hold-kd", type=float, default=1.0,
                     help="--alt-hold damping on the measured vz (default "
                          "%(default)s)")
    fix.add_argument("--alt-hold-max-vz", type=float, default=1.5,
                     help="--alt-hold vz clip [m/s] (default %(default)s)")
    # results/7.10-runner-final-ablation-field-sequence: the ONE switch here
    # whose default is not "today's behaviour", because the field procedure
    # requires it -- see the block comment above MAV_COMP_ID_AUTOPILOT1.
    fix.add_argument("--engage", choices=("offboard", "immediate"),
                     default=None,
                     help="when the policy is allowed to steer. offboard "
                          "(DEFAULT with --state mpa): stream zero velocity "
                          "and the MEASURED heading until PX4's own HEARTBEAT "
                          "says it is in OFFBOARD, then latch the ALIGN "
                          "timer, the --alt-hold reference and the reference "
                          "line's start at THAT instant. immediate (DEFAULT "
                          "with --state synthetic, the pre-7.10 behaviour): "
                          "engage as soon as the goal is applied -- the bench "
                          "and hand-held runs, where no autopilot HEARTBEAT "
                          "exists to wait for")

    jmp = p.add_argument_group("EKF2 reset watch (doc 15.11.6.4)")
    jmp.add_argument("--reset-action", choices=("hold", "log", "off"),
                     default="log",
                     help="on an EKF2 reset (ODOMETRY.reset_counter changed; "
                          "exact deltas from vehicle_local_position). log "
                          "(default, the flight loop behaves as before): report "
                          "it (runner.log + ESTIMATOR_RESET event), change "
                          "nothing. hold: hover, re-anchor the goal (and takeoff "
                          "frame, climb target, --alt-hold reference) to the "
                          "same physical point, end the policy episode, wait for "
                          "an operator RESUME (then ALIGN + a fresh episode). "
                          "off: no watch")

    tko = p.add_argument_group(
        "auto takeoff (doc 15.11.6.4; off unless --auto-takeoff)")
    tko.add_argument("--auto-takeoff", action="store_true",
                     help="on a TAKEOFF command from the goal port (never "
                          "otherwise): request OFFBOARD + ARM, climb to "
                          "--climb-alt, then ALIGN -> POLICY. Needs --state mpa "
                          "and --engage offboard. Once PX4 leaves OFFBOARD after "
                          "that (pilot mode switch / stick override) the runner "
                          "never requests OFFBOARD or arming again.")
    tko.add_argument("--goal-frame", choices=("ekf-ned", "takeoff-flu"),
                     default="ekf-ned",
                     help="how SET_GOAL x y z is read. ekf-ned (default, as "
                          "before): EKF2-local NED metres. takeoff-flu "
                          "(required by --auto-takeoff): x forward / y left / "
                          "z up [m], relative to where the aircraft stood and "
                          "pointed when TAKEOFF was accepted; z is the height "
                          "above that ground point, and the takeoff climbs to it")
    tko.add_argument("--min-goal-z", type=float, default=1.5,
                     help="takeoff-flu: refuse a goal lower than this [m]")
    tko.add_argument("--max-goal-z", type=float, default=15.0,
                     help="takeoff-flu: refuse a goal higher than this [m]")
    tko.add_argument("--climb-rate", type=float, default=1.0,
                     help="[m/s] (depthnav_vel_offboard.py default)")
    tko.add_argument("--arrive-tol", type=float, default=0.3)
    tko.add_argument("--settle-speed", type=float, default=0.2)
    tko.add_argument("--arm-timeout-s", type=float, default=10.0)
    tko.add_argument("--climb-timeout-s", type=float, default=30.0)

    out = p.add_argument_group("output")
    out.add_argument("--bench", type=int, default=0, metavar="N",
                     help="run N ticks on synthetic (or --frames-npz) frames "
                          "with no sockets, print enc/head/tail/total "
                          "percentiles, and exit")
    out.add_argument("--log-csv", default=None,
                     help="one row per tick (never any image data)")
    out.add_argument("--log-plans", default=None, metavar="PATH.npz",
                     help="one row per NETWORK PASS: the head's three "
                          "candidate trajectories in world ENU, the pose they "
                          "were made at, the |alpha| sort order and the "
                          "velocity/heading that went out. Same field names, "
                          "shapes and frame as agile_offboard.py --plan-log, "
                          "so one analysis script reads both. Never any image "
                          "data. Off by default; the flight command is "
                          "unchanged either way.")
    out.add_argument("--log-dir", default="/data/superfly/runs",
                     help="every run writes runner.log + ticks.csv + "
                          "run_meta.json into a NEW directory here "
                          "(<UTC>_<policy>_<pid>); nothing is overwritten")
    out.add_argument("--no-log-dir", action="store_true",
                     help="do not create the per-run directory (an explicit "
                          "--log-csv is still honoured, and still never "
                          "overwritten)")
    out.add_argument("-v", "--verbose", action="store_true")

    prof = p.add_argument_group(
        "profiling / frame capture (doc 15.11.6.5; ALL off by default -- "
        "without these flags the runner behaves exactly as before)")
    prof.add_argument("--profile", action="store_true",
                      help="sample CPU (total + per core), this process, GPU "
                           "busy/clock, cpufreq, MemAvailable and the hottest "
                           "thermal zone in a daemon thread -> profile.csv in "
                           "the run dir (same wall_s clock as ticks.csv)")
    prof.add_argument("--profile-hz", type=float, default=PROFILE_HZ_DEFAULT,
                      help="--profile sample rate [Hz] (default %(default)s)")
    prof.add_argument("--profile-root", default="/",
                      help="prefix of every /proc and /sys path the profiler "
                           "reads (default %(default)s; tests use a fake tree)")
    prof.add_argument("--profile-csv", default=None, metavar="PATH",
                      help="--profile output (default <run dir>/profile.csv)")
    prof.add_argument("--save-frames", type=int, default=0, metavar="N",
                      help="save the exact network input of every N-th network "
                           "pass (+ its tick, the tick its action was applied "
                           "at, and that action) to <run dir>/frames/"
                           "frames_XXXX.npz, %d per file, written by a "
                           "background thread; a full queue (%d) drops and "
                           "counts. 0 = off (default)"
                           % (FRAMES_PER_FILE, FRAME_QUEUE_MAX))
    prof.add_argument("--frames-dir", default=None, metavar="DIR",
                      help="--save-frames output (default <run dir>/frames)")
    args = p.parse_args(argv)
    if args.profile_hz <= 0.0:
        p.error("--profile-hz must be > 0")
    if args.save_frames < 0:
        p.error("--save-frames must be >= 0")
    v = ONBOARD_VARIANTS[args.policy]
    if args.encoder is None:
        args.encoder = os.path.join(args.model_dir, v["encoder"])
    if args.head is None:
        args.head = os.path.join(args.model_dir, v["head"])
    if v["kind"] == "token":
        lo, hi = v["target_speed_band"]
        if args.target_speed is None:
            p.error("--policy %s needs --target-speed (training band %g-%g m/s)"
                    % (args.policy, lo, hi))
        if not lo <= args.target_speed <= hi:
            p.error("--target-speed %g is outside %s's training band %g-%g m/s"
                    % (args.target_speed, args.policy, lo, hi))
    if args.auto_takeoff:
        if args.state != "mpa":
            p.error("--auto-takeoff needs --state mpa (armed/OFFBOARD come off "
                    "PX4's HEARTBEAT on the state pipe)")
        if args.engage == "immediate":
            p.error("--auto-takeoff needs --engage offboard")
        if args.goal_frame != "takeoff-flu":
            p.error("--auto-takeoff needs --goal-frame takeoff-flu (the goal "
                    "x y z are then forward/left/up of the takeoff pose)")
    if args.goal_frame == "takeoff-flu":
        if not args.auto_takeoff:
            p.error("--goal-frame takeoff-flu needs --auto-takeoff (the frame "
                    "is latched when TAKEOFF is accepted)")
        if args.goal_z_mode != "absolute":
            p.error("--goal-frame takeoff-flu carries its own z (height above "
                    "the takeoff point); do not combine it with --goal-z-mode "
                    "%s" % args.goal_z_mode)
        if args.goal is not None:
            p.error("--goal is EKF NED; with --goal-frame takeoff-flu send the "
                    "goal as SET_GOAL x y z")
    # The gate needs a HEARTBEAT, and the only source of one is the MPA state
    # pipe; --state synthetic would hold at zero velocity for ever.
    if args.engage is None:
        args.engage = "offboard" if args.state == "mpa" else "immediate"
    elif args.engage == "offboard" and args.state != "mpa":
        p.error("--engage offboard needs --state mpa: the OFFBOARD gate reads "
                "PX4's HEARTBEAT off the state pipe, and --state synthetic "
                "never opens one")
    return args


def build_stack(args):
    interpreter_class, runtime = tflite_interpreter_class()
    print("[onboard] tflite runtime: %s" % runtime, flush=True)
    print("[onboard] numpy %s, python %s" % (np.__version__,
                                             sys.version.split()[0]), flush=True)
    if ONBOARD_VARIANTS[args.policy]["kind"] == "token":
        return build_token_policy(args, interpreter_class)
    delegates = None
    spec = getattr(args, "encoder_delegate", None)
    if spec:
        opts = parse_delegate_options(getattr(args, "delegate_option", []))
        delegates = [load_tflite_delegate(spec, opts)]
        print("[onboard] encoder delegate: %s %s" % (spec, opts or "{}"),
              flush=True)
    if getattr(args, "encoder_backend", "tflite") == "shim":
        encoder = ShimEncoder(args.encoder, args.shim_lib,
                              num_threads=(args.threads or 4),
                              use_nnapi=not args.shim_cpu)
        print("[onboard] encoder backend: %s (%s)"
              % (encoder.backend, args.shim_lib), flush=True)
    else:
        encoder = OnboardEncoder(args.encoder, interpreter_class,
                                 num_threads=args.threads, delegates=delegates)
    head = TFLiteHead(args.head, interpreter_class, num_threads=args.threads)
    if hasattr(encoder, "interp"):
        n_del, n_nodes = delegated_node_count(encoder.interp)
        if n_nodes is not None:
            print("[onboard] encoder graph: %d node(s), %d delegated"
                  % (n_nodes, n_del), flush=True)
    if hasattr(encoder, "interp"):
        print("[onboard] encoder %s: in %s %s, out %s (%r, %d)"
              % (os.path.basename(args.encoder), encoder.input_dtype,
                 [int(d) for d in encoder._in["shape"]], encoder.output_dtype,
                 encoder.output_scale, encoder.output_zero_point), flush=True)
    print("[onboard] head    %s: visual %s + imu %s -> %s"
          % (os.path.basename(args.head),
             [int(d) for d in head.visual["shape"]],
             [int(d) for d in head.imu["shape"]],
             [int(d) for d in head.out["shape"]]), flush=True)
    print("[onboard] policy %s" % args.policy, flush=True)
    net_every = max(1, int(round(float(args.rate) / max(args.net_rate, 1e-6))))
    policy = OnboardAgilePolicy(
        encoder, head, max_vel=args.max_vel, control_hz=args.rate,
        ref_lookahead_s=args.ref_lookahead_s,
        vel_lookahead_s=args.vel_lookahead_s, max_vel_xy=args.max_vel_xy,
        max_vel_z=args.max_vel_z, planar=args.planar,
        yaw_rate_max=args.yaw_rate_max, net_every=net_every)
    return policy, net_every


def build_token_policy(args, interpreter_class):
    """--policy <variant_id>: a registered token-front-end DepthNav policy."""
    v = ONBOARD_VARIANTS[args.policy]
    if args.encoder_backend == "shim":
        encoder = ShimTokenEncoder(args.encoder, args.shim_lib,
                                   gpu_flags=args.gpu_flags,
                                   num_threads=(args.threads or 4))
    else:
        encoder = TFLiteTokenEncoder(args.encoder, interpreter_class,
                                     num_threads=args.threads)
    head = TokenPolicyHead(args.head, interpreter_class, num_threads=1)
    # Warm-up: the first GPU invoke is several times slower than steady state;
    # pay it here, at start-up, not on the first step after the hand-over.
    t0 = time.perf_counter()
    encoder(np.zeros((1, NET_SIZE, NET_SIZE, 3), np.float32))
    print("[onboard] encoder warm-up invoke %.0f ms" % ((time.perf_counter() - t0) * 1e3),
          flush=True)
    max_xy = float(args.max_vel if args.max_vel_xy is None else args.max_vel_xy)
    max_z = float(args.max_vel if args.max_vel_z is None else args.max_vel_z)
    policy = OnboardTokenPolicy(
        encoder, head, target_speed=args.target_speed, max_vel_xy=max_xy,
        max_vel_z=max_z, planar=args.planar, yaw_slew_deg=args.yaw_slew_deg,
        pipeline=args.pipeline)
    print("[onboard] policy %s (checkpoint sha %s...)"
          % (args.policy, v["checkpoint_sha256"][:8]), flush=True)
    print("[onboard] encoder %s via %s; head %s (1 thread); %s"
          % (os.path.basename(args.encoder), encoder.backend,
             os.path.basename(args.head),
             "PIPELINED (convert || GPU encode, one GRU step per inference)"
             if args.pipeline else "synchronous (convert + encode + head in the tick)"),
          flush=True)
    print("[onboard] target speed %.2f m/s (band %g-%g); |v_xy| cap %.2f m/s, "
          "|vz| cap %.2f m/s (head bounds %.1f / %.1f); yaw slew limit %s"
          % (args.target_speed, v["target_speed_band"][0], v["target_speed_band"][1],
             max_xy, max_z, v["head_max_vel_xy"], v["head_max_vel_z"],
             "none (as trained / SITL)" if args.yaw_slew_deg is None
             else "%.0f deg/s" % args.yaw_slew_deg), flush=True)
    print("[onboard] NOTE: trained at %.0f Hz; the board steps it at the "
          "inference rate (~9.7 Hz pipelined, doc 15.11.6.4), so each GRU step "
          "spans ~100 ms instead of %.1f ms."
          % (v["trained_control_hz"], 1e3 / v["trained_control_hz"]), flush=True)
    return policy, 1


def build_source(args):
    if args.frames_npz:
        return NpzSource(args.frames_npz, fps=args.rate)
    return MPASource(args.pipe)


def _pct(samples):
    a = np.asarray(samples, dtype=np.float64)
    return dict(n=int(a.size), mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p90=float(np.percentile(a, 90)), p99=float(np.percentile(a, 99)),
                min=float(a.min()), max=float(a.max()))


def run_bench(args):
    """N ticks, no sockets, no camera: what does one tick cost on this box?"""
    if ONBOARD_VARIANTS[args.policy]["kind"] == "token" and args.pipeline:
        # A pipelined compute() returns without waiting for the network, so
        # timing it would measure nothing; the per-step cost is the sync one.
        print("[bench] --policy %s: benchmarking the synchronous step "
              "(--no-pipeline implied)" % args.policy, flush=True)
        args.pipeline = False
    policy, net_every = build_stack(args)
    n = int(args.bench)
    if args.frames_npz:
        data = np.load(args.frames_npz)
        key = "frames" if "frames" in data else list(data.keys())[0]
        pool = np.asarray(data[key])
        print("[bench] %d recorded frames from %s" % (len(pool), args.frames_npz),
              flush=True)
    else:
        # Deterministic synthetic frames: a fixed seed so two boards benchmark
        # the same bytes, and 32 of them so the interpreter cannot cache one.
        rng = np.random.RandomState(0)
        pool = rng.randint(0, 256, size=(32, NET_SIZE, NET_SIZE, 3)).astype(np.uint8)
        print("[bench] 32 synthetic frames (RandomState(0))", flush=True)

    pos = np.array([0.0, 0.0, 2.0])
    vel = np.array([1.0, 0.0, 0.0])
    R = np.eye(3)
    ang = np.zeros(3)
    goal = np.array([20.0, 0.0, 2.0])
    imu = hover_imu_state(args.imu_speed) if args.imu_fixed else None

    enc, head_ms, tail, total = [], [], [], []
    for i in range(n + 5):                     # 5 warm-up ticks, discarded
        frame = to_net_frame(pool[i % len(pool)])
        t0 = time.perf_counter()
        cmd = policy.compute(pos, vel, R, ang, goal, frame, imu_override=imu)
        dt_ms = (time.perf_counter() - t0) * 1e3
        if i < 5:
            continue
        if cmd["net_pass"]:
            enc.append(cmd["enc_ms"])
            head_ms.append(cmd["head_ms"])
        tail.append(cmd["tail_ms"])
        total.append(dt_ms)

    print("\n[bench] %d ticks, net every %d tick(s), threads=%s"
          % (n, net_every, args.threads), flush=True)
    rows = [("encoder", enc), ("head", head_ms), ("numpy tail", tail),
            ("TOTAL tick", total)]
    print("%-12s %6s %8s %8s %8s %8s %8s"
          % ("stage", "n", "mean", "p50", "p90", "p99", "max"))
    for name, s in rows:
        if not s:
            continue
        q = _pct(s)
        print("%-12s %6d %8.3f %8.3f %8.3f %8.3f %8.3f"
              % (name, q["n"], q["mean"], q["p50"], q["p90"], q["p99"], q["max"]))
    q = _pct(total)
    print("\n[bench] sustainable rate from p90 total: %.1f Hz "
          "(15 Hz needs p90 <= 66.7 ms)" % (1000.0 / q["p90"]), flush=True)
    print("[bench] last command: v_enu=%s yaw=%.3f alphas=%s"
          % (np.round(cmd["vel_enu"], 4), cmd["yaw"],
             None if cmd.get("alphas") is None else np.round(cmd["alphas"], 4)),
          flush=True)
    return 0


# ===========================================================================
# 7b. Profiling, frame capture and the appended tick columns (doc 15.11.6.5)
# ===========================================================================
# Everything here is OFF unless --profile / --save-frames is typed, except the
# nine columns appended at the END of ticks.csv (TICK_EXTRA_HEADER), which are
# pure reads of values the loop already has. Design rules, because this runs on
# the flight computer next to the control loop:
#   * the profiler is a daemon thread with its OWN file; it shares no lock and
#     no state with the loop, and every read is wrapped: a failing source is
#     reported once and its column stays empty -- it never raises into anything;
#   * the loop hands the frame saver a REFERENCE to an input the policy already
#     computed; the hand-over is a GIL-atomic deque append (no lock), a full
#     queue drops the frame and counts it, and the writer thread does all the
#     conversion and file I/O;
#   * a failure in the appended-column formatting leaves those columns empty
#     (reported once), never the tick.

PROFILE_HZ_DEFAULT = 2.0
FRAMES_PER_FILE = 50
FRAME_QUEUE_MAX = 64

# Appended to ticks.csv, in this order, after `offboard`:
#   pol_vf,pol_vl,pol_vu  the policy's raw velocity in ITS frame (token: START
#                         FLU, action[:3]; agile: plan-time body FLU at 7 m/s)
#   pol_yaw_deg           token: action[3], the yaw offset from the START
#                         heading; agile: empty (its yaw is not a net output)
#   meas_vn,meas_ve,meas_vd  measured velocity, world NED (--state mpa only)
#   roll_deg,pitch_deg    measured attitude, NED/FRD Euler (--state mpa only)
# Empty on ticks where the source says nothing (hold / takeoff override for
# the pol_* columns, --state synthetic for the measured ones).
TICK_EXTRA_HEADER = (",pol_vf,pol_vl,pol_vu,pol_yaw_deg,"
                     "meas_vn,meas_ve,meas_vd,roll_deg,pitch_deg")
TICK_EXTRA_EMPTY = "," * 9
TICK_EXTRA_DOC = {
    "pol_vf,pol_vl,pol_vu": "policy raw velocity in its own frame [m/s]: token = "
                            "action[:3] in the START frame (FLU axes of the "
                            "attitude latched at the first POLICY tick); agile = "
                            "reference velocity of the tracked plan in the body "
                            "FLU frame it was made in, at the net's native "
                            "7 m/s (before scale, --planar, caps). Empty when "
                            "the policy did not tick.",
    "pol_yaw_deg": "token: action[3], yaw offset from the START heading [deg]; "
                   "agile: empty",
    "meas_vn,meas_ve,meas_vd": "newest EKF2 velocity sample, world NED [m/s] "
                               "(--state mpa; empty with --state synthetic)",
    "roll_deg,pitch_deg": "newest EKF2 attitude, NED/FRD ZYX Euler [deg] "
                          "(--state mpa; empty with --state synthetic)",
}


def roll_pitch_from_quat_ned(q_wxyz):
    """Hamilton [w,x,y,z] NED->FRD -> (roll, pitch) [rad], ZYX Euler (the
    inverse of quat_from_euler_ned)."""
    w, x, y, z = (float(v) for v in q_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return 0.0, 0.0
    w, x, y, z = w / n, x / n, y / n, z / n
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


_tick_extra_err = {"done": False}


def tick_extra_fields(policy, policy_ticked, state_last):
    """The appended ticks.csv fields (TICK_EXTRA_HEADER), as one string that
    starts with a comma. Never raises: on any error the fields are left empty
    and the error is printed once."""
    try:
        raw = None
        if policy_ticked and hasattr(policy, "raw_output"):
            raw = policy.raw_output()
        if raw is None:
            s = ",,,,"
        else:
            v, yaw = raw
            s = ",%.5f,%.5f,%.5f,%s" % (float(v[0]), float(v[1]), float(v[2]),
                                        "" if yaw is None
                                        else "%.4f" % math.degrees(yaw))
        if state_last is None:
            return s + ",,,,,"
        vn, ve, vd = (float(c) for c in state_last["vel_ned"])
        roll, pitch = roll_pitch_from_quat_ned(state_last["q_ned_frd"])
        return s + ",%.5f,%.5f,%.5f,%.4f,%.4f" % (
            vn, ve, vd, math.degrees(roll), math.degrees(pitch))
    except Exception as exc:                                  # noqa: BLE001
        if not _tick_extra_err["done"]:
            _tick_extra_err["done"] = True
            try:
                print("[ticks] appended columns left empty: %r (reported once)"
                      % (exc,), flush=True)
            except Exception:                                 # noqa: BLE001
                pass
        return TICK_EXTRA_EMPTY


def _read_text(path):
    with open(path, "r") as f:
        return f.read()


def _cpu_list(text):
    """'0-3,6' -> [0, 1, 2, 3, 6]"""
    out = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _tail_int(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1


class SystemProfiler(object):
    """--profile: whole-board resource sampler, a daemon thread (doc 15.11.6.5 s2.1).

    Writes `profile.csv`, one row per sample, `wall_s` on the SAME clock as
    ticks.csv (seconds since the loop's t0). Every path lives under `root`
    (default "/"; tests point it at a fake tree). A source that is missing at
    start is simply absent (its column stays empty, run_meta says so); one
    that fails later is reported ONCE and its column goes empty. Nothing here
    ever raises into the caller, and no lock or object is shared with the
    flight loop (stop() is the only call from outside, at shutdown).

      cpu_pct, cpu{i}_pct      /proc/stat, busy share between two samples
      proc_cpu_pct             /proc/self/stat utime+stime (can exceed 100)
      proc_rss_mb, proc_threads  /proc/self/status VmRSS / Threads
      gpu_busy_pct             kgsl gpu_busy_percentage, else gpubusy busy/total
      gpu_mhz                  kgsl gpuclk, else devfreq/cur_freq (Hz)
      cpufreq_mhz_policy{k}    cpufreq/policy{k}/scaling_cur_freq (kHz)
      mem_avail_mb             /proc/meminfo MemAvailable
      temp_max_c               max over thermal_zone*/temp (m°C); names in run_meta
    """

    def __init__(self, root="/", hz=PROFILE_HZ_DEFAULT, t0=None, path=None,
                 clk_tck=None):
        self.root = str(root)
        self.hz = float(hz)
        self.t0 = time.time() if t0 is None else float(t0)
        self.path = path
        if clk_tck is None:
            try:
                clk_tck = os.sysconf("SC_CLK_TCK")
            except Exception:                                 # noqa: BLE001
                clk_tck = 100
        self.clk_tck = float(clk_tck)
        self.rows = 0
        self.errors = {}
        self._stop = False
        self._wait = threading.Event()        # private: only stop() sets it
        self._thread = None
        self._prev_cpu = None
        self._prev_proc = None
        self._wall = time.time                # clocks (tests substitute them)
        self._mono = time.monotonic
        self.probe()

    def _p(self, *parts):
        return os.path.join(self.root, *parts)

    def _err(self, key, exc):
        if key in self.errors:
            return
        self.errors[key] = repr(exc)
        try:
            print("[profile] %s unavailable: %r -- column(s) left empty "
                  "(reported once)" % (key, exc), flush=True)
        except Exception:                                     # noqa: BLE001
            pass

    @staticmethod
    def _readable(path):
        try:
            _read_text(path)
            return True
        except Exception:                                     # noqa: BLE001
            return False

    def probe(self):
        """Which sources exist; fixes the column list. Never raises."""
        p = self._p
        self.p_stat = p("proc", "stat")
        self.p_self_stat = p("proc", "self", "stat")
        self.p_self_status = p("proc", "self", "status")
        self.p_meminfo = p("proc", "meminfo")
        kgsl = p("sys", "class", "kgsl", "kgsl-3d0")
        self.p_gpu_busy = None
        self.gpu_busy_kind = None
        for kind, path in (("gpu_busy_percentage", os.path.join(kgsl, "gpu_busy_percentage")),
                           ("gpubusy", os.path.join(kgsl, "gpubusy"))):
            if self._readable(path):
                self.p_gpu_busy, self.gpu_busy_kind = path, kind
                break
        self.p_gpu_freq = None
        for path in (os.path.join(kgsl, "gpuclk"),
                     os.path.join(kgsl, "devfreq", "cur_freq")):
            if self._readable(path):
                self.p_gpu_freq = path
                break
        self.cores = []
        try:
            self.cores = _cpu_list(_read_text(p("sys", "devices", "system", "cpu", "possible")))
        except Exception:                                     # noqa: BLE001
            try:
                self.cores = sorted(int(l.split()[0][3:]) for l in
                                    _read_text(self.p_stat).splitlines()
                                    if re.match(r"^cpu\d+\s", l))
            except Exception:                                 # noqa: BLE001
                self.cores = []
        self.cpufreq = []                    # (k, path, related_cpus)
        base = p("sys", "devices", "system", "cpu", "cpufreq")
        try:
            names = sorted((n for n in os.listdir(base) if re.match(r"^policy\d+$", n)),
                           key=_tail_int)
        except Exception:                                     # noqa: BLE001
            names = []
        for n in names:
            f = os.path.join(base, n, "scaling_cur_freq")
            if self._readable(f):
                try:
                    rel = _read_text(os.path.join(base, n, "related_cpus")).strip()
                except Exception:                             # noqa: BLE001
                    rel = None
                self.cpufreq.append((_tail_int(n), f, rel))
        self.thermal = []                    # (name, temp_path, type)
        base = p("sys", "class", "thermal")
        try:
            names = sorted((n for n in os.listdir(base) if re.match(r"^thermal_zone\d+$", n)),
                           key=_tail_int)
        except Exception:                                     # noqa: BLE001
            names = []
        for n in names:
            f = os.path.join(base, n, "temp")
            if self._readable(f):
                try:
                    typ = _read_text(os.path.join(base, n, "type")).strip()
                except Exception:                             # noqa: BLE001
                    typ = "?"
                self.thermal.append((n, f, typ))
        self.columns = (["wall_s", "cpu_pct"]
                        + ["cpu%d_pct" % i for i in self.cores]
                        + ["proc_cpu_pct", "proc_rss_mb", "proc_threads",
                           "gpu_busy_pct", "gpu_mhz"]
                        + ["cpufreq_mhz_policy%d" % k for k, _, _ in self.cpufreq]
                        + ["mem_avail_mb", "temp_max_c"])
        self.sources = {
            "proc_stat": dict(path=self.p_stat, available=self._readable(self.p_stat)),
            "proc_self_stat": dict(path=self.p_self_stat,
                                   available=self._readable(self.p_self_stat)),
            "proc_self_status": dict(path=self.p_self_status,
                                     available=self._readable(self.p_self_status)),
            "gpu_busy": dict(path=self.p_gpu_busy, kind=self.gpu_busy_kind,
                             available=self.p_gpu_busy is not None),
            "gpu_freq": dict(path=self.p_gpu_freq, available=self.p_gpu_freq is not None),
            "cpufreq": dict(path=p("sys", "devices", "system", "cpu", "cpufreq"),
                            available=bool(self.cpufreq),
                            policies={"policy%d" % k: rel for k, _, rel in self.cpufreq}),
            "meminfo": dict(path=self.p_meminfo, available=self._readable(self.p_meminfo)),
            "thermal": dict(path=p("sys", "class", "thermal"), available=bool(self.thermal),
                            zones={n: typ for n, _, typ in self.thermal}),
        }

    def meta(self):
        return dict(root=self.root, hz=self.hz, path=self.path, clk_tck=self.clk_tck,
                    columns=list(self.columns), sources=self.sources,
                    cores=list(self.cores),
                    note="NPU (Hexagon) occupancy has no readable sysfs; not sampled")

    # -- one sample -------------------------------------------------------
    def _cpu_times(self):
        out = {}
        for line in _read_text(self.p_stat).splitlines():
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            vals = [int(v) for v in parts[1:9]]
            total = sum(vals)
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            out[parts[0]] = (total, idle)
        return out

    def sample(self):
        """-> {column: float or None}. Never raises."""
        row = dict.fromkeys(self.columns)
        row["wall_s"] = self._wall() - self.t0
        mono = self._mono()
        try:
            cur = self._cpu_times()
            prev, self._prev_cpu = self._prev_cpu, cur
            if prev is not None:
                for name, col in ([("cpu", "cpu_pct")]
                                  + [("cpu%d" % i, "cpu%d_pct" % i) for i in self.cores]):
                    if name in cur and name in prev:
                        dt = cur[name][0] - prev[name][0]
                        di = cur[name][1] - prev[name][1]
                        if dt > 0:
                            row[col] = min(100.0, max(0.0, 100.0 * (1.0 - float(di) / dt)))
        except Exception as exc:                              # noqa: BLE001
            self._err("/proc/stat", exc)
        try:
            txt = _read_text(self.p_self_stat)
            rest = txt[txt.rindex(")") + 2:].split()
            cpu_s = (int(rest[11]) + int(rest[12])) / self.clk_tck
            prev, self._prev_proc = self._prev_proc, (mono, cpu_s)
            if prev is not None and mono > prev[0]:
                row["proc_cpu_pct"] = max(0.0, 100.0 * (cpu_s - prev[1]) / (mono - prev[0]))
        except Exception as exc:                              # noqa: BLE001
            self._err("/proc/self/stat", exc)
        try:
            for line in _read_text(self.p_self_status).splitlines():
                if line.startswith("VmRSS:"):
                    row["proc_rss_mb"] = int(line.split()[1]) / 1024.0
                elif line.startswith("Threads:"):
                    row["proc_threads"] = int(line.split()[1])
        except Exception as exc:                              # noqa: BLE001
            self._err("/proc/self/status", exc)
        if self.p_gpu_busy is not None:
            try:
                f = _read_text(self.p_gpu_busy).replace("%", " ").split()
                if self.gpu_busy_kind == "gpu_busy_percentage":
                    row["gpu_busy_pct"] = float(f[0])
                else:
                    busy, total = float(f[0]), float(f[1])
                    row["gpu_busy_pct"] = 100.0 * busy / total if total > 0 else 0.0
            except Exception as exc:                          # noqa: BLE001
                self._err("gpu busy (%s)" % self.p_gpu_busy, exc)
        if self.p_gpu_freq is not None:
            try:
                row["gpu_mhz"] = float(_read_text(self.p_gpu_freq).split()[0]) / 1e6
            except Exception as exc:                          # noqa: BLE001
                self._err("gpu freq (%s)" % self.p_gpu_freq, exc)
        for k, path, _ in self.cpufreq:
            try:
                row["cpufreq_mhz_policy%d" % k] = float(_read_text(path).split()[0]) / 1e3
            except Exception as exc:                          # noqa: BLE001
                self._err(path, exc)
        try:
            for line in _read_text(self.p_meminfo).splitlines():
                if line.startswith("MemAvailable:"):
                    row["mem_avail_mb"] = int(line.split()[1]) / 1024.0
                    break
        except Exception as exc:                              # noqa: BLE001
            self._err("/proc/meminfo", exc)
        temps = []
        for name, path, _ in self.thermal:
            try:
                v = float(_read_text(path).split()[0])
                c = v / 1000.0 if abs(v) >= 1000.0 else v     # m°C (kernel ABI) or °C
                if -50.0 <= c <= 200.0:                       # disabled zones read junk
                    temps.append(c)
            except Exception as exc:                          # noqa: BLE001
                self._err(path, exc)
        if temps:
            row["temp_max_c"] = max(temps)
        return row

    def format_row(self, row):
        out = []
        for c in self.columns:
            v = row.get(c)
            if v is None:
                out.append("")
            elif c == "wall_s":
                out.append("%.4f" % v)
            elif c == "proc_threads":
                out.append("%d" % v)
            else:
                out.append("%.2f" % v)
        return ",".join(out) + "\n"

    # -- the thread ---------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, name="profiler", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        try:
            f = open(self.path, "x")
        except Exception as exc:                              # noqa: BLE001
            self._err("profile csv %s" % self.path, exc)
            return
        try:
            f.write("# onboard_policy_runner profile log (doc 15.11.6.5); "
                    "wall_s = ticks.csv clock\n# t0_epoch=%.6f\n" % self.t0)
            f.write(",".join(self.columns) + "\n")
            f.flush()
            self.sample()                     # baseline for the deltas, no row
            period = 1.0 / max(self.hz, 1e-3)
            nxt = time.monotonic() + period
            while not self._stop:
                self._wait.wait(max(0.0, nxt - time.monotonic()))
                if self._stop:
                    break
                nxt += period
                if nxt < time.monotonic():
                    nxt = time.monotonic() + period
                try:
                    f.write(self.format_row(self.sample()))
                    f.flush()
                    self.rows += 1
                except Exception as exc:                      # noqa: BLE001
                    self._err("profile row", exc)
        except Exception as exc:                              # noqa: BLE001
            self._err("profiler thread", exc)
        finally:
            try:
                f.close()
            except Exception:                                 # noqa: BLE001
                pass

    def stop(self, timeout=2.0):
        self._stop = True
        self._wait.set()
        if self._thread is not None:
            self._thread.join(timeout)


def _stored_frame(x):
    """Network input -> (image to store, input shape, input dtype, exact_u8).

    A float input in [0,1] that is exactly uint8/255 (net_frame_of's own
    arithmetic) is stored as that uint8 image; anything else as float32."""
    a = np.asarray(x)
    shape, dtype = [int(d) for d in a.shape], str(a.dtype)
    img = a[0] if (a.ndim == 4 and a.shape[0] == 1) else a
    if img.dtype == np.uint8:
        return img, shape, dtype, True
    f = np.asarray(img, np.float32)
    u8 = np.clip(np.rint(f.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)
    if np.array_equal(u8.astype(np.float32) / 255.0, f):
        return u8, shape, dtype, True
    return f, shape, dtype, False


class FrameSaver(object):
    """--save-frames N: every N-th network pass, the exact network input.

    Loop side (flight thread): `want()` counts a network pass and says whether
    this one is selected; `push(rec)` appends a record to a deque -- atomic
    under the GIL, no lock -- or, when `max_queue` records are already waiting,
    drops it and counts the drop. Writer side (a daemon thread): pops records,
    converts the image, and writes `frames_XXXX.npz` every `per_file` records
    (atomic rename; never overwrites). Record keys: x, in_tick, in_wall_s,
    enc_wall_s, act_tick, act_wall_s, raw_out[4], sp_ned[4], pass_idx.
    """

    def __init__(self, out_dir, every, per_file=FRAMES_PER_FILE,
                 max_queue=FRAME_QUEUE_MAX, start=True):
        self.out_dir = str(out_dir)
        self.every = max(1, int(every))
        self.per_file = max(1, int(per_file))
        self.max_queue = max(1, int(max_queue))
        self.passes = 0
        self.selected = 0
        self.dropped = 0
        self.saved = 0
        self.lost = 0
        self.files = 0
        self.write_errors = 0
        self.input_shape = None
        self.input_dtype = None
        self.stored_as = None
        self.errors = {}
        self._file_idx = 0
        self._q = collections.deque()
        self._stop = False
        self._wait = threading.Event()        # private: only close() sets it
        os.makedirs(self.out_dir, exist_ok=True)
        self._thread = None
        if start:
            self._thread = threading.Thread(target=self._run, name="frame-saver",
                                            daemon=True)
            self._thread.start()

    # -- flight-loop side ---------------------------------------------------
    def want(self):
        self.passes += 1
        if (self.passes - 1) % self.every:
            return False
        self.selected += 1
        return True

    def push(self, rec):
        if len(self._q) >= self.max_queue:
            self.dropped += 1
            return False
        self._q.append(rec)
        return True

    # -- writer side ----------------------------------------------------------
    def _err(self, key, exc):
        if key in self.errors:
            return
        self.errors[key] = repr(exc)
        try:
            print("[frames] %s: %r (reported once)" % (key, exc), flush=True)
        except Exception:                                     # noqa: BLE001
            pass

    def _run(self):
        chunk = []
        while True:
            try:
                rec = self._q.popleft()
            except IndexError:
                if self._stop:
                    break
                self._wait.wait(0.05)
                continue
            chunk.append(rec)
            if len(chunk) >= self.per_file:
                self._write(chunk)
                chunk = []
        if chunk:
            self._write(chunk)

    def _write(self, chunk):
        try:
            imgs, exact = [], True
            for rec in chunk:
                img, shape, dtype, ok = _stored_frame(rec["x"])
                imgs.append(img)
                exact &= ok
                if self.input_shape is None:
                    self.input_shape, self.input_dtype = shape, dtype
            if exact:
                frames = np.stack(imgs).astype(np.uint8)
                enc = ("uint8; the network input is frames[..]/255 exactly"
                       if self.input_dtype != "uint8" else "uint8; as fed")
            else:
                frames = np.stack([np.asarray(i, np.float32) for i in imgs])
                enc = "float32; as fed"
            self.stored_as = enc

            def col(key, default=float("nan")):
                return np.array([default if r.get(key) is None else r[key] for r in chunk],
                                dtype=np.float64)
            arrays = dict(
                frames=frames, frames_encoding=np.array(enc),
                input_shape=np.array(self.input_shape, np.int64),
                input_dtype=np.array(self.input_dtype),
                in_tick=col("in_tick", -1).astype(np.int64),
                in_wall_s=col("in_wall_s"), enc_wall_s=col("enc_wall_s"),
                act_tick=col("act_tick", -1).astype(np.int64),
                act_wall_s=col("act_wall_s"),
                raw_out=np.array([r.get("raw_out", (np.nan,) * 4) for r in chunk],
                                 dtype=np.float64).reshape(-1, 4),
                sp_ned=np.array([r.get("sp_ned", (np.nan,) * 4) for r in chunk],
                                dtype=np.float64).reshape(-1, 4),
                pass_idx=col("pass_idx", -1).astype(np.int64))
            while True:
                final = os.path.join(self.out_dir, "frames_%04d.npz" % self._file_idx)
                self._file_idx += 1
                if not os.path.exists(final):
                    break
            part = final + ".part"
            with open(part, "xb") as f:
                np.savez(f, **arrays)
            os.replace(part, final)
            self.saved += len(chunk)
            self.files += 1
        except Exception as exc:                              # noqa: BLE001
            self.write_errors += 1
            self.lost += len(chunk)
            self._err("write failed", exc)

    def close(self, timeout=15.0):
        self._stop = True
        self._wait.set()
        if self._thread is not None:
            self._thread.join(timeout)
        return self.stats()

    def stats(self):
        return dict(dir=self.out_dir, every=self.every, per_file=self.per_file,
                    max_queue=self.max_queue, net_passes=self.passes,
                    selected=self.selected, saved=self.saved, dropped=self.dropped,
                    lost_in_write_errors=self.lost, write_errors=self.write_errors,
                    files=self.files, pending=len(self._q),
                    input_shape=self.input_shape, input_dtype=self.input_dtype,
                    stored_as=self.stored_as)


_save_frame_err = {"done": False}


def save_frame_record(saver, policy, tick, wall_s, t0, v_ned, yaw_sp_ned):
    """Flight-loop side of --save-frames, called on a network-pass tick: if
    this pass is selected, queue the policy's `last_net_record` (the input
    that produced the action applied at THIS tick) with both tick numbers,
    the raw action and the setpoint that went out. A reference, no copy, no
    lock. Never raises."""
    try:
        if not saver.want():
            return
        nr = getattr(policy, "last_net_record", None)
        if nr is None:
            return
        x, tag, t_enc = nr
        raw = policy.raw_output() if hasattr(policy, "raw_output") else None
        if raw is None:
            raw_out = (float("nan"),) * 4
        else:
            v, yaw = raw
            raw_out = (float(v[0]), float(v[1]), float(v[2]),
                       float("nan") if yaw is None else float(yaw))
        saver.push(dict(
            x=x, in_tick=None if tag is None else tag[0],
            in_wall_s=None if tag is None else tag[1],
            enc_wall_s=None if t_enc is None else t_enc - t0,
            act_tick=tick, act_wall_s=wall_s, raw_out=raw_out,
            sp_ned=(float(v_ned[0]), float(v_ned[1]), float(v_ned[2]),
                    float(yaw_sp_ned)),
            pass_idx=saver.passes))
    except Exception as exc:                                  # noqa: BLE001
        if not _save_frame_err["done"]:
            _save_frame_err["done"] = True
            try:
                print("[frames] record skipped: %r (reported once)" % (exc,), flush=True)
            except Exception:                                 # noqa: BLE001
                pass


def update_run_meta(run_dir, **sections):
    """Merge `sections` into run_dir/run_meta.json (temp file + rename). Never
    raises; a failure is printed."""
    if not run_dir:
        return
    path = os.path.join(run_dir, "run_meta.json")
    try:
        with open(path) as f:
            meta = json.load(f)
        meta.update(sections)
        with open(path + ".tmp", "w") as f:
            json.dump(meta, f, indent=1, default=str)
        os.replace(path + ".tmp", path)
    except Exception as exc:                                  # noqa: BLE001
        print("[onboard] run_meta.json update failed: %r" % (exc,), flush=True)


# ===========================================================================
# 8. Run logs: every run keeps everything, nothing is ever overwritten
# ===========================================================================
# (user, 2026-09-23) One directory per run under --log-dir, created
# exclusively: runner.log (the whole stdout/stderr), ticks.csv (--log-csv's
# format), run_meta.json (argv, every argument, the sha256 of the runner and
# of every model file, host). An explicit --log-csv / --log-plans path that
# already exists gets a numeric suffix instead of being truncated.

def unique_path(path):
    """`path` if free, else `stem_1.ext`, `stem_2.ext`, ... -- never an existing file."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 1
    while os.path.exists("%s_%d%s" % (stem, k, ext)):
        k += 1
    return "%s_%d%s" % (stem, k, ext)


def make_run_dir(log_dir, tag):
    """<seq>_<UTC>_<policy>_<pid>. The VOXL 2 has no RTC: its clock restarts
    at every boot, so the timestamp is not monotonic across reboots -- the
    5-digit sequence number (1 + the highest already in log_dir) is what
    orders runs; the newest run is always the highest number."""
    seq = 0
    if os.path.isdir(log_dir):
        for name in os.listdir(log_dir):
            m = re.match(r"^(\d{5})_", name)
            if m:
                seq = max(seq, int(m.group(1)))
    base = os.path.join(log_dir, "%05d_%s_%s_%d" % (
        seq + 1, time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), tag, os.getpid()))
    path, k = base, 0
    while True:
        try:
            os.makedirs(path)            # exclusive: fails if it exists
            return path
        except FileExistsError:
            k += 1
            path = "%s_%d" % (base, k)


class _Tee:
    """stdout/stderr -> the terminal AND runner.log (line-buffered)."""

    def __init__(self, stream, fh):
        self._s, self._f = stream, fh

    def write(self, data):
        self._s.write(data)
        try:
            self._f.write(data)
        except ValueError:               # closed at shutdown
            pass
        return len(data)

    def flush(self):
        self._s.flush()
        try:
            self._f.flush()
        except ValueError:
            pass

    def __getattr__(self, name):
        return getattr(self._s, name)


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        return "unreadable: %s" % exc
    return h.hexdigest()


def start_run_logs(args, argv):
    """-> run_dir or None. Sets args.log_csv (default: run_dir/ticks.csv)."""
    if args.log_csv:
        args.log_csv = unique_path(args.log_csv)
    if args.log_plans:
        args.log_plans = unique_path(args.log_plans)
    if args.no_log_dir:
        return None
    run_dir = make_run_dir(args.log_dir, args.policy)
    fh = open(os.path.join(run_dir, "runner.log"), "x", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    if not args.log_csv:
        args.log_csv = os.path.join(run_dir, "ticks.csv")
    files = {"runner": os.path.abspath(__file__), "encoder": args.encoder,
             "head": args.head}
    if getattr(args, "encoder_backend", "") == "shim":
        files["shim_lib"] = args.shim_lib
    meta = dict(argv=list(sys.argv if argv is None else argv),
                args=vars(args), started_utc=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                host=socket.gethostname(), pid=os.getpid(),
                python=sys.version.split()[0],
                files={k: dict(path=v, sha256=_sha256(v)) for k, v in files.items()},
                # doc 15.11.6.5: what the columns appended to ticks.csv mean
                ticks_appended_columns=TICK_EXTRA_DOC)
    with open(os.path.join(run_dir, "run_meta.json"), "x") as f:
        json.dump(meta, f, indent=1, default=str)
    print("[onboard] run logs -> %s (runner.log, ticks.csv, run_meta.json; "
          "never overwritten)" % run_dir, flush=True)
    return run_dir


def main(argv=None):
    args = parse_args(argv)
    if args.bench:
        return run_bench(args)
    run_dir = start_run_logs(args, argv)
    # doc 15.11.6.5: where --profile / --save-frames write (checked before
    # anything is built, so a bad combination fails on the ground).
    profile_csv = frames_dir = None
    if args.profile:
        profile_csv = args.profile_csv or (run_dir and os.path.join(run_dir, "profile.csv"))
        if not profile_csv:
            raise SystemExit("[onboard] --profile with --no-log-dir needs --profile-csv")
        profile_csv = unique_path(profile_csv)
    if args.save_frames > 0:
        frames_dir = args.frames_dir or (run_dir and os.path.join(run_dir, "frames"))
        if not frames_dir:
            raise SystemExit("[onboard] --save-frames with --no-log-dir needs --frames-dir")

    dt = 1.0 / float(args.rate)
    policy, net_every = build_stack(args)
    source = build_source(args)
    yaw_mode = str(args.yaw_out)
    if args.enable_test:
        vel_pub = PrintOnlyPublisher(every=args.test_print_every,
                                     yaw_mode=yaw_mode)
    else:
        vel_pub = MavlinkVelPublisher(host=args.mav_host, port=args.mav_port,
                                      yaw_mode=yaw_mode)
    # doc 12 s7.1: one CommandPublisher per command type, and the policy says
    # which one it needs. `vel_pub` above is only the WIRE -- what goes on it
    # is this publisher's business now.
    vel_yaw_pub = VelocityYawPublisher(
        vel_pub, max_vel_xy=policy.max_vel_xy,
        yaw_rate_max=args.yaw_rate_max, yaw_mode=yaw_mode,
        alt_hold=bool(args.alt_hold) and bool(args.planar),
        alt_hold_kp=args.alt_hold_kp, alt_hold_kd=args.alt_hold_kd,
        alt_hold_max_vz=args.alt_hold_max_vz)
    publishers = {vel_yaw_pub.command_type: vel_yaw_pub}
    # Rule (1): a policy whose command type has no publisher does not fly.
    policy_cmd_type = getattr(policy, "command_type", DEFAULT_COMMAND_TYPE)
    if policy_cmd_type not in publishers:
        raise SystemExit(
            "[onboard] the policy declares command_type %r, for which this "
            "runner has no CommandPublisher (registered: %s). Refusing to "
            "fly." % (policy_cmd_type, ", ".join(sorted(publishers))))
    port = TaskCommandPort(
        host=args.cmd_bind, port=int(args.cmd_port),
        config=dict(max_vel=float(args.max_vel),
                    max_vel_xy=float(policy.max_vel_xy),
                    max_vel_z=float(policy.max_vel_z),
                    planar=bool(args.planar),
                    vel_lookahead_s=float(args.vel_lookahead_s),
                    yaw_rate_max=float(args.yaw_rate_max),
                    rate_hz=float(args.rate), net_every=int(net_every),
                    state=str(args.state), body_rates=str(args.body_rates),
                    enable_test=bool(args.enable_test),
                    encoder=os.path.basename(args.encoder),
                    head=os.path.basename(args.head),
                    net_size=int(NET_SIZE),
                    # 4.7.2 fixes: what the HUD must show, because a flag typed
                    # on the board and not on the laptop is exactly the kind of
                    # divergence the config dict exists to make visible.
                    goal_z_mode=str(args.goal_z_mode),
                    goal_z_max_delta=float(args.goal_z_max_delta),
                    yaw_out=yaw_mode,
                    align_first=bool(args.align_first),
                    alt_hold=bool(args.alt_hold),
                    # 7.10: which instant the policy is allowed to steer from.
                    engage=str(args.engage),
                    # Filled in from the first frame's metadata (audit
                    # defect 7): null until one has actually arrived.
                    cam_width=None, cam_height=None, cam_format=None))
    imu_fixed = hover_imu_state(args.imu_speed) if args.imu_fixed else None
    state_src = (MavlinkStateSource(args.state_pipe)
                 if args.state == "mpa" else None)
    want_rates = args.body_rates == "mavlink"

    # --- the state: dead-reckoned stand-in, or the EKF2 estimate ----------
    # In --state mpa every one of these is overwritten from the newest valid
    # sample before it is used; args.start survives only as the placeholder
    # the banner prints, and the loop holds at zero velocity until the first
    # valid sample lands.
    pos_ned = np.array(args.start, dtype=np.float64)
    yaw_ned = math.radians(args.start_yaw_deg)
    vel_ned = np.zeros(3)
    ang_flu = np.zeros(3)               # body rates, FLU; zero until measured
    R_enu_meas = None                   # set from the measured quaternion only
    goal_ned = None if args.goal is None else np.array(args.goal, dtype=np.float64)
    cruise_d = float(pos_ned[2])

    print("[onboard] camera: %s (%s), control %g Hz, net every %d tick(s) "
          "(~%.1f Hz)" % (getattr(source, "name", "?"),
                          args.frames_npz or args.pipe, args.rate, net_every,
                          args.rate / net_every), flush=True)
    if args.enable_test:
        print("[onboard] *** --enable_test: SMOKE TEST, NOTHING IS SENT TO "
              "PX4. *** Setpoints are printed only; no MAVLink connection is "
              "opened. The vehicle cannot be commanded by this process.",
              flush=True)
    else:
        print("[onboard] setpoints out -> udpout:%s:%d "
              "(SET_POSITION_TARGET_LOCAL_NED, mask 0x%04X, world NED, "
              "yaw as %s)"
              % (vel_pub.addr[0], vel_pub.addr[1],
                 MavlinkVelPublisher.MASK_VEL_YAW if yaw_mode == "angle"
                 else MavlinkVelPublisher.MASK_VEL_YAWRATE,
                 "an ABSOLUTE angle" if yaw_mode == "angle" else "a RATE"),
              flush=True)
    print("[onboard] goals in     <- %s:%d (JSON, world NED)"
          % (args.cmd_bind, port.port), flush=True)
    if state_src is None:
        print("[onboard] STATE IS SYNTHETIC: level attitude, zero body rates, "
              "position dead-reckoned from the commanded velocity, start NED %s "
              "heading %.1f deg. There is no EKF2 feedback."
              % (pos_ned.round(2), math.degrees(yaw_ned)), flush=True)
        if args.freeze_pose:
            print("[onboard] --freeze-pose: the pose stays at start for the "
                  "whole run; the goal can never be reached and the policy "
                  "never holds on arrival (visualisation mode).", flush=True)
    elif args.freeze_pose:
        print("[onboard] --freeze-pose ignored: --state mpa uses the measured "
              "pose.", flush=True)
    else:
        print("[onboard] STATE FROM %s: EKF2 position, velocity, attitude and "
              "body rates, decoded in-process; dead reckoning is OFF. Holding "
              "at zero velocity until a sample arrives with "
              "xy_valid+v_xy_valid+attitude_valid and an age <= %.2f s (start "
              "NED %s is only a placeholder until then)."
              % (state_src.pipe_dir, args.state_stale_s, pos_ned.round(2)),
              flush=True)
        print("[onboard] body rates: %s"
              % ("ATTITUDE_QUATERNION rollspeed/pitchspeed/yawspeed, FRD->FLU"
                 if want_rates else
                 "FORCED TO ZERO by --body-rates zero (pre-rates behaviour)"),
              flush=True)
    if frames_dir is None:
        print("[onboard] frames are held in RAM only; this process never writes "
              "camera data to disk.", flush=True)
    else:
        print("[frames] --save-frames %d: the network input of every %d-th "
              "network pass is written to %s (%d per file, background thread; "
              "queue %d, full = drop and count)."
              % (args.save_frames, args.save_frames, frames_dir, FRAMES_PER_FILE,
                 FRAME_QUEUE_MAX), flush=True)
    print("[onboard] 4.7.2 fixes: goal-z-mode=%s (max delta %.1f m), "
          "yaw-out=%s, align-first=%s (tol %.1f deg, timeout %.1f s), "
          "alt-hold=%s (kp %.2f kd %.2f, |vz| <= %.2f m/s)"
          % (args.goal_z_mode, args.goal_z_max_delta, yaw_mode,
             args.align_first, args.align_tol_deg, args.align_timeout_s,
             args.alt_hold, args.alt_hold_kp, args.alt_hold_kd,
             args.alt_hold_max_vz), flush=True)
    if args.alt_hold and not args.planar:
        print("[onboard] --alt-hold IGNORED without --planar: the net is "
              "still commanding vz and the altitude loop would fight it.",
              flush=True)
    if args.engage == "offboard":
        print("[engage] --engage offboard (7.10): the policy does NOT steer "
              "until PX4's own HEARTBEAT says OFFBOARD. Until then the runner "
              "streams zero velocity and the MEASURED heading, and the ALIGN "
              "timer, the --alt-hold reference and the reference line's start "
              "are all latched at the instant OFFBOARD is entered -- not when "
              "the goal is applied. A HEARTBEAT older than %.1f s counts as "
              "not-offboard." % HEARTBEAT_STALE_S, flush=True)
    else:
        print("[engage] --engage immediate: the policy steers as soon as a "
              "goal is applied (pre-7.10 behaviour; no OFFBOARD gate).",
              flush=True)

    csv = None
    if args.log_csv:
        d = os.path.dirname(os.path.abspath(args.log_csv))
        if d and not os.path.isdir(d):
            os.makedirs(d)
        csv = open(args.log_csv, "x")          # never truncate a previous log
        csv.write("# onboard_policy_runner tick log; vectors world NED\n")
        csv.write("# t0_epoch=%.6f\n" % time.time())
        # The two state columns are appended ONLY in --state mpa, so a
        # synthetic-mode log is byte-identical to what it was before.
        # `yaw_sp_deg` (the absolute NED yaw setpoint, --yaw-out angle) and
        # `tick_dt_s` (the ACTUAL wall gap between ticks -- audit defect 3
        # measured 4.7-11.7 Hz where the loop assumed 15) are APPENDED, so
        # every column an existing reader indexes by name or by position is
        # still where it was.
        csv.write("wall_s,tick,seq,goal_seq,hold,awaiting,reached,"
                  "pos_n,pos_e,pos_d,yaw_ned_deg,vx_n,vy_e,vz_d,yaw_rate,"
                  "speed_xy,goal_n,goal_e,goal_d,dist_xy,"
                  "enc_ms,head_ms,tail_ms,tick_ms,net_pass,alpha0,mode_idx,"
                  "frame_age_s"
                  + ("" if state_src is None else ",state_age_s,state_flags")
                  + ",yaw_sp_deg,tick_dt_s"
                  # 7.10, APPENDED at the end so every existing column keeps
                  # its index: 1/0 = PX4's HEARTBEAT says OFFBOARD / does not,
                  # empty = nothing fresh to say (no beat yet, or > 2 s old).
                  + ",offboard"
                  # doc 15.11.6.5, APPENDED at the end for the same reason:
                  # the policy's raw output in its own frame, the measured NED
                  # velocity and roll/pitch (TICK_EXTRA_HEADER / _DOC).
                  + TICK_EXTRA_HEADER + "\n")

    # Per-network-update candidate-plan log (--log-plans). Independent of
    # --log-csv, which keeps writing exactly what it always wrote.
    plan_log = PlanLogger(args.log_plans) if args.log_plans else None
    if plan_log is not None:
        print("[onboard] logging every network-pass plan to %s"
              % plan_log.path, flush=True)

    awaiting_goal = goal_ned is None
    reached = False
    cmd_hold_prev = False
    frame_hold = False
    frame_stale_since = None
    frame_stale_episodes = 0
    state_last = None
    state_last_t = None
    state_age = None
    state_hold = state_src is not None       # hold until the first valid sample
    state_hold_reason = None
    state_hold_episodes = 0
    state_have_pos = False
    legs_flown = 0
    frames_in = 0
    rgb = None
    rgb_t = None
    stop = {"now": False}
    # --- 4.7.2 fix state --------------------------------------------------
    align_active = False        # a goal was applied and we are still turning
    align_t0 = None             # when the ALIGN phase actually started
    align_tol = math.radians(float(args.align_tol_deg))
    alt_ref_d = None            # the altitude --alt-hold holds, world NED down
    # --- 7.10: the OFFBOARD gate -----------------------------------------
    engage_gate = (args.engage == "offboard")
    offboard_active = False     # PX4's HEARTBEAT says OFFBOARD (unknown=False)
    offboard_csv = ""           # what the tick log records for this tick
    engage_pending = False      # OFFBOARD entered; the latch is still owed
    hb_first_logged = False     # the one-shot "the beat is on this pipe" line
    cam_meta_seen = False
    tick_prev_t = None          # for tick_dt_s
    # --- doc 15.11.6.4: auto takeoff -------------------------------------
    auto = (AutoTakeoff(climb_rate=args.climb_rate,
                        arrive_tol=args.arrive_tol,
                        settle_speed=args.settle_speed,
                        arm_timeout_s=args.arm_timeout_s,
                        climb_timeout_s=args.climb_timeout_s)
            if args.auto_takeoff else None)
    tk = dict(override=False, allow_latch=True, phase=None)
    armed_now = False
    main_now = None
    # The goal-z re-latch at the hand-over (see the latch below): for a policy
    # whose TARGET input carries the goal's z (the token policies), or when
    # the goal was necessarily applied on the ground (--auto-takeoff).
    relatch_goal_z = (args.goal_z_mode == "hold"
                      and (auto is not None
                           or getattr(policy, "takes_raw_frames", False)))
    resetwatch = (EstimatorResetWatch()
                  if (args.reset_action != "off" and state_src is not None) else None)
    if resetwatch is not None:
        q0 = resetwatch.prime()
        print("[reset] EKF2 reset watch: ODOMETRY.reset_counter on the state pipe; "
              "deltas from vehicle_local_position (%s)"
              % ("baseline counters %s" % q0["counters"] if q0 is not None
                 else "NOT readable now -- a reset will still hold, but "
                      "without an exact re-anchor"), flush=True)
    odom_warned = False
    resume_needs_engage = False     # a reset ended the episode; RESUME restarts it
    tframe = None               # TakeoffFrame, latched at the accepted TAKEOFF
    pending_flu = None          # a takeoff-flu goal waiting for the frame
    port.config["goal_frame"] = args.goal_frame
    if auto is not None:
        print("[takeoff] --auto-takeoff armed: SET_GOAL x y z is read as "
              "forward / left / up [m] of the pose the aircraft has when "
              "TAKEOFF is accepted (z = height above that ground point, "
              "%.1f-%.1f m). On TAKEOFF the runner requests OFFBOARD + ARM, "
              "climbs straight up to the goal's z at %.1f m/s, turns to face "
              "the goal, then the policy flies. After PX4 leaves OFFBOARD it "
              "never requests again."
              % (args.min_goal_z, args.max_goal_z, args.climb_rate), flush=True)

    def _on_signal(signum, _frame):
        stop["now"] = True
        print("\n[onboard] signal %d; shutting down." % signum, flush=True)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    def retarget_ned(new_goal_ned, cur_pos_ned):
        """The ONE frame adapter: the goal port is NED, the policy is ENU."""
        policy.retarget(swap_ne(new_goal_ned),
                        None if cur_pos_ned is None else swap_ne(cur_pos_ned))

    def alt_ref_for(goal, pos):
        """The altitude --alt-hold holds: the EKF z at the latching instant
        when the goal's own z is not a trustworthy reference (hold/current),
        the goal's z when it is. One function because 7.10 latches it at TWO
        instants -- when the goal is applied, and again when OFFBOARD is
        entered -- and the two must not be allowed to drift apart."""
        return (float(pos[2]) if args.goal_z_mode in ("hold", "current")
                else float(goal[2]))

    rejects = {"n": 0}

    def on_goal_reject(info):
        """--goal-z-mode current refused a SET_GOAL: say so on every channel.

        stdout for the board log, a UDP event for the operator who is watching
        live, and the STATUS snapshot for the one who reconnects afterwards.
        Nothing about the runner's own state changes -- that is the point.
        """
        rejects["n"] += 1
        port.note_reject(info)
        port.send_event("SET_GOAL_REJECTED", reason="goal_z_delta",
                        goal=[info["x"], info["y"], info["z"]],
                        z=info["z"], pos_z=info["pos_z"],
                        delta=info["delta"], max_delta=info["max_delta"],
                        goal_z_mode=info["mode"])
        print("[cmd] SET_GOAL rejected: z %+.2f is %.1f m from the current "
              "altitude %+.2f (world NED down), more than --goal-z-max-delta "
              "%.1f m; the previous goal and the awaiting state are kept."
              % (info["z"], info["delta"], info["pos_z"], info["max_delta"]),
              flush=True)

    t0 = time.time()
    next_tick = t0
    ticks = 0
    # --- doc 15.11.6.5: off unless --profile / --save-frames --------------
    profiler = saver = None
    if profile_csv is not None:
        profiler = SystemProfiler(root=args.profile_root, hz=args.profile_hz,
                                  t0=t0, path=profile_csv)
        pm = profiler.meta()
        print("[profile] %.1f Hz -> %s; sources: %s; %d thermal zone(s), "
              "%d cpufreq polic(ies), %d core(s)"
              % (args.profile_hz, profile_csv,
                 ", ".join("%s=%s" % (k, "yes" if v["available"] else "NO")
                           for k, v in sorted(pm["sources"].items())),
                 len(profiler.thermal), len(profiler.cpufreq), len(profiler.cores)),
              flush=True)
        update_run_meta(run_dir, profile=pm)
        profiler.start()
    if frames_dir is not None:
        saver = FrameSaver(frames_dir, args.save_frames)
        policy.keep_net_input = True
        save_meta = dict(
            every=args.save_frames, dir=frames_dir, per_file=FRAMES_PER_FILE,
            max_queue=FRAME_QUEUE_MAX,
            policy_input=("float32 [1,224,224,3] in [0,1] (net_frame_of)"
                          if getattr(policy, "takes_raw_frames", False)
                          else "uint8 [224,224,3] (the encoder quantises it)"))
        update_run_meta(run_dir, save_frames=save_meta)
    try:
        while not stop["now"]:
            tick_t0 = time.perf_counter()
            now = time.time()
            elapsed = now - t0
            if args.duration > 0.0 and elapsed >= args.duration:
                print("[onboard] --duration %g s elapsed." % args.duration,
                      flush=True)
                break

            # --- camera: drain whatever is queued, keep the newest ---------
            # Convert ONLY the newest payload. Converting every drained frame
            # (NV12->RGB is ~70 ms of numpy at 1024x768) burned a full core at
            # the camera's native ~26 fps and starved the tick loop to ~1 Hz.
            newest = None
            got = source.read(timeout=0.0)
            while got is not None:
                newest = got
                frames_in += 1
                got = source.read(timeout=0.0)
            if newest is not None:
                meta, payload = newest
                if not cam_meta_seen:
                    # Audit defect 7: the pipeline's resolution, format and
                    # stride were never printed or checked against anything.
                    # One line, on the first frame, so a camera reconfigured
                    # between flights is visible in the log instead of only in
                    # the pictures.
                    cam_meta_seen = True
                    fmt_id = int(meta["format"])
                    fmt_name = FORMAT_NAMES.get(fmt_id, "unknown(%d)" % fmt_id)
                    print("[cam] first frame: %dx%d format=%s stride=%d "
                          "size=%d fps meta=%s"
                          % (int(meta["width"]), int(meta["height"]), fmt_name,
                             int(meta["stride"]), int(meta["size_bytes"]),
                             meta.get("framerate")), flush=True)
                    port.config["cam_width"] = int(meta["width"])
                    port.config["cam_height"] = int(meta["height"])
                    port.config["cam_format"] = fmt_name
                rgb = (RawFrame(meta, payload)
                       if getattr(policy, "takes_raw_frames", False)
                       else payload_to_net_frame(payload, meta))
                rgb_t = time.time()
            if getattr(source, "ended", False):
                print("[onboard] frame source ended.", flush=True)
                break
            frame_age = None if rgb_t is None else (now - rgb_t)

            frame_stale = (args.frame_stale_hold_s >= 0.0
                           and (frame_age is None or frame_age > RGB_STALE_S))
            if frame_stale:
                if frame_stale_since is None:
                    frame_stale_since = now
                if (not frame_hold
                        and now - frame_stale_since >= args.frame_stale_hold_s):
                    frame_hold = True
                    frame_stale_episodes += 1
                    print("[cam] frames stale (%s); holding (episode %d)."
                          % ("no frame yet" if frame_age is None
                             else "%.2fs" % frame_age, frame_stale_episodes),
                          flush=True)
            else:
                if frame_hold:
                    print("[cam] fresh frames again after %.2fs stale; resuming."
                          % (now - frame_stale_since), flush=True)
                frame_stale_since = None
                frame_hold = False

            # --- state: drain the MAVLink pipe, keep the newest sample -----
            # Same discipline as the camera above -- consume everything queued
            # and keep only the last -- because a 50 Hz telemetry stream against
            # a 15 Hz loop would otherwise hand the policy a pose the vehicle
            # left three ticks ago.
            if state_src is not None:
                sample = state_src.drain()
                if sample is not None:
                    if state_last is None:
                        print(first_sample_line(
                            sample, state_src.saw_estimator_status), flush=True)
                    state_last = sample
                    # The arrival clock of the OLDER of the two streams: a
                    # stalled attitude must not ride along on a fresh position.
                    state_last_t = sample["t_arrival"]
                # Staleness is measured against THIS process's clock, not the
                # PX4 timestamp inside the packet: the producer copies
                # VehicleLocalPosition.timestamp verbatim (2d653e5) and the two
                # clocks share no epoch, so only arrival time is comparable.
                state_age = (None if state_last_t is None
                             else max(0.0, time.time() - state_last_t))
                # A hold cause is a KEY plus a detail: the key is what makes
                # an episode (so a growing staleness is one episode, not one
                # per tick), the detail is what gets printed once for it.
                if state_last is None:
                    reason, detail = "no-sample", "no EKF2 sample yet"
                elif not state_flags_ok(state_last["flags"]):
                    reason = "flags"
                    # The raw PX4 word too: which ESTIMATOR_STATUS_FLAGS bits
                    # the autopilot itself is clearing is the actionable fact.
                    detail = ("flags 0x%02x lack %s (PX4 ESTIMATOR_STATUS.flags"
                              "=0x%04x)"
                              % (state_last["flags"],
                                 missing_state_flags(state_last["flags"]),
                                 -1 if state_src.est_raw is None
                                 else state_src.est_raw))
                elif state_age > args.state_stale_s:
                    reason = "stale"
                    detail = ("EKF2 state stale by %.2f s (> --state-stale-s "
                              "%.2f)" % (state_age, args.state_stale_s))
                else:
                    reason, detail = None, None

                if reason is not None:
                    state_hold = True
                    if reason != state_hold_reason:
                        # A DIFFERENT cause is a new episode and gets its own
                        # line: "stale" following "flags lack attitude_valid"
                        # is two things going wrong, not one.
                        state_hold_reason = reason
                        state_hold_episodes += 1
                        print("[state] %s; holding at zero velocity (episode "
                              "%d)." % (detail, state_hold_episodes), flush=True)
                else:
                    if state_hold:
                        print("[state] valid and fresh again (age %.3f s, "
                              "seq=%d); resuming."
                              % (state_age, state_last["seq"]), flush=True)
                    state_hold = False
                    state_hold_reason = None
                    # THE hand-over: everything the loop used to dead-reckon is
                    # now read off the EKF2 estimate instead.
                    pos_ned = np.array(state_last["pos_ned"], dtype=np.float64)
                    vel_ned = np.array(state_last["vel_ned"], dtype=np.float64)
                    R_enu_meas = R_enu_from_ned_frd_quat(state_last["q_ned_frd"])
                    yaw_ned = yaw_ned_from_R_enu(R_enu_meas)
                    # ATTITUDE_QUATERNION gives the rates in body FRD; the
                    # policy's 21-vector wants body FLU, which is the same
                    # diag(1,-1,-1) that R_enu_from_ned_frd_quat applies to the
                    # attitude itself.
                    if want_rates:
                        p_frd, q_frd, r_frd = state_last["rates_frd"]
                        ang_flu = np.array([p_frd, -q_frd, -r_frd],
                                           dtype=np.float64)
                    jump = None
                    if resetwatch is not None and sample is not None:
                        jump = resetwatch.update(sample.get("reset_counter"), pos_ned)
                        if (not odom_warned and not state_src.saw_odometry
                                and state_src.received > 500):
                            odom_warned = True
                            print("[reset] WARNING: no autopilot ODOMETRY on the state "
                                  "pipe -- EKF2 resets are NOT detected.", flush=True)
                    if jump is not None:
                        print("[reset] EKF2 RESET #%d (%s): position %s m, heading "
                              "%+.2f deg%s%s"
                              % (resetwatch.resets, ",".join(jump["kinds"]) or "?",
                                 np.round(jump["dp"], 3), math.degrees(jump["dyaw"]),
                                 "" if jump["exact"] else
                                 " -- NOT EXACT (%s)" % jump["why"],
                                 "" if args.reset_action == "hold"
                                 else " (--reset-action log: continuing)"), flush=True)
                        port.send_event("ESTIMATOR_RESET", kinds=jump["kinds"],
                                        dp=[round(float(v), 4) for v in jump["dp"]],
                                        dyaw_deg=round(math.degrees(jump["dyaw"]), 3),
                                        exact=bool(jump["exact"]),
                                        action=args.reset_action)
                        if args.reset_action == "hold":
                            # The frame moved, not the aircraft: keep every
                            # latched point on the same physical spot.
                            if goal_ned is not None:
                                goal_ned = reanchor_ned(goal_ned, jump)
                                port.config["goal_ekf_ned"] = [round(float(v), 4) for v in goal_ned]
                            if tframe is not None:
                                tframe.reanchor(jump)
                            if auto is not None:
                                auto.reanchor(jump)
                            if alt_ref_d is not None:
                                alt_ref_d += float(jump["p_new"][2] - jump["p_pred"][2])
                            policy.disengage()
                            align_active = False
                            align_t0 = None
                            resume_needs_engage = True
                            port.force_hold()
                            print("[reset] HOLD: goal re-anchored to EKF NED %s; the "
                                  "policy episode is over. Send RESUME to ALIGN and "
                                  "start a fresh one (the pilot can take over any "
                                  "time)." % (None if goal_ned is None
                                              else goal_ned.round(2)), flush=True)
                    if not state_have_pos:
                        state_have_pos = True
                        # "goal without z" means "stay at the altitude the EKF2
                        # says we are at now", not at the CLI placeholder.
                        cruise_d = float(pos_ned[2])

            # --- is PX4 actually in OFFBOARD? (7.10) -----------------------
            # The same pipe, one more message. Everything below keys off
            # `engage_hold`: while it is set the runner is a setpoint STREAM
            # and nothing else, which is exactly what PX4 needs to be willing
            # to enter OFFBOARD in the first place.
            engage_hold = False
            offboard_csv = ""
            if engage_gate and state_src is not None:
                offb, hb_armed, hb_main, hb_age = state_src.offboard_state(now)
                armed_now = bool(hb_armed) and offb is not None
                main_now = hb_main if offb is not None else None
                if state_src.saw_heartbeat and not hb_first_logged:
                    # The counterpart of "[state] first sample": if this line
                    # never appears, the autopilot's HEARTBEAT is not on this
                    # pipe and the gate would hold the policy off for ever.
                    hb_first_logged = True
                    print("[engage] first autopilot HEARTBEAT (compid %d): "
                          "PX4 main mode %d, armed=%s (OFFBOARD is main mode "
                          "%d)" % (MAV_COMP_ID_AUTOPILOT1, hb_main, hb_armed,
                                   PX4_CUSTOM_MAIN_MODE_OFFBOARD), flush=True)
                offboard_csv = "" if offb is None else ("1" if offb else "0")
                offboard_now = bool(offb)      # unknown or stale -> NOT offboard
                if offboard_now != offboard_active:
                    offboard_active = offboard_now
                    if offboard_now:
                        # Latched below, once there is a goal and a pose to
                        # latch it against -- not here, because OFFBOARD can be
                        # entered before either exists.
                        engage_pending = True
                        print("[engage] PX4 IS IN OFFBOARD (armed=%s, "
                              "HEARTBEAT %.2f s old): the policy takes the "
                              "aircraft." % (hb_armed, hb_age), flush=True)
                    else:
                        # The pilot took it back (or the link went quiet).
                        # Everything the hand-over latched is dropped; the
                        # next OFFBOARD entry re-arms it from scratch.
                        engage_pending = False
                        align_active = False
                        align_t0 = None
                        policy.disengage()      # doc 12 s7; no-op for agile
                        print("[engage] PX4 IS NOT IN OFFBOARD (%s): back to "
                              "zero velocity at the measured heading."
                              % ("no autopilot HEARTBEAT yet" if hb_main is None
                                 else "main mode %d, HEARTBEAT %.2f s old"
                                      % (hb_main, hb_age)), flush=True)
                engage_hold = not offboard_active

            # --- task commands, applied between ticks ----------------------
            if state_src is not None and not state_have_pos:
                # Hold queued SET_GOALs until the first valid EKF2 sample. The
                # policy latches its straight reference line from the position
                # handed to retarget(); latching it on the --start placeholder
                # would aim the whole leg from a pose nobody measured.
                # take_goal() is simply not called, so the command stays queued
                # and lands on the first tick that has real state.
                goal_changed = False
            elif args.goal_frame == "takeoff-flu":
                goal_changed = False
                raw = port.take_goal_raw()
                if raw is not None:
                    x, y, z = raw
                    why = None
                    if z is None or not (args.min_goal_z <= float(z) <= args.max_goal_z):
                        why = ("takeoff-flu goal needs z = height above the "
                               "takeoff point in [%.1f, %.1f] m, got %s"
                               % (args.min_goal_z, args.max_goal_z, z))
                    elif math.hypot(float(x), float(y)) <= args.goal_radius:
                        why = ("takeoff-flu goal is %.2f m from the takeoff "
                               "point, inside --goal-radius %.2f m"
                               % (math.hypot(float(x), float(y)), args.goal_radius))
                    if why is not None:
                        port.send_event("SET_GOAL_REJECTED", reason=why)
                        print("[cmd] SET_GOAL rejected: %s" % why, flush=True)
                    else:
                        pending_flu = (float(x), float(y), float(z))
                        print("[cmd] SET_GOAL takeoff-flu forward %.2f, left "
                              "%.2f, up %.2f m%s" % (pending_flu + (
                                  "" if tframe is not None else
                                  "; held until TAKEOFF latches the frame",)),
                              flush=True)
                if pending_flu is not None and tframe is not None:
                    goal_ned = tframe.to_ned(*pending_flu)
                    print("[goal] takeoff-flu (%.2f, %.2f, %.2f) -> EKF NED "
                          "%s; %s" % (pending_flu + (goal_ned.round(3),
                                                     tframe.describe())),
                          flush=True)
                    port.config["goal_takeoff_flu"] = list(pending_flu)
                    port.config["goal_ekf_ned"] = [round(float(v), 4) for v in goal_ned]
                    pending_flu = None
                    retarget_ned(goal_ned, pos_ned)
                    goal_changed = True
            else:
                goal_ned, goal_changed = apply_goal_command_z(
                    port, goal_ned, pos_ned, retarget=retarget_ned,
                    cruise_d=cruise_d, mode=args.goal_z_mode,
                    max_delta=args.goal_z_max_delta, on_reject=on_goal_reject)
            if goal_changed:
                was_awaiting, awaiting_goal = awaiting_goal, False
                reached = False
                legs_flown += 1
                d = float(np.linalg.norm((goal_ned - pos_ned)[:2]))
                print("[cmd] SET_GOAL -> NED %s (seq=%d); %.2f m to run; plan "
                      "not restarted%s."
                      % (goal_ned.round(2), port.seq, d,
                         # Audit defect 9: `awaiting_goal` is also True before
                         # the FIRST goal has ever arrived, and saying
                         # "resuming" there described a hold that never
                         # happened. Only a leg that followed another one can
                         # resume.
                         "; resuming from the post-arrival hold"
                         if (was_awaiting and legs_flown > 1) else ""),
                      flush=True)
                alt_ref_d = alt_ref_for(goal_ned, pos_ned)
                # doc 12 s7: a new leg is a hand-over for a policy that keeps
                # state across ticks. No-op for agile, whose reference line
                # apply_goal_command_z() has already re-latched.
                policy.engage(swap_ne(pos_ned), R_enu_meas, swap_ne(goal_ned))
                if args.align_first:
                    align_active = True
                    align_t0 = None

            # --- doc 15.11.6.4: auto takeoff --------------------------------
            if auto is not None:
                if port.take_takeoff():
                    state_ok = not state_hold and state_have_pos
                    goal_flu = pending_flu
                    cand = (TakeoffFrame(pos_ned, yaw_ned)
                            if (state_ok and goal_flu is not None) else None)
                    target_d = None if cand is None else cand.to_ned(*goal_flu)[2]
                    ok, why = auto.request(now, dict(
                        state_valid=state_ok,
                        camera_fresh=(not frame_hold and rgb is not None),
                        goal_set=goal_flu is not None,
                        autopilot_heartbeat=bool(state_src.saw_heartbeat),
                        on_ground_disarmed=not armed_now,
                        not_in_offboard=not offboard_active), yaw_ned,
                        target_d if target_d is not None else 0.0,
                        main_mode=main_now)
                    if ok:
                        # THE frame latch: the goal is converted on the next
                        # tick's goal block, with exactly this frame.
                        tframe = cand
                        port.config["takeoff_frame"] = dict(
                            origin_ekf_ned=[round(float(v), 4) for v in tframe.p0],
                            heading_ekf_ned_deg=round(math.degrees(tframe.yaw), 3))
                        print("[takeoff] %s" % tframe.describe(), flush=True)
                    port.send_event("TAKEOFF_ACCEPTED" if ok else
                                    "TAKEOFF_REJECTED", reason=why)
                    print("[takeoff] TAKEOFF %s: %s"
                          % ("ACCEPTED" if ok else "REJECTED", why), flush=True)
                tk = auto.step(now, armed_now, offboard_active, pos_ned, vel_ned,
                               main_mode=main_now)
                for what in tk["send"]:
                    vel_pub.request(what)
                for line in tk["log"]:
                    print(line, flush=True)
                    port.send_event("TAKEOFF_" + auto.phase, detail=line)

            # --- 7.10: the hand-over latch --------------------------------
            # OFFBOARD has just been entered. Everything the old code latched
            # when the GOAL landed is latched here instead, because in the
            # field the goal lands on the ground and the hand-over happens
            # tens of seconds later, somewhere else, on another heading:
            #   * the reference line restarts at the pose the vehicle actually
            #     has when control changes hands (not the take-off pad);
            #   * --alt-hold's reference is the altitude at that instant;
            #   * --align-first's timer starts now, so its 8 s cannot be spent
            #     on the ground where the aircraft is not allowed to turn.
            # Deferred until there is both a goal and a measured pose to latch
            # against: OFFBOARD can legitimately be entered before either.
            if (engage_pending and goal_ned is not None and not state_hold
                    and (state_src is None or state_have_pos)
                    and tk["allow_latch"]):
                engage_pending = False
                if relatch_goal_z:
                    # --goal-z-mode hold froze the goal's z when the goal was
                    # APPLIED -- on the ground, for a takeoff. SITL feeds the
                    # policy goal z := the climb altitude (15.5 #1); here the
                    # hand-over altitude is exactly that.
                    goal_ned = np.array([goal_ned[0], goal_ned[1],
                                         float(pos_ned[2])])
                retarget_ned(goal_ned, pos_ned)
                # doc 12 s7: THE hand-over instant. A recurrent policy freezes
                # its START frame and clears its hidden state here; agile's
                # engage() is a no-op and the line above is all it needs.
                policy.engage(swap_ne(pos_ned), R_enu_meas, swap_ne(goal_ned))
                alt_ref_d = alt_ref_for(goal_ned, pos_ned)
                if args.align_first:
                    align_active = True
                    align_t0 = None
                print("[engage] hand-over latch at NED %s: reference line "
                      "restarted here, --alt-hold reference %+.2f, ALIGN %s."
                      % (pos_ned.round(2), alt_ref_d,
                         "armed" if args.align_first else "not requested"),
                      flush=True)

            cmd_hold = port.hold
            if cmd_hold != cmd_hold_prev:
                print("[cmd] %s (seq=%d)" % ("HOLD" if cmd_hold else "RESUME",
                                             port.seq), flush=True)
                cmd_hold_prev = cmd_hold
                if not cmd_hold and resume_needs_engage and goal_ned is not None:
                    resume_needs_engage = False
                    retarget_ned(goal_ned, pos_ned)
                    policy.engage(swap_ne(pos_ned), R_enu_meas, swap_ne(goal_ned))
                    if args.align_first:
                        align_active = True
                        align_t0 = None
                    print("[reset] RESUME after the reset: ALIGN, then a fresh "
                          "policy episode.", flush=True)

            # --- the ALIGN hand-over phase (audit defect 4) -----------------
            # The sim turns to the goal before it lets the policy steer
            # (agile_offboard.py:714-741); the board handed over on whatever
            # heading the pilot's thumb happened to leave, which puts
            # `local_goal` far off the +x the checkpoint was trained along.
            # Zero velocity out, yaw only, and the policy engages when the
            # heading is there or the timeout says stop waiting.
            align_hold = False
            align_yaw_sp = None
            align_yaw_rate = 0.0
            if align_active:
                if (goal_ned is None or state_hold or engage_hold
                        or tk["override"]
                        or (state_src is not None and not state_have_pos)):
                    # Wait for state -- and, since 7.10, for OFFBOARD: while
                    # `engage_hold` is set `align_t0` stays None, so the
                    # timer's first tick is the first tick after the pilot's
                    # thumb moved, not the first tick after the goal landed.
                    align_hold = True       # still zero velocity either way
                else:
                    align_first_tick = align_t0 is None
                    if align_first_tick:
                        align_t0 = now
                    bearing = math.atan2(float(goal_ned[1] - pos_ned[1]),
                                         float(goal_ned[0] - pos_ned[0]))
                    err = wrap_pi(bearing - yaw_ned)
                    waited = now - align_t0
                    if align_first_tick:
                        print("[align] goal bearing %+.1f deg, heading "
                              "%+.1f deg (error %+.1f deg): holding at zero "
                              "velocity until within %.1f deg or %.1f s."
                              % (math.degrees(bearing), math.degrees(yaw_ned),
                                 math.degrees(err), args.align_tol_deg,
                                 args.align_timeout_s), flush=True)
                    if abs(err) <= align_tol:
                        align_active = False
                        print("[align] aligned after %.2f s (error %+.1f deg "
                              "<= %.1f deg); policy ENGAGED."
                              % (waited, math.degrees(err),
                                 args.align_tol_deg), flush=True)
                    elif waited >= args.align_timeout_s:
                        align_active = False
                        print("[align] TIMEOUT after %.2f s with error "
                              "%+.1f deg (> %.1f deg); policy ENGAGED anyway."
                              % (waited, math.degrees(err),
                                 args.align_tol_deg), flush=True)
                    else:
                        align_hold = True
                        align_yaw_sp = wrap_pi(bearing)
                        # A proportional rate, not the err/dt relay the audit
                        # measured saturating at a 3.8 deg error.
                        align_yaw_rate = float(np.clip(
                            1.5 * err, -abs(args.yaw_rate_max),
                            abs(args.yaw_rate_max)))

            hold_now = (frame_hold or cmd_hold or awaiting_goal
                        or goal_ned is None or rgb is None or state_hold
                        or align_hold or engage_hold)

            # Overlay telemetry, body FLU, recomputed from scratch each tick.
            # Both stay None unless this tick actually steers -- Daniel's
            # DELETE-on-hold rule (AirStack daniel/diffaero_ground_control,
            # publish_markers): an arrow left on screen through a hold is worse
            # than no arrow, because it is the one failure where the picture
            # lies about what the aircraft is doing. Null is how the viewer is
            # told to draw nothing.
            viz_goal_dir = viz_vel = viz_out = None

            # doc 12 s7.1 rule (2): a hold carries no policy command to
            # dispatch on, so it leaves through the vehicle's default
            # publisher; the policy branch below replaces this with whatever
            # the command it just produced asks for.
            publisher = vel_yaw_pub
            policy_ticked = False               # ticks.csv pol_* (doc 15.11.6.5)

            if tk["override"]:
                # The takeoff owns the setpoint (ARMING: zero; CLIMB: straight
                # up), heading held as an absolute angle. No policy step.
                v_ned = np.asarray(tk["v_ned"], dtype=np.float64)
                if state_hold or cmd_hold:
                    # No trustworthy altitude (or an operator / jump HOLD):
                    # never climb blind. Hover; the climb resumes when the
                    # sample is valid again / on RESUME.
                    v_ned = np.zeros(3)
                yaw_rate_cmd = 0.0
                yaw_sp_ned = wrap_pi(tk["yaw_sp_ned"])
                alpha0, mode_idx = float("nan"), -1
                net_pass = 0
            elif hold_now:
                v_ned = np.zeros(3)
                yaw_rate_cmd = align_yaw_rate if align_hold else 0.0
                # --yaw-out angle: a hold still has to carry SOME yaw, and the
                # sim's hold carries the MEASURED heading (agile_offboard's
                # hold), not 0 -- 0 would command a turn to due north. During
                # ALIGN it carries the bearing we are turning to.
                yaw_sp_ned = (align_yaw_sp if align_yaw_sp is not None
                              else wrap_pi(yaw_ned))
                alpha0, mode_idx = float("nan"), -1
                net_pass = 0
            else:
                if R_enu_meas is not None:
                    # The measured attitude, converted exactly as
                    # px4_offboard.DroneState does it for agile_offboard.
                    R_enu = R_enu_meas
                else:
                    yaw_enu = swap_yaw(yaw_ned)
                    c, s = math.cos(yaw_enu), math.sin(yaw_enu)
                    R_enu = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                if saver is not None:
                    # --save-frames: this tick's number as ticks.csv will write it
                    policy.tick_tag = (ticks + 1, elapsed)
                cmd = policy.compute(
                    swap_ne(pos_ned), swap_ne(vel_ned), R_enu, ang_flu,
                    swap_ne(goal_ned), rgb, imu_override=imu_fixed)
                policy_ticked = True
                # doc 12 s7.1: dispatch on the command's own type. The whole
                # velocity-specific downstream that used to be written out
                # here is VelocityYawPublisher.encode(), unchanged.
                cmd_type = cmd.get("command_type", DEFAULT_COMMAND_TYPE)
                publisher = publishers.get(cmd_type)
                if publisher is None:
                    raise RuntimeError(
                        "the policy emitted command_type %r, for which this "
                        "runner has no CommandPublisher (registered: %s)."
                        % (cmd_type, ", ".join(sorted(publishers))))
                v_ned, yaw_rate_cmd, yaw_sp_ned = publisher.encode(
                    cmd, dict(pos_ned=pos_ned, vel_ned=vel_ned,
                              yaw_ned=yaw_ned, dt=dt, alt_ref_d=alt_ref_d,
                              state_src=state_src, state_last=state_last))
                # `alphas` / `mode_idx` are the agile head's payload, not part
                # of the Policy contract; the CSV and the overlay degrade to
                # the hold branch's placeholders for a policy without them.
                alphas = cmd.get("alphas")
                alpha0 = (float("nan") if alphas is None
                          else float(np.asarray(alphas).ravel()[0]))
                mode_idx = int(cmd.get("mode_idx", -1))
                net_pass = int(bool(cmd.get("net_pass", False)))

                world_points_per_mode = getattr(
                    policy, "_world_points_per_mode", None)
                if plan_log is not None and net_pass and \
                        world_points_per_mode is not None:
                    # `_world_points_per_mode` is ALREADY world ENU (the policy
                    # tail is ENU-native); everything the loop owns is NED and
                    # is swapped here, at the same boundary the setpoints use.
                    plan_log.log(
                        now, elapsed, swap_ne(pos_ned), R_enu,
                        swap_ne(vel_ned),
                        policy._goal_dir(swap_ne(pos_ned), swap_ne(goal_ned)),
                        alphas, world_points_per_mode,
                        policy._sort_order, mode_idx,
                        swap_ne(v_ned), cmd["yaw"])

                # The two vectors the ground viewer projects. R_enu.T maps
                # world ENU into body FLU, which is what the camera overlay
                # needs -- and it is the same product the policy just consumed
                # (core.AgilePolicy._state_to_model_input), so this is a copy,
                # not a second derivation that could disagree with the first.
                # Publishing body frame is deliberate: the ground never does a
                # frame conversion, so it cannot get one wrong.
                goal_off_enu = swap_ne(goal_ned) - swap_ne(pos_ned)
                goal_norm = float(np.linalg.norm(goal_off_enu))
                if goal_norm > 1e-6:
                    viz_goal_dir = R_enu.T @ (goal_off_enu / goal_norm)
                viz_vel = R_enu.T @ swap_ne(v_ned)

                # The three candidate plans, mapped back into the CURRENT body
                # frame: R_enu.T @ (world_pts - pos). Not the body plan as the
                # net emitted it -- the plan is adopted in WORLD frame and only
                # refreshed every net_every ticks, so on an intermediate tick
                # the drone has moved and turned under it. Re-deriving from the
                # world points with this tick's attitude is what makes the
                # overlay agree with the picture instead of with history.
                wpm = world_points_per_mode
                if wpm is not None:
                    rel = np.asarray(wpm, np.float64) - swap_ne(pos_ned)[None, None, :]
                    # (R_enu.T @ p) for a row vector p is (p @ R_enu), which
                    # broadcasts over (M,T,3) without einsum -- numpy 1.13 on
                    # the board, so the plainest op that works is the right one.
                    modes_body = rel @ R_enu
                    viz_out = {
                        "modes": [[[round(float(c), 3) for c in p]
                                   for p in mode] for mode in modes_body],
                        "mode_idx": mode_idx,
                        "alphas": [round(float(a), 5) for a in
                                   np.asarray(alphas).ravel()],
                        "vel": [round(float(c), 4) for c in viz_vel],
                        "yaw_rate": round(float(yaw_rate_cmd), 4),
                        "enc_ms": round(float(cmd.get("enc_ms",
                                                      float("nan"))), 2),
                        "head_ms": round(float(cmd.get("head_ms",
                                                       float("nan"))), 2),
                    }

            sent_seq = -1
            if args.silent_until_tasked and legs_flown == 0:
                if ticks == 0:
                    print("[onboard] --silent-until-tasked: no setpoints until "
                          "the first SET_GOAL.", flush=True)
            else:
                sent_seq = publisher.send(float(v_ned[0]), float(v_ned[1]),
                                          float(v_ned[2]),
                                          float(yaw_rate_cmd),
                                          yaw=float(yaw_sp_ned))

            # --- dead reckoning (synthetic mode ONLY) ----------------------
            # In --state mpa the pose is whatever the next EKF2 sample says;
            # integrating the command on top of it would double-count.
            # --freeze-pose: the vehicle is on a bench or in someone's hands
            # and is NOT moving, so integrating the command would be fiction
            # -- and fiction that ends the run, because the fake pose reaches
            # the goal and the policy holds. Keep pos/yaw where they started;
            # the goal then stays at a fixed bearing and the net keeps planning
            # against the live camera indefinitely (handheld_viz's model).
            if state_src is None and not args.freeze_pose:
                pos_ned = pos_ned + v_ned * dt
                vel_ned = v_ned
                yaw_ned = math.atan2(math.sin(yaw_ned + yaw_rate_cmd * dt),
                                     math.cos(yaw_ned + yaw_rate_cmd * dt))
            ticks += 1

            dist_xy = (None if goal_ned is None
                       else float(np.linalg.norm((goal_ned - pos_ned)[:2])))

            if (goal_ned is not None and dist_xy < args.goal_radius
                    and not awaiting_goal and not engage_hold
                    and not tk["override"]):
                # `not engage_hold` (7.10): before the hand-over the runner is
                # not flying the aircraft, so it cannot have arrived anywhere.
                reached = True
                awaiting_goal = True
                port.send_reached(goal_ned, pos_ned, dist_xy, phase="POLICY")
                print("\n[onboard] >>> REACHED NED %s at pos=%s (d=%.2f m); "
                      "holding in POLICY for the next SET_GOAL <<<\n"
                      % (goal_ned.round(2), pos_ned.round(2), dist_xy),
                      flush=True)

            # ARMED_WAIT (7.10) outranks ALIGN: while PX4 is not in OFFBOARD
            # the ALIGN phase has not started yet, and a HUD saying ALIGN
            # there would be describing a turn that is not happening.
            port.publish_status(("TAKEOFF_" + tk["phase"]) if tk["override"] else
                                "ARMED_WAIT" if engage_hold else
                                ("ALIGN" if align_hold else "POLICY"),
                                dist_to_goal=dist_xy, reached=reached,
                                awaiting_goal=awaiting_goal, goal=goal_ned,
                                goal_dir_body=viz_goal_dir, vel_body=viz_vel,
                                viz=viz_out,
                                hold_reason=("not-offboard" if engage_hold else
                                             ("align" if align_hold else
                                             (state_hold_reason if state_hold
                                             else ("frame" if frame_hold else
                                                   ("cmd" if cmd_hold else None))))))

            tick_ms = (time.perf_counter() - tick_t0) * 1e3
            # The ACTUAL wall gap between ticks. `tick_ms` is how long the tick
            # took; this is how long since the last one started, which is the
            # number the audit's defect 3 is about (4.7-11.7 Hz measured
            # against a loop that differentiates yaw against a nominal 1/15).
            tick_dt = None if tick_prev_t is None else (now - tick_prev_t)
            tick_prev_t = now
            if csv is not None:
                csv.write(
                    "%.4f,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.2f,"
                    "%.5f,%.5f,%.5f,%.5f,%.5f,%s,%s,%s,%s,"
                    "%.3f,%.3f,%.3f,%.3f,%d,%.4f,%d,%s"
                    % (elapsed, ticks, sent_seq, port.seq, int(hold_now),
                       int(awaiting_goal), int(reached),
                       pos_ned[0], pos_ned[1], pos_ned[2], math.degrees(yaw_ned),
                       v_ned[0], v_ned[1], v_ned[2], yaw_rate_cmd,
                       math.hypot(v_ned[0], v_ned[1]),
                       "" if goal_ned is None else "%.4f" % goal_ned[0],
                       "" if goal_ned is None else "%.4f" % goal_ned[1],
                       "" if goal_ned is None else "%.4f" % goal_ned[2],
                       "" if dist_xy is None else "%.4f" % dist_xy,
                       policy.enc_ms if net_pass else float("nan"),
                       policy.head_ms if net_pass else float("nan"),
                       policy.tail_ms, tick_ms, net_pass, alpha0, mode_idx,
                       "" if frame_age is None else "%.3f" % frame_age)
                    + ("" if state_src is None else
                       ",%s,%s" % ("" if state_age is None
                                   else "%.3f" % state_age,
                                   "" if state_last is None
                                   else "0x%02x" % state_last["flags"]))
                    + ",%.6f,%s" % (math.degrees(yaw_sp_ned),
                                    "" if tick_dt is None else "%.5f" % tick_dt)
                    + ",%s" % offboard_csv
                    + tick_extra_fields(policy, policy_ticked, state_last)
                    + "\n")
                csv.flush()
            if saver is not None and net_pass and policy_ticked:
                save_frame_record(saver, policy, ticks, elapsed, t0, v_ned, yaw_sp_ned)

            if args.verbose or (int(elapsed) != int(elapsed - dt)):
                print("[onboard t=%6.2fs tick=%d] pos=%s v=%s |v_xy|=%.2f "
                      "yaw=%6.1fdeg d=%s hold=%s enc=%.1fms head=%.2fms "
                      "age=%s"
                      % (elapsed, ticks, pos_ned.round(2), v_ned.round(2),
                         math.hypot(v_ned[0], v_ned[1]), math.degrees(yaw_ned),
                         "--" if dist_xy is None else "%.2f" % dist_xy,
                         hold_now, policy.enc_ms, policy.head_ms,
                         "--" if frame_age is None else "%.3f" % frame_age),
                      flush=True)

            next_tick += dt
            sleep = next_tick - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.time()
    finally:
        el = time.time() - t0
        print("[onboard] %d ticks in %.1f s (%.1f Hz), %d net passes (%.1f Hz), "
              "%d frames in, %d goal(s) accepted, final pos NED = %s"
              % (ticks, el, ticks / el if el > 0 else 0, policy.passes,
                 policy.passes / el if el > 0 else 0, frames_in, legs_flown,
                 pos_ned.round(2)), flush=True)
        print("[onboard] setpoints: %d %s, %d send errors; goals: %d "
              "commands, %d rejected, %d refused on goal-z; stale episodes: "
              "%d; speed-cap clips: %d"
              % (vel_pub.sent, "PRINTED (none sent)" if args.enable_test
                 else "sent", vel_pub.send_errors, port.commands, port.errors,
                 rejects["n"], frame_stale_episodes,
                 vel_yaw_pub.cap_violations),
              flush=True)
        if state_src is not None:
            print("[onboard] state pipe: %d records decoded (%d used, %d "
                  "undecodable), %d resync(es), %d superseded within a tick, "
                  "%d hold episode(s)"
                  % (state_src.received, state_src.used, state_src.rejected,
                     state_src.gaps, state_src.dropped, state_hold_episodes),
                  flush=True)
        if profiler is not None:
            profiler.stop()
            print("[profile] %d row(s) -> %s%s"
                  % (profiler.rows, profiler.path,
                     "" if not profiler.errors else
                     "; sources that failed: %s" % ", ".join(sorted(profiler.errors))),
                  flush=True)
            update_run_meta(run_dir, profile=dict(profiler.meta(), rows=profiler.rows,
                                                  errors=profiler.errors))
        if saver is not None:
            fs = saver.close()
            print("[frames] %d network pass(es), %d selected (every %d): %d saved in "
                  "%d file(s), %d DROPPED (queue full), %d lost to %d write error(s)%s"
                  " -> %s" % (fs["net_passes"], fs["selected"], fs["every"], fs["saved"],
                              fs["files"], fs["dropped"], fs["lost_in_write_errors"],
                              fs["write_errors"],
                              "" if not fs["pending"] else
                              ", %d still queued at exit" % fs["pending"], fs["dir"]),
                  flush=True)
            update_run_meta(run_dir, save_frames=dict(save_meta, **fs))
        if csv is not None:
            csv.close()
        if plan_log is not None:
            plan_log.close()
            print("[onboard] wrote %d network-pass plan rows to %s"
                  % (plan_log.rows, plan_log.path), flush=True)
        if hasattr(policy, "close"):
            policy.close()
        vel_pub.close()
        port.close()
        source.close()
        if state_src is not None:
            state_src.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
