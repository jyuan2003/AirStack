"""Nominal high-level policies and conflict scenarios for the CBF swarm.

Ported from ~/drone_soccer (drone_soccer/scenarios.py) and adapted for the
ground controller: scenarios receive plain ``positions`` arrays instead of a
MuJoCo state, and ``initial_positions()`` doubles as the per-drone takeoff
target. The CBF filter projects every nominal velocity onto the safe set, so
a scenario's job is purely to generate interesting conflicts.

Scenarios:

- ``hover``: hold the configured hover positions (the original demo).
- ``random_walk``: each drone holds a fixed-speed velocity and bounces off
  the arena walls -- no goals.
- ``random_goals``: each drone seeks a random goal point and picks a new one
  on arrival.
- ``head_on``: two facing groups fly to swapped goals, re-swapping on arrival
  for perpetual head-on conflicts at the center.
- ``antipodal``: drones on a sphere fly to their antipodes, all crossing the
  center at once.
- ``squeeze``: 3-drone CBF demo -- two "holder" drones goal-track posts
  separated by ``gap_factor * safety_radius`` while the third drone flies
  straight through the gap between them; the holders must yield and return.

- ``rgb_policy``: ONE drone flown by a superfly RGB navigation policy (agile
  or dnav), run on the ground from the copied onboard runner
  (``onboard_policy_runner.py``); frames arrive over UDP from the drone.
  Emits velocity AND a heading target (``yaw_targets``).

Any drone listed in the commander's ``teleop_drones`` has its scenario row
replaced by operator input, so e.g. the squeeze intruder can be hand-flown.
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned arena box the swarm is kept inside.

    Attributes:
        low: shape (3,) minimum corner (m); ``low[2]`` is the floor clearance.
        high: shape (3,) maximum corner (m).
    """

    low: np.ndarray
    high: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.low + self.high)

    @property
    def size(self) -> np.ndarray:
        return self.high - self.low


def seek_velocity(
    positions: np.ndarray,
    goals: np.ndarray,
    max_speed: float,
    approach_gain: float = 2.0,
) -> np.ndarray:
    """Proportional go-to-goal velocity, capped at ``max_speed``.

    Flies straight at each goal at ``max_speed`` when far and eases off within
    ``max_speed / approach_gain`` meters so the drone settles instead of
    overshooting.
    """
    to_goal = goals - positions
    distance = np.linalg.norm(to_goal, axis=-1, keepdims=True)
    speed = np.minimum(max_speed, approach_gain * distance)
    direction = to_goal / np.maximum(distance, 1e-9)
    return direction * speed


def wrap_pi(angle: float) -> float:
    """Wrap an angle into [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion_xyzw(q) -> float:
    """ENU heading of body-x for a ROS (x, y, z, w) FLU->ENU quaternion.

    Same quantity as ``atan2(R[1, 0], R[0, 0])`` of the rotation matrix.
    """
    x, y, z, w = (float(c) for c in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_rate_toward_target(yaw_target: float, yaw_measured: float,
                           kp: float, max_rate: float) -> float:
    """Heading P-loop: ENU yaw rate (CCW+) = clip(kp * wrap(target - meas))."""
    rate = float(kp) * wrap_pi(float(yaw_target) - float(yaw_measured))
    return float(np.clip(rate, -abs(max_rate), abs(max_rate)))


class Scenario(ABC):
    """A takeoff layout plus a nominal go-where policy for the swarm."""

    # True if nominal_velocity() wants velocities / orientations / body rates
    # in addition to positions. The commander passes them only then, so the
    # position-only scenarios keep their one-argument signature.
    needs_full_state = False

    def __init__(
        self,
        num_drones: int,
        bounds: Bounds,
        nominal_speed: float,
        rng: np.random.Generator,
        safety_radius: float,
    ) -> None:
        self.num_drones = int(num_drones)
        self.bounds = bounds
        self.nominal_speed = float(nominal_speed)
        self.rng = rng
        self.safety_radius = float(safety_radius)

    @abstractmethod
    def initial_positions(self) -> np.ndarray:
        """Return shape (N, 3) takeoff targets (non-overlapping, in bounds)."""
        ...

    @abstractmethod
    def nominal_velocity(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        orientations: Optional[list] = None,
        body_rates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return shape (N, 3) nominal (pre-CBF) velocities for this state.

        Args:
            positions: (N, 3) world ENU positions.
            velocities: (N, 3) world ENU velocities (``needs_full_state``
                scenarios only).
            orientations: per drone, a ROS (x, y, z, w) FLU->ENU quaternion,
                or None when that drone has no fresh attitude
                (``needs_full_state`` scenarios only).
            body_rates: (N, 3) body FLU angular rates, rad/s
                (``needs_full_state`` scenarios only).
        """
        ...

    @property
    def goals(self) -> Optional[np.ndarray]:
        """Current per-drone goal points (N, 3) for debugging, or None."""
        return None

    @property
    def yaw_targets(self) -> Optional[list]:
        """Per-drone ENU heading targets (rad), None entries (or None for the
        whole swarm) where the scenario has no opinion. The commander turns a
        target into a yaw rate; no target = zero yaw rate."""
        return None

    @property
    def cbf_exempt_indices(self) -> list:
        """Drone indices whose commands bypass the CBF while the scenario
        runs (deliberate obstacles — everyone else dodges them)."""
        return []

    def _random_positions(self, min_separation: float) -> np.ndarray:
        """Rejection-sample N takeoff positions ``min_separation`` apart."""
        margin = 0.1 * self.bounds.size
        low = self.bounds.low + margin
        high = self.bounds.high - margin
        positions = np.zeros((self.num_drones, 3))
        for index in range(self.num_drones):
            for _attempt in range(4000):
                candidate = self.rng.uniform(low, high)
                if index == 0 or np.all(
                    np.linalg.norm(positions[:index] - candidate, axis=-1)
                    >= min_separation
                ):
                    positions[index] = candidate
                    break
            else:
                positions[index] = candidate
        return positions


class HoverScenario(Scenario):
    """Hold fixed hover positions (the original N-1 hover + teleop demo)."""

    def __init__(self, *args, hover_positions: np.ndarray, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._targets = np.asarray(hover_positions, dtype=float).reshape(-1, 3)
        if self._targets.shape[0] != self.num_drones:
            raise ValueError(
                f'hover scenario needs {self.num_drones} hover positions, '
                f'got {self._targets.shape[0]}')

    def initial_positions(self) -> np.ndarray:
        return self._targets.copy()

    def nominal_velocity(self, positions: np.ndarray) -> np.ndarray:
        return seek_velocity(positions, self._targets, self.nominal_speed)

    @property
    def goals(self) -> Optional[np.ndarray]:
        return self._targets


class RandomWalkScenario(Scenario):
    """Fixed-speed drift with billiard reflection off the arena walls."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        directions = self.rng.normal(size=(self.num_drones, 3))
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        self._velocities = directions * self.nominal_speed

    def initial_positions(self) -> np.ndarray:
        return self._random_positions(min_separation=2.4 * self.safety_radius)

    def nominal_velocity(self, positions: np.ndarray) -> np.ndarray:
        # Reflect any drone that has reached a wall and is still heading out.
        beyond_low = (positions <= self.bounds.low) & (self._velocities < 0.0)
        beyond_high = (positions >= self.bounds.high) & (self._velocities > 0.0)
        self._velocities[beyond_low | beyond_high] *= -1.0
        return self._velocities.copy()


class RandomGoalsScenario(Scenario):
    """Each drone seeks a random goal and resamples a new one on arrival."""

    _ARRIVAL_RADIUS_M = 0.3

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._positions = self._random_positions(
            min_separation=2.4 * self.safety_radius)
        self._goals = self._sample_goals()

    def _sample_goals(self) -> np.ndarray:
        margin = 0.1 * self.bounds.size
        return self.rng.uniform(
            self.bounds.low + margin,
            self.bounds.high - margin,
            size=(self.num_drones, 3),
        )

    def initial_positions(self) -> np.ndarray:
        return self._positions.copy()

    def nominal_velocity(self, positions: np.ndarray) -> np.ndarray:
        reached = (
            np.linalg.norm(positions - self._goals, axis=-1)
            < self._ARRIVAL_RADIUS_M
        )
        if np.any(reached):
            fresh = self._sample_goals()
            self._goals[reached] = fresh[reached]
        return seek_velocity(positions, self._goals, self.nominal_speed)

    @property
    def goals(self) -> Optional[np.ndarray]:
        return self._goals


class HeadOnScenario(Scenario):
    """Two facing rows swap sides, re-swapping on arrival for repeated conflict.

    Drones split into a left group (negative x) and a right group (positive
    x), each stacked along y and z. A small per-drone y offset breaks the
    perfectly-symmetric head-on that would otherwise deadlock the CBF.
    """

    _ARRIVAL_RADIUS_M = 0.4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        left, right = self._build_rows()
        self._positions = np.concatenate([left, right], axis=0)[: self.num_drones]
        # Each drone's goal is its own slot mirrored across x (the far side).
        self._goals = self._positions.copy()
        self._goals[:, 0] *= -1.0

    def _build_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """Lay each group on a y-z grid against its x wall."""
        left_count = (self.num_drones + 1) // 2
        right_count = self.num_drones // 2
        group_size = max(left_count, right_count, 1)

        columns = int(np.ceil(np.sqrt(group_size)))
        layers = int(np.ceil(group_size / columns))
        y_values = np.linspace(
            self.bounds.low[1] + 0.3, self.bounds.high[1] - 0.3, columns
        )
        if layers == 1:
            z_values = np.array([self.bounds.center[2]])
        else:
            z_values = np.linspace(
                self.bounds.low[2] + 0.5, self.bounds.high[2] - 0.5, layers
            )

        x_left = self.bounds.low[0] + 0.4
        x_right = self.bounds.high[0] - 0.4
        symmetry_break = 0.25

        def grid(count: int, x_wall: float, y_shift: float) -> np.ndarray:
            slots = []
            for k in range(count):
                layer, column = divmod(k, columns)
                slots.append([x_wall, y_values[column] + y_shift, z_values[layer]])
            return np.array(slots)

        left = grid(left_count, x_left, -symmetry_break)
        right = grid(right_count, x_right, symmetry_break)
        return left, right

    def initial_positions(self) -> np.ndarray:
        return self._positions.copy()

    def nominal_velocity(self, positions: np.ndarray) -> np.ndarray:
        reached = (
            np.linalg.norm(positions - self._goals, axis=-1)
            < self._ARRIVAL_RADIUS_M
        )
        # Flip x target on arrival so the groups cross back and forth forever.
        self._goals[reached, 0] *= -1.0
        return seek_velocity(positions, self._goals, self.nominal_speed)

    @property
    def goals(self) -> Optional[np.ndarray]:
        return self._goals


class AntipodalScenario(Scenario):
    """Drones on a sphere fly to their antipodes, all crossing the center."""

    _ARRIVAL_RADIUS_M = 0.4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._center = self.bounds.center
        self._radius = 0.45 * float(np.min(self.bounds.size))
        self._positions = self._fibonacci_sphere()
        self._goals = 2.0 * self._center - self._positions

    def _fibonacci_sphere(self) -> np.ndarray:
        indices = np.arange(self.num_drones) + 0.5
        z = 1.0 - 2.0 * indices / self.num_drones
        radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        theta = golden_angle * indices
        unit = np.stack(
            [radius_xy * np.cos(theta), radius_xy * np.sin(theta), z], axis=-1)
        return self._center + self._radius * unit

    def initial_positions(self) -> np.ndarray:
        return self._positions.copy()

    def nominal_velocity(self, positions: np.ndarray) -> np.ndarray:
        reached = (
            np.linalg.norm(positions - self._goals, axis=-1)
            < self._ARRIVAL_RADIUS_M
        )
        if np.any(reached):
            self._goals[reached] = 2.0 * self._center - self._goals[reached]
        return seek_velocity(positions, self._goals, self.nominal_speed)

    @property
    def goals(self) -> Optional[np.ndarray]:
        return self._goals


class SqueezeScenario(Scenario):
    """3-drone CBF showcase: an intruder squeezes between two holders.

    Drones 0 and 1 ("holders") goal-track two explicitly configured posts.
    Drone 2 (the "intruder") shuttles back and forth between two explicitly
    configured waypoints; place the segment so it passes between the posts.

    The expected behavior: as the intruder approaches, the holders' filtered
    velocities push them apart (their nominal keeps pulling them back to
    their posts), the intruder passes through, and the holders settle back
    onto their posts. For the gap to be impassable without yielding, the
    intruder's path must come closer than ``2 * safety_radius`` to a post.

    The intruder is CBF-EXEMPT by default (``intruder_cbf_exempt``): it is
    the deliberate obstacle, and filtering it makes the filter push it
    *backwards* as it approaches the gap (the pair-constraint gradient
    points away from the holders), so it stalls or retreats instead of
    squeezing through. Exempt, it presses on and the holders alone yield.

    To hand-fly the intruder instead, list it in ``teleop_drones`` -- its
    scenario row is then ignored in favor of operator input.
    """

    def __init__(self, *args, holder_positions: np.ndarray,
                 intruder_waypoints: np.ndarray,
                 intruder_cbf_exempt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._intruder_cbf_exempt = bool(intruder_cbf_exempt)
        if self.num_drones != 3:
            raise ValueError(
                f'squeeze scenario requires exactly 3 drones, got {self.num_drones}')
        self._holder_posts = np.asarray(
            holder_positions, dtype=float).reshape(2, 3)
        self._intruder_ends = np.asarray(
            intruder_waypoints, dtype=float).reshape(2, 3)
        post_gap = float(np.linalg.norm(
            self._holder_posts[0] - self._holder_posts[1]))
        if post_gap < 2.0 * self.safety_radius:
            raise ValueError(
                f'holder posts are {post_gap:.2f} m apart, inside their own '
                f'2r keep-out ({2 * self.safety_radius:.2f} m) — the holders '
                'could never both reach their posts')
        self._intruder_goal_index = 1  # start by flying toward waypoint B
        self._arrival_radius = 0.3

    @property
    def holder_posts(self) -> np.ndarray:
        return self._holder_posts.copy()

    @property
    def intruder_waypoints(self) -> np.ndarray:
        return self._intruder_ends.copy()

    @property
    def cbf_exempt_indices(self) -> list:
        return [2] if self._intruder_cbf_exempt else []

    def initial_positions(self) -> np.ndarray:
        return np.vstack([self._holder_posts, self._intruder_ends[0]])

    def nominal_velocity(self, positions: np.ndarray) -> np.ndarray:
        intruder_goal = self._intruder_ends[self._intruder_goal_index]
        if np.linalg.norm(positions[2] - intruder_goal) < self._arrival_radius:
            self._intruder_goal_index = 1 - self._intruder_goal_index
            intruder_goal = self._intruder_ends[self._intruder_goal_index]
        goals = np.vstack([self._holder_posts, intruder_goal])
        return seek_velocity(positions, goals, self.nominal_speed)

    @property
    def goals(self) -> Optional[np.ndarray]:
        return np.vstack(
            [self._holder_posts, self._intruder_ends[self._intruder_goal_index]])


class GoalScenario(Scenario):
    """Each drone seeks a per-drone goal that can be retargeted live.

    Goals default to the configured takeoff layout (``initial_goals``, from
    ``hover_positions``); the commander updates them from
    ``/svg/{name}/goal_command`` while flying, and per-drone speed from
    ``/svg/{name}/speed_command`` (defaults to ``nominal_speed``). All
    goal-tracking drones are CBF-filtered, so commanding two of them at each
    other just makes them dodge. This backs the single- and multi-drone
    tracking tests.
    """

    _APPROACH_GAIN = 1.5

    def __init__(self, *args, initial_goals: np.ndarray, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._goals = np.asarray(
            initial_goals, dtype=float).reshape(self.num_drones, 3)
        self._speeds = np.full(self.num_drones, self.nominal_speed)

    def set_goal(self, index: int, point: np.ndarray) -> None:
        self._goals[index] = np.asarray(point, dtype=float)

    def set_speed(self, index: int, speed: float) -> None:
        self._speeds[index] = max(0.0, float(speed))

    def initial_positions(self) -> np.ndarray:
        return self._goals.copy()

    def nominal_velocity(self, positions: np.ndarray) -> np.ndarray:
        to_goal = self._goals - positions
        distance = np.linalg.norm(to_goal, axis=-1, keepdims=True)
        speed = np.minimum(self._speeds[:, None], self._APPROACH_GAIN * distance)
        direction = to_goal / np.maximum(distance, 1e-9)
        return direction * speed

    @property
    def goals(self) -> Optional[np.ndarray]:
        return self._goals


# Where the gitignored copy of the superfly onboard runner + models lives in
# the robot container (robot/ros_ws/superfly_onboard, see its SOURCES.txt).
DEFAULT_SUPERFLY_DIR = '/root/AirStack/robot/ros_ws/superfly_onboard'


def load_onboard_runner(superfly_dir: str):
    """Import the copied ``onboard_policy_runner`` from ``superfly_dir``.

    The directory also holds the minimal ``superfly`` package tree that
    ``superfly.common.rgb_transport`` needs, so it goes on ``sys.path`` as a
    whole. Refuses a same-named module already imported from elsewhere.
    """
    root = os.path.realpath(superfly_dir)
    path = os.path.join(root, 'onboard_policy_runner.py')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{path} not found; copy the superfly onboard runner into '
            f'{superfly_dir} (see its SOURCES.txt)')
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module('onboard_policy_runner')
    if os.path.realpath(module.__file__) != path:
        raise ImportError(
            f'onboard_policy_runner already imported from {module.__file__}, '
            f'not {path}')
    return module


class RgbPolicyScenario(Scenario):
    """One drone flown by a superfly RGB navigation policy, on the ground.

    Everything policy-side is the ONBOARD runner's own code
    (``onboard_policy_runner.py``, copied into ``superfly_dir``): its argparse
    builds ``args`` exactly as on the board, ``build_stack(args)`` builds the
    variant from ``ONBOARD_VARIANTS`` (``OnboardAgilePolicy`` or
    ``OnboardTokenPolicy``, CPU TFLite), and ``VelocityYawPublisher.encode``
    applies the |v_xy| clamp, ``--alt-hold`` (``altitude_vz_ned``) and the
    absolute heading of ``--yaw-out angle``. This class only replaces the
    runner's I/O: state from the commander's odometry, frames from
    ``RGBSubscriber`` (UDP from ``rgb_bridge_voxl.py``), command out as world
    ENU velocity + an ENU heading target (``yaw_targets``).

    Phases, per episode (an episode starts on the first call after the
    mission was not running for ``episode_gap_s``, i.e. every ~/start):

    1. hand-over: ``retarget(goal, pos)`` + ``engage(...)`` (runner's
       OFFBOARD latch; the dnav START frame + GRU reset happen at its first
       ``compute()``), ``--alt-hold`` reference = the goal's z;
    2. ALIGN: zero velocity, heading target = goal bearing, until within
       ``align_tol_deg`` or ``align_timeout_s`` (``--align-first``);
    3. POLICY: one ``compute()`` per control tick (``--planar``,
       ``--alt-hold``, ``--max-vel``), heading from the policy;
    4. REACHED: horizontal distance < ``goal_radius`` -> zero velocity,
       hold the heading measured on arrival.

    A frame older than ``frame_stale_s`` (or none yet) or missing state
    (no fresh attitude) -> zero velocity and hold the heading, no compute().

    The dnav policy is built UNPIPELINED (``--no-pipeline``): the runner's
    pipelined mode steps the GRU back to back as fast as inference allows
    (~60 Hz on a desktop CPU), not at the trained 15 Hz; synchronous, one
    GRU step per control tick is the trained rate, at ~12-20 ms per tick.
    """

    needs_full_state = True

    ALIGN = 'ALIGN'
    POLICY = 'POLICY'
    REACHED = 'REACHED'

    def __init__(
        self,
        *args,
        policy_variant: str,
        goal,
        hover_positions,
        superfly_dir: str = DEFAULT_SUPERFLY_DIR,
        target_speed: Optional[float] = None,
        max_vel: float = 0.5,
        goal_radius: float = 1.0,
        align_tol_deg: float = 1.0,
        align_timeout_s: float = 30.0,
        control_rate_hz: float = 15.0,
        threads: int = 8,
        alt_hold: bool = True,
        alt_hold_kp: float = 2.0,
        alt_hold_kd: float = 1.0,
        alt_hold_max_vz: float = 1.5,
        rgb_bind: str = '0.0.0.0',
        rgb_port: int = 15003,
        frame_stale_s: float = 0.25,
        episode_gap_s: float = 0.5,
        log=print,
        policy=None,
        frame_source=None,
        clock=time.monotonic,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.num_drones != 1:
            raise ValueError(
                f'rgb_policy scenario flies exactly 1 drone, got {self.num_drones}')
        start = np.asarray(hover_positions, dtype=float).reshape(-1, 3)
        if start.shape[0] != 1:
            raise ValueError(
                f'rgb_policy scenario needs 1 hover position, got {start.shape[0]}')
        self._start = start
        self._goal = np.asarray(goal, dtype=float).reshape(3)
        self._log = log
        self._clock = clock
        self._frame_stale_s = float(frame_stale_s)
        self._episode_gap_s = float(episode_gap_s)

        opr = load_onboard_runner(superfly_dir)
        self.runner = opr
        if policy_variant not in opr.ONBOARD_VARIANTS:
            raise ValueError(
                f'unknown policy variant {policy_variant!r}; registered: '
                f'{sorted(opr.ONBOARD_VARIANTS)}')
        kind = opr.ONBOARD_VARIANTS[policy_variant]['kind']
        # The runner's own command line, as fly_indoor.sh flies it (planar,
        # alt-hold, align-first, absolute heading out), CPU TFLite.
        argv = [
            '--policy', policy_variant,
            '--model-dir', os.path.join(os.path.realpath(superfly_dir), 'models'),
            '--encoder-backend', 'tflite',
            '--threads', str(int(threads)),
            '--max-vel', repr(float(max_vel)),
            '--planar',
            '--rate', repr(float(control_rate_hz)),
            '--goal-radius', repr(float(goal_radius)),
            '--align-first',
            '--align-tol-deg', repr(float(align_tol_deg)),
            '--align-timeout-s', repr(float(align_timeout_s)),
            '--yaw-out', 'angle',
            '--alt-hold-kp', repr(float(alt_hold_kp)),
            '--alt-hold-kd', repr(float(alt_hold_kd)),
            '--alt-hold-max-vz', repr(float(alt_hold_max_vz)),
            '--no-log-dir',
        ]
        if alt_hold:
            argv.append('--alt-hold')
        if kind == 'token':
            if target_speed is None:
                raise ValueError(f'{policy_variant} needs target_speed')
            argv += ['--target-speed', repr(float(target_speed)), '--no-pipeline']
        try:
            self.args = opr.parse_args(argv)
        except SystemExit as e:  # argparse error: already printed to stderr
            raise ValueError(
                f'onboard runner rejected the rgb_policy arguments {argv}') from e
        self.argv = argv
        args = self.args

        if policy is None:
            policy, _net_every = opr.build_stack(args)
            if kind == 'agile':
                # build_token_policy warms its encoder; do the same for agile
                # so the first POLICY tick does not pay the first invoke.
                policy.encoder(np.zeros((opr.NET_SIZE, opr.NET_SIZE, 3), np.uint8))
        self.policy = policy
        # main(): the command-type encoding the runner itself applies.
        self._publisher = opr.VelocityYawPublisher(
            None, max_vel_xy=policy.max_vel_xy,
            yaw_rate_max=args.yaw_rate_max, yaw_mode=args.yaw_out,
            alt_hold=bool(args.alt_hold) and bool(args.planar),
            alt_hold_kp=args.alt_hold_kp, alt_hold_kd=args.alt_hold_kd,
            alt_hold_max_vz=args.alt_hold_max_vz)
        self._dt = 1.0 / float(args.rate)
        self._align_tol = math.radians(float(args.align_tol_deg))

        if frame_source is None:
            from superfly.common.rgb_transport import RGBSubscriber
            frame_source = RGBSubscriber(host=str(rgb_bind), port=int(rgb_port))
        self.frame_source = frame_source

        self._phase = None           # None until the first episode starts
        self._need_engage = True
        self._last_call = None
        self._align_t0 = None
        self._alt_ref_d = None       # --alt-hold reference, world NED down
        self._yaw_target = None
        self._hold_reason = None     # why the current tick holds, or None
        self._hold_yaw = None
        self.last_command = None     # the policy's last compute() dict
        self.compute_ms = float('nan')
        self._log(
            f'rgb_policy: {policy_variant} ({kind}) goal ENU {self._goal.tolist()}'
            f' | runner argv: {" ".join(argv)}')

    # -- Scenario interface --------------------------------------------------
    def initial_positions(self) -> np.ndarray:
        return self._start.copy()

    @property
    def goals(self) -> Optional[np.ndarray]:
        return self._goal[None, :].copy()

    @property
    def yaw_targets(self) -> Optional[list]:
        return [self._yaw_target]

    @property
    def phase(self) -> Optional[str]:
        return self._phase

    def nominal_velocity(self, positions, velocities=None, orientations=None,
                         body_rates=None) -> np.ndarray:
        opr = self.runner
        now = self._clock()
        if self._last_call is None or now - self._last_call > self._episode_gap_s:
            self._need_engage = True
        self._last_call = now
        out = np.zeros((1, 3))

        q = None if orientations is None else orientations[0]
        if q is None or velocities is None:
            self._hold('no fresh state', None)
            return out
        pos = np.asarray(positions[0], dtype=float)
        vel = np.asarray(velocities[0], dtype=float)
        rates = (np.zeros(3) if body_rates is None
                 else np.asarray(body_rates[0], dtype=float))
        x, y, z, w = (float(c) for c in q)
        R_enu = opr.quat_wxyz_to_matrix((w, x, y, z))
        yaw = math.atan2(R_enu[1, 0], R_enu[0, 0])

        if self._need_engage:
            self._engage(pos, R_enu)
        if self._phase == self.REACHED:
            self._yaw_target = self._hold_yaw
            return out

        frame = self._fresh_frame()
        if frame is None:
            self._hold('frame stale or absent', yaw)
        elif self._phase == self.ALIGN and not self._aligned(pos, yaw, now):
            pass                     # zero velocity, heading -> goal bearing
        else:
            out[0] = self._policy_step(pos, vel, R_enu, rates, frame)

        # The runner checks arrival after the tick's command, every tick.
        dist_xy = float(np.linalg.norm((self._goal - pos)[:2]))
        if dist_xy < float(self.args.goal_radius):
            self._phase = self.REACHED
            self._hold_yaw = yaw
            self._log(f'rgb_policy: REACHED goal (d_xy={dist_xy:.2f} m < '
                      f'{self.args.goal_radius} m); holding')
        return out

    # -- internals -------------------------------------------------------------
    def _engage(self, pos, R_enu) -> None:
        """The runner's hand-over latch (main(): OFFBOARD entered)."""
        self._need_engage = False
        self.policy.retarget(self._goal, pos)
        self.policy.engage(pos, R_enu, self._goal)
        # alt_ref_for() with --goal-z-mode absolute: the goal's own z (NED).
        self._alt_ref_d = -float(self._goal[2])
        self._phase = self.ALIGN if self.args.align_first else self.POLICY
        self._align_t0 = None
        self._hold_reason = None
        self._log(f'rgb_policy: hand-over at ENU {np.round(pos, 2).tolist()}, '
                  f'alt-hold ref z={self._goal[2]:.2f} m; {self._phase}')

    def _fresh_frame(self):
        frame, age = self.frame_source.latest_with_age()
        if frame is None or age is None or age > self._frame_stale_s:
            return None
        return self.runner.to_net_frame(frame)

    def _hold(self, reason: str, yaw: Optional[float]) -> None:
        """Zero velocity, heading held at its value when the hold began."""
        if self._hold_reason is None:
            if yaw is not None:
                self._hold_yaw = yaw
            self._log(f'rgb_policy: HOLD ({reason})')
        self._hold_reason = reason
        self._yaw_target = self._hold_yaw if yaw is not None else None

    def _resume(self) -> None:
        if self._hold_reason is not None:
            self._log(f'rgb_policy: hold cleared ({self._hold_reason}); '
                      f'{self._phase}')
            self._hold_reason = None

    def _aligned(self, pos, yaw, now) -> bool:
        """--align-first: True once the heading is within tolerance (or the
        timeout passed) -> POLICY this tick; else turn toward the goal."""
        self._resume()
        bearing = math.atan2(self._goal[1] - pos[1], self._goal[0] - pos[0])
        err = wrap_pi(bearing - yaw)
        if self._align_t0 is None:
            self._align_t0 = now
            self._log(f'rgb_policy: ALIGN bearing {math.degrees(bearing):+.1f} '
                      f'deg, heading {math.degrees(yaw):+.1f} deg')
        waited = now - self._align_t0
        if abs(err) <= self._align_tol or waited >= float(self.args.align_timeout_s):
            self._phase = self.POLICY
            self._log(f'rgb_policy: POLICY engaged after {waited:.2f} s '
                      f'(heading error {math.degrees(err):+.1f} deg)')
            return True
        self._yaw_target = bearing
        return False

    def _policy_step(self, pos, vel, R_enu, rates, frame) -> np.ndarray:
        opr = self.runner
        self._resume()
        t0 = time.perf_counter()
        cmd = self.policy.compute(pos, vel, R_enu, rates, self._goal, frame)
        self.compute_ms = (time.perf_counter() - t0) * 1e3
        self.last_command = cmd
        v_ned, _yaw_rate, yaw_sp_ned = self._publisher.encode(cmd, dict(
            pos_ned=opr.swap_ne(pos), vel_ned=opr.swap_ne(vel),
            yaw_ned=opr.yaw_ned_from_R_enu(R_enu), dt=self._dt,
            alt_ref_d=self._alt_ref_d, state_src=None, state_last=None))
        self._yaw_target = opr.wrap_pi(opr.swap_yaw(yaw_sp_ned))
        return opr.swap_ne(v_ned)


_SCENARIOS = {
    'hover': HoverScenario,
    'goal': GoalScenario,
    'random_walk': RandomWalkScenario,
    'random_goals': RandomGoalsScenario,
    'head_on': HeadOnScenario,
    'antipodal': AntipodalScenario,
    'squeeze': SqueezeScenario,
    'rgb_policy': RgbPolicyScenario,
}


def make_scenario(
    name: str,
    num_drones: int,
    nominal_speed: float,
    bounds: Bounds,
    safety_radius: float,
    seed: int = 7,
    **kwargs,
) -> Scenario:
    """Construct a scenario by name.

    Args:
        name: one of ``hover``, ``random_walk``, ``random_goals``, ``head_on``,
            ``antipodal``, ``squeeze``, ``goal``, ``rgb_policy``.
        num_drones: number of drones (scenario rows match drone_names order).
        nominal_speed: nominal flight speed (m/s).
        bounds: arena box.
        safety_radius: CBF safety radius r (m), used for spacing decisions.
        seed: RNG seed for the randomized scenarios.
        **kwargs: scenario-specific options (``hover_positions`` for hover,
            ``gap_factor`` / ``run_length`` for squeeze, ``policy_variant`` /
            ``goal`` / ... for rgb_policy).

    Raises:
        ValueError: if ``name`` is unknown.
    """
    if name not in _SCENARIOS:
        raise ValueError(
            f'unknown scenario {name!r}; choose from {sorted(_SCENARIOS)}')
    return _SCENARIOS[name](
        num_drones, bounds, nominal_speed, np.random.default_rng(seed),
        safety_radius, **kwargs)
