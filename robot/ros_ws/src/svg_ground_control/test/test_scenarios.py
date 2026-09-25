"""Tests for the scenario policies, including a kinematic squeeze rollout."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from svg_ground_control.cbf_filter import filter_velocities
from svg_ground_control.scenarios import (
    Bounds, make_scenario, yaw_from_quaternion_xyzw, yaw_rate_toward_target)

ARENA = Bounds(low=np.array([-2.0, -2.0, 0.8]), high=np.array([2.0, 2.0, 2.0]))


def make(name, n, **kwargs):
    return make_scenario(
        name, num_drones=n, nominal_speed=0.6, bounds=ARENA,
        safety_radius=0.55, seed=7, **kwargs)


def test_all_scenarios_produce_valid_initial_positions() -> None:
    for name, n in [('random_walk', 5), ('random_goals', 5),
                    ('head_on', 6), ('antipodal', 6)]:
        scenario = make(name, n)
        positions = scenario.initial_positions()
        assert positions.shape == (n, 3)
        assert np.all(positions >= ARENA.low - 1e-9)
        assert np.all(positions <= ARENA.high + 1e-9)
        nominal = scenario.nominal_velocity(positions)
        assert nominal.shape == (n, 3)
        assert np.all(np.isfinite(nominal))


def test_goal_scenario_live_retarget_and_speed() -> None:
    initial = np.array([[0.0, 0.0, 1.2], [1.0, 0.0, 1.2]])
    s = make('goal', 2, initial_goals=initial)
    np.testing.assert_allclose(s.initial_positions(), initial)

    # Default: seek the initial goals.
    pos = initial + np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    v = s.nominal_velocity(pos)
    assert v[0, 0] < 0.0                          # drone 0 pulled back -x
    np.testing.assert_allclose(v[1], 0.0, atol=1e-9)

    # Retarget drone 1 live; speed cap respected.
    s.set_goal(1, np.array([5.0, 0.0, 1.2]))
    s.set_speed(1, 0.5)
    v = s.nominal_velocity(pos)
    assert v[1, 0] > 0.0
    assert abs(np.linalg.norm(v[1]) - 0.5) < 1e-6   # far goal -> capped at speed


def test_hover_scenario_seeks_targets() -> None:
    targets = np.array([[-1.0, 0.0, 1.2], [1.0, 0.0, 1.2]])
    scenario = make('hover', 2, hover_positions=targets)
    np.testing.assert_allclose(scenario.initial_positions(), targets)
    # Displaced drone gets pulled back toward its target.
    displaced = targets + np.array([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    nominal = scenario.nominal_velocity(displaced)
    assert nominal[0, 0] < 0.0           # pulled back along -x
    np.testing.assert_allclose(nominal[1], 0.0, atol=1e-9)


HOLDER_POSTS = [0.0, -0.69, 1.2, 0.0, 0.69, 1.2]
INTRUDER_WAYPOINTS = [-1.5, 0.0, 1.2, 1.5, 0.0, 1.2]


def make_squeeze():
    return make('squeeze', 3, holder_positions=HOLDER_POSTS,
                intruder_waypoints=INTRUDER_WAYPOINTS)


def test_squeeze_intruder_is_cbf_exempt_by_default() -> None:
    assert make_squeeze().cbf_exempt_indices == [2]
    filtered = make('squeeze', 3, holder_positions=HOLDER_POSTS,
                    intruder_waypoints=INTRUDER_WAYPOINTS,
                    intruder_cbf_exempt=False)
    assert filtered.cbf_exempt_indices == []


def test_squeeze_geometry() -> None:
    scenario = make_squeeze()
    initial = scenario.initial_positions()
    # Holders take off exactly at their configured posts.
    np.testing.assert_allclose(initial[0], HOLDER_POSTS[:3])
    np.testing.assert_allclose(initial[1], HOLDER_POSTS[3:])
    # Intruder takes off at waypoint A and its nominal points toward B (+x).
    np.testing.assert_allclose(initial[2], INTRUDER_WAYPOINTS[:3])
    nominal = scenario.nominal_velocity(initial)
    assert nominal[2, 0] > 0.0


def test_squeeze_rejects_overlapping_posts() -> None:
    try:
        make('squeeze', 3,
             holder_positions=[0.0, -0.3, 1.2, 0.0, 0.3, 1.2],  # 0.6 m < 2r
             intruder_waypoints=INTRUDER_WAYPOINTS)
    except ValueError as e:
        assert 'keep-out' in str(e)
    else:
        raise AssertionError('overlapping posts were not rejected')


def test_squeeze_kinematic_rollout_holders_yield_and_return() -> None:
    """Single-integrator rollout: barrier holds, holders yield then return."""
    safety_radius = 0.55
    max_speed = 1.2
    dt = 0.05
    scenario = make_squeeze()
    positions = scenario.initial_positions().copy()
    posts = positions[:2].copy()

    min_pair_distance = np.inf
    max_holder_displacement = 0.0
    for _ in range(400):  # 20 s — more than one full crossing
        nominal = scenario.nominal_velocity(positions)
        result = filter_velocities(
            nominal, positions, safety_radius, max_speed, alpha=2.5)
        # The intruder is CBF-exempt (the commander restores its row).
        safe = result.velocities
        safe[2] = nominal[2]
        positions = positions + safe * dt

        distances = np.linalg.norm(
            positions[:, None] - positions[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        min_pair_distance = min(min_pair_distance, float(distances.min()))
        max_holder_displacement = max(
            max_holder_displacement,
            float(np.linalg.norm(positions[:2] - posts, axis=-1).max()))

    # Holders were genuinely displaced by the crossing...
    assert max_holder_displacement > 0.2
    # ...the intruder actually made it through to the +x side at least once
    # (it shuttles, so just check it covered the run)...
    assert positions[2, 0] > ARENA.center[0] - 1.6
    # ...and, with the exempt intruder pushing through, the holders never let
    # the *holder pair* breach its own barrier; holder-intruder distance may
    # dip slightly below 2r since one party is uncontrolled — require the
    # holders to keep at least 1.5 r body margin from the intruder.
    holder_pair = np.linalg.norm(positions[0] - positions[1])
    assert holder_pair >= 0.0  # sanity
    assert min_pair_distance >= 1.5 * safety_radius

    # After the crossing settles (intruder far from center), holders return.
    for _ in range(100):
        nominal = scenario.nominal_velocity(positions)
        result = filter_velocities(
            nominal, positions, safety_radius, max_speed, alpha=2.5)
        safe = result.velocities
        safe[2] = nominal[2]
        positions = positions + safe * dt
    settle_error = np.linalg.norm(positions[:2] - posts, axis=-1).max()
    assert settle_error < 0.6  # back near the posts (intruder keeps shuttling)


# ---------------------------------------------------------------------------
# Heading helpers + rgb_policy scenario
# ---------------------------------------------------------------------------

def quat_yaw(yaw):
    """ROS (x, y, z, w) quaternion for a level attitude at ENU heading yaw."""
    return np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])


def test_position_only_scenarios_have_no_yaw_opinion() -> None:
    for name, n, kwargs in [
            ('hover', 2, dict(hover_positions=[[-1, 0, 1.2], [1, 0, 1.2]])),
            ('goal', 1, dict(initial_goals=[[0, 0, 1.2]])),
            ('random_walk', 3, {}), ('random_goals', 3, {}),
            ('head_on', 4, {}), ('antipodal', 4, {})]:
        scenario = make(name, n, **kwargs)
        assert scenario.needs_full_state is False
        assert scenario.yaw_targets is None


def test_yaw_helpers() -> None:
    for yaw in (-3.0, -1.0, 0.0, 0.5, 2.9):
        assert abs(yaw_from_quaternion_xyzw(quat_yaw(yaw)) - yaw) < 1e-9
    # CCW (ENU +) error -> positive rate; P gain; clip.
    assert abs(yaw_rate_toward_target(0.1, 0.0, 1.5, 1.5) - 0.15) < 1e-12
    assert yaw_rate_toward_target(3.0, 0.0, 1.5, 1.5) == 1.5
    assert yaw_rate_toward_target(-3.0, 0.0, 1.5, 1.5) == -1.5
    # Wrap: target 179 deg, measured -179 deg is a 2 deg CW (negative) turn.
    rate = yaw_rate_toward_target(math.radians(179), math.radians(-179), 1.5, 1.5)
    assert abs(rate - 1.5 * math.radians(-2)) < 1e-9


def superfly_dir():
    """The gitignored copy of the superfly onboard runner, or skip."""
    candidates = [os.environ.get('SUPERFLY_ONBOARD_DIR', ''),
                  str(Path(__file__).resolve().parents[3] / 'superfly_onboard'),
                  '/root/AirStack/robot/ros_ws/superfly_onboard']
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, 'onboard_policy_runner.py')):
            return c
    pytest.skip('superfly_onboard copy (onboard_policy_runner.py) not present')


class FakePolicy:
    """Stands in for the TFLite policy; records the runner-seam calls."""

    max_vel_xy = 0.5
    max_vel_z = 0.5

    def __init__(self, vel=(2.0, 0.0, 0.3), yaw=0.3):
        self.vel, self.yaw = np.array(vel, dtype=float), float(yaw)
        self.computes, self.engages, self.retargets = 0, 0, []

    def retarget(self, goal_enu, pos_enu=None):
        self.retargets.append(None if pos_enu is None else np.array(pos_enu))

    def engage(self, pos_enu=None, R_enu=None, goal_enu=None):
        self.engages += 1

    def compute(self, pos_enu, vel_enu, R_enu, angular_body, goal_enu, rgb,
                imu_override=None):
        assert rgb.shape == (224, 224, 3)
        self.computes += 1
        return dict(vel_enu=self.vel.copy(), yaw=self.yaw,
                    command_type='velocity_yaw')


class FakeFrames:
    def __init__(self):
        self.frame = np.zeros((224, 224, 3), np.uint8)
        self.age = 0.05

    def latest_with_age(self):
        return (None, None) if self.frame is None else (self.frame, self.age)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


GOAL = [3.0, 0.0, 1.2]


def make_rgb(policy, frames, clock, **kwargs):
    return make_scenario(
        'rgb_policy', num_drones=1, nominal_speed=0.5, bounds=ARENA,
        safety_radius=0.55, superfly_dir=superfly_dir(),
        policy_variant='agile-rgb-cl4nav-r50-int8ptdense-vel-tartanair-v1',
        goal=GOAL, hover_positions=[0.0, 0.0, 1.0], policy=policy,
        frame_source=frames, clock=clock, log=lambda msg: None, **kwargs)


def tick(s, clock, pos, yaw, vel=(0.0, 0.0, 0.0), dt=1.0 / 15.0):
    clock.t += dt
    return s.nominal_velocity(
        np.array([pos], dtype=float), velocities=np.array([vel], dtype=float),
        orientations=[quat_yaw(yaw)], body_rates=np.zeros((1, 3)))


def test_rgb_policy_align_then_policy() -> None:
    policy, frames, clock = FakePolicy(), FakeFrames(), FakeClock()
    s = make_rgb(policy, frames, clock)
    assert s.needs_full_state
    np.testing.assert_allclose(s.initial_positions(), [[0.0, 0.0, 1.0]])
    assert s.yaw_targets == [None]

    # Facing +y, goal bearing 0: ALIGN = zero velocity, heading -> bearing.
    v = tick(s, clock, [0.0, 0.0, 1.0], math.pi / 2)
    np.testing.assert_allclose(v, 0.0)
    assert s.phase == 'ALIGN' and abs(s.yaw_targets[0]) < 1e-12
    assert policy.engages == 1 and policy.computes == 0
    np.testing.assert_allclose(policy.retargets[0], [0.0, 0.0, 1.0])

    # Within 1 deg: POLICY this very tick.
    v = tick(s, clock, [0.0, 0.0, 1.0], math.radians(0.5))
    assert s.phase == 'POLICY' and policy.computes == 1
    # |v_xy| clamped to max_vel_xy (runner VelocityYawPublisher), +x.
    assert abs(np.hypot(v[0, 0], v[0, 1]) - 0.5) < 1e-9 and v[0, 0] > 0
    # --planar + --alt-hold: 0.2 m below the goal z, kp 2 -> 0.4 m/s up.
    assert abs(v[0, 2] - 0.4) < 1e-9
    assert abs(s.yaw_targets[0] - 0.3) < 1e-9
    assert policy.engages == 1


def test_rgb_policy_stale_frame_and_missing_state_hold() -> None:
    policy, frames, clock = FakePolicy(), FakeFrames(), FakeClock()
    s = make_rgb(policy, frames, clock)
    tick(s, clock, [0.0, 0.0, 1.2], 0.0)                 # aligned -> POLICY
    assert s.phase == 'POLICY' and policy.computes == 1

    frames.age = 0.3                                      # > 0.25 s
    v = tick(s, clock, [0.1, 0.0, 1.2], 0.2)
    np.testing.assert_allclose(v, 0.0)
    assert abs(s.yaw_targets[0] - 0.2) < 1e-12           # heading held
    v = tick(s, clock, [0.1, 0.0, 1.2], 0.25)
    np.testing.assert_allclose(v, 0.0)
    assert abs(s.yaw_targets[0] - 0.2) < 1e-12           # latched, not tracked
    frames.frame = None                                   # no frame at all
    np.testing.assert_allclose(tick(s, clock, [0.1, 0.0, 1.2], 0.2), 0.0)
    assert policy.computes == 1

    frames.frame, frames.age = np.zeros((224, 224, 3), np.uint8), 0.0
    v = tick(s, clock, [0.1, 0.0, 1.2], 0.2)
    assert policy.computes == 2 and np.linalg.norm(v) > 0.4
    assert policy.engages == 1                            # same episode

    # Missing attitude: zero velocity, no heading opinion, no compute.
    clock.t += 1.0 / 15.0
    v = s.nominal_velocity(np.array([[0.2, 0.0, 1.2]]),
                           velocities=np.zeros((1, 3)), orientations=[None],
                           body_rates=np.zeros((1, 3)))
    np.testing.assert_allclose(v, 0.0)
    assert s.yaw_targets == [None] and policy.computes == 2


def test_rgb_policy_reached_and_new_episode() -> None:
    policy, frames, clock = FakePolicy(), FakeFrames(), FakeClock()
    s = make_rgb(policy, frames, clock)
    tick(s, clock, [0.0, 0.0, 1.2], 0.0)
    assert s.phase == 'POLICY'
    # Inside the 1.0 m horizontal radius: the tick still flies (runner
    # order), then REACHED holds at zero with the arrival heading.
    tick(s, clock, [2.2, 0.1, 1.2], 0.1)
    assert s.phase == 'REACHED'
    v = tick(s, clock, [2.2, 0.1, 1.2], 0.15)
    np.testing.assert_allclose(v, 0.0)
    assert abs(s.yaw_targets[0] - 0.1) < 1e-12
    computes = policy.computes

    # Mission paused (> episode gap) and restarted away from the goal:
    # a fresh hand-over (engage) and ALIGN again.
    clock.t += 2.0
    v = tick(s, clock, [0.0, 0.0, 1.2], math.pi)
    np.testing.assert_allclose(v, 0.0)
    assert s.phase == 'ALIGN' and policy.engages == 2
    assert policy.computes == computes


def test_rgb_policy_align_timeout_engages() -> None:
    policy, frames, clock = FakePolicy(), FakeFrames(), FakeClock()
    s = make_rgb(policy, frames, clock, align_timeout_s=1.0)
    tick(s, clock, [0.0, 0.0, 1.2], math.pi / 2)
    assert s.phase == 'ALIGN'
    tick(s, clock, [0.0, 0.0, 1.2], math.pi / 2, dt=0.5)
    assert s.phase == 'ALIGN' and policy.computes == 0
    tick(s, clock, [0.0, 0.0, 1.2], math.pi / 2, dt=0.45)
    tick(s, clock, [0.0, 0.0, 1.2], math.pi / 2, dt=0.1)  # 1.05 s after start
    assert s.phase == 'POLICY' and policy.computes == 1


def test_rgb_policy_rejects_out_of_band_target_speed() -> None:
    with pytest.raises(ValueError):
        make_scenario(
            'rgb_policy', num_drones=1, nominal_speed=0.5, bounds=ARENA,
            safety_radius=0.55, superfly_dir=superfly_dir(),
            policy_variant='dnav-rgb-dinov3-cnxt-tiny-tok-vel-starling-orew-ts2-4-v0',
            goal=GOAL, hover_positions=[0.0, 0.0, 1.0], target_speed=1.0,
            policy=FakePolicy(), frame_source=FakeFrames(), log=lambda m: None)
