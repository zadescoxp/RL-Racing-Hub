import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# A hand-designed closed-loop centerline.
# The points are connected cyclically.
TRACK_POINTS = np.array([
    [0.0, -260.0],
    [120.0, -250.0],
    [220.0, -190.0],
    [270.0, -80.0],
    [250.0, 30.0],
    [190.0, 110.0],
    [110.0, 150.0],
    [150.0, 250.0],
    [70.0, 330.0],
    [-60.0, 340.0],
    [-170.0, 300.0],
    [-230.0, 210.0],
    [-250.0, 100.0],
    [-230.0, 0.0],
    [-270.0, -90.0],
    [-250.0, -190.0],
    [-170.0, -260.0],
    [-70.0, -275.0],
], dtype=np.float32)

TRACK_WIDTH = 100.0


def segment_projection(point, a, b):
    """Return projected point and fraction t on segment a->b."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return a.copy(), 0.0
    t = float(np.dot(point - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    return a + t * ab, t


def build_track(points):
    """Precompute segment lengths and cumulative distances."""
    n = len(points)
    lengths = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        lengths[i] = np.linalg.norm(b - a)

    cumulative = np.zeros(n + 1, dtype=np.float32)
    cumulative[1:] = np.cumsum(lengths)
    return lengths, cumulative


SEGMENT_LENGTHS, CUMULATIVE = build_track(TRACK_POINTS)
TRACK_LENGTH = float(CUMULATIVE[-1])


def nearest_track_info(point):
    """
    Find the closest point on the closed centerline.

    Returns:
        distance_from_centerline,
        progress_distance,
        segment_index,
        projected_point,
        tangent_angle
    """
    best_dist = float("inf")
    best_progress = 0.0
    best_i = 0
    best_projection = TRACK_POINTS[0]

    for i in range(len(TRACK_POINTS)):
        a = TRACK_POINTS[i]
        b = TRACK_POINTS[(i + 1) % len(TRACK_POINTS)]
        projection, t = segment_projection(point, a, b)
        dist = float(np.linalg.norm(point - projection))

        if dist < best_dist:
            best_dist = dist
            best_i = i
            best_projection = projection
            best_progress = float(CUMULATIVE[i] + t * SEGMENT_LENGTHS[i])

    a = TRACK_POINTS[best_i]
    b = TRACK_POINTS[(best_i + 1) % len(TRACK_POINTS)]
    tangent = b - a
    tangent_angle = math.atan2(float(tangent[1]), float(tangent[0]))

    return best_dist, best_progress, best_i, best_projection, tangent_angle


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class RacingEnv:
    """
    Continuous 2D racing environment.

    The action controls acceleration/braking and steering.
    The agent itself is a point.

    Training is normally one lap per episode.
    """

    ACTIONS = {
        0: (0.0, 0.0),    # coast
        1: (1.0, 0.0),    # accelerate
        2: (-1.0, 0.0),   # brake/reverse acceleration
        3: (0.0, -1.0),   # left
        4: (0.0, 1.0),    # right
        5: (1.0, -1.0),   # accelerate + left
        6: (1.0, 1.0),    # accelerate + right
    }
    n_actions = len(ACTIONS)

    def __init__(self, max_steps=2500, training_laps=1):
        self.max_steps = max_steps
        self.training_laps = training_laps

        self.dt = 0.10
        self.max_speed = 95.0
        self.acceleration = 34.0
        self.braking = 52.0
        self.turn_rate = 2.5

        self.track_width = TRACK_WIDTH
        self.checkpoint_count = 12
        self.checkpoints = np.linspace(
            0, len(TRACK_POINTS) - 1, self.checkpoint_count,
            dtype=int
        )

        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.speed = 0.0

        self.step_count = 0
        self.laps = 0
        self.total_progress = 0.0
        self.previous_progress = 0.0
        self.last_segment = 0
        self.last_checkpoint = 0

        self.prev_position = self.position.copy()
        self.velocity_from_position = np.zeros(2, dtype=np.float32)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        start = TRACK_POINTS[0]
        next_point = TRACK_POINTS[1]

        self.position = start.copy()
        tangent = next_point - start
        self.heading = math.atan2(float(tangent[1]), float(tangent[0]))
        self.velocity = np.zeros(2, dtype=np.float32)
        self.speed = 0.0

        self.step_count = 0
        self.laps = 0
        self.total_progress = 0.0
        self.previous_progress = 0.0
        self.last_segment = 0
        self.last_checkpoint = 0
        self.prev_position = self.position.copy()
        self.velocity_from_position = np.zeros(2, dtype=np.float32)

        return self.get_observation()

    def _signed_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _respawn(self):
        idx = int(self.checkpoints[self.last_checkpoint])
        self.position = TRACK_POINTS[idx].copy()

        next_idx = (idx + 1) % len(TRACK_POINTS)
        tangent = TRACK_POINTS[next_idx] - TRACK_POINTS[idx]
        self.heading = math.atan2(float(tangent[1]), float(tangent[0]))

        self.speed = 0.0
        self.velocity[:] = 0.0
        self.velocity_from_position[:] = 0.0

        # Small offset into the track keeps the point away from the boundary.
        tangent_unit = tangent / (np.linalg.norm(tangent) + 1e-8)
        normal = np.array([-tangent_unit[1], tangent_unit[0]], dtype=np.float32)
        self.position += normal * 0.0

        _, progress, seg, _, _ = nearest_track_info(self.position)
        self.previous_progress = progress
        self.last_segment = seg

    def _update_checkpoint(self, segment_index):
        # Checkpoints are defined by centerline point indices.
        for cp_idx, segment in enumerate(self.checkpoints):
            # Move checkpoint forward whenever the agent passes it.
            if segment_index == int(segment):
                if cp_idx >= self.last_checkpoint:
                    self.last_checkpoint = cp_idx

    def _advance_progress(self, current_progress):
        """
        Convert wrapped centerline distance into an unwrapped progress value.
        """
        delta = current_progress - self.previous_progress

        # Crossing the start line in the forward direction.
        if delta < -TRACK_LENGTH / 2:
            delta += TRACK_LENGTH
            self.laps += 1

        # A huge positive jump is probably backwards crossing.
        elif delta > TRACK_LENGTH / 2:
            delta -= TRACK_LENGTH

        self.total_progress += delta
        self.previous_progress = current_progress
        return delta

    def get_observation(self):
        """
        Continuous normalized observation for DQN/PPO.

        Includes the actual coordinate information requested by the project,
        plus motion and track-relative features.
        """
        dist, progress, seg, projection, tangent_angle = nearest_track_info(
            self.position
        )

        vx, vy = self.velocity_from_position
        heading_error = self._signed_angle(tangent_angle - self.heading)

        # Track-relative longitudinal/lateral information.
        offset_vec = self.position - projection
        tangent = np.array(
            [math.cos(tangent_angle), math.sin(tangent_angle)],
            dtype=np.float32
        )
        lateral = float(
            offset_vec[0] * (-tangent[1]) +
            offset_vec[1] * tangent[0]
        )

        obs = np.array([
            self.position[0] / 400.0,
            self.position[1] / 400.0,
            vx / self.max_speed,
            vy / self.max_speed,
            self.speed / self.max_speed,
            math.sin(self.heading),
            math.cos(self.heading),
            math.sin(heading_error),
            math.cos(heading_error),
            dist / (self.track_width / 2.0),
            lateral / (self.track_width / 2.0),
            progress / TRACK_LENGTH,
        ], dtype=np.float32)

        return np.clip(obs, -2.0, 2.0)

    def step(self, action):
        action = int(action)
        throttle, steering = self.ACTIONS[action]

        self.step_count += 1
        self.prev_position = self.position.copy()

        # Steering is more effective while moving.
        steer_strength = self.turn_rate * (0.25 + self.speed / self.max_speed)
        self.heading += steering * steer_strength * self.dt

        if throttle > 0:
            self.speed += self.acceleration * self.dt
        elif throttle < 0:
            self.speed -= self.braking * self.dt

        # Natural drag.
        self.speed *= 0.992
        self.speed = float(np.clip(self.speed, 0.0, self.max_speed))

        self.velocity = np.array([
            math.cos(self.heading) * self.speed,
            math.sin(self.heading) * self.speed,
        ], dtype=np.float32)

        self.position += self.velocity * self.dt

        # This is explicitly calculated from delta position / delta time.
        self.velocity_from_position = (
            self.position - self.prev_position
        ) / self.dt

        dist, progress, seg, projection, tangent_angle = nearest_track_info(
            self.position
        )

        progress_delta = self._advance_progress(progress)
        self._update_checkpoint(seg)

        reward = progress_delta * 1.2
        reward -= 0.02  # time penalty

        if progress_delta < -0.2:
            reward -= 0.25

        collision = dist > self.track_width / 2.0
        if collision:
            reward -= 15.0
            self._respawn()

        lap_completed = self.laps >= self.training_laps

        if lap_completed:
            reward += 100.0

        truncated = self.step_count >= self.max_steps
        terminated = bool(lap_completed)

        info = {
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "vx": float(self.velocity_from_position[0]),
            "vy": float(self.velocity_from_position[1]),
            "speed": float(np.linalg.norm(self.velocity_from_position)),
            "distance_from_centerline": float(dist),
            "progress": float(self.total_progress / TRACK_LENGTH),
            "lap": int(self.laps),
            "collision": collision,
            "lap_completed": lap_completed,
            "checkpoint": int(self.last_checkpoint),
        }

        return StepResult(
            self.get_observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def get_track_geometry(self):
        return TRACK_POINTS.copy(), self.track_width
