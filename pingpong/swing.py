"""Paddle velocity estimation from the phone IMU.

Why not double-integrate acceleration
-------------------------------------
1. The error grows as t^2, which is hopeless over the length of a rally.
2. **Web accelerometers are often limited to +/-2g (about 20 m/s^2), and a
   real swing saturates that.** This is rarely noticed, but it means the
   acceleration path is least accurate exactly when accuracy matters most.

Use the gyroscope instead
-------------------------
A stroke is essentially a rotation about the elbow / shoulder with the
phone at the end of the lever. Rigid-body kinematics gives::

    v = w x r

where ``r`` runs from the pivot to the phone, roughly forearm plus handle.
Gyroscopes are typically +/-2000 deg/s so they do not saturate, and there
is no integration drift, so this path yields both the **magnitude and the
direction** of the swing far more reliably than integrating acceleration.

This also fixes an old bug: the previous code removed gravity with
``|a| - 9.81``. Gravity is a vector, ``|a_total| - 9.81 != |a_true|``; the
subtraction has to happen componentwise in world coordinates.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import quat

# Effective distance from the pivot to the blade: forearm + hand + handle.
DEFAULT_ARM_LENGTH = 0.45

# Direction from handle to blade tip in phone body coordinates.
# Holding the phone with the screen facing you, the top is the blade: +Y.
PHONE_HEAD_AXIS = np.array([0.0, 1.0, 0.0])

GRAVITY_WORLD = np.array([0.0, 0.0, 9.81])

# Swing detection thresholds, metres/second
SWING_ENTER_SPEED = 1.8
SWING_EXIT_SPEED = 0.8
# Below these the device counts as still, used for zero-velocity updates
STILL_OMEGA = 0.6      # rad/s
STILL_ACCEL = 1.5      # m/s^2


@dataclass
class SwingSample:
    """Swing estimate at a single instant."""

    time: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    speed: float = 0.0
    omega_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    swinging: bool = False


@dataclass
class SwingEvent:
    """Summary of one complete swing, from onset to peak."""

    start_time: float
    peak_time: float
    peak_speed: float
    peak_velocity: np.ndarray
    duration: float


class SwingEstimator:
    """Turns an IMU stream into paddle velocity and detects swings.

    Typical use::

        est = SwingEstimator()
        sample = est.update(t, omega_body, accel_body, q_phone, calib)
        paddle_velocity = sample.velocity     # feed straight to physics.Paddle
    """

    def __init__(self, arm_length=DEFAULT_ARM_LENGTH, smoothing=0.35):
        self.arm_length = arm_length
        # First-order low-pass coefficient, kept deliberately high (fast
        # response) because a swing is about its peak. The old code used
        # alpha=0.2, a ~75 ms time constant at 60 Hz, which flattened it.
        self.smoothing = smoothing

        self.velocity = np.zeros(3)
        self.speed = 0.0
        self.last_time: Optional[float] = None

        self._swinging = False
        self._swing_start = 0.0
        self._peak_speed = 0.0
        self._peak_velocity = np.zeros(3)
        self._peak_time = 0.0
        self.last_event: Optional[SwingEvent] = None

        self._still_since: Optional[float] = None

    # ------------------------------------------------------------------
    def reset(self):
        self.velocity = np.zeros(3)
        self.speed = 0.0
        self._swinging = False
        self._peak_speed = 0.0
        self.last_event = None

    def update(self, t, omega_body, accel_body, q_phone, calibration=None,
               accel_includes_gravity=False):
        """Feed one IMU sample.

        Parameters
        ----------
        t : float
            Timestamp in seconds.
        omega_body : (3,)
            Gyroscope angular velocity, **body frame, rad/s**.
        accel_body : (3,)
            Accelerometer reading, body frame, m/s^2.
        q_phone : (4,)
            Phone orientation quaternion (x, y, z, w).
        calibration : quat.Calibration, optional
            When given, velocities are expressed in game world coordinates.
        accel_includes_gravity : bool
            Whether the reading still contains gravity (devicemotion's
            accelerationIncludingGravity).
        """
        omega_body = np.asarray(omega_body, dtype=float)
        accel_body = np.asarray(accel_body, dtype=float)

        dt = 0.0 if self.last_time is None else max(0.0, t - self.last_time)
        self.last_time = t

        # ---- orientation: body measurements into world / game frame ----
        if calibration is not None and calibration.ready:
            to_world = calibration.phone_to_world
            omega_world = to_world(q_phone, omega_body)
            accel_world = to_world(q_phone, accel_body)
            head_dir = to_world(q_phone, PHONE_HEAD_AXIS)
        else:
            omega_world = quat.rotate(q_phone, omega_body)
            accel_world = quat.rotate(q_phone, accel_body)
            head_dir = quat.rotate(q_phone, PHONE_HEAD_AXIS)

        # ---- gravity removal: vector subtraction, not magnitude ----
        if accel_includes_gravity:
            linear_accel = accel_world - GRAVITY_WORLD
        else:
            linear_accel = accel_world

        # ---- the core: v = w x r ----
        # r points from the handle to the blade, with the effective arm length.
        r_arm = head_dir * self.arm_length
        v_rot = np.cross(omega_world, r_arm)

        # First-order low-pass: suppress gyro noise but keep the peak
        a = self.smoothing
        self.velocity = (1 - a) * self.velocity + a * v_rot
        self.speed = float(np.linalg.norm(self.velocity))

        # ---- zero-velocity update: bleed off residue when held still ----
        omega_mag = float(np.linalg.norm(omega_world))
        accel_mag = float(np.linalg.norm(linear_accel))
        if omega_mag < STILL_OMEGA and accel_mag < STILL_ACCEL:
            if self._still_since is None:
                self._still_since = t
            elif t - self._still_since > 0.15:
                self.velocity *= 0.5
                self.speed = float(np.linalg.norm(self.velocity))
        else:
            self._still_since = None

        # ---- swing state machine ----
        if not self._swinging:
            if self.speed > SWING_ENTER_SPEED:
                self._swinging = True
                self._swing_start = t
                self._peak_speed = self.speed
                self._peak_velocity = self.velocity.copy()
                self._peak_time = t
        else:
            if self.speed > self._peak_speed:
                self._peak_speed = self.speed
                self._peak_velocity = self.velocity.copy()
                self._peak_time = t
            if self.speed < SWING_EXIT_SPEED:
                self._swinging = False
                self.last_event = SwingEvent(
                    start_time=self._swing_start,
                    peak_time=self._peak_time,
                    peak_speed=self._peak_speed,
                    peak_velocity=self._peak_velocity.copy(),
                    duration=t - self._swing_start,
                )

        return SwingSample(
            time=t,
            velocity=self.velocity.copy(),
            speed=self.speed,
            omega_world=omega_world,
            linear_accel=linear_accel,
            swinging=self._swinging,
        )

    # ------------------------------------------------------------------
    @property
    def swinging(self):
        return self._swinging

    @property
    def peak_speed(self):
        return self._peak_speed
