"""Quaternion math, and the phone-pose -> paddle-pose mapping.

Convention
----------
Internally everything is **(x, y, z, w)** order, matching JS / WebXR.
Panda3D's ``Quat`` takes (w, x, y, z) -- use :func:`to_panda` at the
boundary.

This module fixes three bugs that were stacked on top of each other in
the previous version:

1. **Wrong Euler convention on the phone side.** W3C DeviceOrientation
   is defined as Z-X'-Y'' (intrinsic), i.e. ``R = Rz(alpha)*Rx(beta)*Ry(gamma)``.
   The old code used ZYX *and* treated beta as the Y axis and gamma as
   the X axis, so the axes were swapped. See :func:`from_device_orientation`.

2. **Calibration multiplied on the wrong side.** With calibration pose
   ``R_c`` and current pose ``R``, a world-frame rotation ``dR`` gives
   ``R = dR*R_c``. The old code computed ``R_c^-1 * R = R_c^-1 * dR * R_c``,
   which is dR conjugated into the calibration frame. That is only correct
   when dR shares an axis with R_c -- exactly why "only rotation about the
   vertical axis looked right". The correct world-frame delta is ``R*R_c^-1``.
   See :func:`world_delta`.

3. **Rotation vector used as Euler angles.** The old code converted the
   quaternion to axis-angle and fed ``angle*axis`` straight into
   ``set_hpr()``. A rotation vector is not a set of Euler angles; they
   only agree to first order for small rotations. This module stays in
   quaternions throughout and never touches Euler angles.
"""

import math

import numpy as np

# Which body axis of the phone acts as the paddle face normal.
# (0, 0, -1) is the back of the screen: hold the phone like a paddle,
# screen facing you, back facing the ball.
PHONE_FACE_AXIS = np.array([0.0, 0.0, -1.0])

# The phone's "up" (top of the screen), used as a fallback reference
# when the heading of the face axis degenerates.
PHONE_UP_AXIS = np.array([0.0, 1.0, 0.0])

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])

_EPS = 1e-9


# ------------------------------------------------------------------ basics
def as_quat(q):
    """Coerce any sequence to a float64 (4,) ndarray."""
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape[-1] != 4:
        raise ValueError(f"quaternion must have 4 components, got {arr.shape}")
    return arr


def normalize(q):
    q = as_quat(q)
    n = np.linalg.norm(q)
    if n < _EPS:
        return IDENTITY.copy()
    return q / n


def conjugate(q):
    q = as_quat(q)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def inverse(q):
    """Inverse of a unit quaternion (= conjugate). Input is normalised first."""
    return conjugate(normalize(q))


def multiply(a, b):
    """Hamilton product a*b (applies b's rotation first, then a's)."""
    a = as_quat(a)
    b = as_quat(b)
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def rotate(q, v):
    """Rotate vector v by quaternion q (q treated as a unit quaternion)."""
    q = normalize(q)
    u = q[:3]
    w = q[3]
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def from_axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < _EPS:
        return IDENTITY.copy()
    axis = axis / n
    h = 0.5 * angle_rad
    s = math.sin(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h)])


def to_matrix(q):
    """To a 3x3 rotation matrix (column convention: v_world = M @ v_body)."""
    q = normalize(q)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def to_panda(q):
    """To Panda3D ``Quat`` argument order (w, x, y, z)."""
    q = normalize(q)
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def from_matrix(m):
    """3x3 rotation matrix -> quaternion (Shepperd's method, stable)."""
    m = np.asarray(m, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize(np.array([x, y, z, w]))


def look_quat(forward, up=(0.0, 0.0, 1.0)):
    """Build a paddle pose from a desired face normal.

    The returned quaternion maps the node's local +Y onto ``forward`` and
    keeps local +Z as close to ``up`` as possible. This is the inverse of
    what :class:`Calibration` does, and the AI coach needs it to turn a
    solved contact normal back into a paddle orientation.
    """
    f = np.asarray(forward, dtype=np.float64)
    n = np.linalg.norm(f)
    if n < _EPS:
        return IDENTITY.copy()
    f = f / n

    u = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(f, u))) > 0.999:      # nearly collinear, pick another
        u = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, u)
    rn = np.linalg.norm(r)
    if rn < _EPS:
        r = np.array([1.0, 0.0, 0.0])
    else:
        r = r / rn
    u2 = np.cross(r, f)

    # Columns are the local X / Y / Z axes expressed in world coordinates
    m = np.column_stack([r, f, u2])
    return from_matrix(m)


def slerp(a, b, t):
    """Spherical linear interpolation, used to smooth poses."""
    a = normalize(a)
    b = normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:          # take the shorter path
        b = -b
        dot = -dot
    if dot > 0.9995:       # nearly identical, fall back to lerp
        return normalize(a + t * (b - a))
    theta = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta = math.sin(theta)
    return (math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / sin_theta


def angle_between(a, b):
    """Angle between two orientations, in radians."""
    d = abs(float(np.dot(normalize(a), normalize(b))))
    return 2.0 * math.acos(max(-1.0, min(1.0, d)))


# ------------------------------------------------- W3C DeviceOrientation
def from_device_orientation(alpha_deg, beta_deg, gamma_deg):
    """W3C DeviceOrientation (alpha, beta, gamma) -> quaternion.

    The spec defines this as **Z-X'-Y'' intrinsic**: alpha about Z, beta
    about X', gamma about Y''. This is the Python counterpart of the
    conversion in controller.html; the test suite compares the two.

    Note that beta ~ +/-90 degrees -- which is exactly how you hold the
    phone when using it as a paddle -- is the gimbal-lock singularity of
    this Euler set, where alpha and gamma degenerate. Prefer the sensor's
    native quaternion (AbsoluteOrientationSensor) when it is available.
    """
    d2r = math.pi / 180.0
    _x = (beta_deg or 0.0) * d2r     # beta  -> X
    _y = (gamma_deg or 0.0) * d2r    # gamma -> Y
    _z = (alpha_deg or 0.0) * d2r    # alpha -> Z

    cX, sX = math.cos(_x / 2), math.sin(_x / 2)
    cY, sY = math.cos(_y / 2), math.sin(_y / 2)
    cZ, sZ = math.cos(_z / 2), math.sin(_z / 2)

    return np.array([
        sX * cY * cZ - cX * sY * sZ,   # x
        cX * sY * cZ + sX * cY * sZ,   # y
        cX * cY * sZ + sX * sY * cZ,   # z
        cX * cY * cZ - sX * sY * sZ,   # w
    ])


# ------------------------------------------------------------ calibration
def world_delta(q_current, q_calib):
    """World-frame orientation delta ``dR = R * R_c^-1``.

    This is the fix for bug 2: multiply by the inverse on the **right**,
    not the left.
    """
    return multiply(q_current, inverse(q_calib))


def heading_of(q_phone, face_axis=PHONE_FACE_AXIS, up_axis=PHONE_UP_AXIS):
    """Heading of the phone in the horizontal plane (radians about world Z).

    Uses the world-space projection of face_axis; if that axis is close to
    vertical (degenerate projection) it falls back to up_axis.
    """
    f = rotate(q_phone, face_axis)
    if math.hypot(f[0], f[1]) < 1e-3:
        f = rotate(q_phone, up_axis)
        if math.hypot(f[0], f[1]) < 1e-3:
            return 0.0
    return math.atan2(f[1], f[0])


# Neutral paddle pose: face normal points at +X (towards the opponent).
# A Panda3D node's forward is +Y, so this is a -90 degree yaw.
NEUTRAL_PADDLE = from_axis_angle([0.0, 0.0, 1.0], -math.pi / 2)


class Calibration:
    """Maps phone orientation to paddle orientation in the game.

    The relationship is defined as::

        q_paddle = M * q_phone * B

    * ``M`` is a **pure yaw about Z** that aligns the sensor's ENU world
      frame with the game world frame. Constraining it to pure yaw is the
      important part: it is what makes a rotation about a world horizontal
      axis map to a rotation about a game horizontal axis instead of being
      skewed the way the old code skewed it.
    * ``B`` is a fixed body-frame offset meaning "the paddle is glued to
      the phone at whatever angle you were holding it at calibration".

    At calibration this gives ``M * q_c * B = M * q_c * q_c^-1 * M^-1 * N = N``,
    and an arbitrary world delta dR yields ``M * dR * M^-1 * N``: the delta
    correctly transformed into game coordinates and applied on the left of
    the neutral pose.
    """

    def __init__(self, neutral=None):
        self.neutral = NEUTRAL_PADDLE.copy() if neutral is None else normalize(neutral)
        self.M = IDENTITY.copy()
        self.B = IDENTITY.copy()
        self.ready = False
        self.q_calib = IDENTITY.copy()

    def calibrate(self, q_phone):
        """Establish the mapping from the current phone pose."""
        q_phone = normalize(q_phone)
        self.q_calib = q_phone

        # M: rotate the calibration heading onto game +X (heading zero)
        heading = heading_of(q_phone)
        self.M = from_axis_angle([0.0, 0.0, 1.0], -heading)

        # B = q_c^-1 * M^-1 * N, so the paddle is exactly neutral at calibration
        self.B = multiply(inverse(q_phone), multiply(inverse(self.M), self.neutral))
        self.ready = True
        return self

    def paddle_quat(self, q_phone):
        """Phone pose -> paddle pose in game world coordinates."""
        if not self.ready:
            return self.neutral.copy()
        return normalize(multiply(self.M, multiply(normalize(q_phone), self.B)))

    def paddle_normal(self, q_phone):
        """Paddle face normal in game world coordinates."""
        # The neutral pose already rotates Panda3D's forward (+Y) onto +X,
        # so the face normal is the node's local +Y axis.
        return rotate(self.paddle_quat(q_phone), [0.0, 1.0, 0.0])

    def phone_to_world(self, q_phone, v_body):
        """Rotate a body-frame vector (e.g. acceleration) into game world space.

        This path must go through M as well, otherwise gravity and the swing
        direction end up in different frames.
        """
        return rotate(multiply(self.M, normalize(q_phone)), v_body)
