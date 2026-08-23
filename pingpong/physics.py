"""Ball flight and collision model, NumPy vectorised.

Deliberately free of Panda3D / Bullet:

* Bullet's discrete collisions plus a plain restitution coefficient are
  too crude for a ball this light and this fast, and it cannot be run in
  batch.
* The game and the offline solver share one physics implementation, so a
  policy trained offline behaves the same as what you actually play.
  That is the prerequisite for the AI side of the project to mean anything.

Every function accepts ``(..., 3)`` shaped arrays, so a single ball
``(3,)`` and a batch ``(N, 3)`` both work.

Model
-----
Flight::

    a = g  -  k_d*|v|*v  +  k_m*(w x v)

Drag matters a lot here: the ball is 2.7 g, and at v = 10 m/s the drag
acceleration is about 11 m/s^2, more than gravity. The Magnus sign is
verified too -- topspin (ball travelling +x, w along +y) produces a -z
force, i.e. the ball dips, which matches reality.

Contact: a standard rigid-body impulse model. It first assumes the contact
point sticks and solves for the tangential impulse; if that exceeds
``mu * J_n`` it switches to sliding. Spin falls out of this naturally, with
no special-casing of topspin or backspin.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import constants as C

_EPS = 1e-12


# ------------------------------------------------------------------ helpers
def _norm(v):
    """Norm along the last axis, keeping dimensions."""
    return np.linalg.norm(v, axis=-1, keepdims=True)


def _safe_unit(v):
    n = _norm(v)
    return np.where(n > _EPS, v / np.maximum(n, _EPS), 0.0)


def _dot(a, b):
    return np.sum(a * b, axis=-1, keepdims=True)


# ------------------------------------------------------------------ flight
def acceleration(vel, spin):
    """Gravity + aerodynamic drag + Magnus force."""
    speed = _norm(vel)
    a = np.zeros_like(vel)
    a[..., 2] = -C.GRAVITY
    a = a - C.DRAG_ACCEL_K * speed * vel
    a = a + C.MAGNUS_ACCEL_K * np.cross(spin, vel)
    return a


def step_flight(pos, vel, spin, dt):
    """One RK4 integration step (flight only, no contacts).

    The ball reaches 20-30 m/s, so RK4 is what keeps the trajectory
    accurate at a sane step size; forward Euler drifts noticeably once
    drag is in the picture.
    """
    k1v = acceleration(vel, spin)
    k1p = vel

    k2v = acceleration(vel + 0.5 * dt * k1v, spin)
    k2p = vel + 0.5 * dt * k1v

    k3v = acceleration(vel + 0.5 * dt * k2v, spin)
    k3p = vel + 0.5 * dt * k2v

    k4v = acceleration(vel + dt * k3v, spin)
    k4p = vel + dt * k3v

    new_pos = pos + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
    new_vel = vel + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    # Aerodynamic spin damping is small; spin is roughly conserved per rally
    new_spin = spin * (1.0 - C.SPIN_AIR_DAMP * dt)
    return new_pos, new_vel, new_spin


# ------------------------------------------------------------------ contact
def collide(vel, spin, normal, restitution, friction, surface_vel=None):
    """Impulse solve for a ball hitting a plane (or a paddle).

    Parameters
    ----------
    vel, spin : (..., 3)
        Pre-impact linear and angular velocity, world frame.
    normal : (..., 3)
        Contact normal pointing from the surface towards the ball centre.
    restitution, friction : float or (..., 1)
        Normal restitution and tangential friction coefficients.
    surface_vel : (..., 3), optional
        Velocity of the contact surface itself -- the swinging paddle.

    Returns
    -------
    (vel_out, spin_out)
    """
    n = _safe_unit(normal)
    if surface_vel is None:
        surface_vel = np.zeros_like(vel)

    m = C.BALL_MASS
    r = C.BALL_RADIUS
    inertia = C.BALL_INERTIA

    v_rel = vel - surface_vel
    vn = _dot(v_rel, n)                     # negative means approaching

    # Contact point offset from the ball centre
    r_c = -r * n
    # Relative velocity at the contact point (spin contributes tangentially)
    v_contact = v_rel + np.cross(spin, r_c)
    v_ct = v_contact - _dot(v_contact, n) * n      # tangential part only

    # Normal impulse: delta v_n = -(1+e) * v_n
    jn = -(1.0 + restitution) * m * vn             # along +n, positive when vn<0
    jn = np.maximum(jn, 0.0)

    # Tangential impulse, first assuming a fully sticking contact
    # (contact-point tangential velocity driven to zero). From 1/m + r^2/I
    # this is J_t = -(I/(I + m r^2)) * m * v_ct, i.e. 2/5 for a hollow sphere.
    jt_stick = -C.TANGENT_IMPULSE_FACTOR * m * v_ct
    jt_mag = _norm(jt_stick)
    jt_max = friction * jn

    # Past the friction limit the contact slides and the tangential
    # impulse saturates at mu * J_n.
    sliding = jt_mag > jt_max
    jt = np.where(sliding, -jt_max * _safe_unit(v_ct), jt_stick)

    impulse = jn * n + jt
    vel_out = vel + impulse / m
    spin_out = spin + np.cross(r_c, jt) / inertia

    # Spin about the normal is not covered by the tangential model above,
    # so it is simply damped.
    spin_n = _dot(spin_out, n) * n
    spin_out = (spin_out - spin_n) + spin_n * C.SPIN_NORMAL_DAMP

    # Only apply the result to samples that were actually approaching
    approaching = vn < 0.0
    vel_out = np.where(approaching, vel_out, vel)
    spin_out = np.where(approaching, spin_out, spin)
    return vel_out, spin_out


def bounce_fraction(z_prev, z_now, vz_prev, dt, contact_z=None):
    """Fraction of a step at which the ball crossed the contact height.

    Linear interpolation between the step endpoints is only first-order
    accurate, and that error dominates everything else in the contact time.
    Near the table the ball is in near-free-fall, so z over one step is
    close to a parabola; solving that quadratic instead is exact for
    constant acceleration and costs a square root.

    Measured on a one-bounce prediction at a 1/120 step, this takes the
    reported contact time from ~12 ms out to under 1 ms -- the same accuracy
    a 1/960 step gave with linear interpolation, at an eighth of the cost.

    Works elementwise, so both single balls and batches use it.
    """
    contact_z = C.TABLE_TOP_Z + C.BALL_RADIUS if contact_z is None else contact_z
    c = z_prev - contact_z              # height above contact at the step start
    b = vz_prev * dt                    # first-order term
    a = (z_now - z_prev) - b            # = 0.5 * accel * dt**2

    # First-order fallback, used when the quadratic degenerates
    dz = z_prev - z_now
    safe_dz = np.where(np.abs(dz) > _EPS, dz, 1.0)
    s_lin = np.where(np.abs(dz) > _EPS, c / safe_dz, 0.0)

    disc = b * b - 4.0 * a * c
    ok = (np.abs(a) > _EPS) & (disc >= 0.0)
    sq = np.sqrt(np.where(disc >= 0.0, disc, 0.0))
    den = np.where(ok, 2.0 * a, 1.0)
    r1, r2 = (-b + sq) / den, (-b - sq) / den
    lo, hi = np.minimum(r1, r2), np.maximum(r1, r2)
    # f(0) > 0 and f(1) <= 0, so exactly one root lies in (0, 1]
    s_quad = np.where((lo >= 0.0) & (lo <= 1.0), lo, hi)

    good = ok & (s_quad >= 0.0) & (s_quad <= 1.0)
    return np.clip(np.where(good, s_quad, s_lin), 0.0, 1.0)


TABLE_NORMAL = np.array([0.0, 0.0, 1.0])


def bounce_table(vel, spin):
    """Bounce off the table surface (normal +Z)."""
    n = np.broadcast_to(TABLE_NORMAL, vel.shape)
    return collide(vel, spin, n, C.RESTITUTION_TABLE, C.FRICTION_TABLE)


def on_table_xy(pos):
    """Whether the XY of a position lies within the table surface."""
    return (np.abs(pos[..., 0]) <= C.TABLE_LENGTH / 2) & (
        np.abs(pos[..., 1]) <= C.TABLE_WIDTH / 2
    )


def in_net_span(pos):
    """Whether a position is inside the height and width span of the net."""
    return (
        (pos[..., 2] <= C.NET_TOP_Z + C.BALL_RADIUS)
        & (pos[..., 2] >= C.TABLE_TOP_Z)
        & (np.abs(pos[..., 1]) <= C.TABLE_WIDTH / 2 + C.NET_OVERHANG)
    )


# ------------------------------------------------------------------ paddle
@dataclass
class Paddle:
    """A paddle, possibly mid-swing.

    ``velocity`` is the key to spin: topspin is what the contact model
    produces on its own when a closed face brushes upward through the
    ball. Nothing special-cases it.
    """

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.075          # effective hitting radius (~15 cm blade)
    restitution: float = C.RESTITUTION_PADDLE
    friction: float = C.FRICTION_PADDLE

    def face_normal_toward(self, point):
        """The face normal on the side facing ``point``."""
        n = _safe_unit(np.asarray(self.normal, dtype=float))
        d = np.asarray(point, dtype=float) - self.pos
        return n if float(np.dot(d, n)) >= 0 else -n

    def signed_distance(self, point):
        """Signed distance from the paddle plane along the normal."""
        n = _safe_unit(np.asarray(self.normal, dtype=float))
        return float(np.dot(np.asarray(point, dtype=float) - self.pos, n))

    def within_face(self, point):
        """Whether the point projects inside the circular blade."""
        n = _safe_unit(np.asarray(self.normal, dtype=float))
        d = np.asarray(point, dtype=float) - self.pos
        radial = d - float(np.dot(d, n)) * n
        return float(np.linalg.norm(radial)) <= self.radius + C.BALL_RADIUS


def swept_blade_hit(ball_prev, ball_now, paddle_prev, paddle_now, normal,
                    blade_radius, ball_radius=C.BALL_RADIUS):
    """Fraction of the step at which the ball crosses the blade, or None.

    Testing "is the ball within one radius of the blade plane" once per frame
    does not work at speed: at 60 fps a 5 m/s ball moves 8 cm per frame while
    that capture band is only 2 cm wide, so the ball jumps clean across and
    registers nothing. It tunnels straight through the paddle.

    This tests the swept segment instead, and does it in the paddle's own
    frame so a fast-moving paddle is handled too. It is exact for constant
    velocity over the step, which is what the integrator gives us.
    """
    ball_prev = np.asarray(ball_prev, dtype=float)
    ball_now = np.asarray(ball_now, dtype=float)
    n = _safe_unit(np.asarray(normal, dtype=float))

    # Ball position relative to the paddle, as a line in the step parameter
    a = ball_prev - np.asarray(paddle_prev, dtype=float)
    b = (ball_now - ball_prev) - (np.asarray(paddle_now, dtype=float)
                                  - np.asarray(paddle_prev, dtype=float))

    f0 = float(np.dot(a, n))
    f1 = float(np.dot(a + b, n))

    # No crossing and never close enough to touch
    if f0 * f1 > 0.0 and min(abs(f0), abs(f1)) > ball_radius:
        return None

    denom = f0 - f1
    t = f0 / denom if abs(denom) > _EPS else 0.0
    t = float(np.clip(t, 0.0, 1.0))

    rel = a + b * t
    radial = rel - float(np.dot(rel, n)) * n
    if float(np.linalg.norm(radial)) > blade_radius + ball_radius:
        return None
    return t


def contact_report(vel, spin, normal, paddle_vel,
                   restitution=C.RESTITUTION_PADDLE, friction=C.FRICTION_PADDLE):
    """Break a paddle contact down into the quantities hardware cares about.

    The impulse model already computes all of this internally; this exposes
    it. Two numbers in particular are what you need before asking a machine
    to reproduce a stroke:

    * **required_mu** -- the rubber friction the stroke actually demands,
      ``|J_t| / J_n``. Below the rubber's real coefficient the contact grips
      and the stroke comes out as designed; at or above it the contact slips
      and the spin saturates. A stroke needing mu = 1.2 is not executable
      with rubber that only provides 0.8, whatever the arm does.
    * **drive / brush split** -- the paddle velocity resolved along the blade
      normal and across it. Driving along the normal buys speed; brushing
      across the face buys spin. This is the whole of stroke technique in two
      numbers.
    """
    v = np.asarray(vel, dtype=float)
    w = np.asarray(spin, dtype=float)
    n = _safe_unit(np.asarray(normal, dtype=float))
    u = np.asarray(paddle_vel, dtype=float)

    m, r, inertia = C.BALL_MASS, C.BALL_RADIUS, C.BALL_INERTIA
    v_rel = v - u
    vn = float(np.dot(v_rel, n))
    r_c = -r * n
    v_contact = v_rel + np.cross(w, r_c)
    v_ct = v_contact - float(np.dot(v_contact, n)) * n

    jn = max(-(1.0 + restitution) * m * vn, 0.0)
    jt_stick = -C.TANGENT_IMPULSE_FACTOR * m * v_ct
    jt_mag = float(np.linalg.norm(jt_stick))
    required_mu = jt_mag / jn if jn > _EPS else float("inf")
    sliding = required_mu > friction
    jt = (-friction * jn * _safe_unit(v_ct)) if sliding else jt_stick

    drive = float(np.dot(u, n))                  # along the blade normal
    brush_vec = u - drive * n
    brush = float(np.linalg.norm(brush_vec))

    speed_in = float(np.linalg.norm(v))
    # How square the blade is to the incoming ball: 0 deg means the face is
    # dead-on, 90 deg means a pure graze
    cos_attack = float(np.dot(n, -v / speed_in)) if speed_in > _EPS else 1.0
    attack_deg = math.degrees(math.acos(float(np.clip(cos_attack, -1.0, 1.0))))
    # Blade tilt: 0 deg is a vertical face, positive is closed (top forward)
    tilt_deg = math.degrees(math.asin(float(np.clip(n[2], -1.0, 1.0))))

    return {
        "blade_normal": n,
        "blade_tilt_deg": tilt_deg,
        "angle_of_attack_deg": attack_deg,
        "paddle_velocity": u,
        "paddle_speed": float(np.linalg.norm(u)),
        "drive_speed": drive,
        "brush_speed": brush,
        "brush_direction": _safe_unit(brush_vec),
        "contact_slip_speed": float(np.linalg.norm(v_ct)),
        "normal_impulse_Ns": jn,
        "tangential_impulse_Ns": float(np.linalg.norm(jt)),
        "required_mu": required_mu,
        "sliding": bool(sliding),
        "assumed_mu": friction,
        "assumed_restitution": restitution,
        "peak_force_estimate_N": jn / 0.001,     # ~1 ms contact
    }


def hit_with_paddle(vel, spin, paddle: Paddle, ball_pos):
    """Strike the ball with a paddle, returning (vel_out, spin_out).

    Both the incoming velocity and the incoming spin feed into the result.
    That is precisely what the old version was missing: it set the outgoing
    velocity to ``n * power`` and discarded the incoming ball entirely,
    which is not physics at all.
    """
    n = paddle.face_normal_toward(ball_pos)
    return collide(
        np.asarray(vel, dtype=float),
        np.asarray(spin, dtype=float),
        n,
        paddle.restitution,
        paddle.friction,
        surface_vel=np.asarray(paddle.velocity, dtype=float),
    )


# ------------------------------------------------------------------ sim
LAND_OWN = "own"
LAND_OPPONENT = "opponent"
OUT_SIDE = "out"
OUT_NET = "net"
OUT_FLOOR = "floor"
TIMEOUT = "timeout"


@dataclass
class SimResult:
    outcome: str
    landing: Optional[np.ndarray]
    time: float
    trajectory: Optional[np.ndarray]
    bounces: int


def simulate(
    pos,
    vel,
    spin,
    dt=C.DEFAULT_DT,
    max_time=C.MAX_SIM_TIME,
    stop_on_bounce=True,
    record_trajectory=False,
    record_stride=8,
):
    """Simulate one ball until its first table contact (or out / net).

    With ``stop_on_bounce=True`` it halts at the first bounce and reports
    the landing point, which is what the inverse "where would this land"
    problem needs.
    """
    pos = np.array(pos, dtype=float)
    vel = np.array(vel, dtype=float)
    spin = np.array(spin, dtype=float)

    traj = [pos.copy()] if record_trajectory else None
    t = 0.0
    bounces = 0
    steps = 0
    contact_z = C.TABLE_TOP_Z + C.BALL_RADIUS

    while t < max_time:
        prev_pos = pos.copy()
        prev_vel = vel.copy()
        prev_x = pos[0]
        pos, vel, spin = step_flight(pos, vel, spin, dt)
        t += dt
        steps += 1

        if record_trajectory and steps % record_stride == 0:
            traj.append(pos.copy())

        # ---- net: did we cross the net plane? ----
        if prev_x * pos[0] < 0 and in_net_span(pos):
            return SimResult(OUT_NET, None, t, _finish_traj(traj, pos), bounces)

        # ---- table contact ----
        if pos[2] <= contact_z and vel[2] < 0:
            # Back up to the contact instant, otherwise the discrete step
            # shows up directly as landing-point error.
            frac = float(bounce_fraction(prev_pos[2], pos[2], prev_vel[2],
                                         dt, contact_z))
            contact = prev_pos + (pos - prev_pos) * frac

            if on_table_xy(contact):
                bounces += 1
                landing = contact.copy()
                if stop_on_bounce:
                    # The clock over-counted the part of the step after contact
                    return SimResult(
                        LAND_OWN if landing[0] < 0 else LAND_OPPONENT,
                        landing, t - dt * (1.0 - frac),
                        _finish_traj(traj, landing), bounces,
                    )
                pos = contact
                # Collide with the velocity *at contact*, not at the step end
                vel, spin = bounce_table(prev_vel + (vel - prev_vel) * frac, spin)
                pos = pos + np.array([0.0, 0.0, 1e-4])
                # Fly out the rest of the step. Skipping this loses up to one
                # step of flight at every bounce, which is what made the
                # predicted contact time drift by a whole dt.
                rem = dt * (1.0 - frac)
                if rem > 1e-9:
                    pos, vel, spin = step_flight(pos, vel, spin, rem)
            else:
                # Reached table height but not over the table -> out
                return SimResult(
                    OUT_SIDE, contact, t, _finish_traj(traj, contact), bounces
                )

        # ---- flew too far / hit the floor ----
        if pos[2] < C.FLOOR_Z:
            return SimResult(OUT_FLOOR, None, t, _finish_traj(traj, pos), bounces)
        if (
            abs(pos[0]) > C.TABLE_LENGTH / 2 + C.OUT_MARGIN_XY
            or abs(pos[1]) > C.TABLE_WIDTH / 2 + C.OUT_MARGIN_XY
        ):
            return SimResult(OUT_SIDE, None, t, _finish_traj(traj, pos), bounces)

    return SimResult(TIMEOUT, None, t, _finish_traj(traj, pos), bounces)


def _finish_traj(traj, last):
    if traj is None:
        return None
    traj.append(np.asarray(last, dtype=float).copy())
    return np.array(traj)


@dataclass
class PlaneCrossing:
    """Where and when the ball reaches a given x plane, and its state there."""

    pos: np.ndarray
    vel: np.ndarray
    spin: np.ndarray
    time: float          # seconds from now until it gets there


PREDICT_DT = 1.0 / 120.0


def predict_plane_crossing(pos, vel, spin, x_plane, max_time=2.5,
                           dt=PREDICT_DT):
    """Roll the ball forward until it reaches ``x_plane``.

    Returns the ball's full state there, not just the position: an inverse
    solve needs the incoming velocity and spin **at contact**, and the swing
    controller needs the arrival time to schedule its arc.

    Table bounces along the way are included -- leaving them out throws the
    prediction off completely.

    ``dt`` defaults to a coarser step than the simulator's. At the full
    1/480 this costs about 34 ms per call, which on its own would blow the
    16.7 ms frame budget at 60 fps; the crossing itself is interpolated, so
    the coarser step costs accuracy in the trajectory shape rather than in
    the reported contact point or time.
    """
    pos = np.array(pos, dtype=float)
    vel = np.array(vel, dtype=float)
    spin = np.array(spin, dtype=float)
    if pos[0] <= x_plane:
        return None

    contact_z = C.TABLE_TOP_Z + C.BALL_RADIUS
    t = 0.0
    for _ in range(int(max_time / dt)):
        prev = pos.copy()
        prev_vel = vel.copy()
        pos, vel, spin = step_flight(pos, vel, spin, dt)
        t += dt

        if pos[2] <= contact_z and vel[2] < 0 and on_table_xy(pos):
            frac = float(bounce_fraction(prev[2], pos[2], prev_vel[2], dt, contact_z))
            pos = prev + (pos - prev) * frac
            vel, spin = bounce_table(prev_vel + (vel - prev_vel) * frac, spin)
            pos[2] = contact_z + 1e-4
            rem = dt * (1.0 - frac)      # rest of the step, after the bounce
            if rem > 1e-9:
                pos, vel, spin = step_flight(pos, vel, spin, rem)

        if pos[0] <= x_plane:
            # Interpolate back to the plane, so the contact point and time
            # stay accurate even at a coarse step
            dx = prev[0] - pos[0]
            frac = (prev[0] - x_plane) / dx if dx > _EPS else 1.0
            frac = float(np.clip(frac, 0.0, 1.0))
            cross = prev + (pos - prev) * frac
            return PlaneCrossing(cross, vel, spin, t - dt * (1.0 - frac))
        if pos[2] < C.TABLE_H - 0.6:      # already on the floor
            return None
    return None


def simulate_batch(pos, vel, spin, dt=C.DEFAULT_DT, max_time=C.MAX_SIM_TIME):
    """Simulate N balls to their first table contact, fully vectorised.

    This is the workhorse for generating training data later: no window,
    no 60 fps rallies, just thousands of trajectories at once.

    Returns
    -------
    landing : (N, 3)   landing points (nan where the ball never landed)
    outcome : (N,) int 0=own half 1=opponent half 2=out 3=net 4=timeout
    land_vel, land_spin : (N, 3)
        Velocity and spin as the ball arrives at the table. What the
        opponent actually receives, so these are the controllable
        properties of a shot beyond where it lands.
    """
    pos = np.array(pos, dtype=float)
    vel = np.array(vel, dtype=float)
    spin = np.array(spin, dtype=float)
    n = pos.shape[0]

    landing = np.full((n, 3), np.nan)
    land_vel = np.full((n, 3), np.nan)
    land_spin = np.full((n, 3), np.nan)
    outcome = np.full(n, 4, dtype=np.int8)
    active = np.ones(n, dtype=bool)
    contact_z = C.TABLE_TOP_Z + C.BALL_RADIUS

    n_steps = int(max_time / dt)
    for _ in range(n_steps):
        if not active.any():
            break

        prev_pos = pos.copy()
        prev_vel = vel.copy()
        new_pos, new_vel, new_spin = step_flight(pos, vel, spin, dt)
        pos = np.where(active[:, None], new_pos, pos)
        vel = np.where(active[:, None], new_vel, vel)
        spin = np.where(active[:, None], new_spin, spin)

        # net
        crossed = (prev_pos[:, 0] * pos[:, 0] < 0) & in_net_span(pos) & active
        outcome[crossed] = 3
        active &= ~crossed

        # table contact
        hit = (pos[:, 2] <= contact_z) & (vel[:, 2] < 0) & active
        if hit.any():
            frac = bounce_fraction(prev_pos[:, 2], pos[:, 2], prev_vel[:, 2],
                                   dt, contact_z)[:, None]
            contact = prev_pos + (pos - prev_pos) * frac

            # Velocity at contact, not at the end of the step
            v_at = prev_vel + (vel - prev_vel) * frac

            on_tbl = on_table_xy(contact) & hit
            landing[on_tbl] = contact[on_tbl]
            land_vel[on_tbl] = v_at[on_tbl]
            land_spin[on_tbl] = spin[on_tbl]
            outcome[on_tbl] = np.where(contact[on_tbl][:, 0] < 0, 0, 1)

            off_tbl = hit & ~on_table_xy(contact)
            landing[off_tbl] = contact[off_tbl]
            land_vel[off_tbl] = v_at[off_tbl]
            land_spin[off_tbl] = spin[off_tbl]
            outcome[off_tbl] = 2

            active &= ~hit

        # left the play area
        gone = active & (
            (np.abs(pos[:, 0]) > C.TABLE_LENGTH / 2 + C.OUT_MARGIN_XY)
            | (np.abs(pos[:, 1]) > C.TABLE_WIDTH / 2 + C.OUT_MARGIN_XY)
            | (pos[:, 2] < C.FLOOR_Z)
        )
        outcome[gone] = 2
        active &= ~gone

    return landing, outcome, land_vel, land_spin


def propagate_to_plane_batch(pos, vel, spin, x_plane, dt=PREDICT_DT,
                             max_time=2.5):
    """Advance N balls until each reaches ``x_plane``, fully vectorised.

    The per-ball :func:`predict_plane_crossing` costs milliseconds, which is
    fine once a frame and hopeless for generating a training set -- 200k
    samples that way runs for hours. This does the same job for a whole
    batch at once, bounces included.

    Returns (pos, vel, spin, time, reached, bounced). ``reached`` masks the
    balls that actually got there; ``bounced`` marks those that landed on our
    half on the way, which is what makes a serve legal and hittable.
    """
    pos = np.array(pos, dtype=float)
    vel = np.array(vel, dtype=float)
    spin = np.array(spin, dtype=float)
    n = pos.shape[0]

    out_pos = np.full_like(pos, np.nan)
    out_vel = np.full_like(vel, np.nan)
    out_spin = np.full_like(spin, np.nan)
    out_t = np.full(n, np.nan)
    reached = np.zeros(n, dtype=bool)
    bounced = np.zeros(n, dtype=bool)     # did it land on our half on the way
    active = pos[:, 0] > x_plane

    contact_z = C.TABLE_TOP_Z + C.BALL_RADIUS
    t = 0.0
    for _ in range(int(max_time / dt)):
        if not active.any():
            break
        prev_pos = pos.copy()
        prev_vel = vel.copy()
        new_pos, new_vel, new_spin = step_flight(pos, vel, spin, dt)
        pos = np.where(active[:, None], new_pos, pos)
        vel = np.where(active[:, None], new_vel, vel)
        spin = np.where(active[:, None], new_spin, spin)
        t += dt

        # table bounce along the way
        hit = (pos[:, 2] <= contact_z) & (vel[:, 2] < 0) & on_table_xy(pos) & active
        if hit.any():
            frac = bounce_fraction(prev_pos[:, 2], pos[:, 2], prev_vel[:, 2],
                                   dt, contact_z)[:, None]
            contact = prev_pos + (pos - prev_pos) * frac
            bv, bs = bounce_table(prev_vel + (vel - prev_vel) * frac, spin)
            # finish the rest of the step after the bounce
            rem = dt * (1.0 - frac)
            cpos = contact.copy()
            cpos[:, 2] = contact_z + 1e-4
            rp, rv, rs = step_flight(cpos, bv, bs, rem)
            pos = np.where(hit[:, None], rp, pos)
            vel = np.where(hit[:, None], rv, vel)
            spin = np.where(hit[:, None], rs, spin)
            bounced |= hit & (contact[:, 0] < 0.0)

        # reached the plane
        crossed = active & (pos[:, 0] <= x_plane)
        if crossed.any():
            dx = prev_pos[:, 0] - pos[:, 0]
            f = np.where(dx > _EPS, (prev_pos[:, 0] - x_plane) / np.maximum(dx, _EPS), 1.0)
            f = np.clip(f, 0.0, 1.0)[:, None]
            cross_pos = prev_pos + (pos - prev_pos) * f
            out_pos[crossed] = cross_pos[crossed]
            out_vel[crossed] = vel[crossed]
            out_spin[crossed] = spin[crossed]
            out_t[crossed] = t - dt * (1.0 - f[crossed, 0])
            reached |= crossed
            active &= ~crossed

        # gone: on the floor or off the sides
        dead = active & ((pos[:, 2] < C.TABLE_H - 0.6) | (pos[:, 2] < C.FLOOR_Z)
                         | (np.abs(pos[:, 1]) > C.TABLE_WIDTH / 2 + C.OUT_MARGIN_XY))
        active &= ~dead

    return out_pos, out_vel, out_spin, out_t, reached, bounced


# ------------------------------------------------------------------ live
class BallWorld:
    """Ball state for the game loop: steps per frame and reports events.

    It shares the same flight and contact functions as :func:`simulate`,
    so the trajectory you see in game is identical to the one the AI
    predicts.
    """

    def __init__(self):
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.spin = np.zeros(3)
        self.active = False
        self.bounces = 0
        self.last_event = None
        # Which half each bounce landed on, used for scoring
        self.bounce_side = []

    def reset(self, pos, vel, spin=None):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.spin = np.zeros(3) if spin is None else np.array(spin, dtype=float)
        self.active = True
        self.bounces = 0
        self.last_event = None
        self.bounce_side = []

    def advance(self, dt, substeps=None):
        """Advance dt seconds, sub-stepping internally for contact accuracy.

        Returns the events that happened during the interval, e.g.
        ``[("bounce", pos), ("net", pos), ("out", pos)]``.
        """
        events = []
        if not self.active:
            return events

        if substeps is None:
            substeps = max(1, int(np.ceil(dt / C.DEFAULT_DT)))
        h = dt / substeps
        contact_z = C.TABLE_TOP_Z + C.BALL_RADIUS

        for _ in range(substeps):
            prev = self.pos.copy()
            prev_vel = self.vel.copy()
            self.pos, self.vel, self.spin = step_flight(self.pos, self.vel, self.spin, h)

            # net
            if prev[0] * self.pos[0] < 0 and in_net_span(self.pos):
                self.vel[0] *= -C.RESTITUTION_NET
                self.vel[1] *= C.NET_VELOCITY_DAMP
                self.vel[2] *= C.NET_VELOCITY_DAMP
                self.pos[0] = np.sign(prev[0]) * 1e-3
                events.append(("net", self.pos.copy()))

            # table contact
            if self.pos[2] <= contact_z and self.vel[2] < 0:
                frac = float(bounce_fraction(prev[2], self.pos[2], prev_vel[2],
                                             h, contact_z))
                contact = prev + (self.pos - prev) * frac

                if on_table_xy(contact):
                    self.pos = contact.copy()
                    self.vel, self.spin = bounce_table(
                        prev_vel + (self.vel - prev_vel) * frac, self.spin
                    )
                    self.pos[2] = contact_z + 1e-4
                    rem = h * (1.0 - frac)     # rest of the substep
                    if rem > 1e-9:
                        self.pos, self.vel, self.spin = step_flight(
                            self.pos, self.vel, self.spin, rem
                        )
                    self.bounces += 1
                    self.bounce_side.append("own" if contact[0] < 0 else "opponent")
                    events.append(("bounce", contact.copy()))
                else:
                    self.active = False
                    events.append(("out", contact.copy()))
                    break

            # out of play
            if (
                self.pos[2] < C.FLOOR_Z
                or abs(self.pos[0]) > C.TABLE_LENGTH / 2 + C.OUT_MARGIN_XY
                or abs(self.pos[1]) > C.TABLE_WIDTH / 2 + C.OUT_MARGIN_XY
            ):
                self.active = False
                events.append(("out", self.pos.copy()))
                break

        return events

    def predict(self, max_time=2.0, record=True):
        """Predict the trajectory from the current state (aiming aid / AI)."""
        return simulate(
            self.pos, self.vel, self.spin,
            max_time=max_time,
            record_trajectory=record,
        )
