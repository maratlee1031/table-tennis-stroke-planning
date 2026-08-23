"""Verification suite for the physics and orientation modules.

    python verify_physics.py

No Panda3D required, pure NumPy. Every check prints the actual numbers so
you can see the model behaving like real table tennis, not merely running.
"""

import math
import time

import numpy as np

from pingpong import constants as C
from pingpong import physics, quat

PASS, FAIL = "  [OK]  ", "  [FAIL]"
_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"{PASS if ok else FAIL} {name}")
    if detail:
        print(f"          {detail}")


def section(title):
    print(f"\n=== {title} ===")


# ------------------------------------------------------------------ constants
section("Constant sanity")

check(
    "hollow-sphere inertia factor is 2/3",
    abs(C.BALL_INERTIA_FACTOR - 2 / 3) < 1e-12,
    f"I = {C.BALL_INERTIA:.3e} kg m^2",
)
check(
    "sticking tangential impulse factor is 2/5 (hollow sphere)",
    abs(C.TANGENT_IMPULSE_FACTOR - 0.4) < 1e-12,
    f"a solid sphere would be 2/7 ~ 0.2857; here {C.TANGENT_IMPULSE_FACTOR:.4f}",
)

drag_at_10 = C.DRAG_ACCEL_K * 10.0 ** 2
check(
    "drag acceleration at 10 m/s exceeds gravity",
    drag_at_10 > C.GRAVITY,
    f"a_drag = {drag_at_10:.2f} m/s^2  vs  g = {C.GRAVITY} m/s^2  "
    f"(k_d = {C.DRAG_ACCEL_K:.4f}, literature 0.10-0.14)",
)
check(
    "Magnus coefficient within the literature range 0.003-0.007",
    0.003 < C.MAGNUS_ACCEL_K < 0.007,
    f"k_m = {C.MAGNUS_ACCEL_K:.5f}",
)


# ------------------------------------------------------------------ flight
section("Flight: drag and Magnus")

# Pure drag: a horizontal launch should keep decelerating
pos = np.array([0.0, 0.0, 1.0])
vel = np.array([10.0, 0.0, 0.0])
spin = np.zeros(3)
p1, v1, _ = physics.step_flight(pos, vel, spin, 0.01)
check(
    "drag reduces horizontal speed",
    v1[0] < vel[0],
    f"10.00 -> {v1[0]:.3f} m/s over 0.01 s",
)

# Topspin pushes down, backspin lifts
v0 = np.array([10.0, 0.0, 0.0])
top = np.array([0.0, 200.0, 0.0])     # w along +y with travel +x -> topspin
back = np.array([0.0, -200.0, 0.0])   # backspin
a_top = physics.acceleration(v0, top)
a_back = physics.acceleration(v0, back)
a_none = physics.acceleration(v0, np.zeros(3))
check(
    "topspin produces a downward Magnus force",
    a_top[2] < a_none[2],
    f"a_z: no spin {a_none[2]:.2f} -> topspin {a_top[2]:.2f} m/s^2",
)
check(
    "backspin produces an upward Magnus force",
    a_back[2] > a_none[2],
    f"a_z: no spin {a_none[2]:.2f} -> backspin {a_back[2]:.2f} m/s^2",
)

# Landing: topspin should come down sooner than no spin
start = np.array([-1.0, 0.0, C.TABLE_H + 0.3])
launch = np.array([8.0, 0.0, 0.5])
r_top = physics.simulate(start, launch, top)
r_none = physics.simulate(start, launch, np.zeros(3))
r_back = physics.simulate(start, launch, back)
ok_order = (
    r_top.landing is not None
    and r_none.landing is not None
    and r_top.landing[0] < r_none.landing[0]
)
check(
    "topspin lands shorter than no spin (the ball dives)",
    ok_order,
    f"topspin x={r_top.landing[0]:.3f}  no spin x={r_none.landing[0]:.3f}"
    + (f"  backspin x={r_back.landing[0]:.3f}" if r_back.landing is not None
       else "  backspin carried off the table"),
)


# ------------------------------------------------------------------ bounce
section("Bounce: ITTF drop test")

# ITTF: released from 30 cm above the surface, rebound must be 24-26 cm
drop_h = 0.30
pos = np.array([-0.5, 0.0, C.TABLE_TOP_Z + C.BALL_RADIUS + drop_h])
res = physics.simulate(
    pos, np.zeros(3), np.zeros(3),
    stop_on_bounce=False, record_trajectory=True, record_stride=1,
    max_time=2.0,
)
traj = res.trajectory
# Highest point after the first contact
contact_z = C.TABLE_TOP_Z + C.BALL_RADIUS
zs = traj[:, 2]
touch_idx = int(np.argmax(zs <= contact_z + 1e-6))
apex = float(zs[touch_idx:].max()) - contact_z
check(
    "ITTF drop rebound is 24-26 cm",
    0.235 <= apex <= 0.265,
    f"dropped from 30.0 cm -> rebounds {apex * 100:.1f} cm (e={C.RESTITUTION_TABLE})",
)

# A sliding ball should pick up topspin from table friction
vel_in = np.array([5.0, 0.0, -2.0])
spin_in = np.zeros(3)
v_out, w_out = physics.bounce_table(vel_in, spin_in)
check(
    "a spinless ball gains topspin off the table (friction-spin coupling)",
    w_out[1] > 0,
    f"w_y: 0 -> {w_out[1]:.1f} rad/s, while v_x {vel_in[0]:.2f} -> {v_out[0]:.2f}",
)
check(
    "normal velocity after the bounce matches the restitution",
    abs(v_out[2] - (-C.RESTITUTION_TABLE * vel_in[2])) < 1e-9,
    f"v_z: {vel_in[2]:.2f} -> {v_out[2]:.2f} (expected {-C.RESTITUTION_TABLE * vel_in[2]:.2f})",
)

# The friction model has a sliding and a sticking branch; check both.
# Sticking threshold: TANGENT_FACTOR*|v_ct| <= mu(1+e)|v_n|
#   -> 0.4*v_x <= 0.25*1.93*|v_z|, i.e. v_x <~ 1.21*|v_z|
# Steep enough impact (small v_x, large |v_z|) -> sticks, contact point
# velocity goes to zero, so the ball rolls.
steep_in = np.array([2.0, 0.0, -5.0])
v_s, w_s = physics.bounce_table(steep_in, np.zeros(3))
check(
    "steep impact sticks and satisfies the rolling condition v_x' = r*w_y'",
    abs(v_s[0] - C.BALL_RADIUS * w_s[1]) < 1e-9,
    f"v_x'={v_s[0]:.4f}  r*w_y'={C.BALL_RADIUS * w_s[1]:.4f}",
)
# Shallow impact -> slides, tangential impulse saturates at mu*J_n
jn_flat = (1 + C.RESTITUTION_TABLE) * C.BALL_MASS * abs(vel_in[2])
dvx_expected = -C.FRICTION_TABLE * jn_flat / C.BALL_MASS
check(
    "shallow impact slides, tangential impulse saturates at mu*J_n",
    abs((v_out[0] - vel_in[0]) - dvx_expected) < 1e-9,
    f"dv_x = {v_out[0] - vel_in[0]:.4f} (expected {dvx_expected:.4f}); still "
    f"sliding, so v_x'={v_out[0]:.3f} != r*w_y'={C.BALL_RADIUS * w_out[1]:.3f}",
)


# ------------------------------------------------------------------ paddle
section("Paddle: the incoming ball must reach the result")

paddle = physics.Paddle(
    pos=np.array([-1.2, 0.0, C.TABLE_H + 0.25]),
    normal=np.array([1.0, 0.0, 0.0]),
    velocity=np.zeros(3),
)
# The ball arrives from the far side (+x) heading towards us (-x), so it
# sits on the +x side of the blade
ball_pos = paddle.pos + np.array([0.03, 0.0, 0.0])

slow, _ = physics.hit_with_paddle(np.array([-3.0, 0, 0]), np.zeros(3), paddle, ball_pos)
fast, _ = physics.hit_with_paddle(np.array([-9.0, 0, 0]), np.zeros(3), paddle, ball_pos)
check(
    "the return travels back towards the far side (+x)",
    slow[0] > 0 and fast[0] > 0,
    f"incoming -3 -> return {slow[0]:+.2f} m/s; incoming -9 -> return {fast[0]:+.2f} m/s",
)
check(
    "a faster incoming ball returns faster (the old code discarded this)",
    fast[0] > slow[0] + 3.0,
    f"incoming 3 m/s -> return {slow[0]:.2f} m/s; incoming 9 m/s -> return {fast[0]:.2f} m/s",
)

# Brushing upward should generate topspin
brushing = physics.Paddle(
    pos=paddle.pos,
    normal=np.array([1.0, 0.0, 0.3]),
    velocity=np.array([2.0, 0.0, 6.0]),   # forward and up
)
_, w_brush = physics.hit_with_paddle(
    np.array([-6.0, 0.0, -1.0]), np.zeros(3), brushing, ball_pos
)
check(
    "closed face brushing upward produces topspin",
    w_brush[1] > 0,
    f"w_y = {w_brush[1]:.1f} rad/s (no special case; it falls out of friction)",
)

# Chopping down should generate backspin
chopping = physics.Paddle(
    pos=paddle.pos,
    normal=np.array([1.0, 0.0, 0.5]),
    velocity=np.array([1.0, 0.0, -5.0]),  # forward and down
)
_, w_chop = physics.hit_with_paddle(
    np.array([-6.0, 0.0, -1.0]), np.zeros(3), chopping, ball_pos
)
check(
    "open face chopping downward produces backspin",
    w_chop[1] < 0,
    f"w_y = {w_chop[1]:.1f} rad/s",
)

# Still blade vs swinging blade
still = physics.Paddle(pos=paddle.pos, normal=np.array([1.0, 0, 0]), velocity=np.zeros(3))
swung = physics.Paddle(pos=paddle.pos, normal=np.array([1.0, 0, 0]), velocity=np.array([8.0, 0, 0]))
v_still, _ = physics.hit_with_paddle(np.array([-6.0, 0, 0]), np.zeros(3), still, ball_pos)
v_swung, _ = physics.hit_with_paddle(np.array([-6.0, 0, 0]), np.zeros(3), swung, ball_pos)
check(
    "swing speed converts into ball speed",
    v_swung[0] > v_still[0] + 10.0,
    f"still blade {v_still[0]:.2f} m/s -> swinging at 8 m/s gives {v_swung[0]:.2f} m/s",
)

# Incoming spin has to change the outcome
v_no, _ = physics.hit_with_paddle(
    np.array([-7.0, 0, -1.0]), np.zeros(3), paddle, ball_pos
)
v_top, _ = physics.hit_with_paddle(
    np.array([-7.0, 0, -1.0]), np.array([0.0, 250.0, 0.0]), paddle, ball_pos
)
check(
    "incoming spin changes the return (the blade reads the spin)",
    float(np.linalg.norm(v_top - v_no)) > 0.3,
    f"spinless incoming v_out=({v_no[0]:.2f},{v_no[1]:.2f},{v_no[2]:.2f})  "
    f"topspin incoming v_out=({v_top[0]:.2f},{v_top[1]:.2f},{v_top[2]:.2f})",
)


# ------------------------------------------------------------------ batch
section("Batch simulation (training-data generator)")

N = 4000
rng = np.random.default_rng(0)
bpos = np.tile(np.array([-1.2, 0.0, C.TABLE_H + 0.25]), (N, 1))
bvel = np.stack([
    rng.uniform(4.0, 12.0, N),
    rng.uniform(-2.0, 2.0, N),
    rng.uniform(0.5, 3.5, N),
], axis=1)
bspin = rng.uniform(-150, 150, (N, 3))

t0 = time.perf_counter()
landing, outcome, land_vel, land_spin = physics.simulate_batch(bpos, bvel, bspin)
elapsed = time.perf_counter() - t0

n_opp = int((outcome == 1).sum())
check(
    "a reasonable share of the batch lands on the opponent's half",
    n_opp > N * 0.15,
    f"{n_opp}/{N} opponent, {(outcome == 0).sum()} own half, "
    f"{(outcome == 2).sum()} out, {(outcome == 3).sum()} net",
)
check(
    "throughput is enough to generate training data",
    elapsed < 30.0,
    f"{N} trajectories in {elapsed:.2f} s -> {N / elapsed:,.0f} per second",
)

# Batch and single-ball results must agree
i = int(np.argmax(outcome == 1))
single = physics.simulate(bpos[i], bvel[i], bspin[i])
diff = float(np.linalg.norm(single.landing - landing[i])) if single.landing is not None else 9e9
check(
    "batch and single-ball simulation agree",
    diff < 1e-6,
    f"landing difference {diff:.2e} m",
)


# ------------------------------------------------------------------ quaternion
section("Quaternion: W3C DeviceOrientation conversion")


def rot_matrix(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


worst = 0.0
for alpha, beta, gamma in [
    (0, 0, 0), (30, 0, 0), (0, 45, 0), (0, 0, 60),
    (30, 45, 60), (120, -30, 20), (270, 80, -75), (15, 90, 45),
]:
    q = quat.from_device_orientation(alpha, beta, gamma)
    # Spec: R = Rz(alpha) * Rx(beta) * Ry(gamma)
    expect = rot_matrix("z", alpha) @ rot_matrix("x", beta) @ rot_matrix("y", gamma)
    worst = max(worst, float(np.abs(quat.to_matrix(q) - expect).max()))
check(
    "conversion equals Rz(alpha)*Rx(beta)*Ry(gamma)",
    worst < 1e-10,
    f"worst matrix error over 8 angle sets: {worst:.2e} "
    f"(the old code used ZYX with beta/gamma swapped)",
)


# The old, incorrect conversion, kept to show the size of the difference
def old_euler_to_quat(alpha, beta, gamma):
    d2r = math.pi / 180
    yaw, pitch, roll = alpha * d2r, beta * d2r, gamma * d2r
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


err = math.degrees(quat.angle_between(
    quat.from_device_orientation(30, 45, 60), old_euler_to_quat(30, 45, 60)
))
check(
    "the old conversion really was far off (diagnostic evidence)",
    err > 10.0,
    f"at (alpha,beta,gamma)=(30,45,60) old and new differ by {err:.1f} deg",
)


# ------------------------------------------------------------------ calibration
section("Calibration: world rotation -> game rotation axis mapping")

# A "phone held sideways" calibration pose, i.e. with a lot of roll
q_calib = quat.multiply(
    quat.from_axis_angle([0, 0, 1], math.radians(50)),    # facing some heading
    quat.from_axis_angle([0, 1, 0], math.radians(90)),    # turned on its side
)
cal = quat.Calibration().calibrate(q_calib)

check(
    "the paddle is exactly neutral at calibration",
    quat.angle_between(cal.paddle_quat(q_calib), quat.NEUTRAL_PADDLE) < 1e-9,
    f"error {math.degrees(quat.angle_between(cal.paddle_quat(q_calib), quat.NEUTRAL_PADDLE)):.2e} deg",
)


# The key test: rotating about any world axis must rotate the paddle about
# the corresponding game axis by the same amount
def axis_error(world_axis, deg):
    dR = quat.from_axis_angle(world_axis, math.radians(deg))
    q_new = quat.multiply(dR, q_calib)          # rotated by dR in world space
    got = cal.paddle_quat(q_new)
    # M is a pure yaw, so world horizontal axes map onto game horizontal
    # axes; the expectation is M*dR*M^-1 applied to the neutral pose
    expect = quat.multiply(
        quat.multiply(cal.M, quat.multiply(dR, quat.inverse(cal.M))),
        quat.NEUTRAL_PADDLE,
    )
    return math.degrees(quat.angle_between(got, expect))


max_err = max(
    axis_error([0, 0, 1], 40),
    axis_error([1, 0, 0], 35),
    axis_error([0, 1, 0], 25),
    axis_error([1, 1, 0], 30),
)
check(
    "every world axis maps correctly (not just the vertical one)",
    max_err < 1e-9,
    f"worst deviation {max_err:.2e} deg",
)

# The rotation magnitude must be preserved -- this is what the old
# conjugation broke
dR = quat.from_axis_angle([1, 0, 0], math.radians(35))
q_new = quat.multiply(dR, q_calib)
got_angle = math.degrees(
    quat.angle_between(cal.paddle_quat(q_new), quat.NEUTRAL_PADDLE)
)
check(
    "rotating 35 deg rotates the paddle exactly 35 deg",
    abs(got_angle - 35.0) < 1e-6,
    f"actual {got_angle:.6f} deg",
)

old_rel = quat.multiply(quat.inverse(q_calib), q_new)   # old: q_c^-1 * q
new_rel = quat.world_delta(q_new, q_calib)              # correct: q * q_c^-1
check(
    "world_delta recovers the original world-frame delta",
    quat.angle_between(new_rel, dR) < 1e-9,
    f"error {math.degrees(quat.angle_between(new_rel, dR)):.2e} deg; the old "
    f"left-multiply is off by {math.degrees(quat.angle_between(old_rel, dR)):.1f} deg",
)

# The neutral face normal should point at the opponent
n = cal.paddle_normal(q_calib)
check(
    "the neutral face normal points at the opponent (+X)",
    n[0] > 0.999,
    f"n = ({n[0]:.4f}, {n[1]:.4f}, {n[2]:.4f})",
)


# ------------------------------------------------------------------ result
section("Result")
total, passed = len(_results), sum(_results)
print(f"\n{passed}/{total} checks passed")
if passed != total:
    raise SystemExit(1)
print("All passed.")
