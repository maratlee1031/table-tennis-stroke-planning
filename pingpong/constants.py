"""Physical constants and table geometry.

Coordinate frame (same as the original main_wss.py, Panda3D Z-up):
    +X : from our side towards the opponent (table length)
    +Y : across the table (table width)
    +Z : up
    Table surface sits at z = TABLE_H, the net is the plane x = 0.
    Our half is x < 0, the opponent's half is x > 0.
"""

import math

# ---------------------------------------------------------------- table
TABLE_LENGTH = 2.74      # ITTF standard, metres
TABLE_WIDTH = 1.525
TABLE_HEIGHT = 0.76      # surface height above the floor
NET_HEIGHT = 0.1525      # measured up from the surface
NET_OVERHANG = 0.1525    # how far the net sticks out past each side

# Short aliases (kept for compatibility with the older code)
TABLE_L = TABLE_LENGTH
TABLE_W = TABLE_WIDTH
TABLE_H = TABLE_HEIGHT
NET_H = NET_HEIGHT

TABLE_TOP_Z = TABLE_HEIGHT
NET_TOP_Z = TABLE_HEIGHT + NET_HEIGHT

# ---------------------------------------------------------------- ball
BALL_RADIUS = 0.02       # 40 mm ball
BALL_MASS = 0.0027       # 2.7 g

# A table tennis ball is a hollow shell, so I = (2/3) m r**2, not the
# (2/5) of a solid sphere. This factor directly sets how tangential
# velocity converts into spin on impact, and the difference is large.
BALL_INERTIA_FACTOR = 2.0 / 3.0
BALL_INERTIA = BALL_INERTIA_FACTOR * BALL_MASS * BALL_RADIUS ** 2

# Tangential impulse coefficient for a sticking (rolling) contact,
# derived from 1/m + r**2/I:
#   r**2/I = 1/(BALL_INERTIA_FACTOR * m)  ->  J_t = -FACTOR * m * v_ct
# Hollow sphere (2/3) -> 2/5; solid sphere (2/5) -> 2/7.
TANGENT_IMPULSE_FACTOR = BALL_INERTIA_FACTOR / (1.0 + BALL_INERTIA_FACTOR)

BALL_AREA = math.pi * BALL_RADIUS ** 2

# Legacy names
BALL_R = BALL_RADIUS
BALL_M = BALL_MASS

# ---------------------------------------------------------------- air
AIR_DENSITY = 1.20       # kg/m^3 at 20 C
DRAG_COEFF = 0.40        # sphere Cd at Re ~ 1e4..1e5
MAGNUS_COEFF = 1.00      # slope of lift coeff vs spin parameter S = r*w/v

GRAVITY = 9.81

# Pre-computed acceleration-form coefficients used by the integrator:
#   a_drag   = -DRAG_ACCEL_K * |v| * v
#   a_magnus = +MAGNUS_ACCEL_K * (w x v)
#
# The ball is only 2.7 g, so DRAG_ACCEL_K * v**2 is about 11 m/s^2 at
# v = 10 m/s -- larger than gravity. Trajectories are simply wrong if
# drag is not modelled.
DRAG_ACCEL_K = 0.5 * AIR_DENSITY * DRAG_COEFF * BALL_AREA / BALL_MASS
MAGNUS_ACCEL_K = (
    0.5 * AIR_DENSITY * MAGNUS_COEFF * BALL_AREA * BALL_RADIUS / BALL_MASS
)

# ---------------------------------------------------------------- contact
# ITTF spec: dropped from 30 cm the ball must rebound to 24-26 cm.
# Ignoring air that would give e = sqrt(0.25/0.30) ~ 0.91, but we do
# model drag, which costs energy on both the way down and the way up,
# so the bare restitution has to be slightly higher to reproduce the
# standard test. This value was calibrated with the drop test in
# verify_physics.py (result: 24.7 cm).
RESTITUTION_TABLE = 0.93
FRICTION_TABLE = 0.25        # the surface is fairly slick

RESTITUTION_PADDLE = 0.88    # rubber plus sponge
FRICTION_PADDLE = 0.90       # grippy rubber -- this is what creates spin

RESTITUTION_NET = 0.20       # hitting the net kills almost all the energy
NET_VELOCITY_DAMP = 0.30     # tangential damping after a net contact

# Spin about the contact normal ("drilling") is not covered by the
# tangential model, so it is simply damped.
SPIN_NORMAL_DAMP = 0.95

# Aerodynamic spin damping (1/s). Spin decays slowly over a rally.
SPIN_AIR_DAMP = 0.05

# ---------------------------------------------------------------- sim
DEFAULT_DT = 1.0 / 480.0     # flight integration step; the ball is fast
MAX_SIM_TIME = 5.0           # per-simulation cap, seconds

# Margin used for out-of-bounds tests
OUT_MARGIN_XY = 0.35
FLOOR_Z = 0.0
