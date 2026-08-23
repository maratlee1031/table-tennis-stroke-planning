"""Turn a model's action into a physically executable stroke description.

The networks output a 5-vector -- blade yaw, blade pitch, and a paddle
velocity -- because that is a convenient parameterisation to learn in. It is
not a convenient thing to hand to hardware. This module converts it into the
quantities a robot arm, or a coach, actually works in.

What the model decides
----------------------
* the **blade orientation** at contact (two angles; the third, rotation about
  the blade normal, does not affect a circular blade and is left free -- a
  real arm can use it to stay inside its joint limits)
* the **paddle velocity vector** at contact, which splits into a *drive*
  component along the blade normal (this buys ball speed) and a *brush*
  component across the face (this buys spin)

What the model does **not** decide
----------------------------------
* **friction**. That is a property of the rubber and the ball, not a choice.
  The physics assumes a fixed coefficient, and a stroke computed under that
  assumption is only reproducible on rubber that provides at least the
  friction the stroke demands. ``required_mu`` in the card is that number,
  and it is the first thing to check before trusting a stroke on real
  hardware.
* **restitution**, for the same reason.
* the contact point on the blade -- assumed to be the centre.
"""

import numpy as np

from . import constants as C
from . import dataset, physics, quat


def describe(state, action, goal=None, verify=True):
    """Full physical description of one stroke.

    ``state`` is the 8-vector incoming ball state, ``action`` the 5-vector
    the model produced. Returns a dict of everything needed to execute and
    to sanity-check the stroke.
    """
    state = np.asarray(state, dtype=float).reshape(-1)
    action = np.asarray(action, dtype=float).reshape(-1)

    pos = np.array([dataset.STRIKE_X, state[0], state[1]])
    vel = state[2:5].copy()
    spin = state[5:8].copy()

    normals, pvels = dataset.decode_action(action[None, :])
    n, u = normals[0], pvels[0]

    report = physics.contact_report(vel, spin, n, u)

    out = {
        "incoming": {
            "position": pos,
            "velocity": vel,
            "speed": float(np.linalg.norm(vel)),
            "spin": spin,
            "topspin": float(spin[1]),
            "sidespin": float(spin[2]),
        },
        "blade": {
            "normal": report["blade_normal"],
            # Orientation as a quaternion, so it can go straight to an
            # end-effector pose command. Rotation about the normal is free.
            "quaternion_xyzw": quat.look_quat(report["blade_normal"]),
            "yaw_deg": float(action[0]),
            "pitch_deg": float(action[1]),
            "tilt_from_vertical_deg": report["blade_tilt_deg"],
            "angle_of_attack_deg": report["angle_of_attack_deg"],
        },
        "motion": {
            "velocity": report["paddle_velocity"],
            "speed": report["paddle_speed"],
            "drive_along_normal": report["drive_speed"],
            "brush_across_face": report["brush_speed"],
            "brush_direction": report["brush_direction"],
        },
        "contact": {
            "normal_impulse_Ns": report["normal_impulse_Ns"],
            "tangential_impulse_Ns": report["tangential_impulse_Ns"],
            "required_mu": report["required_mu"],
            "sliding": report["sliding"],
            "slip_speed": report["contact_slip_speed"],
            "peak_force_estimate_N": report["peak_force_estimate_N"],
        },
        "equipment_assumed": {
            "rubber_mu": report["assumed_mu"],
            "rubber_restitution": report["assumed_restitution"],
            "ball_mass_kg": C.BALL_MASS,
            "ball_radius_m": C.BALL_RADIUS,
        },
    }

    if verify:
        achieved, outcome, landing = dataset.apply_actions(
            pos[None, :], vel[None, :], spin[None, :], action[None, :],
            dt=1.0 / 480.0, max_time=3.0)
        out["result"] = {
            "landing": landing[0],
            "land_x": float(achieved[0, 0]),
            "land_y": float(achieved[0, 1]),
            "landing_speed": float(achieved[0, 2]),
            "topspin": float(achieved[0, 3]),
            "sidespin": float(achieved[0, 4]),
            "outcome": int(outcome[0]),
            "landed_in": bool(outcome[0] == 1),
        }
        if goal is not None:
            g = np.asarray(goal, dtype=float).reshape(-1)
            out["result"]["goal"] = g
            out["result"]["error_per_dim"] = achieved[0] - g
            out["result"]["goal_error"] = float(
                dataset.goal_error(achieved[0][None, :], g[None, :])[0])
    return out


def format_card(card):
    """Human-readable stroke card."""
    inc, bl, mo, ct, eq = (card["incoming"], card["blade"], card["motion"],
                           card["contact"], card["equipment_assumed"])
    L = []
    a = L.append
    a("=" * 66)
    a("STROKE CARD")
    a("=" * 66)
    a("")
    a("INCOMING BALL (at the strike plane)")
    a(f"  position        y {inc['position'][1]:+.3f}  z {inc['position'][2]:.3f}  m")
    a(f"  velocity        ({inc['velocity'][0]:+.2f}, {inc['velocity'][1]:+.2f}, "
      f"{inc['velocity'][2]:+.2f})  |v| {inc['speed']:.2f} m/s")
    a(f"  spin            topspin {inc['topspin']:+.0f}   sidespin {inc['sidespin']:+.0f}  rad/s")
    a("")
    a("BLADE AT CONTACT   (what to orient the end effector to)")
    a(f"  normal          ({bl['normal'][0]:+.3f}, {bl['normal'][1]:+.3f}, "
      f"{bl['normal'][2]:+.3f})   unit vector, world frame")
    a(f"  quaternion      ({bl['quaternion_xyzw'][0]:+.4f}, {bl['quaternion_xyzw'][1]:+.4f}, "
      f"{bl['quaternion_xyzw'][2]:+.4f}, {bl['quaternion_xyzw'][3]:+.4f})  xyzw")
    a(f"  tilt            {bl['tilt_from_vertical_deg']:+.1f} deg from vertical "
      f"({'closed' if bl['tilt_from_vertical_deg'] > 0 else 'open'})")
    a(f"  angle of attack {bl['angle_of_attack_deg']:.1f} deg to the incoming ball")
    a("  rotation about the normal is unconstrained -- free for the arm to use")
    a("")
    a("PADDLE MOTION AT CONTACT   (end-effector linear velocity)")
    a(f"  velocity        ({mo['velocity'][0]:+.2f}, {mo['velocity'][1]:+.2f}, "
      f"{mo['velocity'][2]:+.2f})  |u| {mo['speed']:.2f} m/s")
    a(f"  drive           {mo['drive_along_normal']:+.2f} m/s along the normal   -> ball speed")
    a(f"  brush           {mo['brush_across_face']:.2f} m/s across the face      -> spin")
    a(f"  brush direction ({mo['brush_direction'][0]:+.2f}, {mo['brush_direction'][1]:+.2f}, "
      f"{mo['brush_direction'][2]:+.2f})")
    a("")
    a("CONTACT MECHANICS")
    a(f"  normal impulse      {ct['normal_impulse_Ns'] * 1000:.2f} mN s")
    a(f"  tangential impulse  {ct['tangential_impulse_Ns'] * 1000:.2f} mN s")
    a(f"  peak force (approx) {ct['peak_force_estimate_N']:.1f} N over ~1 ms")
    a(f"  REQUIRED rubber mu  {ct['required_mu']:.3f}"
      f"   (assumed {eq['rubber_mu']:.2f}"
      f" -> {'SLIPPING, spin is saturated' if ct['sliding'] else 'grips, stroke is reproducible'})")
    a("")
    a("EQUIPMENT ASSUMED   (not chosen by the model -- properties of the kit)")
    a(f"  rubber friction     mu = {eq['rubber_mu']:.2f}")
    a(f"  rubber restitution  e  = {eq['rubber_restitution']:.2f}")
    a(f"  ball                {eq['ball_mass_kg'] * 1000:.1f} g, "
      f"r = {eq['ball_radius_m'] * 1000:.0f} mm")

    if "result" in card:
        r = card["result"]
        a("")
        a("RESULTING SHOT   (simulated with the full physics)")
        outcome_name = {0: "landed on OUR half -- did not clear the net",
                        1: "IN", 2: "OUT", 3: "into the NET",
                        4: "never came down"}.get(r["outcome"], "?")
        if not np.isfinite(r["land_x"]):
            # The ball never reached the table, so there is nothing to report
            # per dimension; saying "nan" five times would only obscure that.
            a(f"  {outcome_name} -- no landing to measure")
        else:
            a(f"  lands at        x {r['land_x']:+.3f}  y {r['land_y']:+.3f}  m"
              f"   [{outcome_name}]")
            a(f"  landing speed   {r['landing_speed']:.2f} m/s")
            a(f"  spin on arrival topspin {r['topspin']:+.0f}   "
              f"sidespin {r['sidespin']:+.0f}  rad/s")
            if "goal" in r:
                g, e = r["goal"], r["error_per_dim"]
                a("")
                # Three headers over two columns of numbers put the requested
                # value under "achieved", which reads as the model having
                # done the exact opposite of what it did. Print the achieved
                # column that the header always promised.
                a(f"  {'':>22}  {'requested':>10}  {'achieved':>10}"
                  f"  {'error':>10}")
                for i, (name, unit) in enumerate(zip(dataset.GOAL_NAMES,
                                                     dataset.GOAL_UNITS)):
                    a(f"  {name:>22}  {g[i]:+10.2f}  {g[i] + e[i]:+10.2f}"
                      f"  {e[i]:+10.2f}   {unit}")
                a(f"  {'weighted goal error':>22}  {r['goal_error']:>34.3f}")
    a("=" * 66)
    return "\n".join(L)
