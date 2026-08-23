# coach_game.py
# AI coach mode: click a spot on the table and the AI chases the ball,
# solves for the blade angle and swing velocity, and plays the stroke.
#
# Controls:
#   T            toggle AI coach mode
#   Left mouse   click the table to set the target landing spot
#   R            re-serve
#
# Run: python coach_game.py
#
# Differences from the old version
# --------------------------------
# The old solver rebuilt **an entire Bullet world per query** and then ran
# 250 sequential random trials. Now:
#   * it shares pingpong.physics with the game, so the predicted trajectory
#     and the real one are identical;
#   * it uses CEM (cross-entropy method) on top of the vectorised
#     simulate_batch, evaluating hundreds of candidate actions at once,
#     which beats random search on both quality and speed.
#
# The action space also grew from "normal + one scalar power" to
# "normal (yaw, pitch) + swing velocity vector (3)", 5 degrees of freedom,
# which is what makes spin controllable.

import time

import numpy as np
from panda3d.core import LineSegs, TextNode, TransparencyAttrib, Vec3

import main_wss as base
from pingpong import constants as C
from pingpong import physics, quat

# Action bounds: [normal yaw deg, normal pitch deg, swing vx, vy, vz]
PARAM_LOW = np.array([-75.0, -50.0, -6.0, -12.0, -12.0])
PARAM_HIGH = np.array([75.0, 50.0, 16.0, 12.0, 12.0])
PARAM_FLOOR_STD = np.array([2.5, 2.5, 0.35, 0.35, 0.35])

# Two presets for the two-stage solve. Measurements (table below) show
# **iteration count matters far more than population size**: dropping to
# 2 iterations blows the error up from 8 cm to 15-32 cm because CEM has
# not converged, while halving the population and coarsening the
# integration step to 1/120 costs almost nothing.
#
#   iters pop   dt     ms/solve  mean err  max err
#     4   600  1/480      472      5.0cm    7.4cm
#     4   600  1/240      350      3.8cm    6.2cm
#     3   256  1/120       75      8.0cm   13.3cm
#     2   256  1/120       70     15.5cm   32.1cm   <- under-converged
FAST_SOLVE = dict(iters=3, pop=256, sim_dt=1.0 / 120.0)
ACCURATE_SOLVE = dict(iters=4, pop=600, sim_dt=1.0 / 240.0)


def decode_params(params):
    """Parameters -> (face normals, swing velocities)."""
    yaw = np.radians(params[:, 0])
    pitch = np.radians(params[:, 1])
    cp = np.cos(pitch)
    normals = np.stack([cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)], axis=1)
    return normals, params[:, 2:5]


def evaluate_actions(params, ball_pos, ball_vel, ball_spin, target,
                     sim_dt=C.DEFAULT_DT):
    """Score a batch of candidate actions.

    Returns (cost, landing, outcome, normals, paddle_vels). The whole
    population runs through a single vectorised simulation, which is what
    makes CEM viable in place of random search.
    """
    n_pop = params.shape[0]
    normals, pvel = decode_params(params)

    vel = np.tile(ball_vel, (n_pop, 1))
    spin = np.tile(ball_spin, (n_pop, 1))
    out_v, out_w = physics.collide(
        vel, spin, normals, C.RESTITUTION_PADDLE, C.FRICTION_PADDLE, surface_vel=pvel
    )

    start = np.tile(ball_pos, (n_pop, 1)) + normals * (C.BALL_RADIUS * 1.6)
    landing, outcome, _, _ = physics.simulate_batch(start, out_v, out_w, dt=sim_dt)

    err = np.linalg.norm(landing[:, :2] - np.asarray(target)[:2], axis=1)
    err = np.where(np.isnan(err), 4.0, err)
    # outcome: 0=own half 1=opponent half 2=out 3=net 4=timeout
    penalty = np.where(outcome == 1, 0.0, 3.0)
    # Mild preference for the cheaper stroke, which collapses the
    # one-to-many mapping down to a single answer
    effort = 0.01 * np.linalg.norm(pvel, axis=1)
    return err + penalty + effort, landing, outcome, normals, pvel


def solve_stroke(ball_pos, ball_vel, ball_spin, target,
                 iters=4, pop=600, elite_frac=0.12, rng=None,
                 sim_dt=C.DEFAULT_DT):
    """CEM inverse solve: blade angle and swing velocity to reach ``target``.

    This is "option A" from the architecture discussion -- the forward model
    (physics) is already known, so optimise directly in action space, which
    handles the multi-solution nature of the problem naturally.
    """
    rng = rng or np.random.default_rng()
    mean = np.array([0.0, 6.0, 5.0, 0.0, 2.5])
    std = np.array([32.0, 22.0, 4.5, 3.5, 4.0])
    n_elite = max(10, int(pop * elite_frac))
    best = None

    for _ in range(iters):
        params = rng.normal(mean, std, size=(pop, 5))
        params = np.clip(params, PARAM_LOW, PARAM_HIGH)

        cost, landing, outcome, normals, pvel = evaluate_actions(
            params, ball_pos, ball_vel, ball_spin, target, sim_dt=sim_dt
        )
        order = np.argsort(cost)
        top = order[0]
        if best is None or cost[top] < best["cost"]:
            best = dict(
                cost=float(cost[top]),
                normal=normals[top].copy(),
                paddle_vel=pvel[top].copy(),
                landing=landing[top].copy(),
                outcome=int(outcome[top]),
                params=params[top].copy(),
            )

        elites = params[order[:n_elite]]
        mean = elites.mean(axis=0)
        std = elites.std(axis=0) + PARAM_FLOOR_STD

    return best


class CoachGame(base.Game):
    def __init__(self):
        super().__init__()

        self.ai_enabled = False
        self.ai_target = None
        self.ai_plan = None
        self.ai_hit_done = False
        self._last_plan_t = 0.0
        self._final_solved = False
        self._plan_state = None
        self.ai_replan_interval = 0.20
        self.ai_paddle_speed = 3.6          # max paddle travel speed, m/s
        self.rng = np.random.default_rng()

        self.ai_traj_np = None
        self.ai_arrow_np = None
        self.ai_marker_np = None

        self.ai_text = self.hud_text(
            (0.06, -0.58), 0.040, TextNode.ALeft, (0.6, 1.0, 0.9, 1),
            text="AI coach: off (press T)",
        )

        self.accept("t", self.toggle_ai)
        self.accept("mouse1", self.on_click)

        print("[AI] CoachGame loaded. Press T, then click the table to aim.")

    # ------------------------------------------------------------ toggle
    def toggle_ai(self):
        self.ai_enabled = not self.ai_enabled
        self.ai_plan = None
        self.ai_hit_done = False
        if not self.ai_enabled:
            self.ai_target = None
            self._clear_ai_visuals()
            self.ai_text.setText("AI coach: off (press T)")
        else:
            self.ai_text.setText("AI coach: on - click the table to set a target")
        print("[AI] enabled:", self.ai_enabled)

    def serve_ball(self):
        super().serve_ball()
        self.ai_hit_done = False
        self.ai_plan = None
        self._last_plan_t = 0.0
        self._final_solved = False
        self._plan_state = None

    # ------------------------------------------------------------ mouse
    def _mouse_to_table(self):
        if not self.mouseWatcherNode.hasMouse():
            return None
        mpos = self.mouseWatcherNode.getMouse()
        near, far = Vec3(), Vec3()
        self.camLens.extrude(mpos, near, far)
        o = self.render.getRelativePoint(self.camera, near)
        f = self.render.getRelativePoint(self.camera, far)
        d = Vec3(f) - Vec3(o)
        if abs(d.z) < 1e-6:
            return None
        z_plane = C.TABLE_TOP_Z + C.BALL_RADIUS
        t = (z_plane - o.z) / d.z
        if t < 0:
            return None
        p = Vec3(o) + d * t
        if abs(p.x) > C.TABLE_L / 2 or abs(p.y) > C.TABLE_W / 2:
            return None
        return np.array([p.x, p.y, z_plane])

    def on_click(self):
        if not self.ai_enabled:
            return
        target = self._mouse_to_table()
        if target is None:
            self.ai_text.setText("AI coach: click inside the table")
            return
        self.ai_target = target
        self._draw_marker(target)
        self.ai_hit_done = False
        self._last_plan_t = 0.0
        self._final_solved = False
        self._plan_state = None
        self.ai_text.setText(
            f"AI coach: target ({target[0]:+.2f}, {target[1]:+.2f}) - waiting for the ball"
        )

    # ------------------------------------------------------------ predict
    def _predict_intercept(self, x_plane):
        """Ball state when it reaches the paddle's X plane, or None.

        Delegates to physics.predict_plane_crossing, which the base game's
        swing timing also uses -- so the AI and the human aim at the same
        instant, computed the same way.
        """
        c = self.predict_contact() if x_plane == base.PADDLE_STRIKE_X else (
            physics.predict_plane_crossing(
                self.ball.pos, self.ball.vel, self.ball.spin, x_plane
            ) if self.ball.active else None
        )
        return None if c is None else (c.pos, c.vel, c.spin)

    # ------------------------------------------------------------ control
    def _control_paddle(self, st, dt):
        if not self.ai_enabled or self.ai_target is None:
            return super()._control_paddle(st, dt)

        now = time.time()
        intercept = self._predict_intercept(base.PADDLE_STRIKE_X)

        if intercept is not None and not self.ai_hit_done:
            pos_i, vel_i, spin_i = intercept

            # Move towards the intercept point (no teleporting)
            desired = np.array([base.PADDLE_STRIKE_X, pos_i[1], pos_i[2]])
            desired[1] = np.clip(desired[1], -C.TABLE_W / 2 - 0.2, C.TABLE_W / 2 + 0.2)
            desired[2] = np.clip(desired[2], C.TABLE_H + 0.04, C.TABLE_H + 0.95)
            diff = desired - self.paddle.pos
            dist = float(np.linalg.norm(diff))
            if dist > 1e-6:
                step = min(dist, self.ai_paddle_speed * dt)
                self.paddle.pos = self.paddle.pos + diff / dist * step

            # Just before contact, commit with a high-accuracy solve (once)
            ball_dist = float(np.linalg.norm(self.ball.pos - self.paddle.pos))
            final_shot = ball_dist < 0.30 and not self._final_solved

            # Re-solving every frame drops the frame rate into the teens
            # (a single solve is 75-350 ms). The predicted intercept is
            # quite stable, so only re-solve when there is no plan yet or
            # the prediction moved -- the same way a person decides how to
            # play a ball and then executes it.
            moved = (
                self._plan_state is None
                or float(np.linalg.norm(pos_i - self._plan_state)) > 0.06
            )
            due = (now - self._last_plan_t) > self.ai_replan_interval

            if final_shot or (moved and due):
                preset = ACCURATE_SOLVE if final_shot else FAST_SOLVE
                self._last_plan_t = now
                self._plan_state = pos_i.copy()
                if final_shot:
                    self._final_solved = True
                plan = solve_stroke(pos_i, vel_i, spin_i, self.ai_target,
                                    rng=self.rng, **preset)
                if plan is not None:
                    self.ai_plan = plan
                    self._draw_plan(plan, pos_i, vel_i, spin_i)
                    ok = "clears" if plan["outcome"] == 1 else "no solution"
                    tag = "committed" if final_shot else "tracking"
                    self.ai_text.setText(
                        f"AI coach: target ({self.ai_target[0]:+.2f}, "
                        f"{self.ai_target[1]:+.2f})  {ok} [{tag}]\n"
                        f"  blade yaw={plan['params'][0]:+.1f} pitch={plan['params'][1]:+.1f}\n"
                        f"  swing ({plan['paddle_vel'][0]:+.1f}, {plan['paddle_vel'][1]:+.1f}, "
                        f"{plan['paddle_vel'][2]:+.1f}) m/s  |v|={np.linalg.norm(plan['paddle_vel']):.1f}\n"
                        f"  predicted landing ({plan['landing'][0]:+.2f}, "
                        f"{plan['landing'][1]:+.2f})  err {plan['cost']:.3f} m"
                    )
        else:
            # Nothing to intercept -> drift back to the ready position
            home = np.array([base.PADDLE_READY_X, 0.0, C.TABLE_H + 0.28])
            self.paddle.pos = self.paddle.pos + (home - self.paddle.pos) * min(1.0, dt * 3.0)

        # Apply the solved blade angle and swing velocity
        if self.ai_plan is not None:
            self.paddle.normal = self.ai_plan["normal"]
            self.paddle.velocity = self.ai_plan["paddle_vel"]
            pq = quat.look_quat(self.ai_plan["normal"])
        else:
            self.paddle.normal = np.array([1.0, 0.0, 0.0])
            self.paddle.velocity = np.zeros(3)
            pq = quat.NEUTRAL_PADDLE

        self.apply_paddle_transform(pq)

    # ------------------------------------------------------------ visuals
    def _clear_ai_visuals(self):
        for attr in ("ai_traj_np", "ai_arrow_np", "ai_marker_np"):
            np_ = getattr(self, attr, None)
            if np_ is not None:
                np_.remove_node()
                setattr(self, attr, None)

    def _draw_marker(self, pos):
        if self.ai_marker_np is not None:
            self.ai_marker_np.remove_node()
        segs = LineSegs("ai_target")
        segs.set_thickness(3.0)
        segs.set_color(0.25, 1.0, 0.35, 1.0)
        r = 0.06
        n = 24
        for i in range(n + 1):
            a = 2 * np.pi * i / n
            fn = segs.move_to if i == 0 else segs.draw_to
            fn(pos[0] + r * np.cos(a), pos[1] + r * np.sin(a), pos[2] + 0.002)
        self.ai_marker_np = self.render.attach_new_node(segs.create())
        self.ai_marker_np.set_light_off()

    def _draw_plan(self, plan, ball_pos, ball_vel, ball_spin):
        # Predicted trajectory
        out_v, out_w = physics.collide(
            ball_vel, ball_spin, plan["normal"],
            C.RESTITUTION_PADDLE, C.FRICTION_PADDLE,
            surface_vel=plan["paddle_vel"],
        )
        res = physics.simulate(
            ball_pos + plan["normal"] * (C.BALL_RADIUS * 1.6),
            out_v, out_w, record_trajectory=True, max_time=3.0,
        )
        if self.ai_traj_np is not None:
            self.ai_traj_np.remove_node()
            self.ai_traj_np = None
        pts = res.trajectory
        if pts is not None and len(pts) >= 2:
            segs = LineSegs("ai_traj")
            segs.set_thickness(3.0)
            segs.set_color(0.35, 0.95, 1.0, 0.9)
            segs.move_to(*pts[0])
            for p in pts[1:]:
                segs.draw_to(*p)
            self.ai_traj_np = self.render.attach_new_node(segs.create())
            self.ai_traj_np.set_transparency(TransparencyAttrib.M_alpha)
            self.ai_traj_np.set_light_off()

        # Swing direction arrow
        if self.ai_arrow_np is not None:
            self.ai_arrow_np.remove_node()
            self.ai_arrow_np = None
        pv = plan["paddle_vel"]
        mag = float(np.linalg.norm(pv))
        if mag > 0.2:
            d = pv / mag
            o = self.paddle.pos
            tip = o + d * min(0.45, 0.06 + mag * 0.03)
            up = np.array([0.0, 0.0, 1.0])
            side = np.cross(d, up)
            if np.linalg.norm(side) < 1e-6:
                side = np.array([0.0, 1.0, 0.0])
            side = side / np.linalg.norm(side)
            segs = LineSegs("ai_arrow")
            segs.set_thickness(4.0)
            segs.set_color(1.0, 0.75, 0.2, 1.0)
            segs.move_to(*o)
            segs.draw_to(*tip)
            for s in (1, -1):
                segs.move_to(*tip)
                segs.draw_to(*(tip - d * 0.07 + side * 0.045 * s))
            self.ai_arrow_np = self.render.attach_new_node(segs.create())
            self.ai_arrow_np.set_light_off()

    # ------------------------------------------------------------
    def _contact_allowed(self):
        """The AI moves the paddle itself rather than through the swing state
        machine, so its own chase logic is what gates contact."""
        if self.ai_enabled and self.ai_target is not None:
            return True
        return super()._contact_allowed()

    def _try_hit(self, now, ball_prev, paddle_prev):
        before = self.player_hit
        super()._try_hit(now, ball_prev, paddle_prev)
        if self.ai_enabled and self.player_hit and not before:
            self.ai_hit_done = True


if __name__ == "__main__":
    base.launch_wss()
    CoachGame().run()
