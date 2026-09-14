"""Play a rally against a trained policy -- you on the phone, it on the far side.

    python rally_ai.py                                   # best model available
    python rally_ai.py --model mdn_20000_i8p256          # an easier opponent
    python rally_ai.py --list

This is the half the project was missing. `main_wss.py` has a human returning
serves from nothing, `ai_play.py` has a model returning serves with no human
in the room, and `coach_game.py` has the solver commenting on your stroke.
None of them ever put the two players on the same table.

Controls are exactly `main_wss.py`'s: the phone is the paddle, MediaPipe
places it, arrow keys and space work without either. `R` re-serves.

### The opponent solves a mirrored problem

The policies were trained on one situation only: a ball arriving at the
strike plane travelling in -x, to be returned into +x. The opponent faces the
reverse, so its state is rotated 180 degrees about the vertical axis into the
frame the policy knows, and the blade normal and paddle velocity it returns
are rotated back. That is a rigid motion, so the physics is identical and the
policy is never asked to extrapolate -- which it would be if the opposite
half were simply handed to it as-is.
"""

import argparse
import os
import time

import numpy as np
from panda3d.core import LineSegs, TextNode

import main_wss as base
from pingpong import compare, constants as C, datalog, dataset, physics, quat

# Where the opponent meets the ball: our strike plane, mirrored
OPP_STRIKE_X = -base.PADDLE_STRIKE_X

# 180 degrees about z. Applies to positions, velocities and (being a proper
# rotation) to the angular velocity as well.
MIRROR = np.array([-1.0, -1.0, 1.0])

# How far ahead of the crossing the opponent commits to its stroke. It has no
# reaction time to model, but arriving instantly looks like teleporting.
OPP_LEAD = 0.28

CONTACT_REFRESH = 0.06


def mirror(v):
    return np.asarray(v, dtype=float) * MIRROR


class RallyAI(base.Game):
    def __init__(self, contender, difficulty=1.0, seed=0):
        self.contender = contender
        self.difficulty = float(np.clip(difficulty, 0.2, 1.0))
        self.rng = np.random.default_rng(seed)

        self.opp = None
        self.opp_goal = None
        self.opp_action = None
        self.opp_contact = None
        self._opp_contact_t = 0.0
        self._opp_prev = np.zeros(3)
        self._opp_last_hit = 0.0
        self.last_hitter = None
        self.rally_len = 0
        self.best_rally = 0

        super().__init__()

        self.opp = physics.Paddle(pos=np.array([OPP_STRIKE_X, 0.0, C.TABLE_H + 0.25]),
                                  normal=np.array([-1.0, 0.0, 0.0]))
        self.opp_np = self._build_opponent_paddle()

        # These strokes are the model's, and they are played against a human
        # rather than a scripted serve, so they are worth keeping -- but not
        # in the file analyze_strokes.py reads as human play.
        self.logger = datalog.StrokeLogger("runs/rally_strokes.csv",
                                           source=datalog.SRC_HUMAN)
        self.opp_logger = datalog.StrokeLogger("runs/rally_ai_strokes.csv",
                                               source=datalog.SRC_AI)

        self.hud_opp = self.hud_text((0.06, -0.58), 0.040, TextNode.ALeft,
                                     (0.45, 0.90, 1.0, 1))
        self.hud_rally = self.hud_text((0.06, -0.65), 0.040, TextNode.ALeft,
                                       (0.85, 0.87, 0.92, 1))
        print(f"[rally] opponent: {contender.label}")
        print("[rally] it returns to a random spot on your half; "
              "rally as long as you can")

    # ------------------------------------------------------------ scene
    def _build_opponent_paddle(self):
        """Same blade as the player's, in the opponent's colours."""
        np_ = self.render.attach_new_node("opp_paddle")
        face_a = self._make_card(0.15, 0.16)
        face_a.reparent_to(np_)
        face_a.set_pos(0, 0.004, 0)
        face_a.set_color(0.13, 0.35, 0.80, 1)
        face_b = self._make_card(0.15, 0.16)
        face_b.reparent_to(np_)
        face_b.set_h(180)
        face_b.set_pos(0, -0.004, 0)
        face_b.set_color(0.09, 0.09, 0.10, 1)
        edge = self._make_card(0.15, 0.16)
        edge.reparent_to(np_)
        edge.set_p(90)
        edge.set_color(0.85, 0.75, 0.55, 1)
        edge.set_scale(1.02)
        handle = self._make_card(0.035, 0.11)
        handle.reparent_to(np_)
        handle.set_pos(0, 0, -0.135)
        handle.set_color(0.42, 0.28, 0.16, 1)
        seg = LineSegs("opp_normal")
        seg.set_thickness(2.0)
        seg.set_color(0.3, 0.9, 1.0, 0.9)
        seg.move_to(0, 0, 0)
        seg.draw_to(0, 0.22, 0)
        np_.attach_new_node(seg.create())
        return np_

    # ------------------------------------------------------------ rally
    def serve_ball(self):
        super().serve_ball()
        self.opp_goal = None
        self.opp_action = None
        self.opp_contact = None
        self.last_hitter = None
        self.rally_len = 0

    def _new_opp_goal(self):
        """Where the opponent aims, expressed in the frame the policy knows.

        In that frame the target half is always +x, so this is the same goal
        space the models were trained and scored on. It lands on your half
        once the answer is rotated back.
        """
        d = self.difficulty
        return np.array([
            self.rng.uniform(0.40, 0.45 + 0.80 * d),
            self.rng.uniform(-0.30 - 0.32 * d, 0.30 + 0.32 * d),
            self.rng.uniform(4.0, 4.5 + 3.0 * d),
            self.rng.uniform(-60.0 * d, 120.0 + 180.0 * d),
            self.rng.uniform(-90.0 * d, 90.0 * d),
        ], dtype=np.float32)

    def predict_opp_contact(self, now):
        """Ball arrival at the opponent's strike plane, in the policy's frame.

        physics.predict_plane_crossing only looks for a crossing in -x, and
        bails immediately if the ball starts on the far side of the plane --
        it was written for a ball arriving at our end. Rather than give it a
        direction argument, the ball is mirrored first, exactly as the policy
        input is. The crossing then looks like the one the function was
        built for, and the answer comes back already in the frame the policy
        wants; only the paddle placement needs mirroring back.
        """
        if not self.ball.active or self.ball.vel[0] <= 0:
            self.opp_contact = None
            return None
        if (self.opp_contact is None
                or (now - self._opp_contact_t) >= CONTACT_REFRESH):
            self.opp_contact = physics.predict_plane_crossing(
                mirror(self.ball.pos), mirror(self.ball.vel),
                mirror(self.ball.spin), base.PADDLE_STRIKE_X)
            self._opp_contact_t = now
        return self.opp_contact

    # ------------------------------------------------------------ control
    def _control_paddle(self, st, dt):
        super()._control_paddle(st, dt)
        self._opp_prev = self.opp.pos.copy() if self.opp is not None else None
        if self.opp is not None:
            self._control_opponent(time.time())

    def _control_opponent(self, now):
        c = self.predict_opp_contact(now)
        if c is None or self.last_hitter == "ai":
            # Nothing to answer: park it back at the ready position
            self.opp.pos += (np.array([OPP_STRIKE_X, 0.0, C.TABLE_H + 0.25])
                             - self.opp.pos) * 0.12
            self.opp.velocity = np.zeros(3)
            # NEUTRAL_PADDLE faces +x, which is correct for the player and
            # backwards for the opponent -- it would wait for the ball with
            # the blade turned away from it
            self.opp.normal = np.array([-1.0, 0.0, 0.0])
            self._push_opp(quat.look_quat(self.opp.normal))
            return

        # c is already in the policy's frame, so it is fed straight in
        if self.opp_action is None:
            if self.opp_goal is None:
                self.opp_goal = self._new_opp_goal()
            state = dataset.encode_state(c.pos[None, :], c.vel[None, :],
                                         c.spin[None, :]).astype(np.float32)
            g = self.opp_goal[None, :]
            if self.contender.is_oracle:
                a = self.contender.agent.act(state, g, c.pos[None, :],
                                             c.vel[None, :], c.spin[None, :])[0]
            else:
                a = self.contender.agent.act(state, g)[0]
            self.opp_action = np.asarray(a, dtype=float)

        # ... and the answer is rotated back out into the world
        normals, pvels = dataset.decode_action(self.opp_action[None, :])
        self.opp.normal = mirror(normals[0])
        self.opp.velocity = mirror(pvels[0])
        # Slide in over the last stretch instead of appearing at the contact
        # point, so the stroke reads as a stroke
        target = mirror(c.pos)
        f = float(np.clip(1.0 - c.time / OPP_LEAD, 0.0, 1.0))
        self.opp.pos = self.opp.pos + (target - self.opp.pos) * (0.15 + 0.85 * f)
        if f >= 1.0:
            self.opp.pos = target
        self._push_opp(quat.look_quat(self.opp.normal))

    def _push_opp(self, q):
        self.opp_np.set_quat(base.Quat(*quat.to_panda(q)))
        self.opp_np.set_pos(*self.opp.pos)

    # ------------------------------------------------------------ contact
    def _try_hit(self, now, ball_prev, paddle_prev):
        # player_hit is a rally-level flag -- "has the player touched it at
        # all" -- so it only has a rising edge on the first stroke of a
        # rally. last_hit_time moves on every contact, which is what a rally
        # of more than two strokes needs.
        before_t = self.last_hit_time
        super()._try_hit(now, ball_prev, paddle_prev)
        if self.last_hit_time != before_t:
            self.last_hitter = "human"
            self.rally_len += 1
            self._keep_alive(now)
            # The opponent has a new ball coming, so its plan is stale
            self.opp_action = None
            self.opp_goal = None
            self.opp_contact = None
        if self.opp is not None:
            self._try_opponent_hit(now, ball_prev)

    def _try_opponent_hit(self, now, ball_prev):
        if not self.ball.active or (now - self._opp_last_hit) < 0.15:
            return
        if self.last_hitter == "ai" or self.opp_action is None:
            return
        n = self.opp.face_normal_toward(ball_prev)
        t = physics.swept_blade_hit(ball_prev, self.ball.pos,
                                    self._opp_prev, self.opp.pos,
                                    n, base.HIT_RADIUS)
        if t is None:
            return
        self.ball.pos = ball_prev + (self.ball.pos - ball_prev) * t
        in_vel, in_spin, in_pos = (self.ball.vel.copy(), self.ball.spin.copy(),
                                   self.ball.pos.copy())
        out_vel, out_spin = physics.hit_with_paddle(in_vel, in_spin, self.opp,
                                                    self.ball.pos, normal=n)
        self.ball.vel, self.ball.spin = out_vel, out_spin
        self.ball.pos = self.ball.pos + n * (C.BALL_RADIUS * 1.6)
        self.ball.bounces = 0
        self.ball.bounce_side = []
        self._opp_last_hit = now
        self.last_hitter = "ai"
        self.rally_len += 1
        self._keep_alive(now)
        self.invalidate_contact()
        self.opp_contact = None

        self.opp_logger.new_rally()
        self.opp_logger.pending(datalog.StrokeRecord(
            t=now, source=datalog.SRC_AI,
            in_pos=in_pos, in_vel=in_vel, in_spin=in_spin,
            paddle_pos=self.opp.pos.copy(), paddle_normal=n.copy(),
            paddle_vel=self.opp.velocity.copy(),
            out_vel=out_vel.copy(), out_spin=out_spin.copy()))
        print(f"[AI] return {np.linalg.norm(out_vel):4.1f} m/s  "
              f"spin {np.linalg.norm(out_spin):5.0f}  rally {self.rally_len}")

    # ------------------------------------------------------------ scoring
    def _resolve(self, event, pos):
        """Two-sided rally scoring.

        The base class only ever has one player, so a bounce on the far half
        ends the point. Here it is the opponent's turn instead, and the point
        is only decided when someone fails to return.
        """
        if self.rally_over_at:
            return
        # Whoever just erred, the point goes to the *other* side
        if event == "bounce":
            side = "human" if pos[0] < 0 else "ai"
            if self.last_hitter is None:
                return                          # the serve's own first bounce
            if side != self.last_hitter and self.ball.bounces < 2:
                return                          # legal; the other side must play
            # Landing on the far side from whoever hit it is a good shot
            self._settle_ai_log(datalog.RESULT_IN if side != self.last_hitter
                                else datalog.RESULT_OWN_SIDE, pos)
            if side == self.last_hitter:
                # It came down on the hitter's own half -- never cleared
                self._point(side == "ai", "Into your own half"
                            if side == "human" else "AI did not clear the net")
            else:
                # Second bounce on the receiver's side: the receiver missed
                self._point(side == "ai", "AI could not reach it"
                            if side == "ai" else "You missed it")
            self.end_rally()
        elif event == "net":
            self._settle_ai_log(datalog.RESULT_NET, pos)
            self._point(self.last_hitter == "ai",
                        "AI hit the net" if self.last_hitter == "ai"
                        else "Into the net")
            self.end_rally()
        elif event == "out":
            self._settle_ai_log(datalog.RESULT_OUT, pos)
            self._point(self.last_hitter == "ai",
                        "AI hit it out" if self.last_hitter == "ai" else "Out")
            self.ball.active = False
            self.rally_over_at = 0.0

    def _settle_ai_log(self, result, pos):
        """Close out whichever side's stroke record is still open.

        Both loggers hold one pending stroke between the hit and its landing;
        leaving either open means the next serve commits it as a miss.
        """
        if self.last_hitter == "ai" and self.opp_logger.has_pending:
            self.opp_logger.commit(pos, result)
        elif self.last_hitter == "human" and self.logger.has_pending:
            self.logger.commit(pos, result)

    def _keep_alive(self, now):
        """Restart the rally clock on every contact.

        The base class kills the ball RALLY_TIMEOUT seconds after the serve,
        which is right when a rally is one stroke long and wrong here: a good
        exchange was being cut off mid-flight at six seconds. Measuring from
        the last contact instead ends only rallies that have actually
        stopped.
        """
        self.last_serve_time = now
        self.best_rally = max(self.best_rally, self.rally_len)

    def _point(self, to_player, text):
        self.best_rally = max(self.best_rally, self.rally_len)
        self._score(to_player, f"{text}   (rally of {self.rally_len})")

    # ------------------------------------------------------------ loop
    def update(self, task):
        r = super().update(task)
        if self.opp is not None:
            self.hud_opp.setText(f"Opponent: {self.contender.label}")
            self.hud_rally.setText(
                f"Rally {self.rally_len}    best {self.best_rally}")
        return r


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mdn_200000_i8p256",
                    help="checkpoint spec for the opponent; see --list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--difficulty", type=float, default=1.0,
                    help="0.2 gentle and central, 1.0 the full range")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.list:
        for spec, label in compare.available():
            print(f"  {spec:<34} {label}")
        return

    cs = compare.build([args.model])
    if not cs:
        raise SystemExit(f"could not load {args.model}; try --list")

    base.launch_wss()
    RallyAI(cs[0], difficulty=args.difficulty, seed=args.seed).run()


if __name__ == "__main__":
    main()
