"""Paddle motion: ball tracking, hand bias, and a visible swing.

The problem this solves
-----------------------
Hand position means two different things depending on when you look at it.
Before the swing it is *stance*; during the swing it is the stroke itself, a
wide arc covering most of a metre in about 200 ms. Feeding that straight to
the paddle drags the paddle across the table.

The first fix latched the stance at swing onset and arced to the ball from
there, with a hard reach limit. That removed the thrashing but was still
unplayable, for two reasons: the hand-to-paddle mapping spanned a wider range
than the table so small hand movements threw the paddle around, and the assist
was all-or-nothing at the reach boundary.

What is here now is **aim assist**. Hand position still places the paddle --
that is the player's job. The assist measures how far that placement is from
where the ball will actually arrive and pulls the paddle the rest of the way,
fully inside ``aim_full``, fading to nothing by ``aim_none``. Rough placement
is rewarded; being in the wrong place still misses.

Depth is excluded from all of it. The paddle waits at the ready plane and the
swing is what carries it forward, exactly as in real table tennis -- and it
also means a paddle left sitting still cannot return the ball by itself.

The swing itself is a real, visible motion -- backswing, sweep through the
contact point, follow-through -- with amplitude scaled by how hard the player
actually swung, so a gentle push looks gentle and a drive looks explosive.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

READY = "ready"
BACKSWING = "backswing"
FORWARD = "forward"
FOLLOW = "follow"

_EPS = 1e-9


@dataclass
class StrokeConfig:
    """Tuning. All lengths in metres, all times in seconds."""

    # --- aim assist ---------------------------------------------------
    # Hand position is what places the paddle; these decide how much slack
    # it gets. Inside aim_full the paddle snaps onto the ball's arrival
    # point; between aim_full and aim_none the pull fades out smoothly;
    # past aim_none there is no help at all and the ball is missed.
    #
    # This is aim assist, not auto-play: rough placement is rewarded,
    # being in the wrong place is not.
    aim_full: float = 0.26
    aim_none: float = 0.70

    # Time constant for following the aim point while idle.
    track_tau: float = 0.07

    # Swing earlier than this before contact and the stroke completes before
    # the ball arrives -- a real mistime, not a bug.
    max_lead: float = 0.55

    # Backswing takes this share of the time available before contact,
    # capped so it never looks sluggish.
    backswing_share: float = 0.38
    backswing_max: float = 0.16

    # Follow-through duration.
    follow_through: float = 0.20

    # Swing amplitude, scaled by measured swing speed (metres, m per m/s).
    back_base: float = 0.10
    back_per_speed: float = 0.022
    back_max: float = 0.34
    follow_base: float = 0.14
    follow_per_speed: float = 0.032
    follow_max: float = 0.52

    # A swing with no ball to meet still sweeps this far, so the player gets
    # feedback that it registered.
    idle_amplitude: float = 0.30


class StrokeController:
    """Produces the paddle position each frame.

    ``ready`` blends the hand offset with the ball's predicted arrival point;
    a detected swing plays a backswing / sweep / follow-through through that
    point, timed to arrive as the ball does.
    """

    def __init__(self, config: Optional[StrokeConfig] = None, home=None):
        self.cfg = config or StrokeConfig()
        self.state = READY
        self.pos = np.zeros(3) if home is None else np.array(home, dtype=float)

        self._prev_swinging = False
        self._origin = self.pos.copy()      # where the swing started from
        self._back_point = self.pos.copy()
        self._contact_point = self.pos.copy()
        self._follow_point = self.pos.copy()
        self._t_start = 0.0
        self._t_back = 0.0
        self._t_contact = 0.0
        self._t_end = 0.0

        self.connected = False              # was there a ball to meet
        self.time_to_contact = None         # at swing onset, seconds
        self.last_aim_weight = 0.0          # current assist strength, for the HUD

    # ------------------------------------------------------------------
    def reset(self, pos=None):
        self.state = READY
        self._prev_swinging = False
        self.connected = False
        self.time_to_contact = None
        if pos is not None:
            self.pos = np.array(pos, dtype=float)

    # ------------------------------------------------------------------
    def aim_weight(self, hand_point, crossing):
        """How much the assist pulls the paddle onto the ball, 0..1.

        Full inside ``aim_full``, fading smoothly to nothing at ``aim_none``.
        Distance is measured across the table and in height only -- depth is
        the swing's job, not the player's.
        """
        if crossing is None:
            return 0.0
        cfg = self.cfg
        d = float(np.linalg.norm(
            np.asarray(crossing.pos, dtype=float)[1:] - np.asarray(hand_point)[1:]
        ))
        if d <= cfg.aim_full:
            return 1.0
        if d >= cfg.aim_none:
            return 0.0
        t = (cfg.aim_none - d) / max(cfg.aim_none - cfg.aim_full, 1e-6)
        return float(t * t * (3.0 - 2.0 * t))       # smoothstep

    def tracked_point(self, hand_point, crossing):
        """Where the paddle waits for the ball.

        Only the across-table and height axes are assisted. Depth is
        deliberately held at the ready plane: parking the paddle on the
        contact point would let a stationary paddle return the ball by
        itself, which removes the game. Bringing it forward is what the
        swing is for, as in real table tennis.
        """
        base = np.array(hand_point, dtype=float)
        w = self.aim_weight(base, crossing)
        self.last_aim_weight = w
        if w <= 0.0:
            return base
        target = base.copy()
        arrival = np.asarray(crossing.pos, dtype=float)
        target[1:] = base[1:] + (arrival[1:] - base[1:]) * w
        return target

    # ------------------------------------------------------------------
    def _amplitudes(self, speed):
        cfg = self.cfg
        back = min(cfg.back_base + cfg.back_per_speed * speed, cfg.back_max)
        follow = min(cfg.follow_base + cfg.follow_per_speed * speed, cfg.follow_max)
        return back, follow

    def _plan(self, now, target, swing_dir, speed, crossing):
        """Lay out backswing / sweep / follow-through for a swing just started."""
        cfg = self.cfg
        back_amp, follow_amp = self._amplitudes(speed)

        if crossing is not None and crossing.time <= cfg.max_lead:
            # target already carries the aim assist, so a weak assist means
            # the swing genuinely passes wide instead of snapping to the ball
            self.connected = self.last_aim_weight > 0.0
            self.time_to_contact = crossing.time
            contact = np.array(target, dtype=float)
            # Across and up the target decides, but the depth has to be the
            # ball's: waiting happens at the ready plane, and the whole point
            # of the swing is to arrive at the strike plane exactly when the
            # ball does. Leaving the ready depth here put the paddle in the
            # right place at the wrong distance.
            contact[0] = float(crossing.pos[0])
            available = max(crossing.time, 1e-3)
        else:
            # Nothing to meet, or swung far too early: sweep through the
            # current position so the stroke is still visible, and miss.
            self.connected = False
            self.time_to_contact = None if crossing is None else crossing.time
            contact = target + swing_dir * (cfg.idle_amplitude * 0.35)
            available = 0.18

        back_time = min(available * cfg.backswing_share, cfg.backswing_max)

        self._origin = self.pos.copy()
        self._back_point = contact - swing_dir * back_amp
        self._contact_point = contact
        self._follow_point = contact + swing_dir * follow_amp

        self._t_start = now
        self._t_back = now + back_time
        self._t_contact = now + available
        self._t_end = self._t_contact + cfg.follow_through
        self.state = BACKSWING

    # ------------------------------------------------------------------
    def update(self, now, dt, hand_point, swinging, swing_velocity,
               crossing=None):
        """Advance one frame and return the paddle position.

        Parameters
        ----------
        now, dt : float
            Wall-clock seconds and frame time.
        hand_point : (3,)
            Where hand tracking says the paddle should be. This is the
            player's placement; the assist only refines it.
        swinging : bool
            ``SwingEstimator.swinging``; its rising edge starts a stroke.
        swing_velocity : (3,)
            World-frame swing velocity; sets the arc direction and amplitude.
        crossing : physics.PlaneCrossing, optional
            Predicted ball arrival at the strike plane.
        """
        sv = np.asarray(swing_velocity, dtype=float)
        speed = float(np.linalg.norm(sv))
        swing_dir = sv / speed if speed > _EPS else np.array([1.0, 0.0, 0.0])

        target = self.tracked_point(hand_point, crossing)

        started = swinging and not self._prev_swinging
        self._prev_swinging = swinging
        if started and self.state == READY:
            self._plan(now, target, swing_dir, speed, crossing)

        if self.state == BACKSWING:
            if now < self._t_back:
                span = max(1e-4, self._t_back - self._t_start)
                s = float(np.clip((now - self._t_start) / span, 0.0, 1.0))
                # Decelerate into the top of the backswing
                self.pos = self._origin + (self._back_point - self._origin) * (
                    1.0 - (1.0 - s) ** 2
                )
            else:
                self.state = FORWARD

        if self.state == FORWARD:
            if now < self._t_contact:
                span = max(1e-4, self._t_contact - self._t_back)
                s = float(np.clip((now - self._t_back) / span, 0.0, 1.0))
                # Ease in: a real stroke is fastest at contact, and s**2 has
                # its steepest slope at s = 1
                self.pos = self._back_point + (
                    self._contact_point - self._back_point
                ) * (s * s)
            else:
                self.state = FOLLOW

        if self.state == FOLLOW:
            if now < self._t_end:
                span = max(1e-4, self._t_end - self._t_contact)
                s = float(np.clip((now - self._t_contact) / span, 0.0, 1.0))
                self.pos = self._contact_point + (
                    self._follow_point - self._contact_point
                ) * (1.0 - (1.0 - s) ** 2)
            else:
                self.state = READY

        if self.state == READY:
            alpha = 1.0 - np.exp(-dt / max(self.cfg.track_tau, 1e-4))
            self.pos = self.pos + (target - self.pos) * alpha

        return self.pos

    # ------------------------------------------------------------------
    @property
    def busy(self):
        """True while a stroke is playing out."""
        return self.state != READY

    @property
    def swing_phase(self):
        return self.state
