"""Ask for a shot, then watch the models try to play it -- side by side.

    python ai_play.py --compare trainsize      # one method at every data size
    python ai_play.py --compare method         # every method at the top size
    python ai_play.py --compare mdn_200000_i8p256 policy_200000_i8p256 oracle
    python ai_play.py --list                   # what can be compared
    python ai_play.py --model runs/checkpoints/mdn_20000_i8p256.pt   # just one

You choose the shot -- where it should land, how fast it should arrive and
how it should spin -- and the model has to find a stroke that delivers it.
Nothing serves until you ask for it, so there is time to dial the request in.

With --compare, the first planner plays the ball for real and the rest are
planned against the identical incoming ball and the identical request, then
drawn beside it. The spread between the arcs is the difference between the
methods and nothing else. [1]-[9] show and hide them.

Controls:
    click        put the target where you want the ball to land
    arrows       nudge the target
    W / S        landing speed
    E / D        topspin   (D past zero asks for backspin)
    Z / C        sidespin
    enter / R    serve   (right-click does too)
    A            auto-serve on/off (keeps replaying the same request)
    G            random request
    space        replay the last stroke breakdown
    F1           key hints

Each rally runs through four views. The stroke freezes into a slow-motion
breakdown at contact, showing the blade normal and the paddle velocity split
into its drive and brush components -- those two are the whole of the
technique, driving along the blade normal buys speed and brushing across the
face buys spin. Then the camera pulls out and follows the ball over the net
so you can see where it actually goes, and the landing is held on screen
next to what you asked for.
"""

import argparse
import os
import threading
import time

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (CardMaker, Filename, LineSegs, Plane, Point3,
                          TextNode, TransparencyAttrib, Vec3, loadPrcFileData)

import main_wss as base
from pingpong import (compare, constants as C, datalog, dataset, quat,
                      stroke_card)

# The old default window was small enough that the breakdown text and the
# arrow labels ran into each other. This mode is meant to be read, not just
# played, so it asks for room.
loadPrcFileData("", "win-size 1600 900")

# Distinct hues from the validated categorical palette. Each arrow also
# carries a text label, so identity never rests on colour alone.
COL_NORMAL = (0.85, 0.85, 0.90, 1.0)
COL_DRIVE = (0.11, 0.69, 0.48, 1.0)     # aqua
COL_BRUSH = (0.93, 0.63, 0.00, 1.0)     # yellow
COL_IN = (0.92, 0.41, 0.20, 1.0)        # orange
COL_OUT = (0.16, 0.47, 0.84, 1.0)       # blue
COL_ASK = (0.45, 0.90, 1.00, 1.0)       # what you requested
COL_TARGET = (0.25, 1.00, 0.35, 1.0)    # the target ring on the table
COL_LAND = (1.00, 0.42, 0.42, 1.0)      # where it actually landed
COL_DIM = (0.72, 0.74, 0.80, 1.0)

# ---------------------------------------------------------------- phases
SETUP = "setup"          # no ball; you are choosing the shot
INCOMING = "incoming"    # served, on its way to the model
BREAKDOWN = "breakdown"  # frozen at contact, explaining the stroke
FLIGHT = "flight"        # following the return over the net
RESULT = "result"        # landing held next to the request

BREAKDOWN_SECONDS = 2.6
# Dead stop, not slow motion. At 0.12x the breakdown still let 0.34 s of
# flight run, and a 6 m/s return crosses the whole 2.74 m table in that --
# the ball had already landed by the time the camera reached the wide view.
BREAKDOWN_SCALE = 0.0
# The pull-out happens inside the breakdown, while the ball is still stopped
# at the blade, so the flight starts with the camera already in place.
PULLBACK_LEAD = 0.9
# Fast enough not to drag, slow enough to actually watch the arc and the
# spin curve the ball sideways
FLIGHT_SCALE = 0.30
LANDING_BEAT = 0.7       # keep rolling briefly after it touches down
RESULT_SECONDS = 4.0
INCOMING_TIMEOUT = 9.0   # the model somehow never made contact

# On-screen length of the longest arrow, metres. Arrows are scaled to this
# rather than by a fixed metres-per-m/s factor, so a hard stroke stays in
# frame while the ratios between the arrows still read correctly.
ARROW_MAX_LEN = 0.42

# Camera stations, (eye, look-at). The play view sits behind our end looking
# down the table, which is the one angle where the breakdown cannot be read
# -- the blade normal and the drive component both point away from the
# camera and project to almost nothing. The wide view is side-on so the ball
# crossing to the far half is actually visible, which the old fixed contact
# camera never showed.
VIEW_PLAY = ((-3.15, 0.00, C.TABLE_H + 1.05), (0.20, 0.00, C.TABLE_H + 0.12))
VIEW_WIDE = ((-1.55, -3.95, 2.45), (0.15, 0.00, C.TABLE_H + 0.16))

# Editing limits for the request. These are the ranges the random generator
# draws from, i.e. shots of roughly the kind the training goals covered --
# you can still ask for a corner of the box that nothing reaches, which is
# the interesting part.
GOAL_LO = np.array([0.35, -C.TABLE_W / 2 + 0.10, 3.0, -320.0, -180.0])
GOAL_HI = np.array([C.TABLE_L / 2 - 0.10, C.TABLE_W / 2 - 0.10, 8.0, 340.0, 180.0])
GOAL_STEP = np.array([0.05, 0.05, 0.25, 20.0, 20.0])

# Fallback box for [G], used only if the achievable pool cannot be built.
RANDOM_LO = np.array([0.35, -C.TABLE_W / 2 + 0.10, 3.5, -250.0, -140.0])
RANDOM_HI = np.array([C.TABLE_L / 2 - 0.10, C.TABLE_W / 2 - 0.10, 7.5, 320.0, 140.0])

# Spread of arrivals at the strike plane, measured over real rallies:
# y, z, vx, vy, vz, wx, wy, wz. Used to build the pool of requests that [G]
# draws from -- see _build_goal_pool.
ARRIVAL_LO = np.array([-0.55, 0.84, -4.8, -1.2, -2.2, -40.0, -190.0, -70.0])
ARRIVAL_HI = np.array([0.55, 1.06, -1.5, 1.2, 2.2, 40.0, 20.0, 70.0])
GOAL_POOL_N = 512


def arrow(name, origin, vec, colour, thickness=6.0, head=0.075):
    """A 3D arrow as a LineSegs node. Returns (segs, tip)."""
    o = np.asarray(origin, dtype=float)
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    segs = LineSegs(name)
    segs.set_thickness(thickness)
    segs.set_color(*colour)
    if n < 1e-6:
        return segs, o
    d = v / n
    tip = o + v
    segs.move_to(*o)
    segs.draw_to(*tip)
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(d, up)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(d, np.array([0.0, 1.0, 0.0]))
    side /= max(np.linalg.norm(side), 1e-9)
    other = np.cross(d, side)
    for s in (side, -side, other, -other):
        segs.move_to(*tip)
        segs.draw_to(*(tip - d * head + s * head * 0.55))
    return segs, tip


def spin_word(topspin, sidespin):
    """Plain-language reading of a requested spin, for the HUD."""
    if topspin > 40:
        t = "topspin"
    elif topspin < -40:
        t = "BACKSPIN"
    else:
        t = "flat"
    if sidespin > 40:
        return f"{t} + side right"
    if sidespin < -40:
        return f"{t} + side left"
    return t


class AIPlay(base.Game):
    def __init__(self, contenders, goal=None, auto=False, seed=0):
        # The first contender plays the ball for real; the rest are ghosts
        # planned against the identical problem and drawn beside it.
        self.contenders = contenders
        self.primary = contenders[0]
        self.agent = self.primary.agent
        self.agent_label = self.primary.label
        self.shots = []
        self.rng = np.random.default_rng(seed)

        self.goal = None
        self.auto_serve = auto
        self.phase = SETUP
        self.phase_until = 0.0
        self.flight_started = 0.0
        self.landing = None          # (event, position) of the decided rally
        self.landing_at = 0.0

        self.planned_action = None
        self.plan_state = None
        self.plan_ball = None        # (pos, vel, spin) the plan was made for
        self.card = None
        self.time_scale = 1.0
        self._pulled_back = False
        self._hit_pose = None
        self._viz = []               # breakdown arrows, cleared per stroke
        self._cmp_viz = []           # contender arcs, cleared per rally
        self._target_viz = []        # the request marker, cleared per edit
        self._target_label = None
        self._cam_from = None
        self._cam_to = None
        self._cam_t0 = 0.0
        self._cam_dur = 1.0
        self._cam_look = np.asarray(VIEW_PLAY[1], dtype=float)

        super().__init__()

        # The base HUD is built for a human player -- score, connection state,
        # swing speed. None of it applies here and all of it was competing
        # for the same corner as the request, which is what made the screen
        # unreadable. Only the live ball readout survives.
        for h in (self.hud_score, self.hud_state, self.hud_swing, self.hud_conn,
                  self.hud_help):
            h.hide()

        self.hud_title = self.hud_text((0.06, -0.13), 0.058, TextNode.ALeft,
                                       (1, 1, 1, 1))

        # This block sits over the table, where white text on bright blue
        # crossed by the white court lines was barely readable. A backing
        # plate fixes that without darkening the scene itself. It is built
        # first so the text, added after, draws in front of it.
        self._hud_plate = self._make_hud_plate()

        # One line per node so colour can carry meaning: the request is cyan
        # throughout, the achieved shot is blue, and the stroke breakdown
        # matches its arrows.
        def panel(row, size, colour):
            return self.hud_text((0.05, 0.72 - row * 0.062), size,
                                 TextNode.ALeft, colour, corner="bottomleft")

        self.hud_keys = panel(-1.1, 0.036, COL_DIM)
        self.hud_ask = [panel(i, 0.044, COL_ASK) for i in range(5)]
        self.hud_blade = panel(5.4, 0.041, COL_NORMAL)
        self.hud_drive = panel(6.3, 0.041, COL_DRIVE)
        self.hud_brush = panel(7.2, 0.041, COL_BRUSH)
        self.hud_mu = panel(8.1, 0.038, COL_DIM)
        self.hud_got = panel(9.3, 0.043, COL_OUT)
        self.keys_visible = True
        self.hud_cmp = self._make_compare_hud()

        # strokes.csv is the human dataset that analyze_strokes.py reads.
        # These strokes are the model's, so they go somewhere else under
        # their own label rather than quietly diluting it.
        self.logger = datalog.StrokeLogger("runs/ai_play_strokes.csv",
                                           source=datalog.SRC_AI)

        self._bind_keys()
        # Finding reachable requests costs a few seconds of simulation, which
        # is not worth a frozen window on startup. [G] falls back to the
        # plain box until the pool arrives; rebinding the attribute is the
        # only shared state, so no lock is needed.
        self._goal_pool = np.empty((0, 5))
        threading.Thread(target=self._fill_goal_pool, daemon=True).start()

        self.set_goal(np.asarray(goal, dtype=np.float32) if goal is not None
                      else self._random_goal())
        self.enter_setup()
        print(f"[AI] playing: {self.primary.label}")
        for i, c in enumerate(self.contenders[1:], start=2):
            print(f"[AI] comparing against [{i}] {c.label}")
        print("[AI] click the table to place the target, "
              "W S / E D / Z C for speed and spin, then enter to serve")

    # ------------------------------------------------------------ input
    def _bind_keys(self):
        """Rebind the human controls to editing the request.

        The base class points the arrows and space at the paddle, but the
        model places and swings the paddle itself, so those keys are free.
        """
        moves = {"arrow_up": (0, +1), "arrow_down": (0, -1),
                 "arrow_left": (1, +1), "arrow_right": (1, -1)}
        for key, (dim, sign) in moves.items():
            self.accept(key, self.adjust, [dim, sign])
            self.accept(key + "-repeat", self.adjust, [dim, sign])
        for key, (dim, sign) in {"w": (2, +1), "s": (2, -1),
                                 "e": (3, +1), "d": (3, -1),
                                 "z": (4, -1), "c": (4, +1)}.items():
            self.accept(key, self.adjust, [dim, sign])
            self.accept(key + "-repeat", self.adjust, [dim, sign])

        self.accept("enter", self.request_serve)
        self.accept("r", self.request_serve)
        # Point at the spot you want the ball to land. Nudging a target
        # across the table one arrow-key step at a time is not how anyone
        # thinks about placement.
        self.accept("mouse1", self.click_target)
        # Right-click serves, so a whole rally can be set up and started
        # without leaving the mouse
        self.accept("mouse3", self.request_serve)
        for i in range(min(9, len(self.contenders))):
            self.accept(str(i + 1), self.toggle_contender, [i])
        self.accept("g", self.new_goal)
        self.accept("a", self.toggle_auto)
        self.accept("space", self.replay_last)
        self.accept("f1", self.toggle_help)

    # Top of the HUD block, in aspect2d units above the bottom-left corner
    HUD_TOP = 0.85

    def _make_hud_plate(self):
        if self.headless:
            return None
        cm = CardMaker("hud_plate")
        # Unit height, so the plate can be resized to whatever is on screen
        # by scaling rather than rebuilt
        cm.set_frame(-0.02, 2.22, 0.0, 1.0)
        plate = self.a2dBottomLeft.attach_new_node(cm.generate())
        plate.set_color(0.02, 0.03, 0.06, 0.58)
        plate.set_transparency(TransparencyAttrib.M_alpha)
        return plate

    def _fit_plate(self, bottom):
        """Shrink the plate to the lines actually being shown.

        A full-height slab left a large dark rectangle over the table while
        you were still choosing the shot and there was nothing under it.
        """
        if self._hud_plate is None:
            return
        self._hud_plate.set_scale(1, 1, max(0.05, self.HUD_TOP - bottom))
        self._hud_plate.set_pos(0, 0.5, bottom)      # Y is depth in 2D: behind the text

    def _make_compare_hud(self):
        """A colour-keyed row per contender, top right.

        Monospace if the system has it: the whole point is reading a column
        down the page, and the proportional default staggers the digits.
        """
        if self.headless or len(self.contenders) < 2:
            return []
        # Panda3D wants its own path syntax, not the OS's -- handing it
        # "C:/Windows/..." fails with "unable to find font file" and silently
        # leaves the table proportional.
        font = None
        for p in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"):
            if not os.path.exists(p):
                continue
            try:
                f = self.loader.loadFont(Filename.from_os_specific(p))
            except Exception:
                continue
            if f is not None and f.is_valid():
                font = f
                break
        return [OnscreenText(text="", pos=(-1.48, -0.16 - i * 0.058), scale=0.043,
                             fg=(1, 1, 1, 1), align=TextNode.ALeft, mayChange=True,
                             font=font, parent=self.a2dTopRight,
                             shadow=(0, 0, 0, 0.8))
                for i in range(len(self.contenders) + 2)]

    def toggle_contender(self, i):
        if i >= len(self.contenders):
            return
        c = self.contenders[i]
        c.on = not c.on
        # The shots are already planned for every contender, on or off, so a
        # toggle takes effect on the current rally rather than the next one
        if self.shots and self.phase in (FLIGHT, RESULT):
            self.draw_comparison()

    def toggle_help(self):
        self.keys_visible = not self.keys_visible

    def toggle_auto(self):
        self.auto_serve = not self.auto_serve

    # ------------------------------------------------------------ the request
    def _fill_goal_pool(self):
        try:
            pool = self._build_goal_pool()
        except Exception as e:
            print(f"[AI] achievable-goal pool unavailable ({e}); "
                  "[G] keeps sampling the plain box")
            return
        self._goal_pool = pool
        print(f"[AI] [G] now draws from {len(pool)} reachable requests")

    def _build_goal_pool(self, n=GOAL_POOL_N):
        """Requests that some stroke is actually known to deliver.

        Picking the five numbers independently is the trap dataset.py already
        warns about: the action space has five dimensions and so does the
        goal, so at best it is exactly determined, and an arbitrary
        combination of place, speed and spin usually has no solution at all.
        A random button that mostly asks the impossible makes every model
        look equally bad, because the residual is the request's fault.

        So the pool is built the way the training goals were: play a random
        stroke against a plausible arrival and keep whatever it produced.
        The arrival that eventually comes will not be the identical one, but
        it is drawn from the same spread, so the request stays close to
        reachable. Typing an impossible request by hand is still allowed --
        that is the interesting experiment, and it should be a decision.
        """
        s = self.rng.uniform(ARRIVAL_LO, ARRIVAL_HI, size=(n, 8))
        pos = np.stack([np.full(n, base.PADDLE_STRIKE_X), s[:, 0], s[:, 1]], axis=1)
        goals, ok = dataset.achievable_goals(pos, s[:, 2:5], s[:, 5:8], self.rng)
        return goals[ok]

    def _random_goal(self):
        if len(self._goal_pool):
            g = self._goal_pool[self.rng.integers(len(self._goal_pool))]
            return np.clip(g, GOAL_LO, GOAL_HI).astype(np.float32)
        return self.rng.uniform(RANDOM_LO, RANDOM_HI).astype(np.float32)

    def set_goal(self, goal):
        self.goal = np.clip(np.asarray(goal, dtype=np.float32), GOAL_LO, GOAL_HI)
        # The plan is only valid for the goal it was made for. Re-planning is
        # sub-millisecond for a policy, so editing mid-flight simply works.
        self.planned_action = None
        self.draw_target()

    def adjust(self, dim, sign):
        g = self.goal.copy()
        g[dim] += sign * GOAL_STEP[dim]
        self.set_goal(g)

    def click_target(self):
        """Put the target where the mouse is pointing at the table surface.

        The click is intersected with the plane of the table rather than
        picked against geometry, so it still reads correctly from any of the
        camera stations and does not need collision solids.
        """
        if self.headless or not self.mouseWatcherNode.has_mouse():
            return
        near, far = Point3(), Point3()
        self.camLens.extrude(self.mouseWatcherNode.get_mouse(), near, far)
        near = self.render.get_relative_point(self.cam, near)
        far = self.render.get_relative_point(self.cam, far)
        hit = Point3()
        table = Plane(Vec3(0, 0, 1), Point3(0, 0, C.TABLE_TOP_Z))
        if not table.intersects_line(hit, near, far):
            return
        # Clicks off the opponent's half are clamped onto it rather than
        # ignored, so a click near an edge still does the obvious thing
        g = self.goal.copy()
        g[0] = np.clip(hit[0], GOAL_LO[0], GOAL_HI[0])
        g[1] = np.clip(hit[1], GOAL_LO[1], GOAL_HI[1])
        self.set_goal(g)

    def new_goal(self):
        self.set_goal(self._random_goal())

    # ------------------------------------------------------------ phases
    def enter_setup(self):
        self.phase = SETUP
        self.time_scale = 1.0
        self.ball.active = False
        self.landing = None
        self.card = None
        self._hit_pose = None
        self.planned_action = None
        self._clear_viz()
        self._clear_cmp()
        self.draw_target()
        self.cam_move(*VIEW_WIDE, secs=0.7)

    def request_serve(self):
        if self.phase not in (SETUP, RESULT):
            return
        self._clear_viz()
        self.landing = None
        self.card = None
        self._hit_pose = None
        self.serve_ball()
        self.phase = INCOMING
        self.time_scale = 1.0
        self.phase_until = time.time() + INCOMING_TIMEOUT
        self.cam_move(*VIEW_PLAY, secs=0.5)

    def serve_ball(self):
        super().serve_ball()
        # A plan is only valid for the ball it was made for. Without this the
        # auto-serve loop replayed the previous rally's stroke against a
        # freshly served ball.
        self.planned_action = None
        self.plan_state = None
        self.plan_ball = None
        self.shots = []
        self._clear_cmp()

    def enter_breakdown(self):
        self.phase = BREAKDOWN
        self.time_scale = BREAKDOWN_SCALE
        self.phase_until = time.time() + BREAKDOWN_SECONDS
        self._pulled_back = False
        self.draw_breakdown()
        self.cam_move(*self._contact_view(), secs=0.35)
        print("\n" + stroke_card.format_card(self.card))
        self.run_comparison()

    def run_comparison(self):
        """Plan every contender on the ball the primary just hit.

        Done here rather than during the approach because the CEM oracle
        needs a few hundred milliseconds and the breakdown is a still frame,
        so the cost is invisible. The primary is scored on the stroke it
        actually played, not a re-plan.
        """
        if len(self.contenders) < 2 or self.plan_ball is None:
            return
        pos, vel, spin = self.plan_ball
        self.primary.forced_action = self.planned_action
        try:
            self.shots = compare.play(self.contenders, pos, vel, spin, self.goal)
        finally:
            self.primary.forced_action = None
        print("\n" + compare.table(self.shots, self.goal) + "\n")

    def enter_flight(self):
        self.phase = FLIGHT
        self.time_scale = FLIGHT_SCALE
        self.flight_started = time.time()
        # The arrows explained the contact; from here the ball is the subject
        self._clear_viz()
        self.draw_comparison()

    def enter_result(self):
        self.phase = RESULT
        # Effectively frozen, so the numbers can be read against the marks on
        # the table rather than chased across it
        self.time_scale = 0.0
        self.phase_until = time.time() + RESULT_SECONDS
        self.draw_landing()

    def _contact_view(self):
        at = np.asarray(self.card["incoming"]["position"], dtype=float)
        side = -1.0 if at[1] >= 0 else 1.0        # view from the emptier side
        # Close enough that the arrows fill the frame. At the old 2.75 m the
        # whole breakdown sat in one corner with half the screen empty.
        eye = (at[0] + 0.42, at[1] + side * 2.15, at[2] + 0.80)
        return eye, (at[0] + 0.18, at[1], at[2] + 0.10)

    # ------------------------------------------------------------ camera
    def cam_move(self, eye, look, secs=0.6):
        """Glide to a camera station instead of cutting.

        Cutting between the contact close-up and the wide view lost the ball:
        you could not tell that the thing now crossing the net was the same
        one you had just watched leave the blade.
        """
        if self.headless:
            return
        p = self.camera.get_pos()
        self._cam_from = (np.array([p[0], p[1], p[2]], dtype=float),
                          self._cam_look.copy())
        self._cam_to = (np.asarray(eye, dtype=float),
                        np.asarray(look, dtype=float))
        self._cam_t0 = time.time()
        self._cam_dur = max(1e-3, secs)

    def _tick_camera(self):
        if self.headless or self._cam_to is None:
            return
        u = float(np.clip((time.time() - self._cam_t0) / self._cam_dur, 0.0, 1.0))
        s = u * u * (3.0 - 2.0 * u)               # smoothstep, no jerk at either end
        eye = self._cam_from[0] * (1 - s) + self._cam_to[0] * s
        look = self._cam_from[1] * (1 - s) + self._cam_to[1] * s
        self.camera.set_pos(*eye)
        self.camera.look_at(*look)
        self._cam_look = look
        if u >= 1.0:
            self._cam_to = None

    # ------------------------------------------------------------ control
    def _contact_allowed(self):
        # The model positions the paddle itself rather than swinging through
        # the player's stroke state machine
        return self.phase == INCOMING

    def _control_paddle(self, st, dt):
        # Hold the blade exactly as it was at contact for the whole
        # breakdown; snapping it back to neutral contradicted the arrows
        # that were being drawn to explain that very pose.
        if self.phase == BREAKDOWN and self._hit_pose is not None:
            pos, pq, normal = self._hit_pose
            self.paddle.pos = pos.copy()
            self.paddle.normal = normal.copy()
            self.apply_paddle_transform(pq)
            return

        c = self.contact
        if c is None or self.phase != INCOMING:
            self.apply_paddle_transform(quat.NEUTRAL_PADDLE)
            return

        # Plan once per ball, from the predicted state at contact
        if self.planned_action is None:
            state = dataset.encode_state(c.pos[None, :], c.vel[None, :],
                                         c.spin[None, :]).astype(np.float32)
            g = self.goal[None, :]
            # Only the primary plans here. Planning the whole field would
            # stall the frame the ball is still approaching in; the others
            # are planned at contact, where the scene is frozen anyway.
            if self.primary.is_oracle:
                self.planned_action = self.agent.act(
                    state, g, c.pos[None, :], c.vel[None, :], c.spin[None, :])[0]
            else:
                self.planned_action = self.agent.act(state, g)[0]
            self.plan_state = state[0]
            self.plan_ball = (c.pos.copy(), c.vel.copy(), c.spin.copy())

        normals, pvels = dataset.decode_action(self.planned_action[None, :])
        self.paddle.normal = normals[0]
        self.paddle.velocity = pvels[0]
        # Sit exactly where the ball will arrive; this is a demonstration of
        # the stroke, not a test of the positioning
        self.paddle.pos = c.pos.copy()
        self.apply_paddle_transform(quat.look_quat(normals[0]))

    # ------------------------------------------------------------ contact
    def _try_hit(self, now, ball_prev, paddle_prev):
        before = self.player_hit
        super()._try_hit(now, ball_prev, paddle_prev)
        if self.player_hit and not before and self.planned_action is not None:
            self._hit_pose = (self.paddle.pos.copy(),
                              quat.look_quat(self.paddle.normal),
                              self.paddle.normal.copy())
            self.card = stroke_card.describe(self.plan_state, self.planned_action,
                                             goal=self.goal, verify=True)
            self.enter_breakdown()

    def end_rally(self, linger=None):
        # The phase machine owns how long the ball lives. The base class's
        # 1.1 s linger is measured in real time and would cut the flight off
        # part-way through the slow motion.
        super().end_rally(linger=999.0)

    def _resolve(self, event, pos):
        already = bool(self.rally_over_at)
        super()._resolve(event, pos)
        if already or self.landing is not None or not self.player_hit:
            return
        self.landing = (event, np.asarray(pos, dtype=float)
                        if pos is not None else None)
        self.landing_at = time.time()

    def replay_last(self):
        if self.card is not None and self.phase in (FLIGHT, RESULT):
            self.enter_breakdown()

    # ------------------------------------------------------------ visuals
    def _clear_viz(self):
        for np_ in self._viz:
            np_.remove_node()
        self._viz = []

    def _clear_cmp(self):
        for np_ in self._cmp_viz:
            np_.remove_node()
        self._cmp_viz = []

    def draw_comparison(self):
        """Every contender's shot, drawn on the same table at once.

        The ball you watch is the primary's. The others are what the same
        request would have produced from the identical incoming ball, so the
        spread between the arcs is the difference between the methods and
        nothing else.
        """
        self._clear_cmp()
        if len(self.shots) < 2 or self.headless:
            return
        # The base class draws the primary's predicted path in yellow, which
        # here would be a sixth line meaning the same as one of these
        self._clear_traj()

        shown = [s for s in self.shots if s.contender.on]
        for i, s in enumerate(shown):
            col = s.contender.rgba
            primary = s.contender is self.primary
            if s.trajectory is not None and len(s.trajectory) > 1:
                segs = LineSegs("cmp_" + s.contender.label)
                segs.set_thickness(4.0 if primary else 2.2)
                segs.set_color(col[0], col[1], col[2], 1.0 if primary else 0.75)
                segs.move_to(*s.trajectory[0])
                for p in s.trajectory[1:]:
                    segs.draw_to(*p)
                np_ = self.render.attach_new_node(segs.create())
                np_.set_light_off()
                np_.set_transparency(TransparencyAttrib.M_alpha)
                self._cmp_viz.append(np_)

            if s.landing is None:
                continue
            lx, ly = float(s.landing[0]), float(s.landing[1])
            self._cmp_viz.append(self._ring(
                "cmp_land", lx, ly, 0.055, col,
                z=C.TABLE_TOP_Z + 0.008, thickness=3.5 if primary else 2.5))
            # Landings cluster tightly when the methods agree, which is
            # exactly when the labels would overlap, so they are stacked in
            # height with a leader down to the mark
            lift = 0.16 + 0.13 * i
            pole = LineSegs("cmp_pole")
            pole.set_thickness(1.2)
            pole.set_color(col[0], col[1], col[2], 0.5)
            pole.move_to(lx, ly, C.TABLE_TOP_Z + 0.008)
            pole.draw_to(lx, ly, C.TABLE_TOP_Z + lift)
            np_ = self.render.attach_new_node(pole.create())
            np_.set_light_off()
            np_.set_transparency(TransparencyAttrib.M_alpha)
            self._cmp_viz.append(np_)
            self._cmp_viz.append(self._label(
                "cmp_txt", (lx, ly, C.TABLE_TOP_Z + lift),
                f"{s.contender.label}  {s.place_error * 100:.0f}cm",
                col, scale=0.45))

    def _label(self, name, at, text, colour, along=None, scale=0.42):
        """A billboarded text label with a dark plate behind it.

        Plain text over a bright table is close to unreadable, and the labels
        were also being hidden by the paddle, so each one gets a backing card
        and is drawn on top of the scene.
        """
        at = np.asarray(at, dtype=float)
        if along is not None:
            # Push the plate well along the arrow. At the old offset every
            # label piled up around the shared contact point and they
            # overlapped each other into a single unreadable block.
            d = np.asarray(along, dtype=float)
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                at = at + d / n * max(0.16, n * 0.30)
        holder = self.render.attach_new_node("lbl_" + name)
        holder.set_pos(*at)
        if self.headless:
            return holder

        from direct.gui.OnscreenText import OnscreenText
        plate = self._make_card(len(text) * 0.039 + 0.05, 0.10)
        plate.reparent_to(holder)
        plate.set_color(0.03, 0.03, 0.05, 0.82)
        plate.set_transparency(TransparencyAttrib.M_alpha)
        plate.set_pos(0, 0.002, 0)

        txt = OnscreenText(text=text, scale=0.075, fg=colour,
                           align=TextNode.ACenter, mayChange=False)
        txt.reparent_to(holder)
        txt.set_pos(0, 0, -0.025)

        holder.set_billboard_point_eye()
        holder.set_scale(scale)
        holder.set_light_off()
        holder.set_depth_test(False)
        holder.set_bin("fixed", 45)
        return holder

    def _ring(self, name, cx, cy, r, colour, z=None, thickness=3.0):
        segs = LineSegs(name)
        segs.set_thickness(thickness)
        segs.set_color(*colour)
        z = C.TABLE_TOP_Z + 0.004 if z is None else z
        for i in range(41):
            a = 2 * np.pi * i / 40
            (segs.move_to if i == 0 else segs.draw_to)(
                cx + r * np.cos(a), cy + r * np.sin(a), z)
        np_ = self.render.attach_new_node(segs.create())
        np_.set_light_off()
        np_.set_transparency(TransparencyAttrib.M_alpha)
        return np_

    def draw_target(self):
        """The request, drawn on the table where it has to land.

        A number in the corner of the screen is not something you can aim by.
        The ring is on the surface, and the pole above it keeps the target
        findable from the wide camera where a flat ring is nearly edge-on.
        """
        for np_ in self._target_viz:
            np_.remove_node()
        self._target_viz = []
        self._target_label = None
        if self.goal is None or self.headless:
            return
        gx, gy = float(self.goal[0]), float(self.goal[1])

        self._target_viz.append(self._ring("target", gx, gy, 0.09, COL_TARGET,
                                           thickness=3.5))
        self._target_viz.append(self._ring("target_outer", gx, gy, 0.15,
                                           COL_TARGET[:3] + (0.35,), thickness=2.0))

        pole = LineSegs("target_pole")
        pole.set_thickness(2.5)
        pole.set_color(COL_TARGET[0], COL_TARGET[1], COL_TARGET[2], 0.55)
        pole.move_to(gx, gy, C.TABLE_TOP_Z + 0.005)
        pole.draw_to(gx, gy, C.TABLE_TOP_Z + 0.42)
        np_ = self.render.attach_new_node(pole.create())
        np_.set_light_off()
        np_.set_transparency(TransparencyAttrib.M_alpha)
        self._target_viz.append(np_)

        g = self.goal
        self._target_label = self._label(
            "target_txt", (gx, gy, C.TABLE_TOP_Z + 0.52),
            f"{g[2]:.1f} m/s  {spin_word(g[3], g[4])}", COL_TARGET, scale=0.5)
        self._target_viz.append(self._target_label)

    def draw_landing(self):
        """Where it actually went, next to where it was asked to go."""
        self._clear_viz()
        if self.landing is None or self.headless:
            return
        event, pos = self.landing
        if pos is None or event == "net":
            return
        lx, ly = float(pos[0]), float(pos[1])
        self._viz.append(self._ring("landed", lx, ly, 0.07, COL_LAND,
                                    z=C.TABLE_TOP_Z + 0.006, thickness=4.0))

        if event == "bounce":
            # The line between asked and got is the error, drawn at the scale
            # it actually happened rather than quoted in centimetres
            segs = LineSegs("miss")
            segs.set_thickness(2.5)
            segs.set_color(1.0, 0.55, 0.35, 0.8)
            segs.move_to(float(self.goal[0]), float(self.goal[1]),
                         C.TABLE_TOP_Z + 0.006)
            segs.draw_to(lx, ly, C.TABLE_TOP_Z + 0.006)
            np_ = self.render.attach_new_node(segs.create())
            np_.set_light_off()
            np_.set_transparency(TransparencyAttrib.M_alpha)
            self._viz.append(np_)
            d = float(np.hypot(lx - self.goal[0], ly - self.goal[1]))
            text = f"landed  {d * 100:.0f} cm off"
        else:
            text = "off the table"

        self._viz.append(self._label("landed_txt", (lx, ly, C.TABLE_TOP_Z + 0.30),
                                     text, COL_LAND, scale=0.5))

    def draw_breakdown(self):
        """Arrows at the contact point explaining the stroke."""
        self._clear_viz()
        if self.card is None:
            return
        card = self.card
        o = np.asarray(card["incoming"]["position"], dtype=float)
        n = np.asarray(card["blade"]["normal"], dtype=float)
        drive = float(card["motion"]["drive_along_normal"])
        brush_dir = np.asarray(card["motion"]["brush_direction"], dtype=float)
        brush = float(card["motion"]["brush_across_face"])
        v_in = np.asarray(card["incoming"]["velocity"], dtype=float)
        out_v = np.asarray(self.ball.vel, dtype=float)

        # Scale so the longest arrow is always about the same length on
        # screen. A fixed metres-per-m/s factor looked fine for a gentle
        # stroke and sent a hard one straight off the top of the frame,
        # taking its label with it. Ratios between arrows are preserved,
        # which is the part that carries meaning.
        speeds = [abs(drive), brush, card["incoming"]["speed"],
                  float(np.linalg.norm(out_v))]
        S = ARROW_MAX_LEN / max(max(speeds), 1e-6)

        items = [
            ("vin", o - v_in * S, v_in * S, COL_IN,
             f"in  {card['incoming']['speed']:.1f} m/s"),
            ("drive", o, n * drive * S, COL_DRIVE, f"DRIVE  {drive:+.1f}"),
            ("normal", o, n * 0.30, COL_NORMAL, "blade face"),
            ("brush", o, brush_dir * brush * S, COL_BRUSH, f"BRUSH  {brush:.1f}"),
            ("vout", o, out_v * S, COL_OUT,
             f"out  {np.linalg.norm(out_v):.1f} m/s"),
        ]

        # Every arrow starts from the same contact point and they are all
        # roughly the same length, so labels parked at the tips landed on top
        # of each other in one unreadable stripe. Stack them instead, evenly
        # spaced in height, and run a leader line back to the tip each one
        # belongs to. Order in `items` is the order they appear on screen.
        lift = np.linspace(0.30, -0.30, len(items))
        for i, (name, origin, vec, colour, text) in enumerate(items):
            segs, tip = arrow(name, origin, vec, colour)
            np_ = self.render.attach_new_node(segs.create())
            np_.set_light_off()
            np_.set_transparency(TransparencyAttrib.M_alpha)
            # Draw on top: the paddle and table sit right where these arrows
            # start, and half of them were hidden inside the geometry
            np_.set_depth_test(False)
            np_.set_bin("fixed", 40)
            self._viz.append(np_)

            anchor = np.asarray(tip, dtype=float) + np.array([0.0, 0.0, lift[i]])
            leader = LineSegs(name + "_leader")
            leader.set_thickness(1.4)
            leader.set_color(colour[0], colour[1], colour[2], 0.55)
            leader.move_to(*tip)
            leader.draw_to(*anchor)
            lnp = self.render.attach_new_node(leader.create())
            lnp.set_light_off()
            lnp.set_transparency(TransparencyAttrib.M_alpha)
            lnp.set_depth_test(False)
            lnp.set_bin("fixed", 40)
            self._viz.append(lnp)
            self._viz.append(self._label(name, anchor, text, colour, scale=0.38))

    # ------------------------------------------------------------ loop
    def update(self, task):
        now = time.time()
        self._advance_phase(now)

        # Auto-serve and the rally timeout in the base class are both measured
        # in real seconds, which the slow motion and the setup pause both
        # break. Holding the clock still while we are running the phases
        # keeps them from firing.
        if self.phase != INCOMING:
            self.last_serve_time = now

        # Scale time by rewriting last_time, so the base class sees a
        # slowed dt without needing to know about phases at all
        raw = min(1.0 / 30.0, max(1e-4, now - self.last_time))
        self.last_time = now - raw * self.time_scale

        result = super().update(task)
        self._tick_camera()
        self._update_panels()
        return result

    def _advance_phase(self, now):
        if self.phase == INCOMING and now > self.phase_until:
            # Never reached the blade -- nothing to show, go back and re-ask
            self.enter_setup()
        elif self.phase == BREAKDOWN:
            if not self._pulled_back and now > self.phase_until - PULLBACK_LEAD:
                self._pulled_back = True
                self.cam_move(*VIEW_WIDE, secs=PULLBACK_LEAD)
            if now > self.phase_until:
                self.enter_flight()
        elif self.phase == FLIGHT:
            settled = self.landing is not None and now > max(
                self.landing_at, self.flight_started) + LANDING_BEAT
            # A shot that neither lands nor goes out still has to end
            if settled or now > self.flight_started + 8.0:
                self.enter_result()
        elif self.phase == RESULT and now > self.phase_until:
            if self.auto_serve:
                self.request_serve()
            else:
                self.enter_setup()

    # ------------------------------------------------------------ HUD
    def _phase_line(self):
        return {
            SETUP: "choose a shot, then [enter] to serve",
            INCOMING: "ball on its way",
            BREAKDOWN: "STOPPED AT CONTACT  -  the stroke",
            FLIGHT: f"the return, at {FLIGHT_SCALE:.0%} speed",
            RESULT: "where it went",
        }[self.phase]

    def _update_panels(self):
        g = self.goal
        self.hud_title.setText(f"{self.agent_label}\n{self._phase_line()}")

        editing = self.phase in (SETUP, INCOMING)
        k = ("[arrows] ", "[W / S]  ", "[E / D]  ", "[Z / C]  ") if editing else ("",) * 4
        self.hud_ask[0].setText("YOU ASKED FOR")
        self.hud_ask[1].setText(
            f"  {k[0]}land       ({g[0]:+.2f}, {g[1]:+.2f}) m"
            f"      {'long' if g[0] > 0.85 else 'short'}, "
            f"{'right' if g[1] > 0.1 else ('left' if g[1] < -0.1 else 'middle')}")
        self.hud_ask[2].setText(f"  {k[1]}speed      {g[2]:.1f} m/s on arrival")
        self.hud_ask[3].setText(f"  {k[2]}topspin  {g[3]:+7.0f} rad/s"
                                f"      {spin_word(g[3], 0)}")
        self.hud_ask[4].setText(f"  {k[3]}sidespin {g[4]:+7.0f} rad/s")

        if self.keys_visible:
            extra = "   [1-9] show/hide a method" if len(self.contenders) > 1 else ""
            self.hud_keys.setText(
                "[click] set target   [enter] serve   [G] random   "
                f"[A] auto: {'ON' if self.auto_serve else 'off'}   "
                f"[space] replay   [F1] keys{extra}")
        else:
            self.hud_keys.setText("")

        self._update_compare_hud()

        # The floating target caption is on the far half; from the contact
        # close-up it drifts to the screen edge and lands on the text block
        if self._target_label is not None:
            (self._target_label.hide if self.phase == BREAKDOWN
             else self._target_label.show)()

        show_card = self.card is not None and self.phase in (BREAKDOWN, FLIGHT, RESULT)
        self._fit_plate(0.06 if show_card else 0.44)
        if not show_card:
            for h in (self.hud_blade, self.hud_drive, self.hud_brush,
                      self.hud_mu, self.hud_got):
                h.setText("")
            return

        # The stroke itself only matters while it is being explained; during
        # the flight and the result the eye belongs on the ball and the table
        if self.phase == BREAKDOWN:
            c = self.card
            mo, ct, bl = c["motion"], c["contact"], c["blade"]
            face = "closed (top forward)" if bl["tilt_from_vertical_deg"] > 0 \
                else "open (top back)"
            self.hud_blade.setText(
                f"BLADE      tilt {bl['tilt_from_vertical_deg']:+.1f} deg, {face}"
                f"    hitting the ball at {bl['angle_of_attack_deg']:.0f} deg")
            self.hud_drive.setText(
                f"DRIVE      {mo['drive_along_normal']:+.1f} m/s along the face"
                f"    -> pushes the ball forward")
            self.hud_brush.setText(
                f"BRUSH      {mo['brush_across_face']:.1f} m/s across the face"
                f"    -> creates the spin        (total swing {mo['speed']:.1f} m/s)")
            self.hud_mu.setText(
                f"GRIP       needs rubber friction {ct['required_mu']:.2f}"
                f" of the {C.FRICTION_PADDLE:.2f} available"
                f"   -> {'SLIPPING, spin saturated' if ct['sliding'] else 'grips fine'}")
        else:
            for h in (self.hud_blade, self.hud_drive, self.hud_brush, self.hud_mu):
                h.setText("")

        self.hud_got.setText(self._result_text())

    def _update_compare_hud(self):
        """The same numbers as the arcs, in the same colours."""
        if not self.hud_cmp:
            return
        rows = self.hud_cmp
        rows[0].setText(f"{'method':<15}{'land':>7}{'spd':>6}{'top':>6}"
                        f"{'goal':>7}")
        rows[0].setFg((0.72, 0.74, 0.80, 1))

        by_spec = {s.contender.spec: s for s in self.shots}
        for i, c in enumerate(self.contenders):
            row = rows[i + 1]
            s = by_spec.get(c.spec)
            mark = "" if c.on else "  (hidden)"
            if s is None:
                row.setText(f"{i + 1} {c.label:<13}{'--':>7}{mark}")
                row.setFg((0.45, 0.46, 0.50, 1))
                continue
            if s.measured:
                body = (f"{s.place_error * 100:>6.0f}cm"
                        f"{s.error[2]:>+6.1f}"
                        f"{s.error[3]:>+6.0f}"
                        f"{s.goal_error:>7.3f}")
            else:
                body = f"{'--':>8}{'--':>6}{'--':>6}{'--':>7}"
            row.setText(f"{i + 1} {c.label:<13}{body}"
                        f"{'' if s.landed_in else '  MISS'}{mark}")
            r, g, b, _ = c.rgba
            row.setFg((r, g, b, 1.0) if c.on else (r * 0.4, g * 0.4, b * 0.4, 1.0))

        tail = rows[len(self.contenders) + 1]
        tail.setText("lower goal error is better" if self.shots else "")
        tail.setFg((0.55, 0.56, 0.60, 1))

    def _result_text(self):
        """What happened, scored against what was asked for.

        The landing comes from the ball you just watched, not from the card's
        replay, so the text can never disagree with the mark on the table.
        Speed and spin are not measured in flight, so those two errors come
        from the replay of the same stroke.
        """
        if self.phase != RESULT:
            return ""
        verdict = self.last_result_text or "no result"
        r = self.card.get("result") if self.card else None
        has_replay = r is not None and np.isfinite(r["land_x"])
        event, pos = self.landing if self.landing is not None else (None, None)

        if event != "bounce" or pos is None:
            return f"GOT        {verdict}"

        dx, dy = float(pos[0] - self.goal[0]), float(pos[1] - self.goal[1])
        line = f"GOT        land ({pos[0]:+.2f}, {pos[1]:+.2f}) m    [{verdict}]"
        miss = (f"\nMISSED BY  {abs(dx) * 100:.0f} cm long/short,"
                f" {abs(dy) * 100:.0f} cm across")
        if has_replay:
            e = r["error_per_dim"]
            miss += (f", {abs(e[2]):.1f} m/s,"
                     f" {abs(e[3]):.0f} rad/s topspin,"
                     f" {abs(e[4]):.0f} rad/s sidespin")
        return line + miss


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs="+", default=None, metavar="SPEC",
                    help="planners to run side by side on the same ball. "
                         "'trainsize' and 'method' expand to ready-made sets; "
                         "the first one plays the ball for real")
    ap.add_argument("--list", action="store_true",
                    help="show everything that can be compared, then exit")
    ap.add_argument("--model", default=None, help="shorthand for --compare <model>")
    ap.add_argument("--oracle", action="store_true",
                    help="shorthand for --compare oracle")
    ap.add_argument("--goal", type=float, nargs=5, default=None,
                    metavar=("X", "Y", "SPEED", "TOPSPIN", "SIDESPIN"))
    ap.add_argument("--auto", action="store_true",
                    help="keep serving the same request without waiting")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.list:
        print("specs you can pass to --compare:\n")
        for spec, label in compare.available():
            print(f"  {spec:<34} {label}")
        print("\npresets:\n")
        for name, specs in compare.PRESETS.items():
            print(f"  {name:<34} {', '.join(specs)}")
        return

    specs = args.compare
    if not specs:
        specs = ["oracle"] if args.oracle or not args.model else [args.model]
    contenders = compare.build(specs)
    if not contenders:
        print("[AI] none of those could be loaded; try --list")
        return

    base.launch_wss()
    AIPlay(contenders, goal=args.goal, auto=args.auto, seed=args.seed).run()


if __name__ == "__main__":
    main()
