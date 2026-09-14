"""Record the simulator to a GIF for the README.

    python record_demo.py compare --seconds 16
    python record_demo.py rally   --seconds 14
    python record_demo.py play    --seconds 12

Renders offscreen, so it needs no window and no phone. GIF rather than MP4
because GitHub renders an animated GIF inline in a README and treats a
committed .mp4 as a file to download.

`rally` drives *both* paddles with the policy. There is no way to fake a
person convincingly and no reason to try: captioned honestly it shows the
contact model and the mirrored opponent, which is the part worth seeing.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from panda3d.core import Filename, loadPrcFileData

WIDTH, HEIGHT = 960, 540
loadPrcFileData("", f"win-size {WIDTH} {HEIGHT}")
loadPrcFileData("", "window-type offscreen")
loadPrcFileData("", "audio-library-name null")

import ai_play                                    # noqa: E402
import main_wss as base                           # noqa: E402
import rally_ai                                   # noqa: E402
from pingpong import compare, constants as C, dataset, datalog   # noqa: E402

# The interactive wide view leaves headroom above the table, which is right
# when you are tracking a ball by eye and wasted in a fixed-frame recording.
ai_play.VIEW_WIDE = ((-1.30, -3.30, 2.00), (0.08, 0.0, C.TABLE_H + 0.30))

SCRATCH = os.path.join(os.environ.get("TEMP", "."), "pingpong_demo_frames")


def _tmp_logger(name):
    return datalog.StrokeLogger(os.path.join(os.environ.get("TEMP", "."), name))


def build_compare(args):
    cs = compare.build(["mdn_200000_i8p256", "mdn_20000_i8p256",
                        "mdn_2000_i8p256", "oracle"])
    app = ai_play.AIPlay(cs, goal=[0.95, 0.28, 6.0, 200.0, 40.0],
                         auto=True, seed=3)
    app.logger = _tmp_logger("demo_a.csv")
    app.request_serve()
    return app, None


def build_play(args):
    cs = compare.build(["mdn_200000_i8p256"])
    app = ai_play.AIPlay(cs, auto=True, seed=1)
    app.logger = _tmp_logger("demo_b.csv")
    app.request_serve()
    return app, None


def build_rally(args):
    cs = compare.build(["mdn_200000_i8p256"])
    app = rally_ai.RallyAI(cs[0], difficulty=0.55, seed=2)
    app.logger = _tmp_logger("demo_c.csv")
    app.opp_logger = _tmp_logger("demo_d.csv")

    # Both ends are the same policy. The near paddle is placed at the
    # predicted contact and given the stroke the model chose, which is what
    # ai_play already does for the far end.
    agent = cs[0].agent
    plan = {"a": None}
    app._contact_allowed = lambda: True

    def control(st, dt):
        c = app.contact
        if c is not None and app.last_hitter != "human":
            if plan["a"] is None:
                s = dataset.encode_state(c.pos[None, :], c.vel[None, :],
                                         c.spin[None, :]).astype(np.float32)
                goal = np.array([[0.85, 0.10, 5.5, 170.0, 0.0]], np.float32)
                plan["a"] = agent.act(s, goal)[0]
            n, u = dataset.decode_action(plan["a"][None, :])
            app.paddle.pos = c.pos.copy()
            app.paddle.normal = n[0]
            app.paddle.velocity = u[0]
            app.apply_paddle_transform(base.quat.look_quat(n[0]))
        else:
            plan["a"] = None
        app._opp_prev = app.opp.pos.copy()
        app._control_opponent(time.time())

    app._control_paddle = control

    # Side on, so the exchange reads as an exchange. The play camera sits
    # behind the near paddle, which is right when you are the near paddle and
    # useless when the point of the shot is that it comes back.
    app.camera.set_pos(-1.15, -3.55, 2.05)
    app.camera.look_at(0.10, 0.0, C.TABLE_H + 0.28)
    # Nothing is plugged in during a recording, so the connection readout is
    # noise rather than information
    for h in (app.hud_conn, app.hud_help, app.hud_swing):
        h.hide()
    return app, None


SCENES = {"compare": build_compare, "play": build_play, "rally": build_rally}


def capture(app, seconds, stride, warmup=1.0):
    """Step the game, saving every `stride`-th rendered frame."""
    if os.path.isdir(SCRATCH):
        shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH, exist_ok=True)

    t0 = time.time()
    while time.time() - t0 < warmup:          # let the first serve get going
        app.taskMgr.step()

    files, i, n = [], 0, 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.taskMgr.step()
        i += 1
        if i % stride:
            continue
        path = os.path.join(SCRATCH, f"f{n:05d}.png")
        app.win.save_screenshot(Filename.from_os_specific(path))
        files.append(path)
        n += 1
    return files


def to_gif(files, out, fps, scale, colors=192):
    from PIL import Image

    if not files:
        raise SystemExit("no frames captured")
    imgs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)),
                           Image.LANCZOS)
        # One adaptive palette per frame keeps the table gradient clean; the
        # GIF optimiser still de-duplicates unchanged regions between frames
        imgs.append(im.quantize(colors=colors, method=Image.MEDIANCUT))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    return out


def to_mp4(files, out, fps):
    """Best-effort MP4 as well: better quality per byte, for a release asset."""
    if not shutil.which("ffmpeg"):
        return None
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-i", os.path.join(SCRATCH, "f%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
           "-movflags", "+faststart", out]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return out
    except Exception as e:
        print(f"  (mp4 skipped: {e})")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stride", type=int, default=1, help="render frames per saved frame")
    # Offscreen rendering plus a screenshot per frame runs at about 12 fps,
    # so stride 1 at fps 12 plays back at roughly real time
    ap.add_argument("--scale", type=float, default=0.45)
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join("docs", f"demo_{args.scene}.gif")
    app, _ = SCENES[args.scene](args)
    print(f"[demo] recording {args.scene} for {args.seconds:.0f}s ...")
    files = capture(app, args.seconds, args.stride)
    print(f"[demo] {len(files)} frames")

    gif = to_gif(files, out, args.fps, args.scale, args.colors)
    print(f"  wrote {gif}  ({os.path.getsize(gif) / 1e6:.1f} MB)")
    mp4 = to_mp4(files, out.replace(".gif", ".mp4"), args.fps)
    if mp4:
        print(f"  wrote {mp4}  ({os.path.getsize(mp4) / 1e6:.1f} MB)")
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
