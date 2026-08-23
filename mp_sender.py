"""Webcam hand tracking -> UDP, so the game can place the paddle.

    python mp_sender.py

Sends {"x": ., "y": ., "z": .} at webcam rate to 127.0.0.1:5005, where
main_wss.py picks it up. All three are MediaPipe's normalised coordinates
in 0..1.

The game only uses **x and y** (left/right and up/down). MediaPipe's ``z``
is a rough depth relative to the wrist and is far too noisy to position a
paddle with, so depth is assisted by the game instead. It is still sent in
case a future estimator wants it (hand span is a better depth proxy than
the raw z).

Losing the hand
---------------
Tracking drops out constantly during play, and the cause is almost always
**motion blur**: a swinging hand smears across the frame and the detector
finds nothing. The settings here are tuned for that rather than for
accuracy on a still hand:

* ``model_complexity=0`` -- the lite model runs several times faster, so the
  capture rate goes up and each frame is exposed for less time;
* a low ``min_tracking_confidence`` keeps a lock through blurry frames
  instead of dropping and having to re-detect from scratch;
* a small capture resolution at a high frame rate, for the same reason.

The game also holds the last known position through gaps of up to 0.6 s,
so short dropouts no longer move the paddle at all.

Press ESC in the preview window to quit.
"""

import json
import socket
import time

import cv2
import mediapipe as mp

UDP_ADDR = ("127.0.0.1", 5005)

# Landmark 9 is the middle-finger MCP joint, near the centre of the palm.
# Much steadier than a fingertip, which jitters as the fingers move.
PALM_LANDMARK = 9

# Small and fast beats large and sharp here -- short exposure is what keeps
# a swinging hand trackable.
CAP_WIDTH, CAP_HEIGHT, CAP_FPS = 640, 480, 60


def open_camera():
    # CAP_DSHOW avoids the very slow MSMF backend startup on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open the webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # always work on the newest frame
    return cap


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,          # lite model: far higher frame rate
        min_detection_confidence=0.6,
        min_tracking_confidence=0.3,  # hold the lock through blurry frames
    )

    cap = open_camera()
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"camera {w:.0f}x{h:.0f}  ->  udp {UDP_ADDR[0]}:{UDP_ADDR[1]}   (ESC to quit)")

    tracked = False
    frames = 0
    found = 0
    fps = 0.0
    t_fps = time.perf_counter()

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            # Mirror, so moving your hand right moves the paddle right
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False          # lets MediaPipe skip a copy
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                found += 1
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark[PALM_LANDMARK]
                sock.sendto(
                    json.dumps({"x": lm.x, "y": lm.y, "z": lm.z}).encode(),
                    UDP_ADDR,
                )
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )
                if not tracked:
                    tracked = True
            elif tracked:
                tracked = False

            now = time.perf_counter()
            if now - t_fps >= 0.5:
                fps = frames / (now - t_fps)
                rate = found / max(frames, 1) * 100
                frames = found = 0
                t_fps = now
                print(f"\r{fps:5.1f} fps   hand seen {rate:5.1f}% of frames   ",
                      end="", flush=True)

            colour = (80, 200, 80) if tracked else (60, 60, 220)
            cv2.putText(frame, "TRACKING" if tracked else "NO HAND",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            cv2.putText(frame, f"{fps:.0f} fps", (10, int(h) - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("MediaPipe hand tracker", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        print()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        sock.close()


if __name__ == "__main__":
    main()
