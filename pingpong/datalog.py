"""Stroke record logging.

The old states.csv schema was missing too much to train on:

* **No incoming ball state.** The old ``v_hit`` column held the *outgoing*
  velocity, i.e. the model's output; the incoming velocity that should be
  the input was never recorded.
* **No spin at all**, which is the central variable in table tennis.
* **Paddle motion reduced to a scalar "power"**, throwing away direction.
* **No outcome label** (out / net / good), only a landing point or the
  string out_of_bound.
* **No rally / stroke id**, so no sequence modelling and no RL.

The new schema is flat named columns, one row per stroke, readable
directly with pandas.
"""

import csv
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

SCHEMA_VERSION = 2

# Outcome labels
RESULT_IN = "in"              # landed successfully on the opponent's half
RESULT_OWN_SIDE = "own_side"  # landed on our own half (did not clear the net)
RESULT_OUT = "out"            # out of bounds
RESULT_NET = "net"            # into the net
RESULT_MISS = "miss"          # never made contact

# Data sources
SRC_HUMAN = "human"
SRC_AI = "ai"
SRC_RANDOM = "random"         # randomly sampled batch simulation


def _v3(name):
    return [f"{name}_x", f"{name}_y", f"{name}_z"]


COLUMNS = (
    ["schema", "rally_id", "stroke_id", "t", "source"]
    # --- model input: ball state at the moment of contact ---
    + _v3("in_pos") + _v3("in_vel") + _v3("in_spin")
    # --- model input: how the ball was served in the first place ---
    + _v3("serve_pos") + _v3("serve_vel")
    # --- action: the paddle ---
    + _v3("paddle_pos") + _v3("paddle_normal") + _v3("paddle_vel")
    # --- intermediate: ball state right after contact ---
    + _v3("out_vel") + _v3("out_spin")
    # --- label: landing point and outcome ---
    + _v3("landing")
    + ["result", "net_clearance", "flight_time", "bounces"]
)


@dataclass
class StrokeRecord:
    """One complete stroke."""

    rally_id: int = 0
    stroke_id: int = 0
    t: float = 0.0
    source: str = SRC_HUMAN

    in_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    in_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    in_spin: np.ndarray = field(default_factory=lambda: np.zeros(3))

    serve_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    serve_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    paddle_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    paddle_normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    paddle_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))

    out_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    out_spin: np.ndarray = field(default_factory=lambda: np.zeros(3))

    landing: Optional[np.ndarray] = None
    result: str = RESULT_IN
    net_clearance: float = float("nan")
    flight_time: float = float("nan")
    bounces: int = 0

    def to_row(self):
        def three(v):
            if v is None:
                return ["", "", ""]
            arr = np.asarray(v, dtype=float).reshape(3)
            return [f"{x:.6f}" for x in arr]

        row = [SCHEMA_VERSION, self.rally_id, self.stroke_id,
               f"{self.t:.4f}", self.source]
        for name in ("in_pos", "in_vel", "in_spin",
                     "serve_pos", "serve_vel",
                     "paddle_pos", "paddle_normal", "paddle_vel",
                     "out_vel", "out_spin", "landing"):
            row += three(getattr(self, name))
        row += [
            self.result,
            "" if np.isnan(self.net_clearance) else f"{self.net_clearance:.4f}",
            "" if np.isnan(self.flight_time) else f"{self.flight_time:.4f}",
            self.bounces,
        ]
        return row


class StrokeLogger:
    """Writes :class:`StrokeRecord` rows to CSV.

    Two-phase by design: :meth:`pending` registers the stroke at contact,
    then :meth:`commit` fills in the landing point once the ball has
    actually come down, so every row carries a complete label.
    """

    def __init__(self, path="strokes.csv", source=SRC_HUMAN):
        self.path = path
        self.source = source
        self.rally_id = 0
        self.stroke_id = 0
        self._pending: Optional[StrokeRecord] = None
        self._ensure_header()

    def _ensure_header(self):
        need_header = (
            not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        )
        if need_header:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(COLUMNS)

    # ------------------------------------------------------------------
    def new_rally(self):
        self.rally_id += 1
        self.stroke_id = 0
        # An unresolved stroke from the previous rally counts as a miss
        if self._pending is not None:
            self.commit(None, RESULT_MISS)

    def pending(self, record: StrokeRecord):
        """Register a stroke and wait for its landing result."""
        self.stroke_id += 1
        record.rally_id = self.rally_id
        record.stroke_id = self.stroke_id
        record.source = record.source or self.source
        self._pending = record

    def commit(self, landing, result, net_clearance=float("nan"),
               flight_time=float("nan"), bounces=0):
        """Attach the landing point and outcome, then write the row."""
        rec = self._pending
        self._pending = None
        if rec is None:
            return None
        rec.landing = None if landing is None else np.asarray(landing, dtype=float)
        rec.result = result
        rec.net_clearance = net_clearance
        rec.flight_time = flight_time
        rec.bounces = bounces
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(rec.to_row())
        return rec

    @property
    def has_pending(self):
        return self._pending is not None

    def discard(self):
        self._pending = None


# ---------------------------------------------------------------- batch
def write_batch(path, records: Sequence[StrokeRecord]):
    """Append many records at once (for batch-simulated training data)."""
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(COLUMNS)
        for rec in records:
            w.writerow(rec.to_row())


def load(path):
    """Read the dataset back: a DataFrame if pandas is available, else dicts."""
    try:
        import pandas as pd
        return pd.read_csv(path)
    except ImportError:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
