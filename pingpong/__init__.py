"""Table tennis simulation core.

This package deliberately has **no Panda3D dependency** so it can run
headless in batch, which is what the future data-collection and model
training work needs. Rendering and input stay in main_wss.py.

Modules:
    constants -- physical constants and table geometry
    quat      -- quaternion math (W3C DeviceOrientation -> world)
    physics   -- ball flight / collision model, NumPy vectorised
    swing     -- paddle velocity estimation from phone IMU
    stroke    -- stance / swing separation, the paddle motion state machine
    datalog   -- stroke record schema
"""

from . import constants, quat, physics, swing, stroke, datalog  # noqa: F401

__all__ = ["constants", "quat", "physics", "swing", "stroke", "datalog"]
