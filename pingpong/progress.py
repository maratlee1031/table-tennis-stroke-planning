"""Progress bars for the long-running steps.

Data generation runs for tens of minutes; without feedback there is no way to
tell a slow step from a hung one. Wrapped rather than used directly so tqdm
stays optional -- if it is missing everything still runs, just quietly.
"""

import sys
import time

try:
    from tqdm.auto import tqdm as _tqdm
    HAVE_TQDM = True
except ImportError:                                   # pragma: no cover
    _tqdm = None
    HAVE_TQDM = False


class _Fallback:
    """Minimal stand-in: a percentage and an ETA on one rewritten line."""

    def __init__(self, total, desc, unit):
        self.total, self.desc, self.unit = total, desc, unit
        self.n = 0
        self.t0 = time.perf_counter()

    def update(self, k=1):
        self.n += k
        el = time.perf_counter() - self.t0
        frac = self.n / self.total if self.total else 0.0
        eta = (el / frac - el) if frac > 1e-9 else 0.0
        sys.stdout.write(
            f"\r  {self.desc}: {frac * 100:5.1f}%  "
            f"{self.n:,}/{self.total:,} {self.unit}  "
            f"elapsed {el:5.1f}s  eta {eta:5.1f}s   ")
        sys.stdout.flush()

    def set_postfix_str(self, s):
        self.postfix = s

    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def bar(total, desc="", unit="it", leave=True, enabled=True):
    """A progress bar, or a no-op context manager when disabled."""
    if not enabled:
        return _Null()
    if HAVE_TQDM:
        return _tqdm(total=total, desc=desc, unit=unit, leave=leave,
                     ncols=96, bar_format="  {desc:<28} {percentage:5.1f}%|{bar}| "
                                          "{n_fmt}/{total_fmt} [{elapsed}<{remaining}"
                                          "{postfix}]")
    return _Fallback(total, desc, unit)


class _Null:
    def update(self, k=1):
        pass

    def set_postfix_str(self, s):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass
