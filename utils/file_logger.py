"""Minimal drop-in replacement for the subset of wandb used by this repo.

Exposes ``init`` and ``log`` so existing ``wandb.init(...)`` / ``wandb.log(...)``
call sites keep working without the real wandb package. All logged metrics are
streamed as JSON lines to ``<out_dir>/metrics.jsonl`` so they can be replayed
for plotting later.

Usage (as a drop-in shim for ``import wandb``)::

    from utils import file_logger as wandb
    wandb.init(project="...", config={...})
    wandb.log({"loss": 0.1})

Configuration:
  - ``RIDESHARE_LOG_DIR`` env var overrides the output directory
    (default: ``./out``).

File layout (inside the output directory):
  - ``config.json``      -- config dict passed to ``init`` (plus project/run name)
  - ``metrics.jsonl``    -- one JSON object per ``log`` call, with a monotonic
                            ``_step`` field and a wallclock ``_time`` field
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping, Optional


class _Run:
    """Very small stand-in for the object returned by ``wandb.init``."""

    def __init__(self, out_dir: str, project: Optional[str], name: Optional[str]):
        self.out_dir = out_dir
        self.project = project
        self.name = name
        self._step = 0
        self._metrics_path = os.path.join(out_dir, "metrics.jsonl")
        # Truncate on init so repeated runs do not append stale data.
        # If you prefer to preserve history across runs, rename this file
        # before calling init again.
        self._fh = open(self._metrics_path, "w", buffering=1)  # line-buffered

    def log(self, data: Mapping[str, Any], step: Optional[int] = None) -> None:
        if step is None:
            step = self._step
            self._step += 1
        else:
            # keep an internal monotonic counter consistent with explicit steps
            self._step = max(self._step, step + 1)
        record = {"_step": step, "_time": time.time()}
        for k, v in data.items():
            record[k] = _to_jsonable(v)
        self._fh.write(json.dumps(record) + "\n")

    def finish(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass


# Module-level active run. Mirrors wandb's implicit global-run pattern so that
# plain ``wandb.log(...)`` calls (without an explicit run handle) still work.
_active_run: Optional[_Run] = None


def init(
    project: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    dir: Optional[str] = None,
    **_ignored: Any,
) -> _Run:
    """Initialize a file-backed logging run.

    Extra kwargs (e.g. ``entity``, ``tags``) are accepted and ignored for
    compatibility with call sites written against the real ``wandb.init``.
    """
    global _active_run

    out_dir = dir or os.environ.get("RIDESHARE_LOG_DIR", "./out")
    os.makedirs(out_dir, exist_ok=True)

    _active_run = _Run(out_dir=out_dir, project=project, name=name)

    config_payload = {
        "project": project,
        "name": name,
        "config": _to_jsonable(dict(config)) if config is not None else {},
    }
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config_payload, fh, indent=2, default=str)

    return _active_run


def log(data: Mapping[str, Any], step: Optional[int] = None) -> None:
    """Append a metrics record. Auto-initializes a default run if needed."""
    global _active_run
    if _active_run is None:
        _active_run = init()
    _active_run.log(data, step=step)


def finish() -> None:
    global _active_run
    if _active_run is not None:
        _active_run.finish()
        _active_run = None


# -- helpers -----------------------------------------------------------------

def _to_jsonable(x: Any) -> Any:
    """Best-effort conversion of values (incl. numpy/torch scalars) to JSON."""
    # Fast path for common primitive types.
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    # numpy scalars / arrays
    try:
        import numpy as np  # local import to keep this module cheap
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return x.item()
            return x.tolist()
    except Exception:
        pass
    # torch tensors
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.item()
            return x.detach().cpu().tolist()
    except Exception:
        pass
    # containers
    if isinstance(x, Mapping):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # Fallback to string repr; json.dump uses default=str at the top level,
    # but nested dicts still pass through here first.
    return str(x)
