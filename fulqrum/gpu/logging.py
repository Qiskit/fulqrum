# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

_PACKAGE = "fulqrum.gpu"
_TRACE = "fulqrum.gpu.diagnostics"
_LEVELS = {
    "quiet": logging.WARNING,
    "summary": logging.INFO,
    "detailed": logging.DEBUG,
    "trace": logging.DEBUG,
}


class _TierFilter(logging.Filter):
    def __init__(self, mode, rank, ranks):
        super().__init__()
        self.mode, self.rank, self.ranks = mode, rank, ranks

    def filter(self, record):

        if record.levelno >= logging.WARNING:
            return True
        if self.mode == "quiet":
            return False
        if self.mode != "trace" and (
            record.name == _TRACE or record.name.startswith(_TRACE + ".")
        ):
            return False
        rank = getattr(record, "gpu_rank", self.rank)
        return self.ranks is None or rank in self.ranks


def configure_gpu_logging(
    mode="summary", *, stream=None, rank=0, ranks="primary", formatter=None
):
    """Configure GPU log output and return the installed stream handler.

    Modes control which records this handler displays:
        - ``quiet``: warnings and errors only.
        - ``summary``: INFO messages, warnings and errors.
        - ``detailed``: also normal DEBUG messages.
        - ``trace``: also records from ``fulqrum.gpu.diagnostics``.
    Warning and error records bypass rank filtering in every mode. Timing and
    debug configuration flags are unchanged.

    Reconfiguration replaces handlers previously installed by this function.
    Caller-owned handlers remain attached. The ``fulqrum.gpu`` logger's level
    is updated and propagation to parent loggers is disabled; root logger
    settings remain unchanged.

    Args:
        mode: ``quiet``, ``summary`` (default), ``detailed`` or ``trace``.
        stream: Text stream for output; defaults to stderr.
        rank: Fallback rank ID for filtering records without ``gpu_rank``
            metadata; defaults to 0. For MPI, pass this process's communicator rank.
        ranks: Routine-output filter: ``"primary"`` selects rank 0 (default),
            ``"all"`` selects every rank, or supply an iterable of rank IDs.
        formatter: Optional ``logging.Formatter``. The default prints only
            the message, without timestamps or level prefixes.

    Returns:
        The new ``logging.StreamHandler`` attached to ``fulqrum.gpu``.

    Raises:
        ValueError: The mode, rank or rank selection is invalid.
    """
    if mode not in _LEVELS:
        raise ValueError(f"mode must be one of {tuple(_LEVELS)}, got {mode!r}")
    rank = int(rank)
    if isinstance(ranks, str) and ranks == "primary":
        selected = frozenset({0})
    elif isinstance(ranks, str) and ranks == "all":
        selected = None
    elif isinstance(ranks, str):
        raise ValueError("ranks must be 'primary', 'all', or an iterable of rank IDs")
    else:
        selected = frozenset(int(value) for value in ranks)
    handler = logging.StreamHandler(stream)
    handler.setLevel(_LEVELS[mode])
    handler.setFormatter(formatter or logging.Formatter("%(message)s"))
    handler.addFilter(_TierFilter(mode, rank, selected))
    handler._fulqrum_gpu_owned = True
    logger = logging.getLogger(_PACKAGE)
    for previous in tuple(logger.handlers):
        if getattr(previous, "_fulqrum_gpu_owned", False):
            logger.removeHandler(previous)
            previous.close()
    logger.addHandler(handler)
    logger.setLevel(_LEVELS[mode])
    logger.propagate = False
    return handler


def format_memory(nbytes, *, unit="GiB"):

    if nbytes is None:
        return "unavailable"
    divisors = {"GiB": 2**30, "MiB": 2**20}
    return f"{nbytes / divisors[unit]:.3f} {unit}"


def _json_value(value):

    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value


    if type(value).__module__.split(".")[0] == "numpy":
        return _json_value(value.tolist())
    raise TypeError(
        f"Profile data contains an unsupported value: {type(value).__name__}"
    )


def gpu_profile_record(operator):
    """Return a JSON-ready snapshot of existing per-process records; this enables no profiling
    and gathers nothing across MPI ranks."""
    entries = []
    for index, op in enumerate(getattr(operator, "_ranks", (operator,))):
        comm = getattr(op, "_comm", None)
        rank = getattr(op, "_rank", None)
        if rank is None:
            rank = comm.Get_rank() if comm is not None else index
        entry = {
            "rank": int(rank),
            "devices": getattr(op, "_gpu_ids", ()),
            "construction": getattr(op, "construction_timing", {}),
            "last_matvec": getattr(op, "last_timing", {}),
            "configuration": getattr(op, "run_record", {}),
        }
        plan = getattr(op, "plan", None)
        if plan is not None:
            entry["layout"] = plan.to_dict()
            entry["device_storage_bytes"] = op.device_bytes
        entries.append(entry)
    return _json_value(
        {
            "schema_version": 1,
            "event": "gpu_profile_snapshot",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scope": "process",
            "ranks": entries,
            "units": {
                "bytes": "B",
                "*_ms*": "ms",
                "arm_ms": "ms",
                "wall": "s",
                "compute": "s",
                "exchange": "s",
                "sync": "s",
                "h2d": "s",
                "d2h": "s",
            },
        }
    )


def write_gpu_profile(operator, destination):
    """Write an existing profile snapshot as JSON to a path or stream and return it. Use a
    separate file per MPI process."""
    record = gpu_profile_record(operator)
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if hasattr(destination, "write"):
        destination.write(text)
    else:
        Path(destination).write_text(text, encoding="utf-8")
    return record
