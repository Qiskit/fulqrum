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

ERROR_TYPES = {
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
    "MemoryError": MemoryError,
}


def raise_on_every_rank(comm, error, error_type=ValueError, what="check"):


    errors = comm.allgather(error)
    found = [f"rank {rank}: {text}" for rank, text in enumerate(errors) if text]
    if found:
        raise error_type(f"{what} failed: " + "; ".join(found))


def run_on_every_rank(comm, what, function):


    result, error = None, None
    try:
        result = function()
    except tuple(ERROR_TYPES.values()) as exc:
        error = exc
    reports = comm.allgather(
        None if error is None else (type(error).__name__, str(error))
    )
    found = [(rank, report) for rank, report in enumerate(reports) if report]
    if found:
        error_type = ERROR_TYPES[found[0][1][0]]
        message = "; ".join(f"rank {rank}: {text}" for rank, (_name, text) in found)
        raise error_type(f"{what} failed: {message}") from error
    return result
