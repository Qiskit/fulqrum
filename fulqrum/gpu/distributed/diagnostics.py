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

import logging
import os

from .local_comm import mpi_namespace

REPORT_LOGGER = "fulqrum.gpu.diagnostics"


def report(line: str, *, rank: int = 0, matvec=None) -> None:


    logging.getLogger(REPORT_LOGGER).debug(
        line, extra={"gpu_rank": rank, "gpu_matvec": matvec}
    )


def _debug_matvec_enabled() -> bool:

    val = os.environ.get("_FULQRUM_DEBUG_MATVEC", "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _hv_tail_enabled() -> bool:

    return os.environ.get("_FULQRUM_DEBUG_MATVEC", "").strip().lower() == "tail"


def _per_source_enabled() -> bool:

    return os.environ.get("_FULQRUM_DEBUG_MATVEC", "").strip().lower() == "per_source"


def _hv_tail_fingerprint(d_out_local, comm, rank: int, iter_idx: int) -> None:


    if not _hv_tail_enabled():
        return
    if iter_idx not in (1, 2, 3):
        return
    import cupy  # noqa: PLC0415

    MPI = mpi_namespace(comm)

    flat = d_out_local.ravel()
    local_sum = float(flat.sum().get())
    local_sumsq = float((flat * flat).sum().get())
    local_min = float(flat.min().get())
    local_max = float(flat.max().get())
    local_absmax = float(cupy.abs(flat).max().get())
    g_sum = comm.allreduce(local_sum, op=MPI.SUM)
    g_sumsq = comm.allreduce(local_sumsq, op=MPI.SUM)
    g_min = comm.allreduce(local_min, op=MPI.MIN)
    g_max = comm.allreduce(local_max, op=MPI.MAX)
    g_absmax = comm.allreduce(local_absmax, op=MPI.MAX)
    if rank == 0:
        report(
            f"Output vector fingerprint: matvec={iter_idx}, scope=global, "
            f"sum={g_sum:.17e}, sum_squares={g_sumsq:.17e}, "
            f"min={g_min:.17e}, max={g_max:.17e}, max_abs={g_absmax:.17e}",
            matvec=iter_idx,
        )


def _dbg_print(rank: int, msg: str) -> None:

    if _debug_matvec_enabled():
        report(f"{msg} rank={rank}", rank=rank)


def _dbg_hv_fingerprint(
    d_out_local,
    compute_stream,
    comm,
    rank: int,
    phase: str,
    iter_idx: int,
) -> None:


    if not _debug_matvec_enabled():
        return
    if iter_idx not in (1, 2, 3, 4, 5):
        return
    import cupy  # noqa: PLC0415

    MPI = mpi_namespace(comm)

    compute_stream.synchronize()
    flat = d_out_local.ravel()
    local_sum = float(flat.sum().get())
    local_sumsq = float((flat * flat).sum().get())
    local_min = float(flat.min().get())
    local_max = float(flat.max().get())
    local_absmax = float(cupy.abs(flat).max().get())

    g_sum = comm.allreduce(local_sum, op=MPI.SUM)
    g_sumsq = comm.allreduce(local_sumsq, op=MPI.SUM)
    g_min = comm.allreduce(local_min, op=MPI.MIN)
    g_max = comm.allreduce(local_max, op=MPI.MAX)
    g_absmax = comm.allreduce(local_absmax, op=MPI.MAX)

    if rank == 0:
        report(
            f"Output vector fingerprint: matvec={iter_idx}, phase={phase}, scope=global, "
            f"sum={g_sum:.17e}, sum_squares={g_sumsq:.17e}, "
            f"min={g_min:.17e}, max={g_max:.17e}, max_abs={g_absmax:.17e}",
            matvec=iter_idx,
        )


def _dbg_ring_slot_fingerprint(
    d_slot,
    comm,
    rank: int,
    phase: str,
    step: int,
    peer_world: int,
    iter_idx: int,
) -> None:


    if not _debug_matvec_enabled():
        return
    if iter_idx not in (1, 2, 3, 4, 5):
        return
    import cupy  # noqa: PLC0415

    MPI = mpi_namespace(comm)

    flat = d_slot.ravel()
    local_sum = float(flat.sum().get())
    local_sumsq = float((flat * flat).sum().get())
    local_absmax = float(cupy.abs(flat).max().get())
    g_sum = comm.allreduce(local_sum, op=MPI.SUM)
    g_sumsq = comm.allreduce(local_sumsq, op=MPI.SUM)
    g_absmax = comm.allreduce(local_absmax, op=MPI.MAX)
    if rank == 0:
        report(
            f"Received ring slot fingerprint: matvec={iter_idx}, phase={phase}, step={step}, "
            f"scope=global, sum={g_sum:.17e}, sum_squares={g_sumsq:.17e}, "
            f"max_abs={g_absmax:.17e}, rank0_peer={peer_world}",
            matvec=iter_idx,
        )


def _dbg_v_fingerprint(d_v_local, comm, rank: int, iter_idx: int) -> None:


    if not _debug_matvec_enabled():
        return
    if iter_idx not in (1, 2, 3, 4, 5):
        return
    import cupy  # noqa: PLC0415

    MPI = mpi_namespace(comm)

    flat = d_v_local.ravel()
    local_sum = float(flat.sum().get())
    local_sumsq = float((flat * flat).sum().get())
    local_absmax = float(cupy.abs(flat).max().get())
    g_sum = comm.allreduce(local_sum, op=MPI.SUM)
    g_sumsq = comm.allreduce(local_sumsq, op=MPI.SUM)
    g_absmax = comm.allreduce(local_absmax, op=MPI.MAX)
    if rank == 0:
        report(
            f"Input vector fingerprint: matvec={iter_idx}, scope=global, "
            f"sum={g_sum:.17e}, sum_squares={g_sumsq:.17e}, "
            f"max_abs={g_absmax:.17e}, norm={g_sumsq**0.5:.17e}",
            matvec=iter_idx,
        )


def _dbg_ritz_resnorm(
    d_v_local,
    d_out_local,
    compute_stream,
    comm,
    rank: int,
    iter_idx: int,
) -> None:


    if not _debug_matvec_enabled():
        return
    MPI = mpi_namespace(comm)

    compute_stream.synchronize()
    v = d_v_local.ravel()
    h = d_out_local.ravel()
    local_vv = float((v * v).sum().get())
    local_vh = float((v * h).sum().get())
    local_hh = float((h * h).sum().get())
    g_vv = comm.allreduce(local_vv, op=MPI.SUM)
    g_vh = comm.allreduce(local_vh, op=MPI.SUM)
    g_hh = comm.allreduce(local_hh, op=MPI.SUM)
    if rank != 0:
        return
    if g_vv <= 0.0:
        ritz = float("nan")
        resnorm_rel = float("nan")
        resnorm_abs = float("nan")
    else:
        ritz = g_vh / g_vv
        r2 = g_hh - ritz * g_vh
        resnorm_abs = (r2**0.5) if r2 > 0 else 0.0
        resnorm_rel = resnorm_abs / (g_vv**0.5)
    report(
        f"Matvec residual diagnostic: matvec={iter_idx}, "
        f"rayleigh_quotient={ritz:.17e}, residual_norm_abs={resnorm_abs:.6e}, "
        f"residual_norm_per_input_norm={resnorm_rel:.6e}, "
        f"input_norm_squared={g_vv:.6e}, output_norm_squared={g_hh:.6e}, "
        f"input_dot_output={g_vh:.6e}",
        matvec=iter_idx,
    )
