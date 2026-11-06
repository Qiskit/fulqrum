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


import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .aabb_source_timing import format_source_timings, source_class
from .local_comm import mpi_namespace
from .diagnostics import (
    _dbg_hv_fingerprint,
    _dbg_ritz_resnorm,
    _hv_tail_fingerprint,
    _per_source_enabled,
    report,
)


@dataclass(frozen=True)
class RecorderSettings:


    comm_timing: bool
    parent_timing: bool
    source_timing: bool
    arm_timing: bool


class MatvecRecorder:


    def __init__(
        self,
        settings: RecorderSettings,
        comm,
        rank: int,
        iteration: int,
        compute_stream,
        copy_stream,
        source_tile: Callable,
    ):


        self.settings = settings
        self.comm = comm
        self.rank = rank
        self.iteration = iteration
        self.compute_stream = compute_stream
        self.copy_stream = copy_stream
        self.source_tile = source_tile
        self.timing: dict = {}
        self.started = time.perf_counter()
        self.compute_started = None
        self.kernel_events: dict = {}
        self.launches = 0
        self.exchange_pairs: list = []
        self.fetch_pairs: list = []
        self.gather_pairs: list = []
        self.arm_events = [] if settings.arm_timing else None
        self.parent_pairs: list = []
        self.source_records: list = []
        self.per_source = _per_source_enabled() and iteration == 2
        self.per_source_keys: list = []
        self.per_source_values = None

    def exchange_begin(self):

        return self.copy_stream.record() if self.settings.comm_timing else None

    def exchange_end(self, start, detail_pairs=None):


        end = self.copy_stream.record()
        if start is not None and end is not None:
            self.exchange_pairs.append((start, end))
            if detail_pairs is not None:
                detail_pairs.append((start, end))
        return end

    def checkpoint(self, label: str, d_out) -> None:

        _dbg_hv_fingerprint(
            d_out, self.compute_stream, self.comm, self.rank, label, self.iteration
        )

    def source_launched(self, src_rank: int, start, end, d_out, label: str) -> None:


        import cupy  # noqa: PLC0415

        self.kernel_events[src_rank] = (start, end)
        self.launches += 1
        if not self.per_source or len(self.per_source_keys) >= 256:
            return
        if self.per_source_values is None:
            self.per_source_values = cupy.empty((256 * 5,), dtype=cupy.float64)
        base = len(self.per_source_keys) * 5
        with self.compute_stream:
            flat = d_out.ravel()
            self.per_source_values[base + 0] = flat.sum()
            self.per_source_values[base + 1] = (flat * flat).sum()
            self.per_source_values[base + 2] = flat.min()
            self.per_source_values[base + 3] = flat.max()
            self.per_source_values[base + 4] = cupy.abs(flat).max()
        self.per_source_keys.append((int(src_rank), str(label)))

    def compute_done(self) -> None:

        sync_enter = time.perf_counter()
        self.compute_stream.synchronize()
        sync_exit = time.perf_counter()
        self.timing["compute"] = sync_exit - self.compute_started
        self.timing["sync"] = sync_exit - sync_enter
        self.timing["sync_enter_since_start"] = sync_enter - self.started
        self.timing["sync_exit_since_start"] = sync_exit - self.started

    def finish(self, d_in, d_out) -> None:


        _hv_tail_fingerprint(d_out, self.comm, self.rank, self.iteration)
        self._report_per_source()
        _dbg_ritz_resnorm(
            d_in, d_out, self.compute_stream, self.comm, self.rank, self.iteration
        )
        self._report_parent_timing()
        per_source = self._kernel_times()
        if self.settings.comm_timing:
            self._report_comm_timing()
        if self.settings.source_timing:
            self._report_source_timings(per_source)

    def finish_arms(self) -> None:

        if self.arm_events is None:
            return
        import cupy  # noqa: PLC0415

        arm_ms: dict = {}
        for arm, start, end in self.arm_events:
            arm_ms[arm] = arm_ms.get(arm, 0.0) + float(
                cupy.cuda.get_elapsed_time(start, end)
            )
        self.timing["arm_ms"] = arm_ms
        if self.rank == 0:
            report(
                f"Arm kernel timings: matvec={self.iteration}, rank=0, scope=local, "
                + ", ".join(f"{arm}={ms:.3f} ms" for arm, ms in sorted(arm_ms.items())),
                matvec=self.iteration,
            )

    def _report_per_source(self) -> None:

        if not self.per_source or not self.per_source_keys:
            return
        import cupy  # noqa: PLC0415

        MPI = mpi_namespace(self.comm)

        values = cupy.asnumpy(self.per_source_values[: len(self.per_source_keys) * 5])
        for i, (src, label) in enumerate(self.per_source_keys):
            local = values[i * 5 : i * 5 + 5]
            g_sum = self.comm.allreduce(float(local[0]), op=MPI.SUM)
            g_sumsq = self.comm.allreduce(float(local[1]), op=MPI.SUM)
            g_min = self.comm.allreduce(float(local[2]), op=MPI.MIN)
            g_max = self.comm.allreduce(float(local[3]), op=MPI.MAX)
            g_absmax = self.comm.allreduce(float(local[4]), op=MPI.MAX)
            if self.rank == 0:
                report(
                    f"Output vector fingerprint by source: matvec={self.iteration}, record={i:02d}, "
                    f"phase={label}, rank0_source={src}, scope=global, sum={g_sum:.17e}, "
                    f"sum_squares={g_sumsq:.17e}, min={g_min:.17e}, max={g_max:.17e}, "
                    f"max_abs={g_absmax:.17e}",
                    matvec=self.iteration,
                )

    def _report_parent_timing(self) -> None:

        if not self.settings.parent_timing:
            return
        import cupy  # noqa: PLC0415

        parent_ms = sum(
            float(cupy.cuda.get_elapsed_time(start, end))
            for _, start, end in self.parent_pairs
        )
        self.timing["aabb_parent_ms_total"] = parent_ms
        self.timing["aabb_parent_sources"] = len(self.parent_pairs)
        if self.rank == 0:
            report(
                f"Clique kernel timing: matvec={self.iteration}, rank=0, scope=local, "
                f"residual_parent_output={parent_ms:.3f} ms, sources={len(self.parent_pairs)}",
                matvec=self.iteration,
            )

    def _kernel_times(self) -> dict:

        import cupy  # noqa: PLC0415

        per_source = {
            src: float(cupy.cuda.get_elapsed_time(start, end))
            for src, (start, end) in self.kernel_events.items()
        }
        self.timing["kernel_ms_total"] = sum(per_source.values())
        self.timing["kernel_ms_max_per_source"] = (
            max(per_source.values()) if per_source else 0.0
        )
        self.timing["kernel_ms_min_per_source"] = (
            min(per_source.values()) if per_source else 0.0
        )
        return per_source

    def _report_comm_timing(self) -> None:

        import cupy  # noqa: PLC0415

        MPI = mpi_namespace(self.comm)

        self.copy_stream.synchronize()

        def pairs_ms(pairs) -> float:

            return sum(
                float(cupy.cuda.get_elapsed_time(start, end)) for start, end in pairs
            )

        timing = self.timing
        comm_ms = pairs_ms(self.exchange_pairs)
        kernel_ms = float(timing["kernel_ms_total"])
        wall_ms = float(timing["compute"]) * 1e3
        timing["comm_ms_total"] = comm_ms
        timing["comm_exchanges"] = len(self.exchange_pairs)
        fetch_ms = gather_ms = 0.0
        detail = ""
        two_level = timing.get("hierarchical_path", False)
        if two_level:
            fetch_ms = pairs_ms(self.fetch_pairs)
            gather_ms = pairs_ms(self.gather_pairs)
            timing["hierarchical_fetch_ms"] = fetch_ms
            timing["hierarchical_local_share_ms"] = gather_ms
            timing["hierarchical_local_ring_ms"] = gather_ms
            detail = f", remote_fetch={fetch_ms:.1f} ms, local_share={gather_ms:.1f} ms"
        timing["overlap_factor"] = (
            (comm_ms + kernel_ms) / wall_ms if wall_ms > 0 else 0.0
        )
        local = np.array(
            [comm_ms, kernel_ms, wall_ms, fetch_ms, gather_ms], dtype=np.float64
        )
        maxima = np.empty_like(local)
        self.comm.Allreduce(local, maxima, op=MPI.MAX)
        timing["comm_ms_max"] = float(maxima[0])
        timing["kernel_ms_max"] = float(maxima[1])
        timing["wall_ms_max"] = float(maxima[2])
        max_detail = ""
        if two_level:
            timing["hierarchical_fetch_ms_max"] = float(maxima[3])
            timing["hierarchical_local_share_ms_max"] = float(maxima[4])
            timing["hierarchical_local_ring_ms_max"] = float(maxima[4])
            max_detail = (
                f", remote_fetch={maxima[3]:.1f} ms, local_share={maxima[4]:.1f} ms"
            )
        if self.rank == 0:
            report(
                f"Matvec timing: matvec={self.iteration}, rank=0, exchange={comm_ms:.1f} ms, "
                f"kernels={kernel_ms:.1f} ms, kernel_plus_exchange={comm_ms + kernel_ms:.1f} ms, "
                f"compute_region={wall_ms:.1f} ms, overlap_factor={timing['overlap_factor']:.2f}, "
                f"exchange_to_compute_region_ratio={comm_ms / wall_ms if wall_ms > 0 else 0:.2f}, "
                f"exchanges={len(self.exchange_pairs)}{detail}. Independent rank maxima: "
                f"exchange={maxima[0]:.1f} ms, kernels={maxima[1]:.1f} ms, "
                f"compute_region={maxima[2]:.1f} ms{max_detail}",
                matvec=self.iteration,
            )

    def _report_source_timings(self, kernel_ms_per_source: dict) -> None:

        import cupy  # noqa: PLC0415

        grouped: dict = {}
        for record in self.source_records:
            grouped.setdefault(record["src"], []).append(record)
        own_alpha, own_beta = self.source_tile(self.rank)[4:6]
        rows = []
        for src in sorted(set(kernel_ms_per_source) | set(grouped)):
            records = grouped.get(src, [])
            ca, cb = self.source_tile(src)[4:6]
            covered = bool(records) and all(r["start"] is not None for r in records)
            aabb_ms = (
                sum(
                    float(cupy.cuda.get_elapsed_time(r["start"], r["end"]))
                    for r in records
                )
                if covered
                else None
            )
            parents = [r for r in records if r["split"] is not None]
            residual_ms = (
                sum(
                    float(cupy.cuda.get_elapsed_time(r["parent_start"], r["split"]))
                    for r in parents
                )
                if parents
                else None
            )
            parent_ms = (
                sum(
                    float(cupy.cuda.get_elapsed_time(r["split"], r["parent_end"]))
                    for r in parents
                )
                if parents
                else None
            )
            rows.append(
                dict(
                    src=src,
                    **{"class": source_class(ca == own_alpha, cb == own_beta)},
                    path="+".join(sorted({r["path"] for r in records})) or "uncovered",
                    launches=len(records),
                    kernel_ms=kernel_ms_per_source.get(src),
                    aabb_ms=aabb_ms,
                    residual_ms=residual_ms,
                    parent_ms=parent_ms,
                )
            )
        complete = bool(rows) and all(r["aabb_ms"] is not None for r in rows)
        timing = self.timing
        local = dict(
            rank=self.rank,
            sources=rows,
            kernel_ms=timing["kernel_ms_total"],
            comm_ms=timing.get("comm_ms_total"),
            region_ms=timing["compute"] * 1000.0,
            aabb_ms=sum(r["aabb_ms"] for r in rows) if complete else None,
            residual_ms=sum(r["residual_ms"] or 0.0 for r in rows),
            parent_ms=sum(r["parent_ms"] or 0.0 for r in rows),
            covered_sources=sum(r["aabb_ms"] is not None for r in rows),
        )
        timing["aabb_source_rows"] = rows
        timing["aabb_rank_local"] = local
        gathered = self.comm.gather(local, root=0)
        if self.rank == 0:
            report(
                "\n".join(format_source_timings(self.iteration, gathered)),
                matvec=self.iteration,
            )
